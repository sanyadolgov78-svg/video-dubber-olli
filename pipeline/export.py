import os
import subprocess
import json
import tempfile
import shutil
import logging
import math

import numpy as np
import soundfile as sf

from . import transcript

try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None
    ImageFilter = None


def _probe(video_path):
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", video_path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    data = json.loads(out)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return has_audio


def _dubbed_mp4_valid(out_path):
    """Проверяет, что итоговый mp4 реально содержит и видео, и аудио-потоки
    (не пустышка/битый после падения ffmpeg на x264 при нехватке памяти).
    Возвращает True, если структура корректна, иначе False."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", out_path]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        streams = json.loads(out).get("streams", [])
        types = [s.get("codec_type") for s in streams]
        # Нужны и видеопоток, и аудиопоток. Битый экспорт (malloc-fail) даёт
        # файл только с одной дорожкой — он не подходит для повторного использования.
        return "video" in types and "audio" in types
    except Exception:
        return False


def get_video_duration(video_path):
    """Возвращает длительность видео (сек) через ffprobe; при неудаче — None."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        return float(out)
    except Exception:
        return None


def _ref_voice_score(wav_path):
    """Оценка пригодности фрагмента как референса голоса (для XTTS).

    Возвращает (score, snr_db, music_frac) или (0.0, 0.0, 0.0) при ошибке.

    В отличие от чистой энергии учитывает качество записи:
      * плотность речевых окон (доля не-тишины);
      * уровень голоса;
      * SNR — отношение речи к фону внутри фрагмента (по тихим окнам);
      * music_frac — доля «музыкальных/шумовых» окон по спектральной
        плоскостности (у речи выраженные форманты -> низкая плоскостность,
        у шума/музыки -> высокая).
    """
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        return 0.0, 0.0, 0.0
    if audio is None or len(audio) == 0:
        return 0.0, 0.0, 0.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = max(1, int(sr * 0.05))
    n = len(audio) // win
    if n == 0:
        return 0.0, 0.0, 0.0
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return 0.0, 0.0, 0.0
    thr = max(0.01, 0.05 * peak)
    voiced = rms > thr
    if not voiced.any():
        return 0.0, 0.0, 0.0
    density = float(np.mean(voiced))
    level = float(np.mean(rms[voiced]))

    # SNR: речевые окна против фоновых (тихих).
    noise_rms = float(np.mean(rms[~voiced])) if (~voiced).any() else 0.0
    snr_db = 20.0 * float(np.log10((level + 1e-9) / (noise_rms + 1e-9)))

    # Спектральная плоскостность на речевых окнах: речь -> низкая, шум -> высокая.
    vframes = frames[voiced][: min(len(frames[voiced]), 400)]
    music_frac = 0.0
    if len(vframes) > 0:
        try:
            win_fft = vframes * np.hanning(vframes.shape[1])
            mag = np.abs(np.fft.rfft(win_fft, axis=1)) + 1e-10
            flat = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
            music_frac = float(np.mean(flat > 0.30))   # порог откалиброван эмпирически
        except Exception:
            music_frac = 0.0

    # Итог: плотность * уровень, усиленное чистотой (SNR) и без штрафа за музыку.
    snr_factor = min(1.0, max(0.25, snr_db / 20.0))     # 20+ дБ -> 1.0
    clean_factor = max(0.15, 1.0 - music_frac)          # музыка/шум -> понижение
    score = density * level * (0.5 + 0.5 * snr_factor) * clean_factor
    return float(score), float(snr_db), float(music_frac)


def _trim_silence(wav_path):
    """Обрезает тишину по краям референса (XDTT не любит пустые паузы в начале/конце)."""
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        win = max(1, int(sr * 0.05))
        n = len(audio) // win
        if n == 0:
            return wav_path
        rms = np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))
        peak = float(np.max(np.abs(audio)))
        thr = max(0.01, 0.05 * peak) if peak > 0 else 0.01
        voiced = rms > thr
        if not voiced.any():
            return wav_path
        idx = np.flatnonzero(voiced)
        pad = int(0.15 * sr)
        start = max(0, idx[0] * win - pad)
        end = min(len(audio), (idx[-1] + 1) * win + pad)
        keep = audio[start:end]
        if len(keep) >= int(4.0 * sr):
            sf.write(wav_path, keep, sr, subtype="PCM_16")
    except Exception:
        pass
    return wav_path


def _rip_chunk(video_path, st, en, tmp):
    """Вырезает (st,en) из аудио видео в tmp .wav (моно 22050 Гц). True при успехе."""
    dur = en - st
    if dur <= 0:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-ss", "%.3f" % st, "-i", video_path,
        "-t", "%.3f" % dur,
        "-vn", "-ac", "1", "-ar", "22050",
        "-c:a", "pcm_s16le", tmp,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        return bool(proc and proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0)
    except Exception:
        return False


def _merged_chunks(video_path, segments, max_ref=25.0):
    """Собирает непрерывные блоки речи по таймкодам сегментов (склейка реплик < 1с).

    Возвращает отсортированный по длине список (start, end). Если блоков по 3+ сек
    нет — расширяет самый длинный сегмент паузами и возвращает его.
    """
    boxes = []
    for s in segments:
        try:
            st = float(s.get("start", 0.0))
            en = float(s.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if en > st and (s.get("text") or "").strip():
            boxes.append((st, en))
    if not boxes:
        return []

    chunks = []
    cur_s, cur_e = boxes[0]
    for st, en in boxes[1:]:
        if st - cur_e <= 1.0 and en - cur_s <= max_ref:
            cur_e = max(cur_e, en)
        else:
            chunks.append((cur_s, cur_e))
            cur_s, cur_e = st, en
    chunks.append((cur_s, cur_e))
    chunks = [c for c in chunks if c[1] - c[0] >= 3.0]
    chunks.sort(key=lambda c: c[1] - c[0], reverse=True)

    if not chunks:
        # Блоков >= 3 сек нет — берём самый длинный сегмент и расширяем паузами.
        single = max(boxes, key=lambda b: b[1] - b[0])
        st = max(0.0, single[0] - 0.5)
        en = single[1] + 0.5
        total = get_video_duration(video_path) or (en + 1.0)
        en = min(en, total)
        if en - st > max_ref:
            en = st + max_ref
        chunks = [(st, max(st, en))]
    return chunks


def speaker_reference_candidates(video_path, segments, logger, limit=5, min_dur=6.0):
    """Извлекает речевые блоки видео и ранжирует их по качеству записи.

    Оценка: плотность речи * уровень, с поправкой на SNR (сигнал/фон),
    штрафом за музыку/шум (спектральная плоскостность) и учётом длительности
    (XTTS тем точнее клонирует, когда референс не короче min_dur секунд).
    Возвращает список (start, end) — от лучшего к худшему (до limit).
    """
    chunks = _merged_chunks(video_path, segments)
    if not chunks:
        return []

    scored = []  # (score, st, en)
    for st, en in chunks[: max(1, limit * 4)]:
        fd, cand = tempfile.mkstemp(suffix=".wav", prefix="ref_")
        os.close(fd)
        ok = _rip_chunk(video_path, st, en, cand)
        if ok:
            score, snr_db, music = _ref_voice_score(cand)
        else:
            score, snr_db, music = 0.0, 0.0, 0.0
        try:
            if os.path.exists(cand):
                os.remove(cand)
        except Exception:
            pass
        # Умножаем на длительность: короткий референс хуже для клонирования.
        dur_factor = 0.35 + 0.65 * min(1.0, max(0.0, en - st) / max(1.0, min_dur))
        score *= dur_factor
        logger.info(
            "  кандидат %.1f-%.1f с (%.1f с): score=%.4f SNR=%.1f дБ шум/музыка=%.0f%%",
            st, en, en - st, score, snr_db, music * 100.0
        )
        scored.append((score, st, en))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [s for s in scored if s[0] > 0]
    if not chosen:
        return []
    return [(st, en) for _score, st, en in chosen[:limit]]


def extract_speaker_reference(video_path, segments, out_wav, logger, force_range=None):
    """Вырезает чистый речевой фрагмент спикера из видео как референс XTTS.

    По таймкодам сегментов собирает непрерывные блоки речи, извлекает кандидатов
    и выбирает самый речевой по энергии (плотность + громкость), затем обрезает
    тишину по краям. Итог — моно .wav 22050 Гц (формат XTTS).

    force_range=(start,end) заставляет вырезать именно этот блок (используется для
    альтернативного варианта голоса). Возвращает путь к файлу или None.
    Кэшируется: если файл уже существует — не пересоздаёт.
    """
    if out_wav and os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
        logger.info(f"Референс голоса уже есть, использую: {out_wav}")
        return out_wav

    if force_range is not None:
        candidates = [tuple(force_range)]
    else:
        candidates = speaker_reference_candidates(video_path, segments, logger, limit=3)
        if not candidates:
            logger.info("Нет сегментов с таймкодами — референс голоса не вырезан.")
            return None
        # XTTS тем точнее, когда референс длиннее: среди кандидатов берём самый
        # длинный с оценкой выше нуля, иначе — лучший по качеству.
        good = [c for c in candidates if c[1] - c[0] >= 6.0]
        if good:
            st, en = max(good, key=lambda c: c[1] - c[0])
        else:
            st, en = candidates[0]
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="ref_", dir=os.path.dirname(out_wav))
    os.close(fd)

    try:
        if not _rip_chunk(video_path, st, en, tmp):
            logger.info("Не удалось вырезать референс речи (ffmpeg).")
            return None
        dur = get_video_duration(tmp)
        if not dur or dur < 3.0:
            logger.info(f"Референс речи вышел мусором ({dur} сек) — отброшен.")
            return None
        _trim_silence(tmp)
        shutil.move(tmp, out_wav)
        final_dur = get_video_duration(out_wav) or dur
        logger.info(
            f"Референс голоса спикера: {out_wav} ({final_dur:.2f} с, "
            f"фрагмент {st:.1f}–{en:.1f} с видео, отобран по энергии речи)"
        )
        return out_wav
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _srt_max_line_len(srt_path):
    """Максимальная длина строки текста в SRT (число символов одной реплики).

    Позволяет оценить, сколько строк займёт самая длинная реплика на кадре.
    """
    max_len = 0
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return 0
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            line = line.strip()
            if line and "-->" not in line and not line.isdigit():
                max_len = max(max_len, len(line))
    return max_len


def _balance_lines(words, cap, max_lines):
    """Раскладывает слова на минимально возможное число строк, поровну.

    Строк ровно столько, сколько нужно, чтобы текст влез в cap символов
    (но не больше max_lines). Внутри строки слова чередуются равномерно, поэтому
    строки получаются одинаковой длины — без рваных краёв и без растяжки
    пробелов. Возвращает список строк.
    """
    words = [w for w in (words or []) if w.strip()]
    if not words:
        return [""]
    total = sum(len(w) for w in words) + len(words) - 1
    if cap and cap > 0:
        need = max(1, -(-total // cap))
    else:
        need = 1
    n = max(1, min(max_lines or need, need))
    if n == 1:
        return [" ".join(words)]
    # Раскладываем по n строкам: кладём первую строку наполовину короче
    # целевой, чтобы длинные слова ушли в начало, а короткие — в конец.
    target = total / n
    lines, cur, cur_len = [], [], 0
    for i, w in enumerate(words):
        remaining_lines = n - len(lines)
        remaining_words = len(words) - i
        must_fill = remaining_words <= remaining_lines
        add = len(w) + (1 if cur else 0)
        if cur and not must_fill and cur_len + add > target:
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
            continue
        cur.append(w)
        cur_len += add
    if cur:
        lines.append(" ".join(cur))
    # Свести к ровно n строкам (на случай очень длинных слов).
    while len(lines) > n:
        last = lines.pop()
        prev = lines.pop()
        lines.append(prev + " " + last)
    return lines


def _split_replica_into_lines(text, max_lines, capacity=None):
    """Разбивает текст реплики на строки по словам.

    capacity — сколько символов реально помещается в строку кадра при текущем
    кегле. Число строк минимально (в пределах max_lines), строки равной длины:
    слов в строке получается максимум, а края идут ровно.
    """
    words = (text or "").split()
    if not words:
        return [""]
    if max_lines and max_lines > 0:
        return _balance_lines(words, int(capacity) if capacity else 0, max_lines)
    if capacity:
        return _balance_lines(words, int(capacity), len(words))
    return [" ".join(words)]


def _merge_short_words(words, max_lines):
    """Склеивает слова, чтобы вышло ровно не более max_lines строк."""
    if len(words) <= 1:
        return [" ".join(words)]
    per = (len(words) + max_lines - 1) // max_lines if max_lines else len(words)
    lines = []
    for i in range(0, len(words), per):
        lines.append(" ".join(words[i:i + per]))
    if len(lines) > max_lines:
        lines = _merge_extra_lines(lines, max_lines)
    return lines


def _merge_extra_lines(lines, max_lines):
    """Сшивает лишние строки, не превышая max_lines (сливает в последнюю)."""
    out = list(lines)
    while len(out) > max_lines:
        last = out.pop()
        prev = out.pop()
        out.append(prev + " " + last)
    return out


def _rewrite_srt_wrapped(srt_path, max_lines, capacity=None):
    """Переписывает SRT: каждая реплика разбивается на <= max_lines строк.

    capacity — вместимость строки в символах при текущем кегле (None — первый
    проход, вместимость ещё неизвестна). Строки наполняются реальными словами
    под вместимость и балансируются: коротких строк не остаётся, все слова
    стоят обычным одиночным пробелом (текст НЕ растягивается).
    Возвращает максимальную длину одной строки (для подбора кегля).
    Таймкоды и порядок реплик не меняются.
    """
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return 0
    parsed = []
    max_len = 0
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        times = next((l for l in lines if "-->" in l), None)
        text = " ".join(
            l.strip() for l in lines
            if l.strip() and "-->" not in l and not l.strip().isdigit()
        ).strip()
        if times is None or not text:
            parsed.append((block, None, None))
            continue
        wrapped = _split_replica_into_lines(text, max_lines, capacity)
        max_len = max(max_len, max(len(l) for l in wrapped))
        parsed.append((lines[0].strip(), times, wrapped))
    out_blocks = []
    for num, times, wrapped in parsed:
        if times is None or wrapped is None:
            out_blocks.append(num)
            continue
        out_blocks.append(num + "\n" + times + "\n" + "\n".join(wrapped))
    try:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out_blocks) + "\n")
    except Exception:
        pass
    return max_len


def _subtitle_filter(subs_path, cfg=None, video_size=None, subs_segments=None):
    """Строит ffmpeg-фильтр subtitles для отображения субтитров поверх картинки.

    Текст берётся из subs_segments (сегменты прямо из таблицы переводов) либо из
    готового SRT по пути subs_path. В обоих случаях он попадает во временный
    файл: это снимает проблемы с экранированием кириллицы в пути и не требует
    хранить лишние .srt в проекте.

    Если включён блюр нижней зоны (montage_blur_subs_enabled) — текст центрируется
    по вертикали внутри заблюренной полосы (поверх спрятанных родных субтитров).
    Для вертикальных видео (W < H) размер шрифта подбирается так, чтобы самая
    длинная реплика занимала не более subtitles_portrait_max_lines строк
    (по умолчанию 3), но не меньше subtitles_portrait_min_font.
    Возвращает None, если субтитры недоступны.
    """
    if subs_segments:
        fd, tmp = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(transcript.srt_text(subs_segments))
    elif subs_path and os.path.exists(subs_path):
        fd, tmp = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
        shutil.copyfile(subs_path, tmp)
    else:
        return None
    # Экранирование для ffmpeg-фильтра: обратный слэш и двоеточие.
    esc = tmp.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    # Базовый размер шрифта (юниты ASS в системе PlayResY=288).
    base_font = float(cfg.get("subtitles_font_size", 26)) if cfg else 26.0

    vw = vh = None
    if video_size and len(video_size) >= 2:
        vw, vh = video_size[0], video_size[1]

    font = base_font
    portrait = bool(vw and vh and 0 < vw < vh)
    if portrait:
        max_lines = int(cfg.get("subtitles_portrait_max_lines", 3)) if cfg else 3
        if max_lines and max_lines > 0:
            min_font = float(cfg.get("subtitles_portrait_min_font", 7)) if cfg else 7.0
            # Виртуальная ширина ASS при PlayResY=288: 288 * W / H.
            virt_w = 288.0 * vw / vh
            # Средняя ширина глифа в em. Замерено рендером libass на реальных
            # репликах: 0.21 em/симв, а не 0.55 — с завышенным коэффициентом
            # строки получались вдвое короче возможного и слова не использовали
            # ширину кадра.
            char_em = float(cfg.get("subtitles_char_em", 0.22)) if cfg else 0.22
            # Рабочая ширина строки: оставляем небольшие поля по краям кадра.
            width_frac = float(cfg.get("subtitles_width_frac", 0.92)) if cfg else 0.92
            usable_w = virt_w * max(0.4, min(width_frac, 1.0))
            # Общий потолок для вертикальных роликов: весь блок (до max_lines
            # строк) не должен занимать больше subtitles_portrait_block_ratio
            # высоты кадра. Иначе на коротких репликах шрифт раздувается под
            # ширину строки, и текст лезет на пол-экрана.
            block_lines = max(max_lines, 1)
            block_ratio = float(cfg.get("subtitles_portrait_block_ratio", 0.15)) if cfg else 0.15
            max_by_height = block_ratio * 288.0 / (block_lines * 1.2)
            font_scale = float(cfg.get("subtitles_portrait_font_scale", 1.0)) if cfg else 1.0
            if font_scale > 0:
                max_by_height *= font_scale
            font = max(min_font, min(base_font, max_by_height))
            # Первый проход: ровная разбивка на max_lines строк — по ней
            # оцениваем кегль, при котором строки встают в ширину кадра.
            max_len = _rewrite_srt_wrapped(tmp, max_lines)
            cap = 0
            if max_len > 0:
                need = usable_w / (max_len * char_em)
                font = max(min_font, min(font, need))
                # Второй проход: вместимость строки при итоговом кегле —
                # строки наполняются реальными словами, без растяжки.
                cap = int(usable_w / (font * char_em))
                if cap > 0:
                    _rewrite_srt_wrapped(tmp, max_lines, capacity=cap)
            logger = logging.getLogger("dubbing")
            logger.info(
                "Вертикальный формат: реплики до %d строк, кегль %.1f, "
                "вместимость строки ~%d символов (блок ≤ %.1f%% кадра, scale=%.2f)",
                max_lines, font, cap, block_ratio * 100, font_scale
            )

    # Оформление букв (цвет #ffc957 + чёрная обводка, полужирный, Arial) —
    # собирается отдельно, т.к. FontSize пересобирается в двух ветках.
    style = _style_options(cfg)
    force = f"FontSize={font:.2f},Alignment=2"
    if style:
        force += "," + style
    if cfg and cfg.get("montage_blur_subs_enabled") and vh:
        hr = float(cfg.get("montage_blur_subs_height_ratio", 0.25))
        # Вертикальное центрирование текста внутри полосы блюра:
        # отступ снизу = (высота полосы - высота блока) / 2.
        # libass рендерит SRT в виртуальных координатах PlayResY=288; строка
        # шрифта size S занимает ≈ S*1.2 юнитов (межстрочный interline).
        line_h = font * 1.2 * vh / 288.0            # высота строки, px видео
        band_h = vh * hr                             # высота полосы, px видео
        # В портрете реплики жёстко разбиваются до max_lines строк — блок выше
        # одной строки. Отступ считаем по высоте ВСЕГО блока, иначе многострочный
        # текст вылезал бы верхним краем за пределы полосы блюра.
        block_lines = 1
        if portrait:
            block_lines = max(
                int(cfg.get("subtitles_portrait_max_lines", 3)) if cfg else 3, 1)
        block_h = block_lines * line_h
        # Если блок всё равно выше полосы — уменьшаем шрифт, чтобы целиком влезть.
        if block_h > band_h:
            font = max(font * band_h / block_h, 1.0)
            line_h = font * 1.2 * vh / 288.0
            block_h = block_lines * line_h
        # Шрифт мог уменьшиться выше — пересобираем force_style с новым размером,
        # оформление букв (style) терять нельзя — добавляем обратно.
        force = f"FontSize={font:.2f},Alignment=2"
        if style:
            force += "," + style
        # Небольшой запас сверху/снизу, чтобы текст не лип к краям полосы.
        inset = min(max(band_h * 0.04, 0.0), line_h)
        margin_screen = max(band_h - block_h - inset, 0.0) * 0.5
        # Масштаб MarginV в координаты libass: margin_ass = px * 288 / H.
        margin = max(int(round(margin_screen * 288.0 / vh)), 0)
        force += f",MarginV={margin}"
    return "subtitles='%s':force_style='%s'" % (esc, force)


def _ass_color(hex_rgb, alpha=0):
    """'ffc957' -> '&H0057C9FF' (ASS/SSA ждёт &HAABBGGRR, то есть BGR!).

    Порядок каналов в ASS обратный привычному RGB: пользователь задаёт цвет
    как #ffc957 (RGB), а libass читает 57C9FF. Пустая/битая строка -> белый.
    """
    s = str(hex_rgb or "").strip().lstrip("#").strip()
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        s = "FFFFFF"
    r, g, b = s[0:2], s[2:4], s[4:6]
    return "&H%02X%s%s%s" % (int(alpha) & 0xFF, b.upper(), g.upper(), r.upper())


def _style_options(cfg):
    """Оформление букв: цвет, обводка, полужирный, гарнитура -> строка force_style.

    Собирается ОТДЕЛЬНО от кегля, потому что FontSize пересобирается в двух
    местах (портретный режим и подгонка под полосу блюра), а оформление должно
    сохраняться при каждой пересборке.
    """
    if not cfg:
        return ""
    parts = []
    name = str(cfg.get("subtitles_font_name", "") or "").strip()
    if name:
        parts.append("FontName=%s" % name)
    prim = str(cfg.get("subtitles_primary_color", "") or "").strip()
    if prim:
        parts.append("PrimaryColour=%s" % _ass_color(prim))
    outl = str(cfg.get("subtitles_outline_color", "") or "").strip()
    if outl:
        parts.append("OutlineColour=%s" % _ass_color(outl))
    if cfg.get("subtitles_bold"):
        parts.append("Bold=-1")          # ASS: -1 = включить, 0 = выключить
    try:
        ow = float(cfg.get("subtitles_outline_width", 0) or 0)
    except (TypeError, ValueError):
        ow = 0.0
    if ow > 0:
        parts.append("Outline=%.2f" % ow)
    try:
        sh = float(cfg.get("subtitles_shadow", 0) or 0)
    except (TypeError, ValueError):
        sh = 0.0
    if sh > 0:
        parts.append("Shadow=%.2f" % sh)
    return ",".join(parts)


def _corner_box(position, w, h):
    """Прямоугольник угла кадра, в котором ищем уже готовый логотип."""
    if position == "tl":
        return 0, 0, int(w * 0.45), int(h * 0.38)
    if position == "bl":
        return 0, int(h * 0.62), int(w * 0.45), h
    if position == "br":
        return int(w * 0.55), int(h * 0.62), w, h
    return int(w * 0.55), 0, w, int(h * 0.38)  # tr


def _blobs(mask, min_area):
    """Связные области маски площадью не меньше min_area: [(площадь, bbox), ...]."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) >= min_area:
                ys = [p[0] for p in pts]
                xs = [p[1] for p in pts]
                out.append((len(pts), (min(xs), min(ys), max(xs), max(ys))))
    return out


def _corner_logo_probe(video_path, position, n_frames=10, probe_w=640):
    """Ищет уже готовый значок в углу кадра. Возвращает (найден, диагностика).

    Логотип бывает анимированным (наш ромб вращается), поэтому признак «стоит на
    месте» не годится — внутри значка движется. Работает признак ПЕРЕКРЫТИЯ:

      1. В каждом кадре считаем, где пиксель сильно отличается от своего
         локального окружения (медианный фильтр по кадру). У накладки это её
         корпус, и он есть в каждом кадре; у сцены случайные совпадения
         усредняются по кадрам и исчезают.
      2. Собираем связные области этой маски, интересуют только те, что сидят
         у самого угла кадра.
      3. Ключевая проверка — корреляция по времени между яркостью внутри пятна и
         яркостью вокруг него. Накладка закрывает собой сцену, поэтому её
         собственная анимация не связана с движением картинки (corr около 0).
         Любой кусок самой сцены живёт вместе с окружением (corr около 1).
      4. Пятно должно быть графическим объектом: между ним и фоном есть
         ступенька яркости. Плоское пятно (corr не посчитать) тоже подходит —
         это статичный значок, — но тогда ступенька нужна крупнее.

    Если угол почти не движется, честно говорим «не разобрать» и оставляем
    водяной знак: перестраховка лучше двух логотипов в кадре.
    """
    dur = _probe_duration(video_path)
    if not dur or dur <= 1.0 or Image is None:
        return False, "длительность неизвестна"
    tmpdir = tempfile.mkdtemp(prefix="logo_probe_")
    try:
        crops = []
        for i in range(n_frames):
            ts = dur * (0.06 + 0.88 * i / max(1, n_frames - 1))
            png = os.path.join(tmpdir, "f%02d.png" % i)
            proc = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % ts,
                                   "-i", video_path, "-vf",
                                   "scale=%d:-2,format=gray" % probe_w,
                                   "-frames:v", "1", png], capture_output=True)
            if proc.returncode != 0 or not os.path.exists(png):
                continue
            crops.append(np.asarray(Image.open(png).convert("L")).astype(np.float32))
        if len(crops) < 4:
            return False, "не удалось снять кадры"
        fh, fw = crops[0].shape
        x0, y0, x1, y1 = _corner_box(position, fw, fh)
        box = np.stack([c[y0:y1, x0:x1] for c in crops])
        nf, bh, bw = box.shape[0], box.shape[1], box.shape[2]
        if bh < 12 or bw < 12:
            return False, "угол слишком мал"
        med = np.median(box, axis=0)
        if float(np.median(box.std(axis=0))) < 3.0:
            return False, "сцена в углу почти не движется — не разобрать"
        agree = np.zeros((bh, bw), dtype=np.float32)
        for t in range(nf):
            bg = np.asarray(Image.fromarray(box[t].astype(np.uint8))
                            .filter(ImageFilter.MedianFilter(size=21)),
                            dtype=np.float32)
            agree += (np.abs(box[t] - bg) > 25.0)
        agree /= float(nf)
        mask = agree >= 0.75
        cands = _blobs(mask, max(8, int(0.002 * bw * bh)))
        if not cands:
            return False, "устойчивых контрастных пятен в углу нет"
        ax, ay = {"tr": (bw - 1, 0), "tl": (0, 0),
                  "br": (bw - 1, bh - 1), "bl": (0, bh - 1)}.get(position, (bw - 1, 0))

        def corner_dist(bb):
            bx0, by0, bx1, by1 = bb
            dx = (ax - bx1) if ax > bw / 2.0 else (bx0 - ax)
            dy = (ay - by1) if ay > bh / 2.0 else (by0 - ay)
            return max(dx, dy)

        why = "кандидатов нет"
        for area, bb in sorted(cands, key=lambda c: corner_dist(c[1])):
            bx0, by0, bx1, by1 = bb
            bw_box, bh_box = bx1 - bx0 + 1, by1 - by0 + 1
            h_frac, w_frac = bh_box / float(fh), bw_box / float(fw)
            dist = corner_dist(bb)
            area_frac = area / float(bw * bh)
            core = mask[by0:by1 + 1, bx0:bx1 + 1]
            py, px = max(3, int(bh_box * 0.7)), max(3, int(bw_box * 0.7))
            ry0, ry1 = max(0, by0 - py), min(bh, by1 + 1 + py)
            rx0, rx1 = max(0, bx0 - px), min(bw, bx1 + 1 + px)
            ring = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
            ring[by0 - ry0:by1 - ry0 + 1, bx0 - rx0:bx1 - rx0 + 1] = False
            if not ring.any() or not core.any():
                continue
            step = abs(float(np.median(med[by0:by1 + 1, bx0:bx1 + 1][core]))
                       - float(np.median(med[ry0:ry1, rx0:rx1][ring])))
            s_in = np.array([box[t][by0:by1 + 1, bx0:bx1 + 1][core].mean()
                             for t in range(nf)])
            s_ring = np.array([box[t][ry0:ry1, rx0:rx1][ring].mean()
                               for t in range(nf)])
            mv_in, mv_ring = float(s_in.std()), float(s_ring.std())
            flat = mv_in < 2.0
            corr = 0.0
            if not flat and mv_ring > 1e-6:
                corr = float(np.corrcoef(s_in, s_ring)[0, 1])
            info = ("значок=%dpx (%.1f%% угла), размер=%dx%d (%.0f%% высоты кадра), "
                    "до угла=%.0f%% ширины, ступенька яркости=%.0f, "
                    "корреляция с фоном=%.2f, движение внутри/вокруг=%.1f/%.1f"
                    % (area, 100 * area_frac, bw_box, bh_box, 100 * h_frac,
                       100.0 * dist / bw, step, corr, mv_in, mv_ring))
            if dist > 0.20 * bw:
                why = info + " — далеко от самого угла"
                continue
            if h_frac < 0.04 or h_frac > 0.45 or w_frac < 0.03 or w_frac > 0.35:
                why = info + " — размер не похож на значок"
                continue
            if area_frac < 0.002:
                why = info + " — слишком мелкий, чтобы быть значком"
                continue
            if step < 20.0:
                why = info + " — нет ступеньки яркости, это часть кадра"
                continue
            if flat:
                if step < 30.0 or mv_ring < 3.0:
                    why = info + " — плоское пятно без контраста, это часть кадра"
                    continue
                return True, info + " — статичный значок"
            if corr > 0.70:
                why = info + " — двигается вместе со сценой, это не накладка"
                continue
            return True, info
        return False, why
    except Exception as e:
        return False, "ошибка анализа: %s" % e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _logo_settings(cfg, video_path=None, logger=None):
    """Параметры водяного знака: (путь, позиция, отступ) или None.

    Если в углу исходника уже есть свой значок (montage_logo_skip_if_present),
    водяной знак не накладывается — иначе в кадре было бы два логотипа.
    """
    if not cfg.get("montage_logo_enabled"):
        return None
    lp = cfg.get("montage_logo_path", "")
    if not lp or not os.path.exists(lp):
        return None
    position = cfg.get("montage_logo_position", "br")
    if video_path and cfg.get("montage_logo_skip_if_present", True):
        try:
            stamp = int(os.path.getmtime(video_path))
        except OSError:
            stamp = 0
        key = (os.path.abspath(video_path), position, stamp)
        cached = _LOGO_PROBE_CACHE.get(key)
        if cached is None:
            cached = _corner_logo_probe(video_path, position)
            _LOGO_PROBE_CACHE[key] = cached
        found, info = cached
        if found:
            if logger:
                logger.info("В %s уже есть значок лого (%s) — водяной знак не "
                            "накладываем.", _POS_RU.get(position, position), info)
            return None
        if logger:
            logger.info("Свой логотип в углу не найден (%s) — ставим водяной знак.",
                        info)
    return (lp, position, int(cfg.get("montage_logo_margin", 15)))


def _logo_overlay_xy(position, margin):
    """Координаты наложения лого в терминах ffmpeg overlay."""
    if position == "tr":
        return f"W-w-{margin}:{margin}"
    if position == "bl":
        return f"{margin}:H-h-{margin}"
    if position == "tl":
        return f"{margin}:{margin}"
    return f"W-w-{margin}:H-h-{margin}"


def _video_parts(cfg, subs):
    """Строит цепочку фильтров видео: блюр нижней зоны -> субтитры.

    Возвращает (list_parts, label): parts — звенья filter_complex, label —
    входная метка следующего звена ([b] после блюра, [s] после субтитров).
    Лого накладывается отдельной ступенью в export-функциях (его вход -i
    добавляется там же, индекс зависит от состава входов).
    """
    parts, src = [], "[0:v:0]"
    if cfg.get("montage_blur_subs_enabled"):
        hr = float(cfg.get("montage_blur_subs_height_ratio", 0.25))
        st = cfg.get("montage_blur_subs_strength", "15:5")
        parts.append(
            f"[0:v:0]split[bbase][breg];"
            f"[breg]crop=iw:ih*{hr}:0:ih*{1 - hr},boxblur={st}[bblur];"
            f"[bbase][bblur]overlay=0:H*{1 - hr}[b]"
        )
        src = "[b]"
    if subs:
        parts.append(f"{src}{subs}[s]")
        src = "[s]"
    return parts, src


def _probe_video_meta(video_path):
    """(width, height, fps) видео через ffprobe; None при неудаче."""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate",
               "-of", "json", video_path]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        data = json.loads(out).get("streams", [{}])[0]
        w, h = int(data["width"]), int(data["height"])
        rfr = data.get("r_frame_rate", "")
        if "/" in rfr:
            n, d = rfr.split("/")
            fps = float(n) / float(d)
        else:
            fps = float(rfr or 30)
        return w, h, fps
    except Exception:
        return None


def _probe_orientation(w, h):
    """Ориентация видео: 'landscape' | 'portrait' | 'square'."""
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"


def _target_audio_rate(video_path, cfg, has_audio=True):
    """Итоговая частота аудиодорожки.

    Порядок: export_audio_rate из config.json -> частота исходного видео -> 44100.
    Явный выбор обязателен: на цепочке amix(44100 + 24000) -> loudnorm ffmpeg
    оставляет 96 кГц (авто-вставленные aresample вместе с внутренним ресемплингом
    loudnorm дают 24000 -> 48000 -> 96000). Дорожка при этом играет нормально,
    но тратит битрейт вчетверо.
    """
    want = int(cfg.get("export_audio_rate", 0) or 0)
    if want > 0:
        return want
    if has_audio:
        al = _probe_audio_layout(video_path)
        if al:
            return al[0]
    return 44100


def _probe_audio_layout(video_path):
    """(sample_rate, channels) первого аудио-потока; None если аудио нет."""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=sample_rate,channels",
               "-of", "json", video_path]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        st = json.loads(out).get("streams", [{}])[0]
        return int(st["sample_rate"]), int(st["channels"])
    except Exception:
        return None


def _probe_video_timescale(video_path):
    """Знаменатель time_base видео (для -video_track_timescale), 24000 по умолчанию."""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=time_base", "-of", "json", video_path]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        tb = json.loads(out).get("streams", [{}])[0].get("time_base", "")
        if "/" in tb:
            return int(tb.split("/")[1])
    except Exception:
        pass
    return 24000


_LOGO_ANIM_EXT = (".gif", ".mp4", ".webm", ".mov", ".m4v", ".apng")
# Результат проверки «свой ли логотип в углу» — по (файл, угол), чтобы не
# переснимать кадры при каждом экспорте.
_LOGO_PROBE_CACHE = {}
_POS_RU = {"tr": "правом верхнем", "tl": "левом верхнем",
           "br": "правом нижнем", "bl": "левом нижнем"}


def _logo_is_animated(logo):
    """Лого является анимацией (GIF/video) — требует -stream_loop и shortest=1."""
    return bool(logo) and logo[0].lower().endswith(_LOGO_ANIM_EXT)


def _logo_inputs(logo):
    """Аргументы -i для лого: GIF/video бесконечным циклом, PNG/JPG статичный."""
    if not logo:
        return []
    if _logo_is_animated(logo):
        return ["-stream_loop", "-1", "-i", logo[0]]
    return ["-i", logo[0]]


def _logo_chain(vlabel, logo, logo_idx, cfg, video_path):
    """Звенья filter_complex для наложения лого (с масштабом).

    Возвращает список частей-звеньев: scale лого (опц.) -> overlay.
    logo_idx — индекс входа лого (-i) для ссылки [N:v:0].
    """
    parts = []
    sc = float(cfg.get("montage_logo_scale") or 0)
    src = f"[{logo_idx}:v:0]"
    if sc > 0:
        meta = _probe_video_meta(video_path)
        if meta:
            lh = max(1, int(min(meta[0], meta[1]) * sc))
            parts.append(f"{src}scale=-2:{lh}[logo]")
            src = "[logo]"
    xy = _logo_overlay_xy(logo[1], logo[2])
    ov = f"{vlabel}{src}overlay={xy}"
    if _logo_is_animated(logo):
        ov += ":shortest=1"
    parts.append(ov + "[v]")
    return parts


def _pick_intro_video(cfg, ori):
    """Видео-заставка под ориентацию ролика, или None."""
    if not cfg.get("montage_intro_video_enabled"):
        return None
    path = cfg.get("montage_intro_landscape" if ori != "portrait"
                   else "montage_intro_portrait", "")
    if not path or not os.path.exists(path):
        return None
    return path


def _probe_duration(video_path):
    """Длительность файла в секундах; None при неудаче."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", video_path]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def _pick_freeze_frame(video_path):
    """Возвращает момент (сек) самого «выразительного» кадра из основной части.

    Сканирует видео через signalstats (в уменьшенном разрешении для скорости)
    и выбирает кадр с максимальной суммой яркости и контраста. Края ролика
    (первые ~8% и последние ~4%) пропускаются, чтобы не попасть в заставки/титры.
    При неудаче — 0.0 (первый кадр).
    """
    dur = _probe_duration(video_path)
    if not dur or dur <= 1.0:
        return 0.0
    start = min(1.5, dur * 0.08)
    end = dur - max(0.5, dur * 0.04)
    if start >= end:
        return 0.0
    try:
        cmd = ["ffmpeg", "-v", "info", "-i", video_path,
               "-vf", "scale=-2:180,signalstats,metadata=print",
               "-an", "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except Exception:
        return 0.0
    best_ts, best_score = None, -1.0
    pts = yavg = ymax = ymin = None
    for line in proc.stderr.splitlines():
        line = line.strip()
        if "pts_time:" in line:
            try:
                pts = float(line.split("pts_time:")[1].split()[0])
            except Exception:
                pts = None
        elif "YAVG=" in line:
            try:
                yavg = float(line.split("YAVG=")[1].split()[0])
            except Exception:
                yavg = None
        elif "YMAX=" in line:
            try:
                ymax = float(line.split("YMAX=")[1].split()[0])
            except Exception:
                ymax = None
        elif "YMIN=" in line:
            try:
                ymin = float(line.split("YMIN=")[1].split()[0])
            except Exception:
                ymin = None
        if pts is not None and yavg is not None and ymax is not None and ymin is not None:
            if start <= pts <= end:
                score = yavg + (ymax - ymin)
                if score > best_score:
                    best_score = score
                    best_ts = pts
            pts = yavg = ymax = ymin = None
    if best_ts is None:
        return 0.0
    return min(max(best_ts, 0.0), dur - 0.1)


def _subs_present_on_freeze(frame_png, cfg):
    """Определяет, есть ли в нижней зоне стоп-кадра нарисованные чужие субтитры.

    Ищет «выделенный блок текста»: узкое горизонтальное окно (≈4% высоты) с
    плотными краями букв по X, резко контрастирующее со спокойным фоном полосы.
    Это отличает настоящую строку субтитров от тонкой UI-линии (у неё краёв по X
    почти нет) и от детализированной сцены (фон тоже «шумит»). Текст считается
    найденным, если контраст окна к фону больше montage_blur_subs_detect_ratio,
    а само окно содержит достаточно краёв (> montage_blur_subs_detect_min).
    """
    if Image is None:
        return False
    try:
        img = Image.open(frame_png).convert("L")
        if img.width > 640:
            img = img.resize((640, max(1, img.height * 640 // img.width)), Image.LANCZOS)
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape
        hr = float(cfg.get("montage_blur_subs_height_ratio", 0.25))
        hr = min(max(hr, 0.10), 0.60)
        y_band = int(h * (1.0 - hr))
        if y_band >= h - 4:
            return False
        # Плотность краёв по X по строкам: буквы дают много коротких штрихов.
        dx = np.gradient(a, axis=1)
        thr = np.abs(dx).mean() + 1.5 * np.abs(dx).std()
        sig = (np.abs(dx) > thr).astype(int)
        cnt = np.array([int(np.sum(np.abs(np.diff(sig[i])))) for i in range(h)])
        ymax = h - 1
        win = max(2, h // 25)  # полуширина окна ~4% высоты
        best_f = 0.0
        best_wv = 0.0
        for c in range(y_band + win, ymax - win + 1):
            wv = float(cnt[c - win:c + win + 1].mean())
            bg = float(np.concatenate([cnt[y_band:c - win + 1], cnt[c + win:ymax]]).mean())
            f = wv / max(bg, 1.0)
            if f > best_f:
                best_f = f
                best_wv = wv
        ratio = float(cfg.get("montage_blur_subs_detect_ratio", 4.0))
        floor = float(cfg.get("montage_blur_subs_detect_min", 4.0))
        return best_f > ratio and best_wv > floor
    except Exception:
        return False


def _detect_freeze_bottom_trim(frame_png, cfg):
    """Возвращает долю нижней части стоп-кадра для обрезки (0 - не трогать).

    Ищем «хвост интерфейса» у нижней кромки кадра: чёрные поля, тёмную подложку
    панели плеера, сам бегунок длительности. Идём снизу вверх, пока строка темнее
    montage_freeze_ui_dark, и отрезаем всё найденное.

    Граница засчитывается только если:
      - над хвостом идёт СВЕТЛЫЙ блок контента (высотой не менее
        montage_freeze_ui_content от кадра) — то есть это не тёмная сцена;
      - стык резкий (скачок яркости не меньше montage_freeze_ui_edge) — граница
        панели, а не плавный переход;
      - сам хвост не больше 25% кадра.

    Немногое смещение вверх (montage_freeze_ui_margin) добавляется, чтобы край
    панели не «залипал» на 1-2 строках после обрезки.
    """
    if Image is None:
        return 0.0
    try:
        img = Image.open(frame_png).convert("L")
        if img.width > 640:
            img = img.resize((640, max(1, img.height * 640 // img.width)), Image.LANCZOS)
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape
        prof = a.mean(axis=1)
        dark = float(cfg.get("montage_freeze_ui_dark", 100.0) or 100.0)
        edge_min = float(cfg.get("montage_freeze_ui_edge", 25.0) or 25.0)
        need = max(4, int(h * float(cfg.get("montage_freeze_ui_content", 0.08) or 0.08)))
        # 1) Ищем снизу вверх первую строку контента (не темнее порога).
        top = h
        for y in range(h - 1, -1, -1):
            if prof[y] < dark:
                continue
            top = y
            break
        tail = (h - top) / float(h)
        if tail <= 0.0 or top <= 0 or tail > 0.25:
            return 0.0
        # 2) Над хвостом должен идти светлый контент...
        above = prof[max(0, top - need):top]
        if above.size < need or float(above.mean()) < edge_min:
            return 0.0
        # 3) ...и граница между хвостом и контентом — резкая.
        if abs(float(prof[top]) - float(prof[top + 1])) < edge_min:
            return 0.0
        margin = float(cfg.get("montage_freeze_ui_margin", 0.005) or 0.0)
        return min(0.5, tail + margin)
    except Exception:
        return 0.0
def _apply_montage_intro(video_path, out_path, cfg, logger):
    """Вставляет в начало ролика: стоп-кадр + заставку.

    Итоговая структура:
        [первый кадр ролика, удерживается montage_freeze_dur, тишина] +
        [заставка: video mp4 (montage_intro_landscape/portrait) с её звуком
         либо картинка montage_intro_image на montage_intro_hold сек, тишина] +
        [весь ролик целиком].
    Видео-заставка выбирается по ориентации ролика; если не задана — картинка.
    Требует montage_freeze_enabled — иначе пропускается.
    """
    if not cfg.get("montage_freeze_enabled"):
        return out_path
    freeze = float(cfg.get("montage_freeze_dur", 0.5))
    hold = float(cfg.get("montage_intro_hold", 1.5))
    intro_vol = float(cfg.get("montage_intro_volume", 1.0) or 1.0)  # 1.0 = 100%
    card = cfg.get("montage_intro_image", "")
    meta = _probe_video_meta(out_path)
    if not meta:
        logger.info("Заставка пропущена (не удалось определить параметры видео).")
        return out_path
    w, h, fps = meta
    ori = _probe_orientation(w, h)
    intro_video = _pick_intro_video(cfg, ori)
    if not intro_video and (not card or not os.path.exists(card)):
        logger.info("Заставка пропущена: video-заставка и картинка не заданы.")
        return out_path
    if freeze <= 0 and not intro_video and hold <= 0:
        return out_path
    logger.info(f"Формат видео: {ori} ({w}x{h}). Заставка: "
                f"{'video ' + os.path.basename(intro_video) if intro_video else 'картинка'}.")
    tscale = _probe_video_timescale(out_path)
    card_vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        if ori == "landscape" else
        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    )
    has_audio = _probe(out_path)
    al = _probe_audio_layout(out_path) if has_audio else None
    sr = al[0] if al else 44100
    cl = {1: "mono", 2: "stereo"}.get(al[1], "stereo") if al else "stereo"
    crf = cfg.get("export_crf", 18)
    ab = cfg.get("export_audio_bitrate", "192k")
    tmpdir = tempfile.mkdtemp(prefix="montage_intro_")
    try:
        frame0 = os.path.join(tmpdir, "frame0.png")
        frozen = os.path.join(tmpdir, "frozen.mp4")
        intro = os.path.join(tmpdir, "intro.mp4")
        final = os.path.join(tmpdir, "final.mp4")
        lst = os.path.join(tmpdir, "list.txt")
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-r", "%.3f" % fps,
                "-video_track_timescale", str(tscale)]
        aenc = ["-c:a", "aac", "-b:a", ab] if has_audio else ["-an"]

        def _run(args):
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg (заставка):\n" + proc.stderr[-2000:])

        # Стоп-кадр берём из ИСХОДНОГО видео (до наложения субтитров/блюра),
        # чтобы заставка была чистой картинкой: без субтитров и элементов плеера.
        # 1) Обрезаем нижнюю полосу кадра (ползунок плеера, значок громкости).
        # 2) Заблюриваем нижнюю зону, где в исходнике нарисованы ЧУЖИЕ субтитры
        #    (теми же параметрами, что и в основном ролике) — они исчезают.
        #    Блюр применяется ТОЛЬКО если детектор нашёл на стоп-кадре текст
        #    (montage_blur_subs_auto_detect) — пустую зону не размываем.
        freeze_ts = _pick_freeze_frame(video_path)
        crop_bottom = float(cfg.get("montage_freeze_crop_bottom", 0.0) or 0.0)
        # Динамическая обрезка: убираем «бегунок плеера» (тонкую яркую полосу
        # над большой чёрной подложкой), если детектор его нашёл на ПОЛНОМ кадре
        # (до фиксированной обрезки) — иначе выступ сплющивается и не находится.
        det = os.path.join(tmpdir, "det.png")
        try:
            _run(["ffmpeg", "-y", "-ss", "%.3f" % freeze_ts, "-i", video_path,
                  "-vf", "scale=-2:360", "-frames:v", "1", "-y", det])
            auto_trim = _detect_freeze_bottom_trim(det, cfg)
        except Exception:
            auto_trim = 0.0
        if auto_trim > crop_bottom:
            crop_bottom = auto_trim
        if crop_bottom > 0.0 and crop_bottom < 0.5:
            keep = 1.0 - min(crop_bottom, 0.5)
            base_vf = f"crop=iw:ih*{keep:.4f}:0:0,scale={w}:{h}:flags=lanczos"
        else:
            base_vf = f"scale={w}:{h}:flags=lanczos"
        _run(["ffmpeg", "-y", "-ss", "%.3f" % freeze_ts, "-i", video_path,
              "-vf", base_vf, "-frames:v", "1", "-y", frame0])
        notes = []
        if crop_bottom > 0.0 and crop_bottom < 0.5:
            notes.append(f"обрезаны нижние {crop_bottom * 100:.0f}%")
        if cfg.get("montage_blur_subs_enabled"):
            auto = bool(cfg.get("montage_blur_subs_auto_detect", True))
            if auto and not _subs_present_on_freeze(frame0, cfg):
                logger.info("Стоп-кадр: в нижней зоне текст не обнаружен — блюр пропущен.")
                notes.append("субтитров нет — блюр не требуется")
            else:
                hr = float(cfg.get("montage_blur_subs_height_ratio", 0.25))
                st = cfg.get("montage_blur_subs_strength", "15:5")
                blurred = frame0 + ".b.png"
                blur_vf = (f"split[bbase][breg];"
                           f"[breg]crop=iw:ih*{hr}:0:ih*{1 - hr},boxblur={st}[bblur];"
                           f"[bbase][bblur]overlay=0:H*{1 - hr}[fz]")
                _run(["ffmpeg", "-y", "-i", frame0, "-vf", blur_vf,
                      "-frames:v", "1", "-y", blurred])
                shutil.move(blurred, frame0)
                notes.append("чужие субтитры заблюрены")
        logger.info(f"Стоп-кадр из {freeze_ts:.1f} с ({', '.join(notes)}), отмасштабирован к {w}x{h}.")

        def _still_seg(img, dur, outseg, is_frame0):
            cmd = ["ffmpeg", "-y", "-framerate", "%.3f" % fps, "-loop", "1", "-i", img]
            if has_audio:
                cmd += ["-f", "lavfi", "-t", "%.3f" % dur,
                        "-i", "anullsrc=r=%d:cl=%s" % (sr, cl)]
            cmd += ["-t", "%.3f" % dur]
            if is_frame0:
                cmd += ["-map", "0:v:0"]
            else:
                cmd += ["-filter_complex", "[0:v:0]" + card_vf + "[v]",
                        "-map", "[v]"]
            if has_audio:
                cmd += ["-map", "1:a:0"]
            cmd += venc + (aenc if has_audio else []) + ["-movflags", "+faststart", outseg]
            _run(cmd)

        def _video_seg(src, outseg):
            fc = f"[0:v:0]{card_vf},fps={fps:.3f},format=yuv420p[v]"
            cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", fc, "-map", "[v]"]
            if _probe(src):
                # Звук заставки приводим к параметрам основной дорожки: без этого
                # concat с -c copy кладёт пакеты одной частоты в поток другой, и
                # звук заставки превращается в тишину.
                af = []
                if abs(intro_vol - 1.0) > 1e-3:
                    af.append("volume=%.4f" % intro_vol)
                cmd += ["-map", "0:a:0"]
                if af:
                    cmd += ["-af", ",".join(af)]
                if al:
                    cmd += ["-ar", str(sr), "-ac", str(al[1])]
                cmd += ["-c:a", "aac", "-b:a", ab]
            elif has_audio:
                dur = _probe_duration(src) or 1.0
                cmd += ["-f", "lavfi", "-t", "%.3f" % dur,
                        "-i", "anullsrc=r=%d:cl=%s" % (sr, cl),
                        "-map", "1:a:0", "-c:a", "aac", "-b:a", ab]
            cmd += venc + (aenc if has_audio else []) + ["-movflags", "+faststart", outseg]
            _run(cmd)

        _still_seg(frame0, freeze, frozen, True)
        if intro_video:
            _video_seg(intro_video, intro)
        else:
            _still_seg(card, hold, intro, False)
        with open(lst, "w", encoding="utf-8") as f:
            for p in (frozen, intro, out_path):
                f.write("file '%s'\n" % p.replace("\\", "/"))
        # Видео копируем как есть, звук пересобираем с явной частотой/каналами:
        # иначе concat -c copy принимает первую дорожку и молча кладёт остатки
        # в несовместимый поток (заставка звучит тишиной).
        c_a = ["-c:a", "aac", "-b:a", ab]
        if al:
            c_a += ["-ar", str(sr), "-ac", str(al[1])]
        try:
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                  "-c:v", "copy"] + c_a + ["-movflags", "+faststart", final])
        except RuntimeError:
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                  "-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
                 + c_a + ["-movflags", "+faststart", final])
        shutil.move(final, out_path)
        intro_dur = _probe_duration(intro) or (hold if not intro_video else 0)
        logger.info(f"Заставка применена: стоп-кадр {freeze:.1f} с + заставка "
                    f"{intro_dur:.1f} с (длительность +{freeze + intro_dur:.1f} с).")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path




def export_dubbed(video_path, audio_path, out_path, cfg, logger, subs_path=None,
                  subs_segments=None):
    """Склеивает видео и озвучку, с возможностью подмешивания родной дорожки.

    Субтитры берутся из subs_segments (сегменты из таблицы переводов) либо из
    готового SRT по subs_path. Монтаж (config.json): блюр нижней зоны
    (montage_blur_subs_*), водяной знак (montage_logo_*), стоп-кадр
    (montage_freeze_*). Возвращает путь к итоговому файлу.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Озвучка не найдена: {audio_path}")

    has_audio = _probe(video_path)
    # keep_native_track (из GUI) перекрывает удаление родного звука: true = сохранить.
    if cfg.get("keep_native_track") is not None:
        remove = not bool(cfg.get("keep_native_track"))
        logger.info("keep_native_track=%s -> remove_original_audio=%s",
                    cfg.get("keep_native_track"), remove)
    else:
        remove = cfg.get("remove_original_audio", True)
    bg_level = cfg.get("background_original")

    crf = cfg.get("export_crf", 18)
    ab = cfg.get("export_audio_bitrate", "192k")
    ar = _target_audio_rate(video_path, cfg, has_audio)

    vmeta = _probe_video_meta(video_path)
    vh = vmeta[1] if vmeta else None
    subs = _subtitle_filter(subs_path, cfg, vmeta[:2] if vmeta else None, subs_segments)
    logo = _logo_settings(cfg, video_path, logger)
    logo_inputs = _logo_inputs(logo)

    # Видео-цепочка: блюр нижней зоны -> субтитры -> лого -> [v].
    vparts, vlabel = _video_parts(cfg, subs)
    if logo:
        vparts.extend(_logo_chain(vlabel, logo, 2, cfg, video_path))
        vlabel = "[v]"
    else:
        vparts.append(f"{vlabel}null[v]")
        vlabel = "[v]"
    video_fc = ";".join(vparts)

    def _mk_cmd(fc_parts, maps):
        return (["ffmpeg", "-y", "-i", video_path, "-i", audio_path] + logo_inputs
                + ["-filter_complex", ";".join(fc_parts)] + maps
                + ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                   "-c:a", "aac", "-b:a", ab, "-ar", str(ar),
                   "-movflags", "+faststart", out_path])

    # aresample в конце цепочки + -ar на кодировщике: одна частота на весь файл.
    tail = "aresample=%d" % ar
    if has_audio and bg_level is not None:
        audio_fc = (
            f"[0:a:0]volume={float(bg_level):.3f}[bg];"
            f"[bg][1:a:0]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[raw];"
            f"[raw]loudnorm=I=-16:TP=-1.5:LRA=11,{tail}[mix]"
        )
        cmd = _mk_cmd([video_fc, audio_fc], ["-map", "[v]", "-map", "[mix]"])
    elif has_audio and not remove:
        audio_fc = (
            f"[0:a:0]volume=0.08[bg];"
            f"[bg][1:a:0]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[raw];"
            f"[raw]loudnorm=I=-16:TP=-1.5:LRA=11,{tail}[a]"
        )
        cmd = _mk_cmd([video_fc, audio_fc], ["-map", "[v]", "-map", "[a]"])
    else:
        cmd = _mk_cmd([video_fc, f"[1:a:0]loudnorm=I=-16:TP=-1.5:LRA=11,{tail}[a]"],
                      ["-map", "[v]", "-map", "[a]"])

    logger.info("Экспорт итогового видео через ffmpeg...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg завершился с ошибкой:\n{proc.stderr[-2000:]}")

    if not os.path.exists(out_path):
        raise RuntimeError("Итоговый файл не создан.")
    out_path = _apply_montage_intro(video_path, out_path, cfg, logger)
    logger.info(f"Готово: {out_path}")
    return out_path


def export_subtitles_only(video_path, out_path, cfg, logger, subs_path=None,
                          subs_segments=None):
    """Экспортирует видео с субтитрами, сохраняя оригинальное аудио.

    Используется для роликов, отмеченных как «только субтитры» (воен/политика/плен)
    и для видео, уже находящихся на целевом языке (озвучка не требуется).
    Субтитры берутся из subs_segments либо из SRT по subs_path.
    Монтаж (config.json): блюр нижней зоны, водяной знак, стоп-кадр.
    Громкость нативного звука приводится к стандартному уровню (loudnorm,
    ~-16 LUFS), чтобы тихое исходное аудио не звучало еле слышно; поверх
    применяется subtitles_only_original_volume (по умолчанию 1.0).
    Отключить нормализацию: subtitles_only_normalize_audio=false.
    Возвращает путь к out_path.
    """
    crf = cfg.get("export_crf", 18)
    vol = float(cfg.get("subtitles_only_original_volume", 1.0))
    apply_vol = abs(vol - 1.0) > 1e-6
    # Громкость оригинального звука в режиме «только субтитры». Задача — чтобы
    # тихое исходное аудио (например, среднее -34 дБ) не звучало «еле слышно»:
    # уровень приводится к стандартной громкости через один проход loudnorm
    # (EBU R128, ~-16 LUFS, пик до -1.5 дБ), а явный множитель
    # subtitles_only_original_volume (если != 1) применяется поверх.
    # Отключить нормализацию: subtitles_only_normalize_audio = false.
    af_parts = []
    if cfg.get("subtitles_only_normalize_audio", True):
        af_parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if apply_vol:
        af_parts.append("volume=%.3f" % vol)
    af = ",".join(af_parts)
    re_audio = bool(af)
    has_audio = _probe(video_path)
    vmeta = _probe_video_meta(video_path)
    vh = vmeta[1] if vmeta else None
    subs = _subtitle_filter(subs_path, cfg, vmeta[:2] if vmeta else None, subs_segments)
    logo = _logo_settings(cfg, video_path, logger)
    logo_inputs = _logo_inputs(logo)

    active = cfg.get("montage_blur_subs_enabled") or bool(subs) or bool(logo)
    if not active:
        if not has_audio:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-map", "0:v:0",
                   "-c:v", "copy", "-an", "-movflags", "+faststart", out_path]
            logger.info("Экспорт (только субтитры, без аудио) через ffmpeg...")
        else:
            cmd = ["ffmpeg", "-y", "-i", video_path,
                   "-map", "0:v:0", "-map", "0:a?"]
            if af:
                cmd += ["-af", af]
            cmd += ["-c:v", "copy"]
            if re_audio:
                cmd += ["-c:a", "aac", "-b:a", cfg.get("export_audio_bitrate", "192k")]
            else:
                cmd += ["-c:a", "copy"]
            cmd += ["-movflags", "+faststart", out_path]
            if re_audio:
                logger.info("Экспорт (только субтитры, без фильтров): "
                            "оригинальный звук приведён к норме громкости...")
            else:
                logger.info("Экспорт (только субтитры, без фильтров) через ffmpeg...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg завершился с ошибкой:\n{proc.stderr[-2000:]}")
        if not os.path.exists(out_path):
            raise RuntimeError("Итоговый файл не создан.")
        out_path = _apply_montage_intro(video_path, out_path, cfg, logger)
        logger.info(f"Готово (только субтитры): {out_path}")
        return out_path

    vparts, vlabel = _video_parts(cfg, subs)
    if logo:
        vparts.extend(_logo_chain(vlabel, logo, 1, cfg, video_path))
        vlabel = "[v]"
    else:
        vparts.append(f"{vlabel}null[v]")
        vlabel = "[v]"
    video_fc = ";".join(vparts)

    if not has_audio:
        cmd = (["ffmpeg", "-y", "-i", video_path] + logo_inputs
               + ["-filter_complex", video_fc, "-map", "[v]", "-an",
                  "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                  "-movflags", "+faststart", out_path])
        logger.info("Экспорт (только субтитры, без аудио) через ffmpeg...")
    elif re_audio:
        cmd = (["ffmpeg", "-y", "-i", video_path] + logo_inputs
               + ["-filter_complex", video_fc + f";[0:a:0]{af}[a]",
                  "-map", "[v]", "-map", "[a]",
                  "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                  "-c:a", "aac", "-b:a", cfg.get("export_audio_bitrate", "192k"),
                  "-movflags", "+faststart", out_path])
        logger.info("Экспорт (только субтитры): оригинальный звук приведён к "
                    "норме громкости через ffmpeg...")
    else:
        cmd = (["ffmpeg", "-y", "-i", video_path] + logo_inputs
               + ["-filter_complex", video_fc, "-map", "[v]", "-map", "0:a?",
                  "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                  "-c:a", "copy", "-movflags", "+faststart", out_path])
        logger.info("Экспорт (только субтитры) через ffmpeg...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg завершился с ошибкой:\n{proc.stderr[-2000:]}")
    if not os.path.exists(out_path):
        raise RuntimeError("Итоговый файл не создан.")
    out_path = _apply_montage_intro(video_path, out_path, cfg, logger)
    logger.info(f"Готово (только субтитры): {out_path}")
    return out_path

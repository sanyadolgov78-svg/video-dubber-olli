"""Озвучка целевого (например, французского) текста через Coqui XTTS v2 (нейросетевой синтез).

XTTS клонирует голос по референсному аудиофайлу (speaker_wav) вместо выбора
голоса по имени. Модель скачивается один раз с HuggingFace (~1.8 ГБ), после
чего работает Offline.

Алгоритм дорожки полностью повторяет v2 (непрерывный поток реплик с
фиксированными паузами и единым темпом через atempo), чтобы не менять
настроенные паузы и скорость.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

# Модули, которые тянет torch. После сорванного импорта (например, не хватило
# подкачки) они остаются в sys.modules «наполовину инициализированными».
_TORCH_ROOTS = ("torch", "transformers", "kokoro", "misaki", "TTS", "coqui",
                "phonemizer", "espeakng_loader", "num2words")
# torch выгружаем из sys.modules только если его С-расширение не загружено,
# иначе повторный импорт torch в этом же процессе невозможен.
_SAFE_ROOTS = tuple(r for r in _TORCH_ROOTS if r != "torch")


def _purge_torch_modules():
    """Вычищает недозагруженные модули torch/transformers из памяти процесса.

    Сорванный импорт оставляет модули в sys.modules «наполовину» — отсюда ложная
    ошибка 'cannot import name AlbertModel from transformers' вместо настоящей
    причины. После чистки retry честно либо загружает движок, либо сообщает
    исходную ошибку.

    torch._C — скомпилированное расширение: если оно уже загружено в процесс,
    повторный импорт torch невозможен ('already has a docstring'), поэтому такой
    torch не трогаем. Промываем только верхнеуровневые пакеты.
    """
    roots = _SAFE_ROOTS if "torch._C" in sys.modules else _TORCH_ROOTS
    for name in [n for n in sys.modules if n.split(".")[0] in roots]:
        sys.modules.pop(name, None)

# Глобальный загруженный движок XTTS (тяжёлая модель — грузим один раз).
_xtts_engine = None


def _get_engine(cfg, logger):
    global _xtts_engine
    if _xtts_engine is not None:
        return _xtts_engine
    # Подтверждение некоммерческой лицензии Coqui Public Model License (CPML)
    os.environ["COQUI_TOS_AGREED"] = "1"
    try:
        from TTS.api import TTS

        logger.info("Загрузка XTTS v2 (первый запуск может скачать ~1.8 ГБ)...")
        _xtts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    except Exception:
        # Сорванный импорт оставляет torch/transformers «наполовину загруженными» —
        # чистим, иначе следующая попытка упадёт с ложной ошибкой про AlbertModel.
        _xtts_engine = None
        _purge_torch_modules()
        raise
    logger.info("XTTS v2 загружен.")
    return _xtts_engine


def _reference_wav(cfg, gender=None):
    is_f = gender and str(gender).lower() in ("female", "f", "woman", "w")
    if is_f:
        ref = cfg.get("xtts_reference_f") or cfg.get("xtts_reference_wav") or cfg.get("xtts_speaker_wav")
    else:
        ref = cfg.get("xtts_reference_m") or cfg.get("xtts_reference_wav") or cfg.get("xtts_speaker_wav")
    if not ref or not os.path.exists(str(ref)):
        raise RuntimeError(
            "Для XTTS нужен референсный голос: задайте 'xtts_reference_wav' "
            "(или 'xtts_reference_m'/'xtts_reference_f') в config.json "
            "(путь к .wav с речью на целевом языке, 6+ сек)."
        )
    return str(ref)


def _synthesize_one(engine, text, wav_path, cfg, logger, reference_wav=None):
    """Синтезирует одну фразу XTTS в файл wav_path. Возвращает число сэмплов."""
    language = cfg.get("xtts_language", "en")
    emotions = cfg.get("xtts_emotion")  # опционально "Happy", "Sad", ...
    ref = reference_wav or _reference_wav(cfg)
    kwargs = dict(
        text=text,
        speaker_wav=ref,
        language=language,
        split_sentences=False,
        temperature=float(cfg.get("xtts_temperature", 0.7)),
        repetition_penalty=float(cfg.get("xtts_repetition_penalty", 5.0)),
        top_k=int(cfg.get("xtts_top_k", 50)),
        top_p=float(cfg.get("xtts_top_p", 0.8)),
    )
    if emotions:
        kwargs["emotion"] = emotions

    engine.tts_to_file(**kwargs, file_path=wav_path)
    audio, sr = sf.read(wav_path)
    if len(audio) == 0:
        raise RuntimeError("XTTS не сгенерировал аудио.")
    return audio.astype(np.float32), sr


def _atempo_array(audio, sr, ratio):
    """Меняет темп numpy-аудио без сдвига тембра (ffmpeg atempo)."""
    if abs(ratio - 1.0) < 0.001:
        return audio
    if ratio < 0.5 or ratio > 2.0:
        raise ValueError(f"atempo вне диапазона 0.5..2.0: {ratio}")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, audio, sr, subtype="PCM_16")
        out = tmp[:-4] + "_o.wav"
        cmd = ["ffmpeg", "-y", "-i", tmp, "-filter:a", "atempo=%.3f" % ratio, "-ac", "1", out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a2, _ = sf.read(out)
        return a2.astype(np.float32)
    finally:
        for f in (tmp, tmp[:-4] + "_o.wav"):
            if os.path.exists(f):
                os.remove(f)


def synthesize_timed_track(segments, cfg, output_wav, logger, total_duration, reference_wav=None):
    """Синтезирует озвучку, распределяя реплики по таймкодам SRT.

    segments: [{start, end, text}, ...] — субтитры целевого языка.
    total_duration: длительность итоговой дорожки (сек) = длительности ролика.

    Каждая реплика размещается в дорожке на позиции, соответствующей её
    таймкоду в SRT. Если синтезированная реплика длиннее окна — ускоряется;
    если короче — замедляется для заполнения окна. Речь заканчивается ровно
    там, где заканчивается последний сегмент SRT (без хвостовой тишины).
    Возвращает путь output_wav.
    """
    engine = _get_engine(cfg, logger)
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    # Фильтруем сегменты с текстом и таймкодами.
    valid = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", total_duration))
        except (TypeError, ValueError):
            continue
        if end > start:
            valid.append({"text": text, "start": start, "end": end})

    if not valid:
        raise RuntimeError("Нет текста для озвучки.")

    sr = 22050  # XTTS v2 default sample rate
    target_samples = int(round(total_duration * sr))
    min_ratio = float(cfg.get("tts_tempo_min_ratio", 0.75))
    max_ratio = float(cfg.get("tts_tempo_max_ratio", 1.5))

    logger.info(f"XTTS: синтез {len(valid)} реплик по таймкодам...")

    track = np.zeros(target_samples, dtype=np.float32)
    sr = None

    for i, seg in enumerate(valid, 1):
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            a, s = _synthesize_one(engine, seg["text"], tmp, cfg, logger, reference_wav=reference_wav)
        except Exception as e:
            raise RuntimeError(f"XTTS не удался на реплике {i}: {e}")
        finally:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)

        if a is None or len(a) == 0:
            logger.warning(f"  [{i}/{len(valid)}] XTTS вернул пустое аудио — пропущено.")
            continue

        a = np.asarray(a, dtype=np.float32)
        sr = s if sr is None else s

        # Целевое окно для этой реплики.
        target_start = seg["start"]
        target_end = seg["end"]
        target_dur = target_end - target_start

        natural_dur = len(a) / sr
        speed = float(cfg.get("tts_speed", 1.0))  # <1 — голос медленнее, >1 — быстрее

        # Базовый темп под окно SRT (ffmpeg atempo: >1 быстрее, <1 медленнее).
        if natural_dur > target_dur * 1.15:
            # Слишком длинная — ускоряем до целевой длины (не резче max_ratio).
            base = min(natural_dur / target_dur, max_ratio)
        elif natural_dur < target_dur * 0.75:
            # Слишком короткая — замедляем для заполнения окна (не медленнее min_ratio).
            base = max(natural_dur / target_dur, min_ratio)
        else:
            base = 1.0

        ratio = max(min(base * speed, 2.0), 0.5)
        if abs(ratio - 1.0) > 0.01:
            a = _atempo_array(a, sr, ratio)
            if base < 0.99:
                label = "замедлено"
            elif base > 1.01:
                label = "ускорено"
            else:
                label = "темп"
            logger.info(
                f"  [{i}/{len(valid)}] {label} x{ratio:.2f} "
                f"({natural_dur:.1f}с->{natural_dur / ratio:.1f}с)"
            )

        # Размещаем в дорожке на позиции таймкода.
        start_sample = int(round(target_start * sr))
        end_sample = start_sample + len(a)
        if end_sample > target_samples:
            a = a[:target_samples - start_sample]
            end_sample = target_samples
        if start_sample < target_samples and len(a) > 0:
            track[start_sample:start_sample + len(a)] += a

        logger.info(f"  [{i}/{len(valid)}] размещено {target_start:.1f}с–{target_end:.1f}с.")

    if sr is None:
        raise RuntimeError("XTTS не сгенерировал аудио.")

    sf.write(output_wav, track, sr, subtype="PCM_16")

    # Громкость речевой части (без тишины).
    speech_mask = np.abs(track) > 0.005
    speech_sec = np.sum(speech_mask) / sr if sr else 0
    logger.info(
        f"XTTS: речь {speech_sec:.1f}с из {total_duration:.1f}с видео -> {output_wav}"
    )
    return output_wav


_kokoro_engine = None


def _get_kokoro_engine(lang_code, logger):
    """Загружает Kokoro-пайплайн глобально (тяжёлая загрузка — один раз)."""
    global _kokoro_engine
    import os as _os
    # espeak-ng нужен для phonemizer'а (слагается словарь/фонемы). Задаём пути.
    # На Windows это локальная сборка espeak-ng, на Linux — системный пакет
    # (libespeak-ng1 + espeak-ng-data), который ставится через apt, поэтому
    # переменные там НЕ выставляем — phonemizer найдёт его сам.
    if _os.name == "nt":
        _espeak = _os.path.join(_os.environ.get("SKIT_MODELS_DIR") or r"C:\skit_models",
                                "espeak-ng", "eSpeak NG")
        _os.environ.setdefault("ESPEAK_DATA_PATH", _os.path.join(_espeak, "espeak-ng-data"))
        _os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _os.path.join(_espeak, "libespeak-ng.dll"))
    if _kokoro_engine is not None:
        return _kokoro_engine

    try:
        from kokoro import KPipeline

        # misaki.espeak при импорте сам переопределяет пути espeak-ng на пакетные
        # (espeakng_loader), где DLL/данные битые (access violation и «language not
        # supported»). Сбрасываем их в None, чтобы phonemizer взял пути из env
        # (PHONEMIZER_ESPEAK_LIBRARY / data из ESPEAK_DATA_PATH), заданные выше.
        from phonemizer.backend.espeak.wrapper import EspeakWrapper as _EW
        _EW.set_library(None)
        _EW.set_data_path(None)

        logger.info("Загрузка Kokoro-82M (lang=%s)...", lang_code)
        _kokoro_engine = KPipeline(lang_code=lang_code)
    except Exception:
        # Чаще всего это нехватка памяти под torch (WinError 1455). Модули
        # вычищаем, чтобы следующая попытка показала настоящую ошибку, а не
        # ложную 'cannot import name AlbertModel from transformers'.
        _kokoro_engine = None
        _purge_torch_modules()
        raise
    logger.info("Kokoro загружен.")
    return _kokoro_engine


def _kokoro_voice_for_gender(cfg, source_gender):
    """Выбирает голос Kokoro по полу спикера (если 'kokoro_voice_m/f' заданы)."""
    is_f = source_gender and str(source_gender).lower() in ("female", "f", "woman", "w")
    if is_f:
        return cfg.get("kokoro_voice_f") or cfg.get("kokoro_voice", "af_heart")
    return cfg.get("kokoro_voice_m") or cfg.get("kokoro_voice", "am_michael")


def kokoro_synthesize_timed(segments, cfg, output_wav, logger, total_duration, voice):
    """Синтезирует озвучку через Kokoro с ЕДИНЫМ темпом для всех фраз.

    Каждая фраза озвучивается одним множителем темпа kokoro_speed (<1 = медленнее),
    без пофразовой деформации. Реплики центрируются в своих окнах SRT; при выходе
    за границы ролика — обрезаются. Возвращает путь output_wav.
    """
    import numpy as _np
    import soundfile as _sf

    lang_code = cfg.get("kokoro_lang", "a")
    sr = 24000
    engine = _get_kokoro_engine(lang_code, logger)

    ratio = float(cfg.get("kokoro_speed", 0.9))
    ratio = max(min(ratio, 1.15), 0.70)

    target_samples = int(round(total_duration * sr))
    track = _np.zeros(target_samples, dtype=_np.float32)

    # Шаг 1. Синтезируем все фразы с базовым темпом и запоминаем их длительности.
    phrases = []  # (start, next_start, end, audio_after_base)
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", total_duration))
        except (TypeError, ValueError):
            continue
        if end <= start or start < 0.0:
            continue
        chunks = []
        for _gs, _ps, audio in engine(text, voice=voice):
            a = _np.asarray(audio, dtype=_np.float32)
            if a.ndim > 1:
                a = a.mean(axis=1)
            chunks.append(a)
        if not chunks:
            continue
        a = _np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if abs(ratio - 1.0) > 0.01:
            a = _atempo_array(a, sr, ratio)
        phrases.append((start, end, a))

    if not phrases:
        raise RuntimeError("Kokoro не сгенерировал аудио.")

    # Шаг 2. Мягкие границы: фраза может звучать вплоть до начала СЛЕДУЮЩЕЙ фразы
    # (а не до конца своего окна) — это позволяет говорить спокойно и оставляет
    # меньше пауз. Для этого находим фактический лимит каждой фразы.
    limits = []  # доступная длительность для каждой фразы (сек)
    for idx, (_s, _e, a) in enumerate(phrases):
        if idx + 1 < len(phrases):
            nxt = phrases[idx + 1][0]
        else:
            nxt = total_duration
        avail = max(nxt - _s - 0.10, 0.15)  # минус небольшой зазор перед следующей фразой
        limits.append(avail)

    # Единый темп для ВСЕХ фраз: минимальное ускорение, чтобы каждая фраза
    # вписалась в свой лимит. Оно не агрессивное: cap по умолчанию 1.35.
    # Если все фразы свободно помещаются — темп остаётся естественным (ratio).
    need_max = 1.0
    for (_s, _e, a), lim in zip(phrases, limits):
        need_max = max(need_max, (len(a) / sr) / lim)
    cap = float(cfg.get("kokoro_max_tempo", 1.35))
    uniform = min(max(need_max, 1.0), cap)
    if uniform > 1.01:
        logger.info(f"Kokoro: единый темп x{ratio * uniform:.2f} по самым узким местам (доп. x{uniform:.2f})")
        phrases = [(s, e, _atempo_array(a, sr, uniform)) for s, e, a in phrases]

    # Шаг 3. Размещаем фразы подряд в начале своих зон (без центрирования,
    # поэтому пауз становится заметно меньше).
    def _place(a, start, first_end):
        """Размещает фразу в дорожке в позиции start, не трогая зону следующей."""
        ss = int(round(start * sr))
        # Граница зоны — конец окна (следующая фраза), за неё не вылезаем.
        se_end = int(round(first_end * sr))
        se2 = ss + len(a)
        if se2 > se_end:
            a = a[: se_end - ss]
            se2 = se_end
        if se2 > target_samples:
            a = a[: target_samples - ss]
            se2 = target_samples
        if ss < target_samples and len(a) > 0:
            track[ss : ss + len(a)] += a

    for i, (_start, _end, a) in enumerate(phrases, 1):
        final = len(a) / sr
        lim = limits[i - 1]
        if a.size and final > lim:
            # Страховка: даже при равномерном темпе фраза выходит за лимит —
            # плавно обрежем по границе зоны (не по жёсткому концу окна).
            keep = max(0, int(lim * sr))
            if a.size > keep:
                fade = min(keep, int(0.05 * sr))
                a = a[:keep].copy()
                if fade > 0:
                    a[-fade:] *= _np.linspace(1.0, 0.0, fade, dtype=_np.float32)
                final = len(a) / sr
        pos = _start  # прижимаем фразу к началу её зоны — пауз меньше
        _place(a, pos, _end)
        logger.info(
            f"  [{i}/{len(phrases)}] фраза {final:.1f}с, "
            f"окно {_start:.1f}-{_end:.1f}с, размещено {pos:.1f}с"
        )

    _sf.write(output_wav, track, sr, subtype="PCM_16")
    speech = _np.mean(_np.abs(track) > 0.005) * len(track) / sr if len(track) else 0.0
    logger.info(f"Kokoro: речь {speech:.1f}с из {total_duration:.1f}с видео -> {output_wav}")
    return output_wav


def _edge_voice_for_gender(cfg, source_gender, lang="fr"):
    """Выбирает голос Edge-TTS по полу спикера (edge_voice_m/f из cfg)."""
    is_f = source_gender and str(source_gender).lower() in ("female", "f", "woman", "w")
    pref = cfg.get(f"edge_voice_{lang}_{'f' if is_f else 'm'}")
    pref = pref or cfg.get(f"edge_voice_{'f' if is_f else 'm'}")
    if pref:
        return pref
    bases = {"fr": ("fr-FR-DeniseNeural", "fr-FR-HenriNeural"),
             "ru": ("ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"),
             "en": ("en-US-JennyNeural", "en-US-GuyNeural")}
    f, m = bases.get(lang, ("", "fr-FR-HenriNeural"))
    return f if is_f else m


def _edge_synthesize_phrase(text, voice, rate, tmp_wav, logger):
    """Синтезирует одну фразу через Microsoft Edge-TTS (по сети).

    Возвращает (audio_float32, sr). Использует временный .mp3 и ffmpeg
    для переконвертации в 24 кГц mono WAV (как в остальном пайплайне).
    """
    import asyncio
    import edge_tts

    mp3 = tmp_wav[:-4] + ".mp3"
    for f in (tmp_wav, mp3):
        if os.path.exists(f):
            os.remove(f)

    async def _go():
        comm = edge_tts.Communicate(text, voice=voice, rate=rate)
        await comm.save(mp3)

    asyncio.run(_go())
    if not os.path.exists(mp3) or os.path.getsize(mp3) == 0:
        raise RuntimeError("Edge-TTS не сгенерировал аудио.")

    cmd = ["ffmpeg", "-y", "-i", mp3, "-ar", "24000", "-ac", "1",
           "-c:a", "pcm_s16le", tmp_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(tmp_wav)
    return audio.astype(np.float32), sr


def edge_synthesize_timed(segments, cfg, output_wav, logger, total_duration, voice,
                          source_gender=None):
    """Синтезирует озвучку через Microsoft Edge-TTS (по сети) с единым темпом.

    Повторяет тайминг-логику Kokoro: фразы озвучиваются с единым множителем
    темпа, прижимаются к началу зон SRT, не вылезая за границы следующих фраз.
    Возвращает путь output_wav.
    """
    import numpy as _np

    sr = 24000
    rate = str(cfg.get("edge_rate", "+0%"))
    # edge_tts: "-10%" замедляет темп речи (<0 = медленнее, >0 = быстрее).
    speed = float(cfg.get("edge_speed", 1.0))
    if abs(speed - 1.0) > 0.005:
        pct = int(round((speed - 1.0) * 100))
        rate = ("+%d%%" if pct >= 0 else "-%d%%") % abs(pct)
        logger.info(f"Edge-TTS: темп речи {rate} (edge_speed={speed})")
    target_samples = int(round(total_duration * sr))
    track = _np.zeros(target_samples, dtype=_np.float32)

    # Шаг 1. Синтезируем фразы базовым темпом, запоминаем длительности.
    phrases = []  # (start, end, audio)
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", total_duration))
        except (TypeError, ValueError):
            continue
        if end <= start or start < 0.0:
            continue
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            a, s = _edge_synthesize_phrase(text, voice, rate, tmp, logger)
        except Exception as e:
            logger.warning(f"  Edge-TTS: фраза пропущена ({e})")
            continue
        finally:
            for f in (tmp, tmp[:-4] + ".mp3"):
                if os.path.exists(f):
                    os.remove(f)
        a = _np.asarray(a, dtype=_np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if len(a) == 0:
            continue
        phrases.append((start, end, a))

    if not phrases:
        raise RuntimeError("Edge-TTS не сгенерировал аудио.")

    # Шаг 2. Мягкие границы: фраза может звучать вплоть до начала следующей.
    limits = []
    for idx, (_s, _e, a) in enumerate(phrases):
        nxt = phrases[idx + 1][0] if idx + 1 < len(phrases) else total_duration
        avail = max(nxt - _s - 0.10, 0.15)
        limits.append(avail)

    need_max = 1.0
    for (_s, _e, a), lim in zip(phrases, limits):
        need_max = max(need_max, (len(a) / sr) / lim)
    cap = float(cfg.get("edge_max_tempo", 1.35))
    uniform = min(max(need_max, 1.0), cap)
    if uniform > 1.01:
        logger.info(f"Edge-TTS: единый темп x{uniform:.2f} по самым узким местам")
        phrases = [(s, e, _atempo_array(a, sr, uniform)) for s, e, a in phrases]

    # Шаг 3. Размещение фраз подряд от начала зон.
    def _place(a, start, first_end):
        ss = int(round(start * sr))
        se_end = int(round(first_end * sr))
        se2 = ss + len(a)
        if se2 > se_end:
            a = a[: se_end - ss]
            se2 = se_end
        if se2 > target_samples:
            a = a[: target_samples - ss]
            se2 = target_samples
        if ss < target_samples and len(a) > 0:
            track[ss : ss + len(a)] += a

    for i, (_start, _end, a) in enumerate(phrases, 1):
        final = len(a) / sr
        lim = limits[i - 1]
        if a.size and final > lim:
            keep = max(0, int(lim * sr))
            if a.size > keep:
                fade = min(keep, int(0.05 * sr))
                a = a[:keep].copy()
                if fade > 0:
                    a[-fade:] *= _np.linspace(1.0, 0.0, fade, dtype=_np.float32)
                final = len(a) / sr
        pos = _start
        _place(a, pos, _end)
        logger.info(
            f"  [{i}/{len(phrases)}] Edge фраза {final:.1f}с, "
            f"окно {_start:.1f}-{_end:.1f}с, размещено {pos:.1f}с"
        )

    sf.write(output_wav, track, sr, subtype="PCM_16")
    speech = _np.mean(_np.abs(track) > 0.005) * len(track) / sr if len(track) else 0.0
    logger.info(f"Edge-TTS: речь {speech:.1f}с из {total_duration:.1f}с видео -> {output_wav}")
    return output_wav


def resolve_tts_engine(cfg, source_gender=None):
    """Определяет движок озвучки с учётом пола спикера.

    'auto' (или нераспознанное) ->
        женский спикер: kokoro (ff_siwis — единственный французский голос);
        мужской/неизвестный: edge (Microsoft Edge-TTS, локально стабилен,
        работает по сети) — если edge включён в конфиге (edge_enabled),
        иначе xtts (клонирование голоса по референсу).
    'kokoro'/'xtts'/'edge' -> принудительно, независимо от пола.
    """
    engine = str(cfg.get("tts_engine", "auto")).lower().strip()
    if engine not in ("auto", "kokoro", "xtts", "edge"):
        engine = "auto"
    if engine != "auto":
        return engine
    # По полу: ff_siwis — женский голос, поэтому мужика отправляем в XTTS/Edge.
    is_f = source_gender and str(source_gender).lower() in ("female", "f", "woman", "w")
    if not is_f and cfg.get("edge_enabled"):
        return "edge"
    return "kokoro" if is_f else "xtts"


def synth_timed_with_retry(segments, cfg, voice_dir, output_wav, source_gender, logger, total_duration):
    """Синтезирует озвучку через выбранный движок (cfg['tts_engine'] / 'auto').

    'kokoro' -> Kokoro-82M (единый темп, по голосу из kokoro_voice_m/f;
                ref-клонирование не используется, voice_dir игнорируется).
    'xtts'   -> Coqui XTTS v2 по референсу спикера.
    'edge'   -> Microsoft Edge-TTS (по сети; локально стабилен, не требует
                тяжёлых моделей; голос edge_voice_m/f).
    'auto'   -> по полу спикера: female -> kokoro, иначе edge (если
                edge_enabled=true иначе xtts).
    Возвращает (wav_path, used_voice|None).
    """
    engine_name = resolve_tts_engine(cfg, source_gender)
    if engine_name == "kokoro":
        try:
            voice = _kokoro_voice_for_gender(cfg, source_gender)
            wav = kokoro_synthesize_timed(segments, cfg, output_wav, logger, total_duration, voice)
            return wav, voice
        except Exception as e:
            logger.error(f"Озвучка Kokoro провалилась: {e}")
            return None, None
    if engine_name == "edge":
        try:
            lang = str(cfg.get("xtts_language", "fr")).split("-")[0].lower()
            voice = _edge_voice_for_gender(cfg, source_gender, lang)
            logger.info(
                f"Edge-TTS: голос для '{source_gender}' -> {voice} (lang={lang})"
            )
            wav = edge_synthesize_timed(
                segments, cfg, output_wav, logger, total_duration, voice,
                source_gender=source_gender,
            )
            return wav, voice
        except Exception as e:
            logger.error(f"Озвучка Edge-TTS провалилась: {e}")
            return None, None
    # По умолчанию и для 'xtts' — XTTS v2 (клонирование голоса по референсу).
    try:
        ref = _reference_wav(cfg, source_gender)
        logger.info(f"XTTS: голос для '{source_gender}' -> {os.path.basename(str(ref))}")
        wav = synthesize_timed_track(segments, cfg, output_wav, logger, total_duration, reference_wav=ref)
        return wav, ref
    except Exception as e:
        logger.error(f"Озвучка XTTS провалилась: {e}")
        return None, None

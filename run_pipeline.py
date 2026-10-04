"""
Главный конвейер автоматической озвучки видео на целевой язык (по умолчанию французский).

Полный поток (одноэтапная транскрибация):
  1. Забирает каждый видеофайл из папки input/
  2. Транскрибирует речь (faster-whisper), определяет язык и пол говорящего
  3. Сразу переводит реплики на ДВА языка (русский + целевой) и собирает
     одну таблицу .rtf на проверку носителю
  4. Ждёт правки таблицы; после сохранения файла читает её обратно
  5. Полишинг перевода через LLM (Ollama) — опционально
  6. Озвучивает текст (Kokoro / XTTS / Edge по полу спикера)
  7. Экспортирует итоговое видео (ffmpeg) в папку output/

Запуск:
    python run_pipeline.py            # обработать все видео в input/ за один проход
    python run_pipeline.py --watch    # авто: новые видео + правки таблицы -> продолжение без ручного запуска
    python run_pipeline.py some.mp4   # обработать конкретный файл
"""

import os
import re
import sys
import time
import shutil
import argparse
import hashlib
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config
from pipeline import transcribe
from pipeline import transcript
from pipeline import rtf_table
from pipeline import sensitive
from pipeline import translate
from pipeline import polish
from pipeline import tts
from pipeline import export

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv"}


def _translate_both(source_segments, cfg, source_lang, target_lang, logger):
    """Переводит сегменты сразу на два языка: на русский и на целевой.

    Нога 1 (оригинал -> русский) идёт через mbart (50 языков) — если оригинал
    уже русский, текст берётся как есть. Нога 2 (русский -> целевой) идёт
    через направленную пару opus-mt из конфига, поэтому русский перевод
    используется повторно, а не пересчитывается каскадом заново.
    """
    src = (source_lang or "").lower().strip()
    ru_lang = "ru"

    if src == ru_lang:
        logger.info("Оригинал на русском — русская колонка = исходный текст.")
        ru_segments = [dict(s, text=(s.get("text") or "").strip()) for s in source_segments]
    else:
        logger.info(f"Перевод оригинала ({src or 'auto'}) на русский (mbart, 50 языков)...")
        cfg_ru = dict(cfg)
        cfg_ru["translate_model"] = "mbart"
        ru_segments = translate.translate_segments(
            source_segments, cfg_ru, src, logger, tgt_lang=ru_lang
        )
    translate.unload_models()  # освобождаем mbart (~1.5 ГБ) до загрузки opus

    if ru_lang == target_lang:
        logger.info(f"Целевой язык совпадает с русским ({target_lang}).")
        target_segments = [dict(s) for s in ru_segments]
    else:
        logger.info(f"Перевод русского на {target_lang} (opus-mt)...")
        cfg_tgt = dict(cfg)
        target_segments = translate.translate_segments(
            ru_segments, cfg_tgt, ru_lang, logger, tgt_lang=target_lang
        )
    translate.unload_models()

    ru_count = sum(1 for s in ru_segments if (s.get("text") or "").strip())
    tgt_count = sum(1 for s in target_segments if (s.get("text") or "").strip())
    logger.info(f"Переводы готовы: русский — {ru_count} реплик, "
                f"{target_lang} — {tgt_count} реплик.")
    if not tgt_count:
        logger.error("Целевой перевод пуст — таблица будет без текста озвучки.")
    return ru_segments, target_segments


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("dubbing")


def _notify_done(logger, label):
    """Звуковой сигнал + отметка ГОТОВО в лог по завершении озвучки видеофайла."""
    try:
        import winsound
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            winsound.Beep(880, 250)
            winsound.Beep(1174, 350)
    except Exception:
        pass
    logger.info(f"\n{'='*60}\n[ГОТОВО] Озвучка завершена: {label}\n{'='*60}\n")


def _outputs_up_to_date(table_rtf, dubbed_wav, final_mp4):
    """Озвучка и итоговый mp4 уже соответствуют текущей версии таблицы.

    Проверка нужна ДО полишинга: иначе watcher на каждом цикле опроса заново
    прогонял LLM-полишинг для ролика, который давно готов, и заваливал Ollama
    ошибками 500, хотя переводы в таблице не менялись.
    """
    if not (os.path.exists(dubbed_wav) and os.path.exists(final_mp4)):
        return False
    if os.path.getmtime(table_rtf) > os.path.getmtime(dubbed_wav):
        return False
    if not getattr(export, "_dubbed_mp4_valid", lambda p: True)(final_mp4):
        return False
    return os.path.getmtime(dubbed_wav) <= os.path.getmtime(final_mp4)


def _retranslate_on_source_edit(table_rtf, table_orig, table_json, meta, cfg, logger):
    """Правка столбца «Оригинал» (исходник транскрибации) -> пересчёт французского.

    Текущий RTF сверяется с контрольной копией автогенерации по текстам столбцов:
      * если правился ТОЛЬКО оригинал — французский перевод устарел: прогоняем
        перевод заново (mbart до русского, затем opus до целевого), переписываем
        третий столбец таблицы, таймкоды из сегментов сохраняются;
      * если правился французский столбец (в т.ч. вместе с оригиналом) — таблица
        не трогается, дальше работает прежний алгоритм.
    Возвращает {"source","ru","target"} для продолжения обработки, либо None.
    """
    if cfg.get("retranslate_on_source_edit", True) is False:
        return None
    if not meta or not (meta.get("language") or "").strip():
        logger.warning("Язык источника не известен (meta.json) — правку оригинала без "
                       "повторного перевода применить нельзя.")
        return None

    target_lang = (cfg.get("target_lang") or "fr").lower().strip()
    for p in (table_rtf, table_orig):
        if not os.path.exists(p):
            return None

    def _read(path):
        r = rtf_table.read_translation_table(
            path, table_json, ru_lang="ru", target_lang=target_lang)
        return r if r else None

    cur = _read(table_rtf)
    base = _read(table_orig)
    if not cur or not base:
        return None
    if not any((s.get("text") or "").strip() for s in cur["source"]):
        return None

    def _index(segs):
        return {(round(float(s["start"]), 3), round(float(s["end"]), 3)):
                (s.get("text") or "").strip()
                for s in segs if s.get("start") is not None and s.get("end") is not None}

    c_src, c_tgt = _index(cur["source"]), _index(cur["target"])
    b_src, b_tgt = _index(base["source"]), _index(base["target"])
    src_changed = (set(c_src) != set(b_src)) or any(
        c_src.get(k, None) != b_src.get(k, None) for k in set(c_src) & set(b_src))
    tgt_changed = (set(c_tgt) != set(b_tgt)) or any(
        c_tgt.get(k, None) != b_tgt.get(k, None) for k in set(c_tgt) & set(b_tgt))

    if not (src_changed and not tgt_changed):
        return None

    detected = (meta.get("language") or "").lower().strip() or "auto"
    source_segments = cur["source"]
    logger.info("Обнаружена правка столбца «Оригинал» — перевожу французский "
                "заново, таймкоды сохраняю...")
    ru_segments, target_segments = _translate_both(
        source_segments, cfg, detected, target_lang, logger)
    rtf_table.write_translation_table(
        table_rtf, table_json, source_segments, ru_segments, target_segments,
        source_lang=detected or "auto", ru_lang="ru", target_lang=target_lang)
    # Правка потреблена: повторно пересчитывать не нужно, пока таблицу не меняют
    # снова. Такой же механизм, как у автокопии при первом создании.
    shutil.copyfile(table_rtf, table_orig)
    logger.info(f"Французский столбец обновлён после правки оригинала: {table_rtf}")
    return {"source": source_segments, "ru": ru_segments, "target": target_segments}


def _import_txt_edits(work_dir, base, table_rtf, table_json, target_lang, logger):
    """Переносит правки носителя из .transcript_otpravka.txt в таблицу .table.rtf.

    Возвращает True, если в txt есть реальные изменения относительно таблицы и они
    записаны в rtf (mtime таблицы становится свежее .orig — правка потреблена).
    """
    txt = os.path.join(work_dir, base + ".transcript_otpravka.txt")
    if not os.path.isfile(txt):
        return False
    pat = re.compile(r"^\[[0-9.]+\]\s+(RU|FR):\s*(.*)$")
    ru_lines, fr_lines = [], []
    with open(txt, encoding="utf-8") as f:
        for raw in f:
            m = pat.match(raw.strip())
            if not m:
                continue
            if m.group(1) == "RU":
                ru_lines.append(m.group(2).strip())
            else:
                fr_lines.append(m.group(2).strip())
    if not fr_lines:
        return False
    data = rtf_table.read_translation_table(
        table_rtf, table_json, ru_lang="ru", target_lang=target_lang)
    if not data:
        return False
    new_ru = [dict(s) for s in (data.get("ru") or [])]
    new_tgt = [dict(s) for s in (data.get("target") or [])]
    changed = False
    for i in range(min(len(fr_lines), len(new_tgt))):
        t = fr_lines[i]
        if (new_tgt[i].get("text") or "").strip() != t:
            new_tgt[i]["text"] = t
            changed = True
    for i in range(min(len(ru_lines), len(new_ru))):
        t = ru_lines[i]
        if (new_ru[i].get("text") or "").strip() != t:
            new_ru[i]["text"] = t
            changed = True
    if not changed:
        return False
    try:
        rtf_table.write_translation_table(
            table_rtf, table_json, data.get("source") or new_ru, new_ru, new_tgt,
            source_lang=data.get("source_lang") or "auto",
            ru_lang="ru", target_lang=target_lang)
    except Exception as e:
        logger.error("Не удалось применить правки из %s: %s", txt, e)
        return False
    logger.info("Правки из %s перенесены в таблицу %s (реплик FR: %d)",
                os.path.basename(txt), table_rtf, len(fr_lines))
    return True


def process_file(video_path, cfg, logger):
    base = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = os.path.join(config.WORK_DIR, base)
    os.makedirs(work_dir, exist_ok=True)

    dubbed_wav = os.path.join(work_dir, "dubbed.wav")
    # Суффикс имени выходного файла = целевой язык (_en / _fr)
    _tlang = (cfg.get("target_lang") or "en").lower().strip()
    target_lang = _tlang
    final_mp4 = os.path.join(config.OUTPUT_DIR, base + "_" + _tlang + ".mp4")

    # Метаданные распознавания (язык/пол) — нужны для выбора голоса
    meta_json = os.path.join(work_dir, base + ".meta.json")

    # 1. Одноэтапная транскрибация: один проход whisper -> сразу перевод
    #    (на русский для полишинга/sensitive и на целевой для озвучки)
    #    в одной таблице RTF на проверку носителю. Отдельные SRT не создаются:
    #    источник текста и таймкодов для озвучки — французский столбец таблицы.
    table_rtf = os.path.join(work_dir, base + ".table.rtf")
    table_orig = table_rtf + ".orig"
    table_json = os.path.join(work_dir, base + ".table.json")

    logger.info(f"\n===== Обработка: {os.path.basename(video_path)} =====")

    if not os.path.exists(table_rtf):
        tr = transcribe.transcribe(video_path, cfg, logger)
        if not tr["segments"]:
            logger.warning("Не удалось разобрать речь — файл пропущен.")
            return None

        transcript.save_meta(meta_json, language=tr["language"], gender=tr["gender"])
        detected = (tr["language"] or "").lower().strip()
        logger.info(f"Язык оригинала: {tr['language']}, пол говорящего: {tr['gender']}")

        ru_segments, target_segments = _translate_both(
            tr["segments"], cfg, detected, target_lang, logger
        )
        rtf_table.write_translation_table(
            table_rtf, table_json, tr["segments"], ru_segments, target_segments,
            source_lang=detected or "auto", ru_lang="ru", target_lang=target_lang,
        )
        shutil.copyfile(table_rtf, table_orig)

        logger.info("\n=== ТАБЛИЦА ПЕРЕВОДОВ СОЗДАН (RTF) ===")
        logger.info(f"Файл: {table_rtf}")
        logger.info("Столбцы: таймкоды | оригинал | перевод на "
                    f"{target_lang}.")
        logger.info("Правьте текст в Word и сохраните файл — конвейер продолжится")
        logger.info("сам: полишинг -> озвучка -> экспорт.")
        return None

    # 2. Таблица ещё не отредактирована носителем — ждём правки. Правки могут
    #    прийти и из текстового транскрипта (.transcript_otpravka.txt) — если он
    #    изменён, переносим их в таблицу и продолжаем.
    if not rtf_table.table_is_edited(table_rtf, table_orig):
        if not _import_txt_edits(work_dir, base, table_rtf, table_json, target_lang, logger):
            logger.info(f"Таблица переводов ещё не отредактирована: {table_rtf}")
            logger.info("Правьте переводы в Word (.table.rtf) или Блокноте "
                        "(.transcript_otpravka.txt), сохраните файл — конвейер продолжится сам.")
            return None

    table = rtf_table.read_translation_table(
        table_rtf, table_json, ru_lang="ru", target_lang=target_lang
    )
    if not table or not any((s.get("text") or "").strip() for s in table["target"]):
        logger.warning("В таблице нет ни одной реплики с переводом — файл пропущен.")
        return None

    source_segments = table["source"] or table["ru"]
    ru_segments = table["ru"] or source_segments
    target_segments = table["target"]
    logger.info(f"Таблица отредактирована: {table_rtf} "
                f"(реплик: {len(target_segments)}).")

    meta = transcript.load_meta(meta_json) or {}
    gender = meta.get("gender") or "unknown"

    # Правка столбца «Оригинал» (исходник транскрибации): французский столбец
    # пересчитывается заново с сохранением таймкодов. Если правился французский
    # столбец — таблица остаётся как есть, работает прежний алгоритм.
    rerun = _retranslate_on_source_edit(table_rtf, table_orig, table_json, meta, cfg, logger)
    if rerun:
        source_segments = rerun["source"]
        ru_segments = rerun["ru"]
        target_segments = rerun["target"]

    # Детект sensitive по РУССКОМУ тексту (именно русский словарь ключевых слов
    # использует модуль sensitive) — он всегда есть в боковой подложке таблицы.
    sensitive_result = sensitive.detect(ru_segments, cfg, work_dir, base, logger)
    sensitive_data = sensitive_result

    # Принудительный пол (из GUI "Озвучка") перекрывает авто-определение по whisper.
    forced_gender = (cfg.get("forced_gender") or "").strip().lower()
    if forced_gender in ("female", "f", "woman", "w", "male", "m", "man"):
        gender = forced_gender
        logger.info(f"Принудительный голос задан через GUI: forced_gender={forced_gender}")

    # Уже озвученный и корректно экспортированный ролик пропускаем сразу,
    # не затрагивая LLM: таблица не менялась с момента последней озвучки.
    if cfg.get("dub_enabled", True) \
            and _outputs_up_to_date(table_rtf, dubbed_wav, final_mp4):
        logger.info(
            f"Видео уже озвучено и корректно экспортировано: {final_mp4}. "
            "Внесите правки в таблицу переводов — озвучка обновится сама."
        )
        return None

    # 3. Полишинг переводчика: правит целевой (французский) текст, сверяясь с
    #    русским из подложки таблицы. Результат идёт сразу в озвучку и субтитры;
    #    сама таблица остаётся единственным источником правок носителя.
    polished_segments = polish.polish_segments(ru_segments, target_segments, cfg, logger)
    if polished_segments is None:
        logger.error("Конвейер остановлен: полишинг обязателен, но Ollama недоступна.")
        return None
    target_segments = polished_segments

    # Длительность ролика — чтобы распределить озвучку на весь тайминг.
    total_duration = export.get_video_duration(video_path)
    if not total_duration:
        total_duration = max((s.get("end", 0.0) for s in target_segments), default=0.0)
    if total_duration <= 0:
        total_duration = len(target_segments) * 5.0

    # Субтитры вшиваются из тех же сегментов, что и озвучка (текст таблицы).
    show_subs = cfg.get("show_subtitles", True)
    subs_segments = target_segments if show_subs else None

    # 3б. Sensitive-видео: по умолчанию озвучиваем как обычные
    #     (dub_sensitive_videos=true в config.json). Если флаг выключен —
    #     только субтитры без озвучки (прежнее поведение).
    if sensitive_data.get("sensitive") and not cfg.get("dub_sensitive_videos", True):
        if os.path.exists(final_mp4) and os.path.getmtime(table_rtf) <= os.path.getmtime(final_mp4):
            logger.info(f"Видео в режиме «только субтитры» уже готово (таблица не менялась): {final_mp4}")
            return final_mp4
        export.export_subtitles_only(video_path, final_mp4, cfg, logger,
                                     subs_segments=subs_segments)
        logger.info(f"Видео классифицировано как sensitive, озвучка отключена "
                    f"(dub_sensitive_videos=false): {final_mp4}")
        return final_mp4

    # 3а. Озвучку можно отключить из GUI («Не озвучивать»): распознавание, перевод и
    #     полишинг остаются, но аудио не заменяется и mp4 не создаётся.
    if not cfg.get("dub_enabled", True):
        logger.info(
            f"Озвучка отключена (dub_enabled=false): перевод проверен в таблице {table_rtf}. "
            "Шаг озвучки/экспорта пропущен."
        )
        return None

    # 3б. Если таблица уже озвучена (носитель её после этого не правил) —
    #     повторно не озвучиваем. Сравниваем по таблице: иначе LLM-правки
    #     заставляли бы переозвучивать ролик каждый цикл.
    if os.path.exists(dubbed_wav) and \
            os.path.getmtime(table_rtf) <= os.path.getmtime(dubbed_wav):
        # Итоговый mp4 мог остаться битым после падения ffmpeg на x264 (malloc) —
        # не переиспользуем его вслепую: переэкспортируем прямо здесь, оставив
        # готовую озвучку и таблицу нетронутыми.
        logger.info(
            f"Озвучка актуальна (таблица не менялась), но итоговый mp4 битый "
            f"или устарел — переэкспортирую из готового dubbed.wav: {final_mp4}"
        )
        if os.path.exists(final_mp4):
            os.remove(final_mp4)
        export.export_dubbed(video_path, dubbed_wav, final_mp4, cfg, logger,
                             subs_segments=subs_segments)
        return final_mp4

    # 3в. Озвучка по сегментам, распределённая по таймкодам ролика (с ретраями голосов).
    #    Двухвариантная озвучка: основной голос + спокойный вариант (если включено).
    #    Референсы спикера достаются внутри _synth_variant/_obtain_ref.
    wav_main = _dub_variants(video_path, target_segments, cfg, target_lang,
                             gender, total_duration, work_dir, base, source_segments, logger)
    if wav_main is None:
        return None

    _notify_done(logger, final_mp4)

    return final_mp4


def _dub_variants(video_path, target_segments, cfg, target_lang, gender,
                  total_duration, work_dir, base, source_segments, logger):
    """Озвучивает текст на целевом языке в основной и (опционально) спокойный варианты.

    Варианты ведут раздельные файлы, чтобы основной результат не затирался альтернативой:
      - основной:  ref_speaker.wav  -> dubbed.wav     -> output/<имя>_<lang>.mp4
      - спокойный: ref_speaker2.wav -> dubbed_alt.wav -> output/<имя>_<lang>_alt.mp4
    Оба используют референсы спикера, отобранные из видео по энергии речи.

    Субтитры вшиваются прямо из target_segments (текст таблицы переводов) во
    временный SRT, отдельные файлы переводов в проекте не нужны.
    Возвращает путь основного wav (или None при неудаче).
    """
    # Субтитры вшиваются только если show_subtitles != false (из GUI).
    show_subtitles = cfg.get("show_subtitles", True)
    subs_segments = target_segments if show_subtitles else None

    # Транскрипты (из GUI): генерация читаемых .txt рядом с роликом в work/.
    if cfg.get("show_ru_transcript", False):
        with open(os.path.join(work_dir, base + ".transcript_ru.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(sg.get("text", "") for sg in source_segments))
        logger.info("Транскрипт (русский) записан: {0}.transcript_ru.txt".format(base))
    if cfg.get("show_tgt_transcript", False):
        with open(os.path.join(work_dir, base + "." + target_lang + ".transcript_tgt.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(sg.get("text", "") for sg in target_segments))
        logger.info("Транскрипт ({0}) записан: {1}.{0}.transcript_tgt.txt".format(target_lang, base))

    # Основной вариант.
    wav_main = _synth_variant(
        video_path, target_segments, cfg, target_lang, gender, total_duration,
        work_dir, base,
        ref_name="ref_speaker.wav", wav_name="dubbed.wav",
        out_label="основной", alt_index=None, temp_key="xtts_temperature",
        subs_segments=subs_segments, logger=logger,
    )
    if wav_main is None:
        logger.error("Озвучка основного варианта не удалась. Файл пропущен.")
        return None

    # Альтернативный (спокойный) вариант: второй по энергии референс + низкая температура.
    if cfg.get("alt_voice_enabled", True):
        wav_alt = _synth_variant(
            video_path, target_segments, cfg, target_lang, gender, total_duration,
            work_dir, base,
            ref_name="ref_speaker2.wav", wav_name="dubbed_alt.wav",
            out_label="спокойный (альтернативный)", alt_index=2, temp_key="xtts_temperature_alt",
            subs_segments=subs_segments, logger=logger,
        )
        if wav_alt:
            logger.info("Альтернативный (спокойный) вариант готов.")
        else:
            logger.info("Альтернативный вариант не создан — оставлен только основной.")

    return wav_main


def _synth_variant(video_path, target_segments, cfg, target_lang, gender, total_duration,
                   work_dir, base, ref_name, wav_name, out_label, alt_index, temp_key,
                   subs_segments, logger):
    """Рендерит один вариант озвучки и экспортирует его в output/.

    alt_index=None -> основной референс (лучший по энергии);
    alt_index=2    -> второй по энергии референс (спокойный) + температура temp_key.
    Возвращает путь dubbed .wav или None при неудаче.
    """
    ref_path = os.path.join(work_dir, ref_name)
    final_ref = None
    engine_name = tts.resolve_tts_engine(cfg, gender)
    # Kokoro не клонирует голос спикера — референс/вырезка не нужны (и XTTS-переменные не трогаем).
    uses_xtts = engine_name != "kokoro"
    if uses_xtts and cfg.get("clone_speaker_voice", True):
        final_ref = _obtain_ref(video_path, target_segments, ref_path, alt_index, logger)
        if final_ref is None:
            logger.info("Референс спикера недоступен для '{0}' — вариант пропущен.".format(out_label))
            return None

    cfg_tts = dict(cfg)
    cfg_tts["xtts_language"] = target_lang
    if final_ref:
        # Пер-видео референс спикера перекрывает статичные reference_m/f.
        cfg_tts["xtts_reference_wav"] = final_ref
        cfg_tts["xtts_reference_m"] = final_ref
        cfg_tts["xtts_reference_f"] = final_ref

    temp_val = cfg.get(temp_key)
    if temp_val is not None:
        cfg_tts["xtts_temperature"] = float(temp_val)

    # Альтернативный вариант: отдельный темп Edge-TTS (как xtts_temperature_alt для XTTS).
    if alt_index == 2:
        speed_alt = cfg_tts.get("edge_speed_alt")
        if speed_alt is not None:
            cfg_tts["edge_speed"] = float(speed_alt)
            logger.info(f"Альтернативный вариант: темп Edge-TTS {cfg_tts['edge_speed']} "
                        f"(edge_speed_alt из конфига).")

    wav_path = os.path.join(work_dir, wav_name)
    wav, used_voice = tts.synth_timed_with_retry(
        target_segments, cfg_tts, config.VOICES_DIR, wav_path, gender, logger, total_duration
    )
    if wav is None:
        logger.error("Озвучка не удалась ни одним голосом. Файл пропущен.")
        return None

    final_mp4 = os.path.join(config.OUTPUT_DIR,
                             base + "_" + target_lang + ("" if alt_index is None else "_alt") + ".mp4")
    export.export_dubbed(video_path, wav, final_mp4, cfg, logger,
                         subs_segments=subs_segments)
    logger.info("Вариант '{0}' готов: {1}".format(out_label, final_mp4))
    return wav


def _obtain_ref(video_path, segments, ref_path, alt_index, logger):
    """Достаёт референс спикера (кэш или перевырезание).

    alt_index=None -> основной кандидат (лучший по энергии);
    alt_index=2    -> второй по энергии кандидат (для спокойной озвучки).
    Возвращает путь к .wav или None.
    """
    if alt_index == 2:
        if os.path.exists(ref_path) and os.path.getsize(ref_path) > 0:
            logger.info("Альтернативный референс (кэш): {0}".format(ref_path))
            return ref_path
        cands = _speaker_candidates(video_path, segments, logger)
        if cands and len(cands) >= 2 and cands[1] is not None:
            st, en = cands[1]
            if export.extract_speaker_reference(video_path, segments, ref_path, logger,
                                                force_range=(st, en)) is None:
                return None
            return ref_path if os.path.exists(ref_path) else None
        # Второго речевого блока нет — берём основной референс спикера:
        # различие в эмоции достигается пониженной температурой (спокойный вариант).
        main_ref = os.path.join(os.path.dirname(ref_path), "ref_speaker.wav")
        if os.path.exists(main_ref):
            logger.info("Отдельного спокойного блока в видео нет — "
                        "спокойный вариант рендерится на том же голосе с низкой температурой.")
            return main_ref
        return None

    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 0:
        logger.info("Референс голоса спикера (кэш): {0}".format(ref_path))
        return ref_path
    export.extract_speaker_reference(video_path, segments, ref_path, logger)
    return ref_path if os.path.exists(ref_path) else None


def _speaker_candidates(video_path, segments, logger):
    """Возвращает кандидатов (st, en) отсортированных по энергии речи (убывание).

    Использует тот же разбор, что extract_speaker_reference: вырезает каждый
    непрерывный блок (до N), оценивает по энергии и сортирует от лучшего к худшему.
    """
    try:
        return export.speaker_reference_candidates(video_path, segments, logger)
    except Exception:
        logger.debug("Не удалось оценить кандидатов референса.", exc_info=True)
        return []


def gather_videos(input_dir):
    out = []
    for name in sorted(os.listdir(input_dir)):
        if os.path.splitext(name)[1].lower() in VIDEO_EXT:
            out.append(os.path.join(input_dir, name))
    return out


def _unique_target(dest_dir, name):
    """Имя в dest_dir без коллизий: file.mp4 -> file (2).mp4 -> file (3).mp4 ..."""
    root, ext = os.path.splitext(name)
    cand = os.path.join(dest_dir, name)
    n = 2
    while os.path.exists(cand):
        cand = os.path.join(dest_dir, f"{root} ({n}){ext}")
        n += 1
    return cand


def _ingest_watch_dirs(cfg, logger):
    """Переносит новые видео из приёмных папок в input/ (полу-авто приём из МАКС и др.).

    Папки: штатный INBOX_DIR + любые пути из config["watch_dirs"]. Файлы, которые
    ещё пишутся (размер меняется) или залочены, пропускаются — сторож вернётся
    к ним на следующем цикле. Разовый перенос: файл исчезает из приёмной папки,
    поэтому повторные циклы не дублируют работу.
    """
    dirs = []
    os.makedirs(config.INBOX_DIR, exist_ok=True)
    for d in [config.INBOX_DIR] + list(cfg.get("watch_dirs") or []):
        if d and os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    if not dirs:
        return
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    moved = 0
    for d in dirs:
        for name in sorted(os.listdir(d)):
            if os.path.splitext(name)[1].lower() not in VIDEO_EXT:
                continue
            src = os.path.join(d, name)
            try:
                if time.time() - os.path.getmtime(src) < 3.0:
                    continue  # ещё догружается
            except OSError:
                continue
            dst = _unique_target(config.INPUT_DIR, name)
            try:
                shutil.move(src, dst)
                moved += 1
                logger.info(f"Приём: {src} -> {dst}")
            except (OSError, shutil.Error):
                continue  # занят процессом / пишется — попробуем в следующем цикле
    if moved:
        logger.info(f"Перенесено в input/: {moved} видео.")


def _build_cfg(args):
    """Загружает config.json и применяет --set key=value оверрайды (с type coercion)."""
    cfg = config.load_config()
    for item in getattr(args, "set", []):
        if "=" in item:
            k, v = item.split("=", 1)
            if v.lower() in ("true", "yes", "1"):
                v = True
            elif v.lower() in ("false", "no", "0"):
                v = False
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            cfg[k] = v
    # Пресет языка применяется ПОСЛЕ --set, иначе --set translate_model=...
    # перезаписал бы модель, а остальные ключи языка остались бы от прежнего.
    lang = getattr(args, "lang", None)
    if lang:
        config.apply_language_preset(cfg, lang)
    return cfg


def _pid_alive(pid):
    """Жив ли сторож: True только если PID принадлежит именно ему.

    Одной проверки «есть ли такой PID» мало: номер мог достаться постороннему
    процессу (например, после перезапуска Windows), и сторож отказывался
    стартовать, считая, что он уже запущен.
    """
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            proc = psutil.Process(int(pid))
            cmdline = " ".join(proc.cmdline() or []).lower()
            if "run_pipeline" not in cmdline:
                return False
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False
    try:
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


WATCH_LOCK_MAX_AGE = 600  # 10 минут: старше — считаем застрявшим и забираем


def _acquire_watch_lock(logger):
    """Не даёт запустить второй сторож одновременно (предотвращает порчу выходов).

    Блокировка АТОМАРНАЯ (O_CREAT|O_EXCL): даже одновременный старт двух демонов
    не приводит к двум владельцам — ровно один пишет pid-файл, остальные выходят.
    Если pid-файл живого.. нет: старый PID считаем мёртвым, застрявший (старше
    10 минут) — тоже. Иначе ждём полсекунды и пробуем заново (гонка).
    """
    pid_file = os.path.join(config.WORK_DIR, ".watcher.pid")
    os.makedirs(config.WORK_DIR, exist_ok=True)
    while True:
        try:
            fd = os.open(pid_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            old = None
            try:
                with open(pid_file, "r", encoding="utf-8") as f:
                    old = int(f.read().strip())
            except (ValueError, OSError, IOError):
                pass
            try:
                age = time.time() - os.path.getmtime(pid_file)
            except OSError:
                age = WATCH_LOCK_MAX_AGE + 1
            alive = old is not None and _pid_alive(old)
            if alive and age < WATCH_LOCK_MAX_AGE:
                logger.error(
                    f"Сторож уже запущен (PID {old}) — второй экземпляр НЕ запускаю. "
                    "Завершите первый сторож или удалите work\\.watcher.pid, если PID мёртвый."
                )
                return False
            msg = (f"pid-файл {pid_file} от {('PID ' + str(old)) if old else 'неизвестного'}"
                   f" (возраст {int(age)} с) — забираю блокировку.")
            logger.info(msg)
            try:
                os.remove(pid_file)
            except OSError:
                pass
            time.sleep(0.5)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True


def main():
    parser = argparse.ArgumentParser(description="Автоматическая озвучка видео на целевой язык")
    parser.add_argument("paths", nargs="*", help="Конкретные файлы (опционально)")
    parser.add_argument("--watch", action="store_true", help="Режим слежения за папкой input/")
    parser.add_argument("--interval", type=int, default=15, help="Интервал проверки в watch-режиме (сек)")
    parser.add_argument("--set", action="append", default=[],
                        help="Переопределение конфига: --set key=value (можно несколько раз)")
    parser.add_argument("--lang", choices=config.available_languages(),
                        help="Язык озвучки/перевода: fr или en (перекрывает config.json)")
    args = parser.parse_args()

    logger = setup_logger()
    cfg = _build_cfg(args)

    config.ensure_dirs(cfg)

    if cfg.get("source_lang") != "auto":
        pass  # явный язык источника

    def run_once():
        _ingest_watch_dirs(cfg, logger)
        targets = args.paths if args.paths else gather_videos(config.INPUT_DIR)
        if not targets:
            logger.info("Нет видео для обработки.")
            return
        for v in targets:
            try:
                process_file(v, cfg, logger)
            except Exception as e:
                logger.error(f"Ошибка обработки {v}: {e}")
                logger.debug(traceback.format_exc())

    if args.watch:
        if not _acquire_watch_lock(logger):
            return
        logger.info(
            f"Режим слежения ({config.INPUT_DIR}): новые видео транскрибируются и сразу "
            "переводятся на два языка в таблицу .rtf, правки таблицы "
            "(а значит и озвучка/экспорт) применяются автоматически, без повторного "
            "запуска. Остановка: Ctrl+C."
        )
        # Конфиг перечитывается из config.json на каждой итерации, чтобы настройки
        # GUI применялись к уже работающему сторожу без перезапуска.
        busy = set()
        while True:
            cfg = _build_cfg(args)
            _ingest_watch_dirs(cfg, logger)
            for v in gather_videos(config.INPUT_DIR):
                base = os.path.splitext(os.path.basename(v))[0]
                if base in busy:
                    continue
                # Обрабатываем ВСЕ видео каждый цикл: process_file сам решает,
                # какой этап нужен (новый транскрипт / перевод / озвучка / экспорт)
                # и возвращает None, пока не наступило время следующего шага.
                busy.add(base)
                try:
                    final = process_file(v, cfg, logger)
                    if final:
                        logger.info(f"Готовый ролик: {final}")
                except Exception as e:
                    logger.error(f"Ошибка обработки {v}: {e}")
                    logger.debug(traceback.format_exc())
                finally:
                    busy.discard(base)
            time.sleep(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    main()

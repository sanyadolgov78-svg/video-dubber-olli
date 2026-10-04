import os


def detect_gender_from_segments(segments):
    """Простейшая оценка пола по средней высоте тона (f0).
    Женские голоса обычно выше (150-300 Гц), мужские ниже (80-180 Гц).
    Здесь использует знание из whisper-сегментов (нет f0), поэтому делаем
    грубую эвристику по длительности/характеру — реальный f0 считается в transcribe().
    Возвращает "male", "female" или "unknown".
    """
    # Заглушка: настоящая оценка пола выполняется в transcribe() по аудио.
    return "unknown"


def compute_voice_gender(pitch_map):
    """pitch_map: список значений f0 (Гц). Возвращает 'male'|'female'|'unknown'."""
    import numpy as np
    f0 = [p for p in pitch_map if p and p > 0]
    if not f0:
        return "unknown"
    med = float(np.median(f0))
    if med < 165:
        return "male"
    elif med > 175:
        return "female"
    return "unknown"


def transcribe(video_path, cfg, logger):
    """Транскрибирует видео с таймкодами и определяет пол говорящего.

    Возвращает dict:
      {
        "language": "ru",
        "gender": "male"|"female"|"unknown",
        "segments": [ {start, end, text}, ... ],
        "full_text": "..."
      }
    """
    from faster_whisper import WhisperModel

    device = cfg["whisper_device"]
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute_type = cfg["whisper_compute_type"]
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(f"Загрузка whisper-модели '{cfg['whisper_model']}' ({device}/{compute_type})...")
    model = WhisperModel(cfg["whisper_model"], device=device, compute_type=compute_type)

    logger.info("Транскрибация (это может занять время)...")

    decode_options = dict(
        language=None if cfg.get("source_lang", "auto") == "auto" else cfg.get("source_lang", "auto"),
        vad_filter=True,
        beam_size=int(cfg.get("whisper_beam_size", 5)),
        temperature=float(cfg.get("whisper_temperature", 0.0)),
    )
    prompt = cfg.get("whisper_prompt", "")
    if prompt:
        decode_options["initial_prompt"] = prompt
    segments_iter, info = model.transcribe(video_path, **decode_options)

    source_lang = info.language
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })

    full_text = " ".join(s["text"] for s in segments if s["text"])

    # Оценка пола по питчу аудио (librosa). Видео не читается librosa напрямую,
    # поэтому сначала извлекаем звуковую дорожку в WAV через ffmpeg.
    gender = "unknown"
    try:
        import subprocess
        import tempfile
        wav = os.path.join(tempfile.gettempdir(), "skit_pitch_" + str(os.getpid()) + ".wav")
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
                      "-t", str(cfg.get("pitch_analysis_max_sec", 120)), wav]
        subprocess.run(ffmpeg_cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import librosa
        y, sr = librosa.load(wav, sr=16000, mono=True)
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        gender = compute_voice_gender(f0)
        if os.path.exists(wav):
            os.remove(wav)
    except Exception as e:
        logger.warning(f"Не удалось оценить пол по питчу: {e}")

    logger.info(f"Язык: {source_lang}, пол говорящего: {gender}")

    # Автодробление слишком длинных сегментов (whisper_max_segment_dur, сек).
    # Разбиваем по границам клауз (запятые/точки) с пропорциональными таймингами.
    max_seg_dur = float(cfg.get("whisper_max_segment_dur", 0) or 0)
    if max_seg_dur > 0:
        try:
            from pipeline import transcript as _tr
            before = len(segments)
            segments = _tr.split_long_segments(segments, max_dur=max_seg_dur)
            if len(segments) != before:
                logger.info(f"Автодробление: сегментов стало {before} -> {len(segments)} "
                            f"(макс. длительность {max_seg_dur:.1f} с).")
        except Exception as e:
            logger.warning(f"Автодробление не выполнено: {e}")

    return {
        "language": source_lang,
        "gender": gender,
        "segments": segments,
        "full_text": full_text,
    }

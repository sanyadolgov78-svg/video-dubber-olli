"""Тесты переключения языка озвучки через LANGUAGE_PRESETS.

Проверяем, что смена target_lang действительно меняет модель перевода,
язык XTTS и голоса Kokoro, и что неизвестный язык отвергается, а не
молча оставляет прошлые настройки.

Тесты не требуют моделей, GPU и сети.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config, translate


def _cfg(lang):
    cfg = dict(config.DEFAULT_CONFIG)
    return config.apply_language_preset(cfg, lang)


# ---------- пресеты ----------

def test_available_languages():
    langs = config.available_languages()
    assert "fr" in langs
    assert "en" in langs


def test_french_preset():
    cfg = _cfg("fr")
    assert cfg["target_lang"] == "fr"
    assert cfg["translate_model"] == "opus-mt-ru-fr"
    assert cfg["xtts_language"] == "fr"
    assert cfg["kokoro_lang"] == "f"
    # Французский голос в Kokoro один, оба пола на него.
    assert cfg["kokoro_voice_m"] == cfg["kokoro_voice_f"] == "ff_siwis"


def test_english_preset():
    cfg = _cfg("en")
    assert cfg["target_lang"] == "en"
    assert cfg["translate_model"] == "opus-mt-ru-en"
    assert cfg["xtts_language"] == "en"
    assert cfg["kokoro_lang"] == "a"


def test_english_uses_xtts():
    """Английский по умолчанию через XTTS — голос клонируется у спикера."""
    cfg = _cfg("en")
    assert cfg["tts_engine"] == "xtts"
    assert cfg.get("clone_speaker_voice", True) is True


def test_french_auto_engine():
    """Французский по умолчанию через auto (женский -> Kokoro)."""
    assert _cfg("fr")["tts_engine"] == "auto"


def test_english_kokoro_voices_differ():
    """У английского в Kokoro есть и мужской, и женский голос."""
    cfg = _cfg("en")
    assert cfg["kokoro_voice_m"] != cfg["kokoro_voice_f"]
    assert cfg["kokoro_voice_m"].startswith("am_")
    assert cfg["kokoro_voice_f"].startswith("af_")


# ---------- главное: переключение не оставляет хвостов от прошлого языка ----------

def test_switch_clears_previous_language():
    """После fr -> en не должно остаться модели и языка от французского."""
    cfg = _cfg("fr")
    config.apply_language_preset(cfg, "en")
    assert cfg["translate_model"] == "opus-mt-ru-en"
    assert "ru-fr" not in cfg["translate_model"]
    assert cfg["xtts_language"] == "en"
    assert cfg["kokoro_lang"] == "a"


def test_switch_back_and_forth():
    """Обратное переключение тоже должно быть чистым."""
    cfg = _cfg("en")
    config.apply_language_preset(cfg, "fr")
    assert cfg["translate_model"] == "opus-mt-ru-fr"
    assert cfg["xtts_language"] == "fr"
    config.apply_language_preset(cfg, "en")
    assert cfg["translate_model"] == "opus-mt-ru-en"
    assert cfg["xtts_language"] == "en"


# ---------- выбор модели перевода ----------

def test_translate_model_resolves_per_language():
    """_choose_model_name отдаёт разные модели для разных языков."""
    fr = translate._choose_model_name(_cfg("fr"))
    en = translate._choose_model_name(_cfg("en"))
    assert fr != en
    assert fr.endswith("ru-fr")
    assert en.endswith("ru-en")


# ---------- ошибки ----------

def test_unknown_language_rejected():
    try:
        config.apply_language_preset(dict(config.DEFAULT_CONFIG), "de")
    except ValueError as e:
        assert "de" in str(e)
        assert "fr" in str(e) and "en" in str(e)
    else:
        raise AssertionError("неизвестный язык должен отвергаться")


def test_unknown_language_does_not_corrupt_cfg():
    """Неизвестный язык не должен оставлять конфиг в промежуточном виде."""
    cfg = _cfg("fr")
    before = dict(cfg)
    try:
        config.apply_language_preset(cfg, "xx")
    except ValueError:
        pass
    assert cfg == before


# ---------- суффикс выходного файла ----------

def test_output_suffix_follows_language():
    """Файл называется по целевому языку: _fr.mp4 или _en.mp4."""
    for lang in config.available_languages():
        cfg = _cfg(lang)
        name = "video_" + cfg["target_lang"] + ".mp4"
        assert name.endswith("_" + lang + ".mp4")


def test_preset_does_not_touch_unrelated_keys():
    """Пресет не должен затирать настройки озвучки/субтитров."""
    cfg = dict(config.DEFAULT_CONFIG)
    style_before = cfg["subtitles_font_size"]
    engine_before = cfg["kokoro_speed"]
    config.apply_language_preset(cfg, "en")
    assert cfg["subtitles_font_size"] == style_before
    assert cfg["kokoro_speed"] == engine_before


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
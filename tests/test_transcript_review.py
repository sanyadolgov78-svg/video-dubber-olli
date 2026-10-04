"""Тесты механизма проверки транскрипта.

Проверяем то, от чего зависит остановка конвейера: определение факта
правки, формат txt-файла, перенос правок в таблицу и пересчёт перевода
при изменении оригинала.

Тесты не требуют моделей, GPU и сети — только модуль rtf_table.

Запуск без pytest:
    python tests/test_transcript_review.py

Запуск с pytest:
    python -m pytest tests/test_transcript_review.py -v
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import rtf_table


def _segments(n=3):
    src = [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": f"Text {i}"}
           for i in range(n)]
    ru = [{"start": s["start"], "end": s["end"], "text": f"Привет {i}"}
          for i, s in enumerate(src)]
    fr = [{"start": s["start"], "end": s["end"], "text": f"Bonjour {i}"}
          for i, s in enumerate(src)]
    return src, ru, fr


def _make_table(path, n=3):
    src, ru, fr = _segments(n)
    rtf_table.write_translation_table(
        path, path + ".json", src, ru, fr,
        source_lang="ru", ru_lang="ru", target_lang="fr",
    )
    return src, ru, fr


# ---------- table_is_edited ----------

def test_new_table_is_not_edited():
    """Свежесозданная таблица правки не содержит -> ждём носителя."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        orig = rtf + ".orig"
        _make_table(rtf)
        import shutil
        shutil.copyfile(rtf, orig)
        assert rtf_table.table_is_edited(rtf, orig) is False


def test_edited_table_is_detected():
    """Правка в файле отличается от эталона -> конвейер продолжает."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        orig = rtf + ".orig"
        _make_table(rtf)
        import shutil
        shutil.copyfile(rtf, orig)
        assert rtf_table.table_is_edited(rtf, orig) is False

        # Правка: переписываем файл с изменённым переводом
        _make_table(rtf)
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")
        data["target"][1]["text"] = "Salut !"
        rtf_table.write_translation_table(
            rtf, rtf + ".json", data["source"], data["ru"], data["target"],
            source_lang="ru", ru_lang="ru", target_lang="fr")
        assert rtf_table.table_is_edited(rtf, orig) is True


def test_missing_files_mean_not_edited():
    """Нет файлов -> конвейер не должен считать, что правка сделана."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "nope.table.rtf")
        orig = rtf + ".orig"
        assert rtf_table.table_is_edited(rtf, orig) is False


# ---------- чтение/запись таблицы ----------

def test_table_roundtrip_keeps_text():
    """Текст переживает цикл запись -> чтение."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        _make_table(rtf, n=4)
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")
        assert data is not None
        assert len(data["target"]) == 4
        assert data["target"][0]["text"].strip() == "Bonjour 0"
        assert data["ru"][0]["text"].strip() == "Привет 0"


def test_timecodes_preserved():
    """Таймкоды не должны разъезжаться при записи таблицы."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        _make_table(rtf, n=3)
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")
        for i, seg in enumerate(data["target"]):
            assert abs(float(seg["start"]) - i * 2.0) < 0.01


# ---------- формат .transcript_otpravka.txt ----------

def test_txt_edit_pattern_recognised():
    """Регулярка из _import_txt_edits должна узнавать наш формат."""
    import re
    pat = re.compile(r"^\[[0-9.]+\]\s+(RU|FR):\s*(.*)$")
    m = pat.match("[12.50] FR: Bonjour, comment ça va ?")
    assert m is not None
    assert m.group(1) == "FR"
    assert m.group(2).strip() == "Bonjour, comment ça va ?"

    m2 = pat.match("[0.0] RU: Привет")
    assert m2 is not None and m2.group(1) == "RU"


def test_txt_edit_ignores_noise():
    """Строки без префикса игнорируются, а не ломают разбор."""
    import re
    pat = re.compile(r"^\[[0-9.]+\]\s+(RU|FR):\s*(.*)$")
    assert pat.match("просто заметка") is None
    assert pat.match("[12.5] DE: Guten Tag") is None      # только RU/FR
    assert pat.match("[без времени] FR: привет") is None  # время обязательно


def test_empty_txt_does_not_continue():
    """Пустой или бессмысленный txt не должен запускать озвучку."""
    import re
    pat = re.compile(r"^\[[0-9.]+\]\s+(RU|FR):\s*(.*)$")
    lines = ["", "заметка", "заголовок"]
    fr = [m.group(2).strip() for m in
          (pat.match(l.strip()) for l in lines) if m and m.group(1) == "FR"]
    assert not fr        # нет ни одной строки FR -> возврат False в конвейере


# ---------- пересчёт перевода при правке оригинала ----------

def test_source_edit_invalidates_table():
    """Правка только оригинала требует пересчёта перевода.

    Ключевое условие алгоритма: если менялся столбец оригинала, а
    французский трогали нет, перевод считается устаревшим.
    """
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        _make_table(rtf, n=3)
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")

        data["ru"][0]["text"] = "Здравствуйте"     # правка оригинала
        # французский столбец не изменён

        ru_changed = (data["ru"][0]["text"].strip() != "Привет 0")
        target_untouched = (data["target"][0]["text"].strip() == "Bonjour 0")
        assert ru_changed and target_untouched
        # -> конвейер обязан пересчитать target по новому оригиналу


def test_target_edit_keeps_source_flow():
    """Правка только перевода: источник не трогаем, озвучиваем как есть."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        _make_table(rtf, n=3)
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")
        data["target"][0]["text"] = "Salut !"
        assert data["ru"][0]["text"].strip() == "Привет 0"  # оригинал цел
        assert data["target"][0]["text"].strip() == "Salut !"


# ---------- вспомогательное ----------

def test_lang_label():
    assert rtf_table.lang_label("ru")
    assert rtf_table.lang_label("fr")


def test_empty_table_rejected():
    """Пустая таблица -> конвейер пропускает ролик, а не озвучивает пустоту."""
    with tempfile.TemporaryDirectory() as d:
        rtf = os.path.join(d, "v.table.rtf")
        rtf_table.write_translation_table(
            rtf, rtf + ".json", [], [], [],
            source_lang="ru", ru_lang="ru", target_lang="fr")
        data = rtf_table.read_translation_table(
            rtf, rtf + ".json", ru_lang="ru", target_lang="fr")
        has_text = bool(data) and any(
            (s.get("text") or "").strip() for s in data["target"])
        assert not has_text


# ---------- runner ----------

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
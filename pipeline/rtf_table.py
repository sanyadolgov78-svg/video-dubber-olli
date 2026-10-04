# -*- coding: utf-8 -*-
"""Таблица переводов в формате RTF: таймкоды + оригинал + перевод.

Заменяет двухэтапную схему (SRT оригинала, потом SRT перевода) одной таблицей:
после транскрибации алгоритм сразу переводит реплики, отдаёт носителю .rtf на
правку, а после сохранения файла читает перевод обратно и продолжает конвейер
(полишинг -> озвучка -> экспорт).

Столбцы: Таймкоды | Оригинал (<язык видео>) | Французский.
Русский перевод носителю не показывается, но остаётся в json-подложке: он нужен
полишингу (сверка с русским) и автодетекту sensitive (русский словарь ключевых слов).
Таймкоды берутся из столбца «Таймкоды», а при его отсутствии — из подложки.
"""
import json
import os
import re

from . import transcript

# Ширины столбцов (твипы) и поля — по образцу «Таблица 2 в 1»:
# колонка таймкодов узкая, текстовые шире, поля 1370 твипов (~2.4 см).
COL_WIDTHS = (1360, 3700, 4100)
MARGIN_LR = 1370

LANG_LABELS = {
    "ru": "Русский",
    "fr": "Французский",
    "en": "Английский",
    "de": "Немецкий",
    "es": "Испанский",
    "it": "Итальянский",
    "uk": "Украинский",
    "pl": "Польский",
    "be": "Белорусский",
}


def lang_label(code):
    return LANG_LABELS.get((code or "").lower(), (code or "?").upper())


def _esc(text):
    """Экранирует текст для RTF; не-ASCII уходит через \\uN? (юникод)."""
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ch in "\r\n\t":
            out.append(" ")
        elif code < 128:
            out.append(ch)
        elif code > 0xFFFF:
            c = code - 0x10000
            out.append("\\u%d?\\u%d?" % (0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF)))
        else:
            out.append("\\u%d?" % code)
    return "".join(out)


# Рамка ячейки: пишется ОТДЕЛЬНО для каждой ячейки. Word не применяет
# границы уровня строки к 2-й и 3-й ячейкам — при одном блоке на строку
# таблица рисуется только первой колонкой, а остальные выглядят как обычный
# текст (Word при сохранении проставляет им \brdrnone).
_CELL_EDGE = ("\\clvertalt"
              "\\clbrdrt\\brdrs\\brdrw15"
              "\\clbrdrl\\brdrs\\brdrw15"
              "\\clbrdrb\\brdrs\\brdrw15"
              "\\clbrdrr\\brdrs\\brdrw15")


def _cell_defs(shaded=False):
    """Определение строки таблицы: рамка и \\cellx для каждой ячейки."""
    defs = "\\trowd\\trgaph72\\trleft0\\trftsWidth1\\trpaddl72\\trpaddr72"
    pos = 0
    for w in COL_WIDTHS:
        pos += w
        cell = _CELL_EDGE + "\\cltxlrtb\\clftsWidth3\\clwWidth%d" % w
        if shaded:
            cell += "\\clcbpat2"
        defs += cell + "\\cellx%d" % pos
    return defs


def write_translation_table(rtf_path, json_path, source_segments, ru_segments,
                            target_segments, source_lang="auto", ru_lang="ru",
                            target_lang="fr"):
    """Создаёт RTF-таблицу (таймкоды | оригинал | перевод) и json-подложку.

    Таймкоды берутся из source_segments; русский перевод в таблицу не выводится,
    но сохраняется в подложку для полишинга и автодетекта sensitive.
    Возвращает путь к .rtf.
    """
    os.makedirs(os.path.dirname(rtf_path) or ".", exist_ok=True)

    rows = [["Таймкоды",
             "Оригинал (%s)" % lang_label(source_lang),
             lang_label(target_lang)]]

    count = max(len(source_segments), len(ru_segments), len(target_segments))
    timings = []
    ru_texts = []
    for i in range(count):
        src = source_segments[i] if i < len(source_segments) else None
        ru = ru_segments[i] if i < len(ru_segments) else None
        tgt = target_segments[i] if i < len(target_segments) else None
        seg = src or ru or tgt or {}
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        timings.append([start, end])
        ru_texts.append((ru or {}).get("text", "") or "")
        stamp = "%s --> %s" % (transcript.fmt_ts(start), transcript.fmt_ts(end))
        rows.append([
            stamp,
            (src or {}).get("text", "") or "",
            (tgt or {}).get("text", "") or "",
        ])

    out = ["{\\rtf1\\ansi\\ansicpg1251\\uc1\\deff0"]
    out.append("{\\fonttbl{\\f0\\fswiss\\fcharset204 Calibri;}}")
    out.append("\\fs20\\margl%d\\margr%d\\margt1134\\margb1134" % (MARGIN_LR, MARGIN_LR))
    for n, cells in enumerate(rows):
        header = n == 0
        out.append(_cell_defs(shaded=header))
        for c in cells:
            # \pard\intbl перед текстом обязателен: иначе граница ячейки
            # \cellxN слипается с текстом, начинающимся цифрой (таймкод).
            align = "\\b\\qc " if header else "\\ql "
            out.append("\\pard\\intbl" + align + _esc(c) + "\\cell ")
        out.append("\\pard\\intbl\\ql\\b0\\row\n")
    out.append("}")

    with open(rtf_path, "w", encoding="ascii", errors="strict", newline="\r\n") as f:
        f.write("".join(out))

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"timings": timings,
                       "ru": ru_texts,
                       "columns": rows[0],
                       "source_lang": source_lang,
                       "ru_lang": ru_lang,
                       "target_lang": target_lang}, f, ensure_ascii=False, indent=1)
    return rtf_path


_UNI_ESC = re.compile(r"\\u(-?\d+)[ ]?\??")
_HEX_ESC = re.compile(r"\\'([0-9a-fA-F]{2})")
_CTRL_WORD = re.compile(r"\\([a-zA-Z]+)(-?\d+)?[ ]?")
_CELLX = re.compile(r"\\cellx(\d+)")
_LIT_ESC = re.compile(r"\\([^a-zA-Z])")
_TIME_SEP = re.compile(r"\s*(?:-->|->|–—>|—>)\s*")
_FONT_DEF = re.compile(r"\{\\?\**\\?(?:a|f)(\d+)[^{}]*?\\fcharset(\d+)")
_ANSI_CPG = re.compile(r"\\ansicpg(\d+)")

# \fcharset -> кодовая страница. 0 = ANSI (для русского документа это latin-1,
# и именно в ней Word хранит французские акценты), 204 = 1251 и т.д.
_CHARSET_CP = {
    0: "cp1252", 2: "cp1250", 161: "cp1253", 162: "cp1254", 163: "cp1258",
    177: "cp1255", 178: "cp1256", 186: "cp1257", 204: "cp1251", 238: "cp1252",
}

# Символические control words RTF (Word так пишет тире и типографские кавычки).
_SYMBOLS = {
    "emdash": "—", "endash": "–", "emspace": " ", "enspace": " ",
    "qmspace": " ", "bullet": "•", "lquote": "‘", "rquote": "’",
    "ldblquote": "“", "rdblquote": "”", "ltrmark": "‎", "rtlmark": "‏",
}

# Граница ячейки не может превышать 32767 твипов (~5.8 см). Если цифры параметра
# \cellx «съели» начало текста ячейки (текст сразу начинается с цифры, например
# таймкод), лишние цифры отбрасываются по этому ограничению.
_CELLX_MAX = 32767


def _strip_star_groups(text):
    """Вырезает группы-метаданные {\\*\\generator ...}, {\\*\\bkmkstart ...} и т.п."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{\\*", i):
            level = 0
            j = i
            while j < n:
                if text[j] == "{":
                    level += 1
                elif text[j] == "}":
                    level -= 1
                    if level == 0:
                        j += 1
                        break
                j += 1
            i = j
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _decode_cell(raw, fonts=None, default_cp="cp1251"):
    """Декодирует содержимое ячейки RTF (\\uN?, \\'XX, служебные слова — в ноль).

    fonts — карта «номер шрифта -> \\fcharset» из таблицы шрифтов документа,
    нужна для верной кодировки \\'hh (кириллица и латиница в одном файле).
    """
    fonts = fonts or {}
    out = []
    i = 0
    n = len(raw)
    font = None
    while i < n:
        ch = raw[i]
        if ch != "\\":
            if ch in "{}":
                i += 1
                continue
            # Word переносит длинные строки RTF по всему файлу — переносы строк
            # внутри текста не являются содержимым и не должны становиться
            # пробелами (иначе слова рвутся посередине).
            if ch in "\r\n":
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        m = _UNI_ESC.match(raw, i)
        if m:
            code = int(m.group(1))
            if code < 0:
                code += 0x10000
            nxt = m.end()
            if 0xD800 <= code <= 0xDBFF:
                m2 = _UNI_ESC.match(raw, nxt)
                if m2:
                    low = int(m2.group(1))
                    if low < 0:
                        low += 0x10000
                    if 0xDC00 <= low <= 0xDFFF:
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        nxt = m2.end()
            out.append(chr(code) if 0 <= code <= 0x10FFFF else "?")
            i = nxt
            continue
        m = _HEX_ESC.match(raw, i)
        if m:
            # \'hh читается в кодировке текущего шрифта: кириллица Word пишет
            # шрифтом с \fcharset204 (cp1251), а французские акценты — шрифтом
            # с \fcharset0 (cp1252). Без учёта шрифта \'e9 превращается в «й».
            out.append(bytes([int(m.group(1), 16)]).decode(_charset_cp(fonts, font, default_cp),
                                                          "replace"))
            i = m.end()
            continue
        m = _CELLX.match(raw, i)
        if m:
            # Параметр \cellx физически ограничен; лишние цифры — уже текст ячейки.
            digits = m.group(1)
            while len(digits) > 1 and int(digits) > _CELLX_MAX:
                digits = digits[:-1]
            i = m.start(1) + len(digits)
            continue
        m = _CTRL_WORD.match(raw, i)
        if m:
            word = m.group(1)
            if word in ("f", "af"):
                font = m.group(2)
            elif word in ("par", "line", "tab"):
                out.append(" ")
            elif word in _SYMBOLS:
                out.append(_SYMBOLS[word])
            i = m.end()
            continue
        m = _LIT_ESC.match(raw, i)
        if m:
            out.append(m.group(1))
            i = m.end()
            continue
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _fonttbl_block(text):
    """Вырезает блок {\\fonttbl ... } целиком (с учётом вложенных групп)."""
    start = text.find("{\\fonttbl")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:start + 8000]


def _parse_fonts(text):
    """Извлекает из документа соответствие «номер шрифта -> \\fcharset» и
    кодовую страницу документа по \\ansicpg."""
    fonts = {}
    for f in _FONT_DEF.finditer(_fonttbl_block(text)):
        fonts[f.group(1)] = int(f.group(2))
    default_cp = "cp1251"
    c = _ANSI_CPG.search(text)
    if c:
        try:
            default_cp = "cp" + c.group(1)
        except ValueError:
            pass
    return fonts, default_cp


def _charset_cp(fonts, font, default_cp):
    cs = fonts.get(font) if font is not None else None
    if cs is None:
        return default_cp
    return _CHARSET_CP.get(cs, default_cp)


def _iter_rows(text):
    """Отдаёт строки таблицы как списки ячеек (строки без \\cell пропускаются).

    В конце каждой строки RTF добавляет служебный пустой абзац (« terminator
    » с \\row), который даёт лишнюю пустую ячейку — она отбрасывается.
    """
    fonts, default_cp = _parse_fonts(text)
    body = _strip_star_groups(text)
    start = body.find("\\trowd")
    if start >= 0:
        body = body[start:]
    for raw_row in re.split(r"\\row(?![a-zA-Z])", body):
        if "\\cell" not in raw_row:
            continue
        cells = [_decode_cell(c, fonts, default_cp)
                 for c in re.split(r"\\cell(?![a-zA-Z])", raw_row)]
        while len(cells) > 1 and not cells[-1].strip():
            cells.pop()
        yield cells


def _column_map(header, ru_lang, target_lang):
    """Определяет номера столбцов по заголовку; при неудаче — по позициям по умолчанию."""
    idx = {"time": 0, "source": 1, "target": 2}
    if not header:
        return idx
    low = [c.lower() for c in header]
    tgt_lbl = lang_label(target_lang).lower()

    def find(*needles, **kw):
        after = kw.get("after", -1)
        for i, name in enumerate(low):
            if i <= after:
                continue
            if any(nd in name for nd in needles):
                return i
        return None

    i = find("тайм", "время", "time", "duration")
    if i is not None:
        idx["time"] = i
    i = find("оригинал", "исходник", "source", "original", "транскрипт", after=idx["time"])
    if i is not None:
        idx["source"] = i
    # Ищем целевой столбец строго ПОСЛЕ оригинала: при совпадении языков
    # (французский оригинал -> французский перевод) обе подписи содержат
    # одно и то же название, и первый матч уезжал в столбец «Оригинал».
    i = find(tgt_lbl, "франц", "target", "перевод", after=idx["source"])
    if i is not None:
        idx["target"] = i
    else:
        idx["target"] = len(header) - 1 if len(header) > 2 else 2
    return idx


def _parse_time_cell(text):
    """'00:00:01,500 --> 00:00:03,000' -> (start, end) в секундах."""
    if not text:
        return None
    parts = _TIME_SEP.split(text.strip())
    if len(parts) < 2:
        return None
    try:
        return transcript.parse_ts(parts[0].strip()), transcript.parse_ts(parts[1].strip())
    except (ValueError, IndexError):
        return None


def read_translation_table(rtf_path, json_path=None, ru_lang="ru", target_lang="fr"):
    """Читает отредактированную RTF-таблицу (таймкоды | оригинал | перевод).

    Возвращает {"source": [...], "ru": [...], "target": [...]} — списки сегментов
    {start, end, text}. Оригинал и перевод берутся из таблицы, русский текст —
    из json-подложки (в таблице его нет), таймкоды — из столбца, иначе из подложки.
    """
    if not os.path.exists(rtf_path):
        return None
    rows = list(_iter_rows(_read_text(rtf_path)))
    if not rows:
        return None

    header = rows[0]
    ncols = max(3, len(header))
    cmap = _column_map(header, ru_lang, target_lang)

    timings = []
    ru_texts = []
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            timings = data.get("timings") or []
            ru_texts = data.get("ru") or []
        except (ValueError, OSError):
            timings, ru_texts = [], []

    out = {"source": [], "ru": [], "target": []}
    for n, cells in enumerate(rows[1:]):
        cells = cells[-ncols:] if len(cells) > ncols else cells

        def cell(key):
            i = cmap.get(key, -1)
            return cells[i] if 0 <= i < len(cells) else ""

        span = _parse_time_cell(cell("time"))
        if span is None and n < len(timings):
            try:
                span = (float(timings[n][0]), float(timings[n][1]))
            except (TypeError, ValueError, IndexError):
                span = None
        if span is None:
            continue
        start, end = span
        if end <= start:
            continue
        ru_text = ru_texts[n] if n < len(ru_texts) else ""
        out["source"].append({"start": start, "end": end, "text": cell("source")})
        out["ru"].append({"start": start, "end": end, "text": ru_text})
        out["target"].append({"start": start, "end": end, "text": cell("target")})
    return out


def table_is_edited(rtf_path, orig_path):
    """True, если таблица отличается от контрольной копии автогенерации."""
    if not os.path.exists(rtf_path) or not os.path.exists(orig_path):
        return False
    try:
        with open(rtf_path, "rb") as a, open(orig_path, "rb") as b:
            return a.read() != b.read()
    except OSError:
        return False

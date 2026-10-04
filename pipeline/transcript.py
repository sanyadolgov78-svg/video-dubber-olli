import os
import re


def _clauses(text):
    """Делит текст на клаузы по знакам , ; : . ! ? … (разделитель остаётся в конце части)."""
    parts = re.split(r"(?<=[,;:])|(?<=[.!?…])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def split_long_segments(segments, max_dur=6.0):
    """Разбивает сегменты длиннее max_dur на куски по границам клауз.

    Тайминги новых кусков распределяются пропорционально длине текста
    внутри исходного окна (начало/конец исходного сегмента сохраняются).
    Если дробление невозможно (нет разделителей) — сегмент остаётся целиком.
    """
    out = []
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = (seg.get("text") or "").strip()
        dur = end - start
        if dur <= max_dur or not text:
            out.append(dict(seg))
            continue
        clauses = _clauses(text)
        if not clauses or len(clauses) < 2:
            out.append(dict(seg))
            continue
        total_chars = sum(len(c) for c in clauses)
        per_char = dur / total_chars if total_chars else 0.0
        chunks = []
        cur, cur_chars = [], 0
        limit_chars = max_dur / per_char if per_char > 0 else 1e9
        for cl in clauses:
            cl_chars = len(cl)
            if cur and cur_chars + cl_chars > limit_chars:
                chunks.append(" ".join(cur))
                cur, cur_chars = [], 0
            cur.append(cl)
            cur_chars += cl_chars
        if cur:
            chunks.append(" ".join(cur))
        acc = start
        for c in chunks:
            c_start = max(start, acc)
            c_end = min(end, acc + len(c) * per_char)
            if c_end > c_start:
                out.append({"start": c_start, "end": c_end, "text": c})
            acc = c_end
        if acc < end and out:
            out[-1]["end"] = end
        if not out:
            out.append(dict(seg))
    return out


def fmt_ts(seconds):
    """Секунды -> строка SRT (HH:MM:SS,mmm)."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_text(segments):
    """Собирает содержимое SRT по сегментам [{start, end, text}, ...]."""
    parts = []
    num = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        num += 1
        parts.append(f"{num}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{text}\n")
    return "\n".join(parts)


def write_srt(segments, srt_path):
    """Записывает SRT из сегментов [{start, end, text}, ...].

    Возвращает путь к файлу.
    """
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text(segments))
    return srt_path


def read_srt_text(srt_path):
    """Извлекает текст из (исправленного) SRT в один сплошной текст.

    Пропускает строки с номерами и таймкодами; собирает строки-реплики
    через пробел.
    """
    lines = []
    with open(srt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\n")
        # Чистим возможные номера/таймкоды и собираем текст реплики
        text_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p.isdigit():
                continue
            if "-->" in p or "–—>" in p:
                continue
            text_parts.append(p)
        if text_parts:
            lines.append(" ".join(text_parts))
    return " ".join(lines).strip()


def load_meta(meta_path):
    """Загружает сохранённые метаданные (язык/пол), если есть."""
    import json
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_meta(meta_path, **fields):
    import json
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)


def read_srt_segments(srt_path):
    """Читает SRT и возвращает список сегментов [{start, end, text}, ...].

    Таймкоды парсятся для сохранения соответствия при переводе/озвучке.
    Пропускает пустые реплики.
    """
    ts_to_sec = parse_ts

    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\n")
        start = end = None
        text_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if "-->" in p:
                left, _, right = p.partition("-->")
                start = ts_to_sec(left.strip())
                end = ts_to_sec(right.strip())
                continue
            if p.isdigit():
                continue
            text_parts.append(p)
        text = " ".join(text_parts).strip()
        if text and start is not None and end is not None:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def parse_ts(s):
    """'HH:MM:SS,mmm' -> секунды (float)."""
    hh, mm, rest = s.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


# Старые приватные имена — на случай обращений из внешнего кода.
_fmt_ts = fmt_ts
_parse_ts = parse_ts


def transcript_is_edited(src_srt, orig_srt):
    """True, если SRT был отредактирован пользователем после автогенерации.

    Сравнивает текущий файл с контрольной (исходной) копией.
    """
    if not os.path.exists(src_srt) or not os.path.exists(orig_srt):
        # Если контрольной копии нет — считаем файл не готовым (нужна генерация)
        return False
    try:
        with open(src_srt, "r", encoding="utf-8") as a, open(orig_srt, "r", encoding="utf-8") as b:
            return a.read() != b.read()
    except Exception:
        return False

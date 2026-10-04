# -*- coding: utf-8 -*-
"""Автодетект敏感ный контент (военные/политики/пленные).

После редактирования русского транскрипта классифицирует видео
как敏感ное (только субтитры без озвучки) или нейтральное.
Использует ключевые слова + опциональный LLM (Ollama) для подтверждения.
"""
import hashlib
import json
import os
import time
import urllib.request


DEFAULT_TAGS = [
    # военные
    "войн", "военн", "солдат", "армия", "фронт", "боец",
    "обстрел", "окоп", "оружие", "артиллери", "дрон", "бпла",
    "минобороны", "спецoperаци", "снайпер", "гаубиц",
    # политика
    "политик", "политическ", "президент", "депутат", "парламент",
    "правительств", "министр", "губернатор", "мэр", "посол", "канцлер", "сенатор",
    # плен / освобожденные
    "плен", "пленн", "заложник", "обмен пленн", "освобожд", "репатриант",
]


def _make_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path(work_dir, base):
    return os.path.join(work_dir, base + ".sensitive.json")


def _write_json(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _llm_query(ru_text, cfg, logger):
    """Классифицирует контент через Ollama (возвращает True=敏感ный/ДА, False=НЕТ, None=ошибка)."""
    from . import polish

    url = cfg.get("ollama_url", "http://localhost:11434")
    if not polish._ensure_ollama(url, logger, start_attempts=1):
        return None

    prompt = (
        "Ты — классификатор контента видео. Определи по русской транскрипции, относится ли видео к одной из тем:\n"
        "1) военные действия, военная тематика;\n"
        "2) политики, политические деятели;\n"
        "3) люди, освобождённые из плена, заложники, обмен пленными.\n"
        "Если да — ответь «ДА». Если нет — ответь «НЕТ». Отвечай строго одним словом.\n\n"
        f"Транскрипт:\n{ru_text[:8000]}"
    )
    payload = json.dumps({
        "model": cfg.get("ollama_model", "qwen2.5:7b"),
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,  # выгружаем модель сразу, чтобы освободить RAM перед переводом
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 4},
    }).encode("utf-8")
    api_url = url.rstrip("/") + "/api/generate"
    try:
        req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ans = data.get("response", "").strip().upper()
        if "ДА" in ans:
            return True
        if "НЕТ" in ans:
            return False
    except Exception as e:
        logger.warning(f"LLM-классификация: ошибка ({e})")
    return None


def detect(source_segments, cfg, work_dir, base, logger):
    """Классифицирует контент видео по русской транскрипции.

    Результат кэшируется в work/<base>.sensitive.json (пересчёт только при изменении текста).
    Если LLM включён (`subtitles_only_llm`) и есть совпадения по тегам —
    подтверждает/опровергает решение через Ollama; иначе — по ключевым словам.
    Возвращает dict{ sensitive: bool, method: "keywords"/"llm"/"none", tags: [...] }.
    """
    if not cfg.get("subtitles_only_enabled", True):
        return {"sensitive": False, "method": "disabled", "tags": []}

    text = " ".join((s.get("text") or "").lower() for s in source_segments or [])
    text_hash = _make_hash(text)

    cache = _read_json(_cache_path(work_dir, base))
    if cache and cache.get("hash") == text_hash:
        return {
            "sensitive": bool(cache.get("sensitive", False)),
            "method": cache.get("method", "cached"),
            "tags": cache.get("tags", []),
        }

    tags = [t.lower() for t in cfg.get("subtitles_only_tags", DEFAULT_TAGS)]
    min_tags = int(cfg.get("subtitles_only_min_tags", 1))
    matched = [t for t in tags if t in text]
    keyword_hit = len(matched) >= min_tags

    method = "keywords" if keyword_hit else "none"
    sensitive = keyword_hit

    # LLM-подтверждение, если включено и есть keyword-hit ( снижает false positives).
    if cfg.get("subtitles_only_llm", True) and keyword_hit:
        llm_result = _llm_query(text, cfg, logger)
        if llm_result is not None:
            sensitive = llm_result
            method = "llm"
        # иначе — fallback на keyword_hit

    result = {
        "sensitive": sensitive,
        "method": method,
        "tags": matched[:20],
        "hash": text_hash,
        "ts": time.time(),
    }
    _write_json(_cache_path(work_dir, base), result)
    logger.info(
        "Классификация контента: sensitive=%s (метод=%s, теги=%s)",
        sensitive, method, matched[:8],
    )
    return result


def load(work_dir, base):
    """Читает кэш классификации. Возвращает dict{ sensitive: bool, ... }."""
    cache = _read_json(_cache_path(work_dir, base))
    if cache:
        return {"sensitive": bool(cache.get("sensitive", False)), "method": cache.get("method", "cached"), "tags": cache.get("tags", [])}
    return {"sensitive": False, "method": "none", "tags": []}

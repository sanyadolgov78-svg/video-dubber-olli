import json
import os
import subprocess
import time
import urllib.request


def _ollama_candidate_paths():
    """Типовые пути установки Ollama (Windows). Возвращает абсолютные пути."""
    local = os.environ.get("LOCALAPPDATA", "")
    prog = os.environ.get("ProgramFiles", "")
    home = os.environ.get("USERPROFILE", "")
    cands = []
    for base in (local, prog, home, "C:\\Program Files", "C:\\Program Files (x86)"):
        if base:
            cands.append(os.path.join(base, "Programs", "Ollama", "ollama.exe"))
            cands.append(os.path.join(base, "Ollama", "ollama.exe"))
    # из PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p:
            cands.append(os.path.join(p, "ollama.exe"))
    seen = set()
    out = []
    for c in cands:
        c = c.strip('"')
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_ollama():
    for cand in _ollama_candidate_paths():
        if os.path.isfile(cand):
            return cand
    return None


def _is_up(url, logger, timeout=3):
    try:
        import requests
        r = requests.get(url + "/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False


def _start_ollama(logger):
    """Пытается запустить сервис Ollama. Возвращает True, если удалось."""
    exe = _find_ollama()
    if not exe:
        logger.warning("Ollama не найдена на диске — автоустановка невозможна.")
        return False
    logger.info(f"Запускаю Ollama: {exe}")
    try:
        startupinfo = None
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.Popen([exe], startupinfo=subprocess.STARTUPINFO() if False else None,
                         creationflags=flags, close_fds=True)
    except Exception as e:
        logger.warning(f"Не удалось запустить Ollama: {e}")
        return False
    return True


def _ensure_ollama(url, logger, start_attempts=1):
    """Гарантирует, что Ollama доступна. При необходимости запускает и ждёт. Возвращает bool."""
    if _is_up(url, logger):
        return True
    logger.warning("Ollama недоступна — пробую запустить.")
    exe = _find_ollama()
    if not exe:
        logger.warning("Ollama не установлена (ollama.exe не найдена).")
        return False
    for attempt in range(start_attempts):
        _start_ollama(logger)
        # ждём готовности до ~40 сек (первый старт грузит модель)
        deadline = time.time() + 40
        while time.time() < deadline:
            if _is_up(url, logger):
                logger.info("Ollama запущена и отвечает.")
                return True
            time.sleep(1.5)
    return _is_up(url, logger)


# Промпты для полишинга по языкам.
_POLISH_PROMPTS = {
    "en": (
        "You are a professional English editor. This is a machine translation "
        "of a video transcription into English. Fix spelling and grammar mistakes "
        "and make the text natural, fluent and clear for a native English speaker, "
        "without changing the meaning. Reply ONLY with the corrected text, no comments."
    ),
    "fr": (
        "Tu es un professionnel de l'edition francaise. Ceci est une traduction "
        "automatique d'une transcription video en francais. Corrige les fautes "
        "d'orthographe et de grammaire, et rends le texte naturel, fluide et "
        "clair pour un francophone, sans changer le sens. Reponds UNIQUEMENT "
        "avec le texte corrige, sans commentaires."
    ),
}


# Сегментные промпты по языкам (посегментная правка перевода).
_POLISH_PROMPT_SEGMENT = {
    "en": (
        "You are a professional subtitle localizer for an English-speaking audience. "
        "A Russian video line and its machine translation into English are given below.\n"
        "Write back ONE natural, idiomatic English line exactly as a native speaker "
        "would say it in a voiced video (spoken register, correct grammar, word order, "
        "word choice).\n"
        "Rules:\n"
        "1) Keep the exact meaning; do not add or cut information.\n"
        "2) Keep numbers, dates, personal names, place names and brand names unchanged.\n"
        "3) Avoid literal calques and machine-translation artifacts.\n"
        "4) Do not explain; do not add commentary, quotation marks or brackets.\n"
        "Reply with ONLY the corrected English line. If it is already natural, return it unchanged."
    ),
    "fr": (
        "Tu es un localisateur professionnel de sous-titres pour un public francophone. "
        "Une replique russe de la video et sa traduction automatique en francais sont "
        "donnees ci-dessous.\n"
        "Redige UNE replique francaise naturelle et idiomatique, exactement comme un "
        "locuteur natif la dirait dans une video commentee (registre oral, grammaire "
        "correcte, ordre des mots, choix des mots).\n"
        "Regles :\n"
        "1) Conserve le sens exact ; n'ajoute ni ne retire d'information.\n"
        "2) Conserve tels quels les nombres, dates, noms propres, toponymes et marques.\n"
        "3) Evite les calques litteraux et les artefacts de traduction automatique.\n"
        "4) N'explique pas ; n'ajoute ni commentaire, ni guillemets, ni parentheses.\n"
        "Reponds UNIQUEMENT avec la replique francaise corrigee. Si elle est deja "
        "naturelle, renvoie-la telle quelle."
    ),
}


def _polish_one_line(ru_text, en_text, cfg, logger, url):
    """Исправляет одну строку перевода через Ollama (ru + целевой язык). Возвращает "" при ошибке."""
    lang = (cfg.get("target_lang") or "en").lower().strip()
    segment_prompt = _POLISH_PROMPT_SEGMENT.get(lang) or _POLISH_PROMPT_SEGMENT["en"]
    prompt = (
        f"{segment_prompt}\n\n"
        f"RUSSIAN SOURCE:\n{ru_text}\n\n"
        f"MACHINE TRANSLATION:\n{en_text}"
    )
    payload = json.dumps({
        "model": cfg["ollama_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(cfg.get("polish_temperature", 0.1)),
            "num_ctx": 4096,
            "num_predict": 512,
        },
    }).encode("utf-8")
    api_url = url.rstrip("/") + "/api/generate"
    try:
        req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ошибка полишинга сегмента ({e}); строка остаётся без правок.")
        return ""


def _unload_ollama(url, logger):
    """Просит Ollama выгрузить модель из памяти (освобождает RAM)."""
    try:
        payload = json.dumps({
            "model": "x",  # любой; keep_alive=0 выгружает все неиспользуемые модели
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }).encode("utf-8")
        req = urllib.request.Request(url.rstrip("/") + "/api/generate",
                                     data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        logger.info("Модель Ollama выгружена из памяти.")
    except Exception as e:
        logger.debug(f"Не удалось выгрузить Ollama: {e}")


def polish_segments(ru_segments, en_segments, cfg, logger):
    """Исправляет перевод каждого сегмента на целевом языке, сверяясь с русским текстом.

    Возвращает список сегментов [{start, end, text}] с исправленным переводом
    (тексты могут измениться — они идут в озвучку). Если Ollama недоступна при
    обязательном полишинге — возвращает None (конвейер должен остановиться).
    """
    if not cfg.get("polish_enabled", True):
        return en_segments

    url = cfg.get("ollama_url", "http://localhost:11434")
    if not _ensure_ollama(url, logger):
        required = bool(cfg.get("polish_required", True))
        if required:
            logger.error(
                "Ollama недоступна и не запустилась, а полишинг обязателен "
                "(polish_required=true). Останавливаю конвейер."
            )
            return None
        logger.warning("Ollama недоступна — пропускаю полишинг (перевод остаётся как есть).")
        return en_segments

    ru_texts = [(s.get("text") or "").strip() for s in ru_segments or []]
    out = []
    for idx, en in enumerate(en_segments):
        en_text = (en.get("text") or "").strip()
        en = dict(en, text=en_text)
        if not en_text:
            out.append(en)
            continue
        ru = ru_texts[idx] if idx < len(ru_texts) else ""
        corrected = _polish_one_line(ru, en_text, cfg, logger, url)
        if corrected:
            en["text"] = corrected
        out.append(en)
    logger.info("Полишинг переводчика применён по сегментам.")
    _unload_ollama(url, logger)
    return out


def polish_english(text, cfg, logger):
    """Исправляет перевод через локальную LLM (Ollama) для указанного языка.

    Если Ollama недоступна — при `polish_required` возвращает None (признак
    того, что конвейер должен остановиться), иначе возвращает исходный текст.

    Возвращает текст, либо None если полишинг обязателен и Ollama недоступна.
    """
    if not cfg.get("polish_enabled", True):
        return text

    lang = (cfg.get("target_lang") or "en").lower().strip()

    url = cfg.get("ollama_url", "http://localhost:11434")
    if not _ensure_ollama(url, logger):
        required = bool(cfg.get("polish_required", True))
        if required:
            logger.error(
                "Ollama недоступна и не запустилась, а полишинг обязателен "
                "(polish_required=true). Останавливаю конвейер, чтобы перевод "
                "не ушёл без проверки носителем. Запустите Ollama и повторите."
            )
            return None
        logger.warning("Ollama недоступна — пропускаю исправление текста (перевод остаётся как есть).")
        return text

    template = _POLISH_PROMPTS.get(lang) or _POLISH_PROMPTS["en"]
    prompt = f"{template}\n\nTEXT:\n{text}"

    payload = json.dumps({
        "model": cfg["ollama_model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")

    api_url = url.rstrip("/") + "/api/generate"
    try:
        req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        corrected = data.get("response", "").strip()
        if corrected:
            logger.info(f"Текст ({lang}) исправлен через LLM.")
            return corrected
    except Exception as e:
        logger.warning(f"Ошибка исправления через Ollama ({e}); верну исходный перевод.")

    return text

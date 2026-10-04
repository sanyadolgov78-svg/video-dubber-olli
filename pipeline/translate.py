"""Перевод текста на целевой язык через facebook/mbart / Helsinki opus (локально).

Модель скачивается с Hugging Face Hub (бесплатно) и работает Offline.
Поддерживает 50 языков в обе стороны через единую модель — лучше качество
по сравнению с opus-mt (отдельные пары).
"""

import os
import re
import threading

# Карта кодов whisper (2-буквенные) → коды mbart-large-50 (составные).
# Коды выверены по фактическому токенизатору локальной модели
# (MBART50TokenizerFast.lang_code_to_id) — перечислены только те, что есть
# в модели (иначе перевод уходит в fallback ru_RU и ломается).
# Полный реальный список модели (52): af_ZA ar_AR az_AZ bn_IN cs_CZ de_DE
# en_XX es_XX et_EE fa_IR fi_FI fr_XX gl_ES gu_IN he_IL hi_IN hr_HR id_ID
# it_IT ja_XX ka_GE kk_KZ km_KH ko_KR lt_LT lv_LV mk_MK ml_IN mn_MN mr_IN
# my_MM ne_NP nl_XX pl_PL ps_AF pt_XX ro_RO ru_RU si_LK sl_SI sv_SE sw_KE
# ta_IN te_IN th_TH tl_XX tr_TR uk_UA ur_PK vi_VN xh_ZA zh_CN.
WHISPER_TO_MBART = {
    "ru": "ru_RU",
    "uk": "uk_UA",
    "en": "en_XX",
    "de": "de_DE",
    "es": "es_XX",
    "fr": "fr_XX",
    "it": "it_IT",
    "pt": "pt_XX",
    "nl": "nl_XX",
    "pl": "pl_PL",
    "cs": "cs_CZ",
    "sv": "sv_SE",
    "fi": "fi_FI",
    "tr": "tr_TR",
    "ar": "ar_AR",
    "zh": "zh_CN",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "id": "id_ID",
    "hi": "hi_IN",
    "he": "he_IL",
    "vi": "vi_VN",
    "ro": "ro_RO",
    "hr": "hr_HR",
    "sl": "sl_SI",
    "et": "et_EE",
    "lv": "lv_LV",
    "lt": "lt_LT",
    "mn": "mn_MN",
    "az": "az_AZ",
    "kk": "kk_KZ",
    "km": "km_KH",
    "mk": "mk_MK",
    "mr": "mr_IN",
    "my": "my_MM",
    "ne": "ne_NP",
    "ps": "ps_AF",
    "si": "si_LK",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "af": "af_ZA",
    "bn": "bn_IN",
    "fa": "fa_IR",
    "gl": "gl_ES",
    "gu": "gu_IN",
    "ka": "ka_GE",
    "ml": "ml_IN",
    "ur": "ur_PK",
    "xh": "xh_ZA",
}

# Карта целевых языков (whisper 2-буквенные → mbart коды).
TARGET_TO_MBART = {
    "en": "en_XX",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "it": "it_XX",
    "pt": "pt_XX",
    "ru": "ru_RU",
}

MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
OPUS_RU_EN = "Helsinki-NLP/opus-mt-ru-en"
OPUS_RU_FR = "Helsinki-NLP/opus-mt-ru-fr"

# Тип весов при загрузке mbart: 50-языковая модель не помещается в память в
# fp32 (2.4 ГБ весов + 2.44 ГБ файла при ~2.7 ГБ свободного commit).
# bfloat16 занимает 1.2 ГБ и не снижает качество перевода заметно.
MBART_LOAD_DTYPE = "bfloat16"

# Текущая модель перевода (имя, затем локальная папка).
CURRENT_TRANSLATOR = None

# Кэш загруженных моделей.
_model_cache = {}
_token_cache = {}
_cache_lock = threading.Lock()
CURRENT_CFG = None


def _local_path(cfg, model_name):
    """Возвращает путь к локальной папке модели (в кэше), если она есть."""
    local_dir = os.path.join(cfg["model_cache_dir"], model_name.split("/")[-1])
    if os.path.isdir(local_dir):
        return local_dir
    return None


def _choose_model_name(cfg):
    """Имя модели перевода: по конфигу (translate_model) либо mbart по умолчанию."""
    name = (cfg.get("translate_model") or "mbart").strip().lower()
    if name in ("opus", "opus-mt-ru-en", "marian"):
        return OPUS_RU_EN
    if name in ("opus-mt-ru-fr",):
        return OPUS_RU_FR
    return MBART_MODEL


def _resolve_mbart_dtype(logger):
    """Возвращает torch-тип весов для загрузки mbart.

    Модель на 50 языков весит 2.4 ГБ в fp32. На машине с ~2 ГБ свободной
    памяти fp32 не помещается, поэтому по умолчанию грузим веса в bf16
    (1.2 ГБ): диапазон значений как у fp32, теряется только мантисса,
    что для перевода безопасно. Переопределяется переменной окружения
    SKIT_MBART_DTYPE = bfloat16 | float16 | float32.
    """
    import torch

    name = os.environ.get("SKIT_MBART_DTYPE", MBART_LOAD_DTYPE).strip().lower()
    table = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    if name not in table:
        logger.warning("mbart: неизвестный SKIT_MBART_DTYPE=%r, беру bfloat16", name)
        name = "bfloat16"
    logger.info("mbart: тип весов при загрузке: %s", name)
    return table[name]


def _mbart_manual_load(local_dir, mbart_name, logger):
    """Загрузка mbart без весового из transformer: from_pretrained ломается
    на этой машине (access violation в safetensors->torch mmap при большом
    файле 2.3 ГБ).

    Раньше модель собиралась из AutoConfig сразу в fp32 — это отдельное
    выделение 2.4 ГБ под случайные веса, и вместе с mmap файла (2.44 ГБ)
    пик доходил до 4.8 ГБ при ~2.7 ГБ свободного commit. Отсюда 0xC0000005.

    Теперь модель собирается на устройстве meta: ни байта памяти не
    выделяется, все параметры — пустые заглушки. Реальные веса читаются из
    mmap по одному тензору и сразу подставляются через load_state_dict(
    assign=True) в bf16. Пик памяти падает примерно до 1.2 ГБ, а общие
    (tied) веса остаются одним тензором. Все 50 языков сохраняются.
    """
    import gc
    import json
    import mmap
    import struct

    import numpy as np

    import torch

    safetensors_path = os.path.join(local_dir, "model.safetensors")
    tok = _load_mbart_tokenizer(local_dir, logger)

    # Парсим заголовок safetensors вручную (избегаем полного read в память).
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    start = 8 + header_len
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    from transformers import MBartForConditionalGeneration, AutoConfig

    torch_dtype = _resolve_mbart_dtype(logger)

    logger.info("mbart: сборка модели из AutoConfig на meta (без выделения RAM)...")
    config = AutoConfig.from_pretrained(local_dir)
    with torch.device("meta"):
        model = MBartForConditionalGeneration(config)

    DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
              "F64": np.float64, "I8": np.int8, "U8": np.uint8, "BF16": np.float16}
    TIED_ALIASES = {
        "model.encoder.embed_tokens.weight": "model.shared.weight",
        "model.decoder.embed_tokens.weight": "model.shared.weight",
        "lm_head.weight": "model.shared.weight",
    }

    fsize = os.path.getsize(safetensors_path)
    loaded = 0
    with open(safetensors_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), fsize, access=mmap.ACCESS_READ)
        try:
            with torch.no_grad():
                new_sd = {}
                for name, meta in tensors.items():
                    off, end = meta["data_offsets"]
                    src_dtype = DTYPES[meta["dtype"]]
                    shape = list(meta["shape"])
                    buf = memoryview(mm)[start + off: start + end]
                    arr = np.frombuffer(buf, dtype=src_dtype).reshape(shape)
                    tensor = torch.from_numpy(np.ascontiguousarray(arr).copy()).to(torch_dtype)
                    del arr, buf
                    gc.collect()
                    new_sd[name] = tensor
                    for alias, canonical in TIED_ALIASES.items():
                        if canonical == name:
                            new_sd[alias] = tensor
                    loaded += 1

                missing, unexpected = model.load_state_dict(new_sd, strict=False, assign=True)
                del new_sd
                gc.collect()

                # Всё, что осталось на meta (буферы position_ids и т.п.),
                # материализуем нулями — иначе будет ошибка при generate.
                for mod in model.modules():
                    for bname, buf_t in list(mod.named_buffers(recurse=False)):
                        if buf_t is not None and getattr(buf_t, "is_meta", False):
                            mod._buffers[bname] = torch.zeros(
                                buf_t.shape, dtype=buf_t.dtype, device="cpu")
                    for pname, par in list(mod.named_parameters(recurse=False)):
                        if par is not None and getattr(par, "is_meta", False):
                            mod._parameters[pname] = torch.zeros(
                                par.shape, dtype=torch_dtype, device="cpu")
        finally:
            mm.close()

    if unexpected:
        logger.warning("mbart: лишние ключи в файле: %s", len(unexpected))
    if missing:
        logger.warning("mbart: не найдено в файле: %s", missing)
    logger.info("mbart: загружено тензоров: %s, dtype=%s", loaded, torch_dtype)
    model.eval()
    return model, tok


def _load_mbart_tokenizer(local_dir, logger):
    from transformers import MBart50TokenizerFast

    tok = None
    try:
        tokfile = os.path.join(local_dir, "sentencepiece.bpe.model")
        if os.path.exists(tokfile):
            tok = MBart50TokenizerFast.from_pretrained(local_dir)
        else:
            tok = MBart50TokenizerFast.from_pretrained(local_dir)
    except Exception as _e:
        logger.warning("mbart: токенизатор не загрузился (%s)", _e)
    return tok


def _load_model(local_dir, model_name, logger):
    with _cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name], _token_cache[model_name]

        # На машинах с малым объёмом RAM (и CPU-класса Ryzen) перебора потоков
        # torch недостаточно памяти -> access violation при пересборке. Ограничиваем.
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass

        if "mbart" in model_name and local_dir:
            # from_pretrained ломается на большом safetensors (0xC0000005).
            model, tok = _mbart_manual_load(local_dir, model_name, logger)
        else:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            if local_dir:
                logger.info(f"Загрузка модели для перевода из локального кэша: {local_dir}")
                tok = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
            else:
                logger.info(f"Скачивание модели mbart ({model_name}, ~1.5 ГБ)...")
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                # Материализуем кэш (без симлинков HF)
                try:
                    from huggingface_hub import snapshot_download
                    local = _local_path(CURRENT_CFG or {}, model_name)
                    if local:
                        snapshot_download(model_name, local_dir=local)
                except Exception:
                    pass
        model.eval()
        _model_cache[model_name] = model
        _token_cache[model_name] = tok
        return model, tok


def unload_models():
    """Освобождает переведённые модели из памяти (для малого объёма RAM)."""
    import gc
    with _cache_lock:
        _model_cache.clear()
        _token_cache.clear()
    gc.collect()


_ENTITY_RE = re.compile(
    r"&(?:apos|amp|quot|lt|gt|nbsp|rsquo|lsquo|#39|#x27|#160);", re.IGNORECASE)
_ENTITY_MAP = {
    "&apos;": "'", "&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">",
    "&nbsp;": " ", "&rsquo;": "'", "&lsquo;": "'",
    "&#39;": "'", "&#x27;": "'", "&#160;": " ",
}


def _clean_translation(text):
    """Разворачивает HTML-сущности, которые opus-mt выдаёт вместо знаков.

    Модель обучена на корпусах с HTML-экранированием и иногда вместо апострофа
    печатает "&apos;". В таблице это мусор, а Edge TTS произносит такое вслух
    («and apos»), поэтому сущности разворачиваем сразу после перевода.
    """
    if not text:
        return text
    return _ENTITY_RE.sub(lambda m: _ENTITY_MAP.get(m.group(0).lower(), m.group(0)), text)


def _translate_batch(local_dir, model_name, text, src_lang_code, tgt_code, logger):
    """Переводит текст → целевой язык через выбранную seq2seq модель."""
    import torch

    model, tok = _load_model(local_dir, model_name, logger)
    is_mbart = "mbart" in model_name

    if is_mbart:
        # Язык-источник задаётся в токенизаторе (mbart — многоязычная модель).
        tok.src_lang = src_lang_code
        tgt_id = tok.convert_tokens_to_ids(tgt_code)
        if tgt_id == tok.unk_token_id:
            logger.warning(f"Токен '{tgt_code}' не найден; пробую без forced_bos.")
            tgt_id = None
    else:
        # opus/marian — направленная пара ru→<target>: языковые коды не нужны.
        tgt_id = None

    # Делим на фрагменты (лимит модели ~512 токенов, берём с запасом)
    max_len = 400
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return ""

    inputs = tok(chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)

    gen_kwargs = dict(
        max_length=512,
        num_beams=5,
        early_stopping=True,
    )
    if tgt_id is not None:
        gen_kwargs["forced_bos_token_id"] = tgt_id

    with torch.no_grad():
        translated = model.generate(**inputs, **gen_kwargs)

    outputs = tok.batch_decode(translated, skip_special_tokens=True)
    return _clean_translation(" ".join(o.strip() for o in outputs if o.strip()).strip())


def translate_segments(segments, cfg, source_language, logger, tgt_lang=None):
    """Переводит каждый сегмент [{start, end, text}] на целевой язык по отдельности.

    tgt_lang=None -> берётся cfg['target_lang']; иначе перевод идёт на tgt_lang
    (так делаются обе ноги таблицы переводов: на русский и на целевой).

    Сохраняет таймкоды оригинала, возвращает список сегментов:
      [{start, end, text: <перевод>}, ...]
    """
    global CURRENT_CFG, CURRENT_TRANSLATOR
    CURRENT_CFG = cfg

    src = (source_language or "").lower().strip()
    tgt_lang = (tgt_lang or cfg.get("target_lang") or "en").lower().strip()
    tgt_code = TARGET_TO_MBART.get(tgt_lang, "en_XX")

    # Источник совпадает с целевым — возвращаем без изменений
    if src == tgt_lang:
        logger.info(f"Исходный язык '{src}' совпадает с целевым — перевод не требуется.")
        return [dict(s, text=(s.get("text") or "").strip()) for s in segments]

    model_name = _choose_model_name(cfg)
    CURRENT_TRANSLATOR = model_name
    local_dir = _local_path(cfg, model_name)

    # Определяем mbart-код исходного языка (не нужен для opus; оставляем для лога)
    if src in WHISPER_TO_MBART:
        src_code = WHISPER_TO_MBART[src]
    else:
        logger.warning(f"Язык '{src}' не найден в карте mbart; пробую ru_RU...")
        src_code = "ru_RU"

    logger.info(f"Перевод: {model_name} ({src_code} → {tgt_code})")

    out = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        translated = _translate_batch(local_dir, model_name, text, src_code, tgt_code, logger)
        seg_en = dict(seg)
        seg_en["text"] = (translated or "").strip()
        out.append(seg_en)
    return out


def translate_segments_via_russian(segments, cfg, source_language, logger):
    """Переводит сегменты каскадом: иностранный язык → русский → целевой.

    Каскад «через русский» позволяет переводить на французский с ЛЮБОГО из
    ~50 языков mbart, а не только с русского (opus-mt-ru-fr переводит только
    ru→fr). Русский — промежуточный шаг, чтобы использовать качественную пару
    ru→fr (opus), а первую ногу (иностр. → ru) покрывает mbart.

    Сохраняет таймкоды оригинала. Возвращает [{start, end, text}, ...] на
    целевом языке.
    """
    global CURRENT_CFG, CURRENT_TRANSLATOR

    src = (source_language or "").lower().strip()
    tgt_lang = (cfg.get("target_lang") or "en").lower().strip()
    if src == tgt_lang:
        logger.info(f"Исходный язык '{src}' совпадает с целевым — перевод не требуется.")
        return [dict(s) for s in segments]
    if src == "ru":
        # Уже русский — используем только пару ru → цель (opus), без каскада.
        logger.info("Исходный язык русский — прямой перевод ru → {0}.".format(tgt_lang))
        return translate_segments(segments, cfg, src, logger)

    # Нога 1: иностранный → русский через mbart (по сегментам, с сохранением таймкодов).
    logger.info(f"Каскад: перевод {src} → русский (mbart, 50 языков)...")
    cfg_mid = dict(cfg)
    cfg_mid["target_lang"] = "ru"
    cfg_mid["translate_model"] = "mbart"
    mid_segments = translate_segments(segments, cfg_mid, src, logger)
    if not mid_segments or not any((s.get("text") or "").strip() for s in mid_segments):
        logger.error("Каскад: русский промежуточный перевод пуст — останавливаюсь.")
        return []
    unload_models()  # освобождаем mbart (~1.5 ГБ) до загрузки opus

    # Нога 2: русский → целевой (fr) через opus-mt-ru-fr.
    logger.info(f"Каскад: перевод русский → {tgt_lang} (opus-mt-ru-fr)...")
    cfg_tgt = dict(cfg)
    cfg_tgt["source"] = "ru"
    final_segments = translate_segments(mid_segments, cfg_tgt, "ru", logger)
    return final_segments


def translate_text(text, cfg, source_language, logger):
    """Переводит текст на целевой язык через выбранную модель.

    source_language: код языка оригинала, определённый whisper'ом (напр. "ru", "en", "uk").
    """
    global CURRENT_CFG, CURRENT_TRANSLATOR
    CURRENT_CFG = cfg

    src = (source_language or "").lower().strip()
    tgt_lang = (cfg.get("target_lang") or "en").lower().strip()
    tgt_code = TARGET_TO_MBART.get(tgt_lang, "en_XX")

    if src == tgt_lang:
        logger.info(f"Исходный язык '{src}' совпадает с целевым — перевод не требуется.")
        return text.strip()

    model_name = _choose_model_name(cfg)
    CURRENT_TRANSLATOR = model_name
    local_dir = _local_path(cfg, model_name)

    if src in WHISPER_TO_MBART:
        src_code = WHISPER_TO_MBART[src]
    else:
        logger.warning(f"Язык '{src}' не найден; пробую ru_RU...")
        src_code = "ru_RU"

    logger.info(f"Перевод: {model_name} ({src_code} → {tgt_code})...")
    return _translate_batch(local_dir, model_name, text, src_code, tgt_code, logger)

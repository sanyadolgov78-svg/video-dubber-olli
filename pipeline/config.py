import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "input")
# Приёмная папка: видео из неё автоматически переносятся в input/ и обрабатываются.
# Удобно для «сохранил из мессенджера → алгоритм подхватил».
INBOX_DIR = os.path.join(BASE_DIR, "inbox_max")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
# Каталог моделей и голосов. Путь должен быть ASCII: sentencepiece/piper/espeak
# не открывают файлы из путей с не-ASCII символами (кириллица).
#   Windows: C:\skit_models, Linux: ~/skit_models. Переопределяется переменной
#   окружения SKIT_MODELS_DIR.
MODELS_ROOT = os.environ.get("SKIT_MODELS_DIR") or (
    r"C:\skit_models" if os.name == "nt" else os.path.expanduser("~/skit_models"))
MODELS_ROOT = os.path.expanduser(MODELS_ROOT)
# Голоса Piper и модели перевода храним в каталоге БЕЗ кириллицы, т.к.
# sentencepiece/piper не открывают файлы из путей с не-ASCII символами.
VOICES_DIR = os.path.join(MODELS_ROOT, "voices")
WORK_DIR = os.path.join(BASE_DIR, "work")
# Локальный кэш скачанных моделей перевода (реальные файлы, без HF-симлинков)
MODEL_CACHE_DIR = os.path.join(MODELS_ROOT, "translation")

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    # --- Общие ---
    "source_lang": "auto",          # "auto" = определять язык автоматически (рекомендуется)
    "target_lang": "fr",            # целевой язык перевода (французский)
    # Дополнительные папки-приёмники: новые видео из них сторож переносит в input/
    # и обрабатывает как обычные. Путь вроде r"C:\Users\...\Downloads\MAX".
    "watch_dirs": [],

    # --- Транскрибация (faster-whisper) ---
    "whisper_model": "small",  # tiny/base/small/medium/large-v3-turbo/large-v3/large-v2 (чем больше, тем точнее) — эта машина (ГБ) не тянет large-v3-turbo
    "whisper_device": "auto",           # "auto" | "cpu" | "cuda"
    "whisper_compute_type": "auto",     # "auto" | "int8" | "float16" | "float32"
    "whisper_beam_size": 5,             # размер луча декодера (больше=точнее, медленнее)
    "whisper_temperature": 0.0,         # 0 = детерминированный декод
    "whisper_prompt": "",               # подсказка для пунктуации (русская для русских роликов), "" = выключено
    "whisper_max_segment_dur": 6.0,     # автодробление слишком длинных сегментов, сек (0 = выключено)

    # --- Перевод (локально: opus-mt-ru-fr / mbart-large-50) ---
    # Каталог для скачанных моделей перевода (реальные файлы).
    "model_cache_dir": MODEL_CACHE_DIR,
    "translate_model": "opus-mt-ru-fr",   # "opus-mt-ru-fr" | "opus-mt-ru-en" | "mbart" (см. _choose_model_name)
    "translate_via_russian": True,  # True = каскад «иностр.(50 языков) → русский → целевой» через mbart+opus
    "dub_enabled": True,            # False = НЕ озвучивать (только распознать+перевести+полишить SRT)

    # --- Полишинг перевода (опционально, через Ollama LLM) ---
    "polish_enabled": True,         # True = полишинг через LLM
    "polish_required": True,      # True = остановиться, если Ollama недоступна,         # True = исправлять перевод через локальную LLM (нужен Ollama)
    "ollama_model": "qwen2.5:7b",   # модель Ollama, понимающая целевой язык
    "ollama_url": "http://localhost:11434",
    "polish_temperature": 0.1,      # стабильность полишинга: низкая => правки консервативны

    # --- Озвучка (движок TTS) ---
    # tts_engine: "auto" = по полу спикера (female -> kokoro, male/unknown -> xtts);
    #             "kokoro" (быстрый, статичные голоса) или "xtts" (клонирование голоса спикера).
    "tts_engine": "auto",
    "kokoro_speed": 0.9,            # единый темп речи Kokoro (<1 = медленнее)
    "kokoro_max_tempo": 1.35,       # максимум доп. ускорения под узкие окна (выше = быстрее/резче)
    "kokoro_lang": "f",               # язык Kokoro (f = французский fr-fr)
    "kokoro_voice_m": "ff_siwis",     # мужской голос Kokoro (французских мужских нет — заглушка ff_siwis)
    "kokoro_voice_f": "ff_siwis",     # женский голос Kokoro (единственный французский)

    # --- Озвучка (Coqui XTTS v2, нейросетевой синтез / клонирование голоса) ---
    # XTTS синтезирует по РЕФЕРЕНСНОМУ голосу (аудиофайл с речью на целевом языке),
    # а не по имени голоса. Путь к .wav должен быть ASCII (без кириллицы).
    "xtts_reference_wav": os.path.join(MODELS_ROOT, "xtts", "reference.wav"),
    "xtts_language": "fr",          # язык синтеза (код XTTS: "en", "fr", "ru", ...)
    "xtts_temperature": 0.8,        # 0.0 = стабильно/монотонно, 1.0 = вариативно (0.8 = живо)
    "clone_speaker_voice": True,    # True = клонировать голос спикера видео (вырезка референса из ролика)
    "alt_voice_enabled": True,      # True = дополнительно рендерить спокойный вариант (*_alt.mp4)
    "xtts_temperature_alt": 0.55,   # температура спокойного варианта (ниже = ровнее)
    "edge_speed": 1.0,              # единый темп речи Edge-TTS (<1 = медленнее, >1 = быстрее)
    "edge_speed_alt": 1.15,         # темп альтернативного варианта (*_alt.mp4), как xtts_temperature_alt
    "edge_rate": "+0%",             # запасной процент производства речи (если edge_speed=1.0)
    "edge_max_tempo": 1.35,         # максимум доп. ускорения под узкие окна (выше = быстрее/резче)
    "tts_speed": 1.0,
    "background_original": 0.08,    # уровень нативного аудио в миксе с озвучкой (0.08 = 8%)
    "subtitles_only_original_volume": 1.0,  # громкость исходного звука в роликах БЕЗ озвучки (1.0 = 100%)
    "subtitles_font_size": 26,      # базовый размер шрифта субтитров (юниты ASS, PlayResY=288)
    "subtitles_portrait_max_lines": 3,  # вертикальное видео: максимум строк на реплику
    "subtitles_portrait_min_font": 10,   # вертикальное видео: мин. размер шрифта (не уменьшать ниже)
    "subtitles_portrait_block_ratio": 0.12,  # вертикальное видео: макс. доля высоты кадра под блок субтитров (0.12 = 12%)
    "subtitles_portrait_font_scale": 1.0,  # множитель шрифта в вертикальном видео (1.0 = без изменений)
    "subtitles_char_em": 0.22,         # ширина глифа в em (замерено libass: ~0.21) — от неё считается вместимость строки
    "subtitles_width_frac": 0.92,      # доля ширины кадра под строку субтитров (поля по краям)
    # Оформление букв (принято пользователем, менять только по его просьбе):
    #   цвет #ffc957 (оранжевый) + чёрная обводка, полужирный, Arial.
    "subtitles_font_name": "Arial",        # гарнитура (Arial; при отсутствии libass берёт похожую)
    "subtitles_primary_color": "ffc957",   # цвет букв без решётки, RGB hex (#ffc957 = оранжевый)
    "subtitles_outline_color": "000000",   # цвет обводки букв
    "subtitles_outline_width": 1.6,        # толщина обводки (0 = без обводки)
    "subtitles_bold": True,                # True = полужирное начертание
    "subtitles_shadow": 0.0,               # тень под буквами (0 = нет)

    # --- Автодетект чувствительного контента: только субтитры ---
    "subtitles_only_enabled": True,
    "subtitles_only_llm": True,
    "subtitles_only_min_tags": 1,
    "subtitles_only_tags": [
        "войн", "военн", "солдат", "армия", "фронт", "боец",
        "обстрел", "окоп", "оружие", "артиллери", "дрон", "бпла",
        "минобороны", "спецопераци", "снайпер", "гаубиц",
        "политик", "политическ", "президент", "депутат", "парламент",
        "правительств", "министр", "губернатор", "мэр", "посол", "канцлер", "сенатор",
        "плен", "пленн", "заложник", "обмен пленн", "освобожд", "репатриант"
    ],
    "dub_sensitive_videos": True,   # True = озвучивать и sensitive-видео (воен/полит/плен);
                                    # False = только субтитры без озвучки

    # --- Монтаж (применяется на этапе экспорта) ---
    "montage_freeze_enabled": False,     # True = в начало ролика: стоп-кадр (первый кадр) + заставка
    "montage_freeze_dur": 0.5,           # длительность стоп-кадра первого кадра в начале, сек
    "montage_freeze_crop_bottom": 0.0,   # доля нижней части стоп-кадра для обрезки (элементы плеера), 0 = выключено
    "montage_intro_image": "",           # путь к картинке заставки (PNG/JPG), ASCII; "" = заставки нет
    "montage_intro_hold": 1.5,           # длительность показа картинки-заставки, сек
    "montage_intro_video_enabled": False,  # True = вставлять видео-заставку (mp4) вместо картинки
    "montage_intro_landscape": "",       # видео-заставка для горизонтальных роликов (mp4)
    "montage_intro_portrait": "",        # видео-заставка для вертикальных роликов (mp4)
    "montage_logo_enabled": False,       # True = наложить водяной знак (лого)
    "montage_logo_path": os.path.join(MODELS_ROOT, "logo.png"),  # PNG/GIF/mp4 (ASCII путь)
    "montage_logo_position": "br",       # br | tr | bl | tl
    "montage_logo_margin": 15,           # отступ от краёв, px
    "montage_logo_scale": 0.0,           # масштаб лого как доля высоты кадра (0 = без масштаба)
    "montage_blur_subs_enabled": False,  # True = блюрить нижнюю зону (чужие субтитры)
    "montage_blur_subs_height_ratio": 0.16,  # доля высоты кадра снизу
    "montage_blur_subs_strength": "15:5",    # сила блюра boxblur (радиус:степени)
    "montage_blur_subs_auto_detect": True,   # True = на стоп-кадре блюрить ТОЛЬКО если детектор нашёл текст
    "montage_blur_subs_detect_ratio": 4.0,   # «блок текста» контрастнее спокойного фона во столько раз → текст есть
    "montage_blur_subs_detect_min": 8.0,     # мин. плотность краёв в окне текста, ниже которой текст не ищем

    # --- Экспорт ---
    "export_crf": 18,               # качество видео (меньше = лучше, 18-23 норм)
    "export_audio_bitrate": "192k",
    "remove_original_audio": True,  # True = заменить родную дорожку озвучкой (дубляж)

    # --- Обработка ошибок озвучки ---
    "max_dub_retries": 3            # сколько разных голосов перебрать при неудаче
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        return merged
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


def ensure_dirs(cfg=None):
    for d in (INPUT_DIR, OUTPUT_DIR, VOICES_DIR, WORK_DIR, INBOX_DIR):
        os.makedirs(d, exist_ok=True)

#!/usr/bin/env bash
# Установка пайплайна на свежий Ubuntu-сервер (Oracle Free Tier, ARM aarch64).
# Запуск:  bash setup_server.sh
# Скрипт идемпотентный: можно запускать повторно, он ничего не ломает.
set -e

PROJ="${PROJ:-$HOME/video}"
MODELS="${MODELS:-$HOME/skit_models}"

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }

say "1/8  Системные пакеты"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    git build-essential cmake pkg-config \
    ffmpeg python3 python3-pip python3-venv python3-dev \
    espeak-ng libespeak-ng1 curl ca-certificates fonts-dejavu-core \
    >/dev/null

say "2/8  Проверка версий"
python3 --version
ffmpeg -version | head -1
# faster-whisper/ctranslate2 на ARM требуют wheels под cp310-cp314.
python3 - <<'PY'
import sys
v = sys.version_info
print("Python", "%d.%d" % (v.major, v.minor))
if not (3, 10) <= (v.major, v.minor) <= (3, 13):
    print("ВНИМАНИЕ: нужна версия 3.10-3.13 (для ctranslate2/torch есть wheels aarch64)")
else:
    print("OK: версия поддерживается ARM-сборками")
PY

say "3/8  Каталоги"
mkdir -p "$MODELS"/{voices,translation,xtts} "$PROJ"/{input,output,work,inbox_max}

say "4/8  Виртуальное окружение"
cd "$PROJ"
[ -d venv ] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel >/dev/null

say "5/8  Зависимости Python (ARM wheels, без компиляции)"
pip install \
    "faster-whisper>=1.0.0" \
    "transformers>=4.40" sentencepiece protobuf \
    librosa soundfile numpy \
    edge-tts \
    requests huggingface-hub safetensors \
    2>&1 | tail -5

say "6/8  Torch (нужен для перевода opus-mt; ~200 МБ)"
pip install --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -3

say "7/8  Ollama для полишинга (опционально)"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1 || true
fi
if command -v ollama >/dev/null 2>&1; then
    (nohup ollama serve >/dev/null 2>&1 &) || true
    sleep 3
    ollama pull qwen2.5:7b 2>&1 | tail -2 || echo "модель не скачалась — полишинг будет пропущен"
else
    echo "Ollama не установился — полишинг будет пропущен (на это есть polish_required=false)"
fi

say "8/8  Проверка импортов"
python - <<'PY'
import platform
print("архитектура:", platform.machine())
ok = True
for mod in ("numpy", "faster_whisper", "transformers", "sentencepiece", "librosa", "edge_tts"):
    try:
        __import__(mod)
        print(f"  OK      {mod}")
    except Exception as e:
        ok = False
        print(f"  ОШИБКА  {mod}: {e}")
try:
    import torch
    print(f"  OK      torch {torch.__version__}")
except Exception as e:
    print(f"  ОШИБКА  torch: {e}")
print("ИТОГ:", "всё ставится" if ok else "есть проблемы — смотрите строки ОШИБКА")
PY

say "Готово"
cat <<EOF

Как запустить:

  cd $PROJ
  source venv/bin/activate
  cp config_linux.json config.json     # настройки под сервер (если файл есть)
  python run_pipeline.py --watch --interval 10

Как закинуть видео:

  scp C:/путь/видео.mp4 ubuntu@ВАШ_IP:~/video/input/

Как забрать готовый ролик:

  scp ubuntu@ВАШ_IP:~/video/output/ИМЯ_fr.mp4 C:/путь/

Полезно:
  nvidia-smi     — нет, ускорителя нет, всё на CPU
  free -h        — память (должно быть ~20 ГБ свободно)
  ollama list    — какие модели скачались

EOF

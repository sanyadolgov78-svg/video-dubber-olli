# Скрипт установки конвейера локальной озвучки видео на французский.
# Запуск:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "=== 1/3 Установка ffmpeg (через winget) ==="
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpeg) {
    Write-Host "ffmpeg не найден. Устанавливаю (потребуется подтверждение) ..."
    winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Write-Host "ffmpeg установлен. Откройте НОВЫЙ терминал, чтобы PATH обновился."
} else {
    Write-Host "ffmpeg уже установлен: $($ffmpeg.Source)"
}

Write-Host "=== 2/3 Установка Python-пакетов ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "=== 3/3 Создание папок ==="
New-Item -ItemType Directory -Force -Path "input","output","voices","work" | Out-Null

Write-Host ""
Write-Host "Готово к использованию."
Write-Host ""
Write-Host "Голоса Piper для французского скачаются автоматически при первом запуске."
Write-Host "Если хотите исправление французского через ИИ - установите Ollama:"
Write-Host "  https://ollama.com  затем:  ollama pull qwen2.5:7b"
Write-Host ""
Write-Host "Как запустить:"
Write-Host "  python run_pipeline.py                 # обработать вход/ один раз"
Write-Host "  python run_pipeline.py --watch         # следить за папкой вход/"

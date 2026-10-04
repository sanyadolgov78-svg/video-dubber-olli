# -*- coding: utf-8 -*-
"""
V3 Dubbing GUI — tkinter, запускает run_pipeline.py --watch --interval N
Цветовая схема: тёмно-синий фон + золотые элементы
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pipeline import config

# --- Цвета ---
BG = "#1a2744"       # тёмно-синий фон
FG = "#ffd700"       # золотой текст
BG_ENTRY = "#0f1a2e" # поле ввода
BG_BTN = "#ffd700"   # кнопка (золотая)
FG_BTN = "#1a2744"   # текст кнопки (синий)
BG_FRAME = "#12203a" # рамка
SEL_BG = "#2a4070"   # выделение списка
LOG_BG = "#0a1220"   # фон лога

PYTHON = os.path.join(config.BASE_DIR, ".venv", "Scripts", "python.exe")
PIPELINE = os.path.join(config.BASE_DIR, "run_pipeline.py")

# --- Функции ---

def run_cmd(cmd, log_cb, finish_cb=None):
    def _target():
        try:
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
                cwd=config.BASE_DIR, env=os.environ.copy(),
            )
            for line in p.stdout:
                log_cb(line)
            p.wait()
            if finish_cb:
                finish_cb(p.returncode)
        except Exception as exc:
            log_cb(f"\n[ошибка запуска] {exc}\n")
            if finish_cb:
                finish_cb(-1)
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


class DubApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("V3 Dubbing Pipeline")
        self.root.configure(bg=BG)
        self.root.minsize(780, 560)

        cfg = config.load_config()
        self.cfg = dict(cfg)

        # Состояние
        self._watching = False
        self._watch_proc = None
        self._next_running = False

        self._build_ui()

        # Загружаем текущие значения из config.json в элементы интерфейса,
        # чтобы GUI отражал фактическое состояние конвейера (и фонового сторожа).
        self._target_lang = (self.cfg.get("target_lang") or "fr").lower().strip()
        if not bool(self.cfg.get("dub_enabled", True)):
            self._gender.set("off")
        else:
            self._gender.set("auto")
            forced = (self.cfg.get("forced_gender") or "auto").strip().lower()
            self._gender.set(forced if forced in ("female", "male") else "auto")
        self._subs_var.set(bool(self.cfg.get("show_subtitles", True)))
        # Текст живёт в таблице .table.rtf, отдельные файлы транскриптов
        # по умолчанию не создаются — включаются вручную при необходимости.
        self._transcript_ru_var.set(bool(self.cfg.get("show_ru_transcript", False)))
        self._transcript_tgt_var.set(bool(self.cfg.get("show_tgt_transcript", False)))

        self._refresh_file_list()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI ----

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG, foreground=FG, font=("Segoe UI", 13, "bold"))
        style.configure("Gold.TButton", background=BG_BTN, foreground=FG_BTN,
                         font=("Segoe UI", 10, "bold"))
        style.map("Gold.TButton", background=[("active", "#e6c200")])
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TRadiobutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.map("TRadiobutton", background=[("active", BG)])
        style.configure("TScale", background=BG, troughcolor=BG_ENTRY)
        style.configure("TCombobox", fieldbackground=BG_ENTRY, background=BG_ENTRY,
                         foreground=FG)

        # ---- Верх: видео-список + кнопки ----
        top = ttk.Frame(self.root)
        top.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Видео в input/", style="Header.TLabel").pack(anchor="w")

        lst_frame = ttk.Frame(left)
        lst_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.file_list = tk.Listbox(
            lst_frame, selectmode="extended", bg=BG_ENTRY, fg=FG,
            selectbackground=SEL_BG, selectforeground="#ffffff",
            font=("Consolas", 10), relief="flat", bd=0,
        )
        scr = ttk.Scrollbar(lst_frame, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scr.set)
        self.file_list.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(4, 0))

        self._next_btn = ttk.Button(btn_frame, text="Далее", style="Gold.TButton",
                                    command=self._next_stage)
        self._next_btn.pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Открыть видео...", style="Gold.TButton",
                   command=self._pick_video).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Обновить список", style="Gold.TButton",
                   command=self._refresh_file_list).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="Открыть output/", style="Gold.TButton",
                   command=self._open_output).pack(side="left")
        ttk.Button(btn_frame, text="Открыть work/", style="Gold.TButton",
                   command=self._open_work).pack(side="left", padx=(4, 0))

        # ---- Правая панель: настройки ----
        right = ttk.Frame(top, style="TFrame")
        right.pack(side="right", fill="y", padx=(8, 0))

        ttk.Label(right, text="Настройки", style="Header.TLabel").pack(anchor="w", pady=(0, 4))

        # Целевой язык берётся из config.json (fr по умолчанию, метка динамическая).
        self._target_lang = (self.cfg.get("target_lang") or "fr").lower().strip()
        _lang_names = {"fr": "Французский", "en": "Английский"}
        ttk.Label(right, text="Целевой язык: " + _lang_names.get(self._target_lang, self._target_lang.upper())).pack(anchor="w", pady=(0, 6))

        # Озвучка (радио): авто по полу / женский / мужской / выкл. (не озвучивать)
        self._gender = tk.StringVar(value="auto")
        ttk.Label(right, text="Озвучка:").pack(anchor="w")
        voice_frame = ttk.Frame(right)
        voice_frame.pack(anchor="w", pady=(0, 6))
        ttk.Radiobutton(voice_frame, text="Авто (по полу)", variable=self._gender,
                        value="auto", command=self._persist_settings).pack(side="left")
        ttk.Radiobutton(voice_frame, text="Женский", variable=self._gender,
                        value="female", command=self._persist_settings).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(voice_frame, text="Мужской", variable=self._gender,
                        value="male", command=self._persist_settings).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(voice_frame, text="Не озвучивать", variable=self._gender,
                        value="off", command=self._persist_settings).pack(side="left", padx=(6, 0))

        # Чекбоксы
        self._subs_var = tk.BooleanVar(value=True)
        self._transcript_ru_var = tk.BooleanVar(value=False)
        self._transcript_tgt_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(right, text="Субтитры в видео", variable=self._subs_var,
                        command=self._persist_settings).pack(anchor="w")
        ttk.Checkbutton(right, text="Транскрипт русский", variable=self._transcript_ru_var,
                        command=self._persist_settings).pack(anchor="w")
        ttk.Checkbutton(right, text="Транскрипт целевой язык", variable=self._transcript_tgt_var,
                        command=self._persist_settings).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

        # Интервал
        int_frame = ttk.Frame(right)
        int_frame.pack(fill="x", pady=(0, 2))
        ttk.Label(int_frame, text="Интервал (сек):").pack(side="left")
        self._interval = tk.IntVar(value=5)
        ttk.Scale(int_frame, from_=2, to=30, variable=self._interval,
                  orient="horizontal").pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._int_label = ttk.Label(int_frame, textvariable=self._interval, width=3)
        self._int_label.pack(side="left")

        # Доп. параметры (--set key=value)
        ttk.Label(right, text="Доп. параметры:").pack(anchor="w")
        self._extra = tk.StringVar(value="")
        tk.Entry(right, textvariable=self._extra, bg=BG_ENTRY, fg=FG,
                 insertbackground=FG, font=("Consolas", 9)).pack(fill="x")

        # Сторож
        self._watch_btn = ttk.Button(right, text="Запустить сторож",
                                     style="Gold.TButton", command=self._toggle_watch)
        self._watch_btn.pack(fill="x", pady=(8, 2))
        ttk.Button(right, text="Остановить сторож", style="Gold.TButton",
                   command=self._stop_watch).pack(fill="x")

        # ---- Низ: лог ----
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        ttk.Label(bottom, text="Лог", style="Header.TLabel").pack(anchor="w")

        self.log = tk.Text(
            bottom, bg=LOG_BG, fg=FG, insertbackground=FG,
            font=("Consolas", 9), relief="flat", bd=0, wrap="word",
            state="disabled",
        )
        scr2 = ttk.Scrollbar(bottom, command=self.log.yview)
        self.log.configure(yscrollcommand=scr2.set)
        self.log.pack(side="left", fill="both", expand=True)
        scr2.pack(side="right", fill="y")

    # ---- Список видео ----

    def _refresh_file_list(self):
        self.file_list.delete(0, "end")
        inp = config.INPUT_DIR
        if not os.path.isdir(inp):
            return
        for name in sorted(os.listdir(inp)):
            if name.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
                self.file_list.insert("end", name)

    def _selected_videos(self):
        return [self.file_list.get(i) for i in self.file_list.curselection()]

    # ---- Выбор/загрузка видео в input/ ----

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov *.webm"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        name = os.path.basename(path)
        dest = os.path.join(config.INPUT_DIR, name)
        try:
            os.makedirs(config.INPUT_DIR, exist_ok=True)
            if os.path.abspath(path) == os.path.abspath(dest):
                self._log(f"\n[ЗАГРУЗКА] Файл уже в input/: {name}\n")
            else:
                shutil.copy2(path, dest)
                self._log(f"\n[ЗАГРУЗКА] Скопирован: {name} -> input/\n")
        except Exception as exc:
            self._log(f"\n[ЗАГРУЗКА] Ошибка копирования: {exc}\n")
            messagebox.showerror("Ошибка", f"Не удалось скопировать файл:\n{exc}")
            return
        self._refresh_file_list()
        self._log("[ЗАГРУЗКА] Файл готов к обработке (сторож обработает автоматически).\n")

    # ---- Продолжение алгоритма (кнопка «Далее») ----

    def _next_stage(self):
        """Выполняет следующий доступный этап для выбранных видео."""
        if getattr(self, "_next_running", False):
            self._log("\n[ДАЛЕЕ] Следующий этап уже выполняется — дождитесь завершения.\n")
            return
        sel = self._selected_videos()
        if not sel:
            self._log("\n[ДАЛЕЕ] Сначала выберите видео в списке.\n")
            return
        paths = [os.path.join(config.INPUT_DIR, name) for name in sel]
        cmd = [PYTHON, PIPELINE] + paths
        self._log("\n[ДАЛЕЕ] Запуск следующего этапа:\n  "
                  + " ".join('"' + c + '"' if " " in c else c for c in cmd) + "\n")
        self._next_running = True
        self._next_btn.configure(state="disabled")

        def _finish(_rc):
            self._next_running = False
            self.root.after(0, lambda: self._next_btn.configure(state="normal"))

        run_cmd(cmd, self._log, finish_cb=_finish)

    # ---- Лог ----

    def _log(self, text):
        def _do():
            self.log.configure(state="normal")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")
        self.root.after(0, _do)

    # ---- Сторож ----

    def _build_watch_cmd(self):
        # Настройки GUI персистятся в config.json (см. _persist_settings), а сторож
        # перечитывает config.json на каждой итерации. Поэтому здесь передаём только
        # интервал и ручные --set из поля «Доп. параметры» (легит. переопределения).
        cmd = [PYTHON, PIPELINE, "--watch", "--interval", str(self._interval.get())]
        extra = (self._extra.get() or "").strip()
        if extra:
            for pair in extra.split():
                if "=" in pair:
                    cmd += ["--set", pair]
        return cmd

    def _toggle_watch(self):
        if self._watching:
            self._stop_watch()
            return
        cmd = self._build_watch_cmd()
        self._log(f"\n{'='*60}\n[СТОРОЖ] Запуск: {' '.join(cmd)}\n{'='*60}\n")
        self._watching = True
        self._watch_btn.configure(text="Остановить сторож")

        def _finish(rc):
            self._watching = False
            self.root.after(0, lambda: self._watch_btn.configure(text="Запустить сторож"))
            self._log(f"\n[СТОРОЖ] Завершён (код {rc})\n")

        self._watch_thread = run_cmd(cmd, self._log, finish_cb=_finish)

    def _stop_watch(self):
        if not self._watching:
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "python.exe", "/FI",
                 f"WINDOWTITLE eq *run_pipeline*"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        self._watching = False
        self._watch_btn.configure(text="Запустить сторож")
        self._log("\n[СТОРОЖ] Остановлен\n")

    # ---- Каталоги ----

    def _open_output(self):
        path = config.OUTPUT_DIR
        if os.path.isdir(path):
            os.startfile(path)

    def _open_work(self):
        path = config.WORK_DIR
        if os.path.isdir(path):
            os.startfile(path)

    # ---- Персист настроек GUI в config.json ----
    # Каждое изменение элемента интерфейса сохраняется в config.json, чтобы
    # даже фоновый сторож (автозапуск) применял выбор пользователя без перезапуска.

    def _persist_settings(self):
        self.cfg["target_lang"] = self._target_lang
        g = self._gender.get()
        if g == "off":
            # Озвучка выключена: распознавание/перевод/полишинг остаются, аудио не создаётся.
            self.cfg["dub_enabled"] = False
            self.cfg.pop("forced_gender", None)
        else:
            self.cfg["dub_enabled"] = True
            if g != "auto":
                self.cfg["forced_gender"] = g
            else:
                self.cfg.pop("forced_gender", None)
        self.cfg["show_subtitles"] = bool(self._subs_var.get())
        self.cfg["show_ru_transcript"] = bool(self._transcript_ru_var.get())
        self.cfg["show_tgt_transcript"] = bool(self._transcript_tgt_var.get())
        try:
            config.save_config(self.cfg)
            voice = "выключена" if g == "off" else f"озвучка={g}"
            self._log(f"\n[НАСТРОЙКА] Сохранено: target_lang={self.cfg['target_lang']}, "
                      f"{voice}, субтитры={self.cfg['show_subtitles']}\n")
        except Exception as exc:
            self._log(f"\n[НАСТРОЙКА] Ошибка сохранения config.json: {exc}\n")

    # ---- Запуск ----

    def _on_close(self):
        if self._watching:
            self._stop_watch()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DubApp().run()

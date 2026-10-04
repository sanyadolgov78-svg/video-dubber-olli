# -*- coding: utf-8 -*-
"""Синхронизация с мессенджером МАКС (робот maxbot).

Замыкает конвейер на живой мессенджер:
  * забирает НОВЫЕ видео из чата «ПРЕДЛОЖКА/Proposer FR» — жму «Скачать»,
    в окне «Сохранить как» сохраняет в inbox_max/ (дальше watcher-пайплайн
    сам переносит в input/ и обрабатывает);
  * когда конвейер создал таблицу переводов work/<имя>/<имя>.table.rtf —
    отправляет транскрипт (RU + FR) на проверку в «Избранное»;
  * когда озвучка завершена (output/<имя>_<язык>.mp4) — отправляет готовый
    ролик в «Избранное».

Дубликаты не отправляются дважды: журнал подписей ведётся в
work/max_state.json.
"""
import os
import sys
import json
import time
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import maxbot  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline import rtf_table  # noqa: E402

CHAT_PREDLOZHKA = "ПРЕДЛОЖКА"
CHAT_FAVORITE = "Избранное"
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv"}


def _clean_name(text):
    import re
    out = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    out = out.strip("_.- ") or ""
    return out[:48]


def _looks_video(path):
    """Быстрая проверка: MP4/MOV (ftyp) или достаточно крупный файл."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        if b"ftyp" in head or b"moov" in head or b"mdat" in head:
            return True
        return os.path.getsize(path) > 200_000
    except OSError:
        return False


def _norm_base(name):
    return os.path.splitext(os.path.basename(name))[0]


class MaxSync:
    def __init__(self, bot=None, logger=None):
        self.bot = bot or maxbot.MaxBot()
        self.log = logger or logging.getLogger("max_sync")
        self.state_file = os.path.join(config.WORK_DIR, "max_state.json")
        self.state = self._load_state()
        self.paused_reported = False

    # ---------- журнал ----------
    def _load_state(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"downloaded": {}, "transcripts": {}, "videos": {}}

    def _save_state(self):
        os.makedirs(config.WORK_DIR, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    # ---------- сканирование ПРЕДЛОЖКИ ----------
    _TIME_RE = __import__("re").compile(r"^\d{1,3}[:.]?\d{2}$")

    def _scroll_by(self, wheel_dist):
        from pywinauto import mouse
        try:
            mouse.scroll(coords=(1200, 600), wheel_dist=wheel_dist)
        except Exception:
            pass
        time.sleep(0.35)

    def _line_candidates(self, words):
        """Карточки-кандидаты по правому токену времени (ЧЧ:ММ или «1646»).
        Завершается классификацией по меню при скачивании.

        Тип A: время на одной строке с подписью (подпись слева) — старый жёсткий.
        Тип B: время отдельно (внизу превью), над ним — описание (в пределах
        превью до ~200px). Клик по типу B калиброван по живому ПКМ пользователя:
        центр токена + (−124, +98)."""
        _TIME_RE = self._TIME_RE
        out = []
        for t, cx, cy, rc in words:
            if rc[0] <= 560 or not _TIME_RE.match(t) or rc[3] > 1000 or rc[1] > 880:
                continue
            top = rc[1]
            line = [w for w in words if abs(w[3][1] - top) <= 8 and w[3][0] > 560]
            line.sort(key=lambda w: w[3][0])
            if not line:
                continue
            if any(w[3][0] > rc[2] for w in line):   # время не в конце строки
                continue
            left = min(w[3][0] for w in line)
            right = max(w[3][2] for w in line)
            cap_words = [w[0] for w in line if w[3][2] <= rc[0]]
            if cap_words:
                base = _clean_name(" ".join(cap_words)) or "predlozka"
                out.append({
                    "sig": f"{t}|{base}",
                    "base": base[:48],
                    "caption": " ".join(w[0] for w in line),
                    "click": (int(left + 0.58 * (right - left)), int(top + 62)),
                    "order_y": top,
                    "kind": "A",
                })
                continue
            # тип B: токен отдельно (бейдж длительности), выше — описание.
            # Превью ВИДЕО — ПОД токеном/описанием; точка по двум живым ПКМ
            # пользователя (Ирландия-ролик: бейдж (1048,307) -> клик (841,382);
            # НАТО-ролик: бейдж (1044,394) -> клик (920,492)): середина превью.
            desc = [w for w in words
                    if w[3][0] > 560
                    and top - 200 <= w[3][3] < top - 4
                    and w[3][0] < rc[2] + 12]
            desc.sort(key=lambda q: (q[3][1], q[3][0]))
            if not desc:
                continue
            dbase = _clean_name(" ".join(w[0] for w in desc)) or "predlozka"
            cxp, cyp = int(cx - 160), int(cy + 105)
            cxp = max(640, min(1150, cxp))
            cyp = max(250, min(960, cyp))
            out.append({
                "sig": f"{t}|{dbase}",
                "base": dbase[:48],
                "caption": " ".join(w[0] for w in desc),
                "click": (cxp, cyp),
                "order_y": top,
                "kind": "B",
            })
        out.sort(key=lambda c: c["order_y"])
        return out

    def scan_video_cards(self):
        """Открывает ПРЕДЛОЖКУ, листает от верха вниз и собирает карточки-кандидатов
        (строки, заканчивающиеся временем). Возвращает кандидатов, которых ещё не
        обрабатывали (не скачаны и не отвергнуты как «не видео»)."""
        if not self.bot.app:
            self.bot.connect(timeout=15)
        if not self.bot.open_chat(CHAT_PREDLOZHKA):
            self.log.error("Чат «%s» не найден.", CHAT_PREDLOZHKA)
            return []
        time.sleep(1.5)
        for _ in range(18):          # на самый верх
            self._scroll_by(300)
        seen = set()
        cards = []
        stale = 0
        for page in range(40):
            cands = self._line_candidates(self.bot.ocr_whole())
            newp = 0
            for c in cands:
                s = c["sig"]
                if s in seen:
                    continue
                seen.add(s)
                if s in self.state["downloaded"] or s in self.state.get("nonvideo", {}):
                    continue
                cards.append(c)
                newp += 1
            self.log.debug("  page %d: кандидатов %d, новых %d", page, len(cands), newp)
            stale = 0 if newp else stale + 1
            if stale >= 3:
                break
            self._scroll_by(-200)
        cards.sort(key=lambda c: c["order_y"], reverse=True)  # свежие (ниже) — первыми
        return cards

    def _open_save_menu(self, point, tries=3):
        """Правый клик, ждём пункт «Сохранить видео как» (первый в меню).
        При промахе пробует со сдвигом. Возвращает (центр пункта, слово).
        В конце закрывает меню, если оно не то (ESC)."""
        from pywinauto import mouse
        for dx, dy in ((0, 0), (70, 20), (-60, -15)):
            x, y = point[0] + dx, point[1] + dy
            try:
                mouse.click(button="right", coords=(x, y))
            except Exception:
                continue
            time.sleep(1.2)
            best = self._find_save_item()
            if best:
                return (best[1], best[2]), best
        self.bot.press("{ESC}")
        return None, None

    def _find_save_item(self):
        """Пункт «Сохранить видео как» — первый пункт контекстного меню видео.
        Берём самый верхний «сохранить» в колонке меню (x>560), у которого в той
        же колонке есть пункты ниже (меню, а не слово в тексте сообщения)."""
        words = self.bot.ocr_whole()
        cands = [w for w in words if w[0].lower() == "сохранить"
                 and w[3][0] > 560 and 150 < w[3][1] < 980]
        for w in sorted(cands, key=lambda o: o[3][1]):
            if any(o[3][0] > 560 and abs(o[3][0] - w[3][0]) <= 30
                   and o[3][1] > w[3][3] for o in words):
                return w
        return None

    def download_card(self, card):
        """Полный цикл: контекстное меню → «Сохранить видео как» → диалог
        «Сохранить как» → файл в inbox_max. Возвращает путь или None."""
        cpoint, _ = self._open_save_menu(card["click"])
        if cpoint is None:
            self.log.warning("Меню «Сохранить видео как» не открылось: %s", card["sig"])
            return None
        self.bot.click(*cpoint)
        saved = maxbot.save_file_to_inbox(default_name=card["base"] + (".mp4" if not card["base"].lower().endswith((".mp4", ".mov", ".mkv")) else ""),
                                          timeout=50)
        if saved is None:
            self.log.warning("Диалог сохранения не появился: %s", card["sig"])
            return None
        if not _looks_video(saved):
            os.remove(saved)
            self.log.warning("Скачанный файл не похож на видео (удалён): %s", saved)
            return None
        self.log.info("Скачано в inbox_max: %s (%d байт)", saved, os.path.getsize(saved))
        return saved

    def download_new(self, cards=None):
        new = 0
        if cards is None:
            cards = self.scan_video_cards()
        for card in cards:
            path = self.download_card(card)
            if path:
                self.state.setdefault("downloaded", {})[card["sig"]] = {
                    "path": path, "ts": time.time(), "size": os.path.getsize(path),
                    "caption": card["caption"],
                }
                self._save_state()
                new += 1
                continue
            # решётка методом исключения: меню «Сохранить видео как» не открылось.
            # Тип A (подпись на строке) — верные «не видео» (текст) — чёрный список.
            # Тип B (бейдж длительности + описание) — похоже на видео: НЕ баним,
            # оставляем на следующую попытку (чат живой, карточка могла уехать).
            if card.get("kind") != "B":
                self.state.setdefault("nonvideo", {})[card["sig"]] = {"ts": time.time()}
                self._save_state()
        return new

    # ---------- транскрипт на проверку ----------
    def _combined_transcript(self, base):
        """Собирает читаемый транскрипт (RU + FR) из работы конвейера."""
        work = os.path.join(config.WORK_DIR, base)
        rtf = os.path.join(work, base + ".table.rtf")
        jso = os.path.join(work, base + ".table.json")
        if not os.path.exists(rtf):
            return None
        tgt_lang = config.load_config().get("target_lang", "fr")
        data = rtf_table.read_translation_table(rtf, jso, ru_lang="ru", target_lang=tgt_lang)
        if not data:
            return None
        ru = data.get("ru") or []
        tgt = data.get("target") or []
        lines = [f"Проверка транскрипта и перевода ({base})"]
        for i, s in enumerate(ru):
            st = tgt[i] if i < len(tgt) else {}
            ts = f"[{float(s.get('start') or 0):.1f}]"
            lines.append(f"\n{ts} RU: {(s.get('text') or '').strip()}"
                         f"\n{ts} {tgt_lang.upper()}: {(st.get('text') or '').strip()}")
        out = os.path.join(work, base + ".transcript_otpravka.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out

    def send_transcripts(self):
        sent = 0
        for name in sorted(os.listdir(config.WORK_DIR)):
            base = name
            if base in self.state["transcripts"]:
                continue
            work = os.path.join(config.WORK_DIR, base)
            rtf = os.path.join(work, base + ".table.rtf")
            if not os.path.isfile(rtf):
                continue
            txt = self._combined_transcript(base)
            if not txt:
                continue
            try:
                self.bot.send_file(txt, CHAT_FAVORITE)
                self.state["transcripts"][base] = {"ts": time.time(), "file": txt}
                self._save_state()
                sent += 1
                self.log.info("Транскрипт отправлен в «Избранное»: %s", txt)
            except Exception as e:
                self.log.error("Не удалось отправить транскрипт %s: %s", base, e)
        return sent

    # ---------- готовый ролик ----------
    def send_ready_videos(self, min_age=30):
        sent = 0
        cfg = config.load_config()
        tgt = (cfg.get("target_lang") or "fr").lower().strip()
        for name in sorted(os.listdir(config.OUTPUT_DIR)):
            if not os.path.splitext(name)[1].lower() in VIDEO_EXT:
                continue
            if not name.endswith("_" + tgt + ".mp4"):
                continue
            full = os.path.join(config.OUTPUT_DIR, name)
            if time.time() - os.path.getmtime(full) < min_age:
                continue
            base = name[: -len("_" + tgt + ".mp4")]
            tbl = os.path.join(config.WORK_DIR, base, base + ".table.rtf")
            entry = self.state["videos"].get(name)
            if entry is not None:
                last = entry.get("tbl_mtime")
                if last is None:
                    last = entry.get("ts") or 0
                if not (os.path.exists(tbl) and os.path.getmtime(tbl) > last + 2):
                    continue
            try:
                self.bot.send_file(full, CHAT_FAVORITE)
                self.state["videos"][name] = {
                    "ts": time.time(), "file": full, "mtime": os.path.getmtime(full),
                    "tbl_mtime": os.path.getmtime(tbl) if os.path.exists(tbl) else 0}
                self._save_state()
                sent += 1
                self.log.info("Готовый ролик отправлен в «Избранное»: %s", full)
            except Exception as e:
                self.log.error("Не удалось отправить ролик %s: %s", name, e)
        return sent

    # ---------- весь цикл ----------
    def _pause_marker(self):
        return os.path.join(config.WORK_DIR, ".robot_pause")

    def _stop_marker(self):
        return os.path.join(config.WORK_DIR, ".robot_stop")

    def run_once(self):
        res = {}
        if os.path.exists(self._stop_marker()):
            if not self.paused_reported:
                self.log.info("Робот остановлен флагом %s — действий не выполняю.",
                              self._stop_marker())
                self.paused_reported = True
            return res
        if os.path.exists(self._pause_marker()):
            if not self.paused_reported:
                self.log.info(
                    "Робот на паузе по загрузке (файл-флаг %s): новые посты не тяну, "
                    "но готовое отправляю в «Избранное».", self._pause_marker())
                self.paused_reported = True
        else:
            self.paused_reported = False
        try:
            self.bot.connect(timeout=15).focus()
        except Exception as e:
            self.log.error("МАКС недоступен: %s", e)
            res["error"] = str(e)
            return res
        if not os.path.exists(self._pause_marker()):
            res["downloaded"] = self.download_new()
        else:
            res["downloaded"] = 0
        res["transcripts"] = self.send_transcripts()
        res["videos"] = self.send_ready_videos()
        return res

    def run_watch(self, interval=30, max_cycles=None):
        n = 0
        while max_cycles is None or n < max_cycles:
            try:
                res = self.run_once()
                self.log.info("Цикл: %s", res)
            except Exception as e:
                self.log.error("Сбой цикла: %s", e)
            n += 1
            time.sleep(interval)


def _acquire_robot_lock():
    """Ограничение: ровно один робот в watch-режиме (файл .max_robot.pid)."""
    import ctypes
    import psutil
    lock = os.path.join(config.WORK_DIR, ".max_robot.pid")
    os.makedirs(config.WORK_DIR, exist_ok=True)
    OK = False
    while not OK:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            old = None
            try:
                with open(lock, "r", encoding="utf-8") as f:
                    old = int(f.read().strip())
            except (ValueError, OSError):
                pass
            alive_robot = False
            if old:
                try:
                    cmd = psutil.Process(old).cmdline() or []
                    alive_robot = any("max_robot.py" in str(a) for a in cmd)
                except Exception:
                    alive_robot = False
            if alive_robot:
                logging.getLogger("max_sync").warning(
                    "Робот уже работает (PID %s) — второй экземпляр не запускаю.", old)
                return False
            try:
                os.remove(lock)
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        OK = True
    return True


def _release_robot_lock():
    lock = os.path.join(config.WORK_DIR, ".max_robot.pid")
    try:
        pid = int(open(lock, "r").read().strip())
        if pid == os.getpid():
            os.remove(lock)
    except (ValueError, OSError):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Робот МАКС: ПРЕДЛОЖКА -> конвейер -> Избранное")
    parser.add_argument("--watch", action="store_true", help="Цикл без остановки")
    parser.add_argument("--interval", type=int, default=30, help="Пауза между циклами (сек)")
    parser.add_argument("--cycles", type=int, default=None, help="Число циклов (без --watch)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    if args.watch and not _acquire_robot_lock():
        return 1
    try:
        s = MaxSync()
        if args.watch:
            s.run_watch(interval=args.interval)
        else:
            cycles = args.cycles or 1
            for _ in range(cycles):
                print(s.run_once())
                if cycles > 1:
                    time.sleep(args.interval)
    finally:
        if args.watch:
            _release_robot_lock()


if __name__ == "__main__":
    main()
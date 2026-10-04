# -*- coding: utf-8 -*-
"""Робот для мессенджера МАКС: навигация по окну + OCR-чтение WebEngine зон.

Возможности:
  * connect()/focus() — привязка к окну MAX
  * chat_buttons() — список чатов (имя, rect)
  * open_chat(name) — открыть чат кликом по кнопке
  * open_media_tab(name) — клик по вкладке (Медиа/Файлы/...)
  * shot(left, top, right, bottom) -> PNG-файл (обрезок экрана)
  * ocr_words(left, top, right, bottom) -> [(word, cx, cy, rect)]  (координаты экрана)
  * click(x, y), right_click(x, y), press(keys), type_text(s)
  * regions под каждый чат/меню
"""
import os, sys, time, asyncio, tempfile

import pywinauto
import pywinauto.mouse as mouse
import pywinauto.keyboard as kb
from pywinauto import Desktop, Application

PIL = None
try:
    from PIL import Image, ImageGrab
    PIL = True
except Exception:
    PIL = False

from winsdk.windows.storage import StorageFile, FileAccessMode
from winsdk.windows.graphics.imaging import BitmapDecoder
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.globalization import Language

ROOT = os.path.dirname(os.path.abspath(__file__))
SHOT_TMP = os.path.join(ROOT, "work", "_ocr_tmp.png")


class MaxBot:

    def __init__(self):
        self.app = None
        self.win_rect = None

    # ---------- подключение ----------
    def connect(self, timeout=20):
        t0 = time.time()
        while True:
            for w in Desktop(backend="uia").windows():
                try:
                    if w.window_text() == "MAX" and w.is_visible():
                        self.app = w
                        r = w.rectangle()
                        self.win_rect = (r.left, r.top, r.right, r.bottom)
                        return self
                except Exception:
                    continue
            if time.time() - t0 > timeout:
                raise RuntimeError("окно МАКСа не найдено (запустите и залогиньтесь)")
            time.sleep(2)

    def focus(self):
        self.app.set_focus()
        time.sleep(0.6)
        return self

    # ---------- примитивы окна ----------
    def _buttons(self, left=90, right=560):
        out = []
        try:
            desc = list(self.app.descendants(control_type="Button"))
        except Exception:
            desc = []
        for b in desc:
            try:
                r = b.rectangle()
                if r and r.left >= left and r.left < right and r.bottom > 0:
                    out.append((b.window_text().strip(), (r.left, r.top, r.right, r.bottom)))
            except Exception:
                continue
        return out

    def chat_buttons(self):
        return sorted([x for x in self._buttons(90, 560) if x[0]], key=lambda t: t[1][1])

    def open_chat(self, name, scroll_tries=6):
        self.focus()
        for _ in range(scroll_tries):
            for nm, r in self.chat_buttons():
                if name.lower() in nm.lower():
                    self.click_rect(r)
                    time.sleep(1.2)
                    return True
            # список чатов мог быть прокручен мимо искомой плитки — листаем вниз
            try:
                mouse.scroll(coords=(320, 500), wheel_dist=-260)
            except Exception:
                pass
            time.sleep(0.6)
        return False

    def verify_open_chat(self, name):
        """Подтвердить по заголовку области сообщений, что открыт чат <name>.
        Название текущего чата — верхняя плашка (Static) над сообщениями."""
        words = self.ocr_words(590, 45, 1910, 128)
        for t, cx, cy, r in words:
            if name.lower() in t.lower():
                return True
        return False

    def click_chat_coords(self, name):
        """Вернуть rect чата по имени без клика."""
        for nm, r in self.chat_buttons():
            if name.lower() in nm.lower():
                return r
        return None

    def click_rect(self, r):
        mouse.click(coords=((r[0] + r[2]) // 2, (r[1] + r[3]) // 2))
        time.sleep(0.6)

    def right_click_rect(self, r):
        mouse.right_click(coords=((r[0] + r[2]) // 2, (r[1] + r[3]) // 2))
        time.sleep(0.8)

    def click(self, x, y):
        mouse.click(coords=(int(x), int(y)))
        time.sleep(0.6)

    def right_click(self, x, y):
        mouse.right_click(coords=(int(x), int(y)))
        time.sleep(0.8)

    def press(self, keys):
        kb.send_keys(keys)
        time.sleep(0.4)

    def type_text(self, s):
        kb.type_keys(s, pause=0.02)
        time.sleep(0.4)

    # ---------- скриншот и OCR ----------
    def shot(self, left, top, right, bottom):
        if PIL:
            img = ImageGrab.grab(bbox=(int(left), int(top), int(right), int(bottom)))
            img.save(SHOT_TMP)
        else:
            # запасной путь: PowerShell + System.Drawing
            import subprocess
            ps = (f"Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
                  f"$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
                  f"$img=New-Object System.Drawing.Bitmap($b.Width,$b.Height);"
                  f"$g=[System.Drawing.Graphics]::FromImage($img);"
                  f"$r=New-Object System.Drawing.Rectangle({int(left)},{int(top)},{int(right-left)},{int(bottom-top)});"
                  f"$g.CopyFromScreen($r.Location,[System.Drawing.Point]::Empty,$r.Size);"
                  f"$img.Save(r'{SHOT_TMP}');$g.Dispose();$img.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
        return SHOT_TMP

    @staticmethod
    def _ocr_impl(png):
        async def _run():
            f = await StorageFile.get_file_from_path_async(png)
            stream = await f.open_async(FileAccessMode.READ)
            dec = await BitmapDecoder.create_async(stream)
            bmp = await dec.get_software_bitmap_async()
            eng = OcrEngine.try_create_from_language(Language("ru"))
            if eng is None:
                eng = OcrEngine.try_create_from_user_profile_languages()
            if eng is None:
                return []
            res = await eng.recognize_async(bmp)
            out = []
            for line in res.lines:
                for w in line.words:
                    r = w.bounding_rect
                    out.append((w.text, int(r.x), int(r.y), int(r.width), int(r.height)))
            return out
        return asyncio.run(_run())

    def ocr_words(self, left, top, right, bottom):
        """OCR прямоугольника экрана; координаты слов — АБСОЛЮТНЫЕ (экранные)."""
        png = self.shot(left, top, right, bottom)
        words = self._ocr_impl(png)
        return [(t, left + x + w // 2, top + y + h // 2, (left + x, top + y, left + x + w, top + y + h))
                for (t, x, y, w, h) in words]

    def ocr_whole(self):
        r = self.win_rect
        return self.ocr_words(r[0], r[1], r[2], r[3])

    def find_word(self, words, *targets):
        """Вернуть (центр_x, центр_y) слова, содержащего одну из целей (без учёта регистра)."""
        tl = [t.lower() for t in targets]
        for t, cx, cy, r in words:
            if any(x.lower() in t.lower() for x in tl):
                return (cx, cy, r)
        return None

    # ---------- отправка файла в чат (калибровка «Избранное») ----------
    @staticmethod
    def _find_open_dialog(timeout):
        hwnd = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            for w in Desktop(backend="win32").windows():
                try:
                    if w.class_name() == "#32770" and w.window_text().strip() == "Выбрать файл":
                        return w.handle
                except Exception:
                    continue
            time.sleep(0.5)
        return None

    def _open_paperclip_menu(self):
        """Открыть меню скрепки и вернуть центр пункта «Файл» либо None.
        Клик по скрепке тогглит меню, поэтому после клика верифицируем слова."""
        for _ in range(4):
            self.click(607, 978)
            time.sleep(0.7)
            for t, cx, cy, r in self.ocr_whole():
                if t.lower().startswith("файл") and 600 < cx < 760 and 770 < cy < 870:
                    return (cx, cy)
            time.sleep(0.5)
        return None

    def _fill_open_dialog(self, path, dlg_hwnd):
        app = Application(backend="win32").connect(handle=dlg_hwnd)
        dlg = app.window(class_name="#32770")
        edit = None
        for e in dlg.descendants(class_name="Edit"):
            r = e.rectangle()
            if 230 < r.left < 280 and 530 < r.top < 575:
                edit = e
                break
        if edit is None:
            raise RuntimeError("поле «Имя файла» в диалоге не найдено")
        edit.set_focus()
        edit.set_edit_text(path)
        time.sleep(0.6)
        dlg["Открыть"].click()

    def send_file_to_fav(self, path):
        """Отправить файл в чат «Избранное» (прежний вызов)."""
        return self.send_file(path, "Избранное")

    def send_file(self, path, chat="Избранное"):
        """Отправить файл в чат: скрепка → «Файл» → нативный диалог
        «Выбрать файл» (путь в поле «Имя файла», Открыть) → Enter.

        Координаты привязки (скрепка 607,978, пункт меню 657,817, поле диалога)
        соответствуют текущей раскладке окна МАКС 1920x1080.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if not self.open_chat(chat, scroll_tries=3):
            raise RuntimeError("чат «%s» не найден в списке чатов" % chat)

        item = self._open_paperclip_menu()
        if item is None:
            raise RuntimeError("меню скрепки не открылось")
        self.click(*item)

        hwnd = self._find_open_dialog(40)
        if hwnd is None:
            raise RuntimeError("диалог «Выбрать файл» не появился")
        self._fill_open_dialog(path, hwnd)
        time.sleep(2.5)

        self.click(686, 979)
        time.sleep(0.6)
        self.press("{ENTER}")
        time.sleep(2.0)
        return True


def _win32_dlg(title_contains=("Сохранить", "Скачать")):
    """Наверх-менеджер: вернуть handle активного win32-диалога #32770 по маске заголовка."""
    import pywinauto
    from pywinauto import Desktop as _D
    for w in _D(backend="win32").windows():
        try:
            if not w.is_visible():
                continue
            r = w.rectangle()
            if r.right <= 200 or r.bottom <= 200:
                continue
            c = w.class_name()
            t = w.window_text().strip()
            if c == "#32770" and (not title_contains or any(k.lower() in t.lower() for k in title_contains)):
                return w.handle
        except Exception:
            continue
    return None


def save_file_to_inbox(default_name=None, overwrite=False, inbox_dir=None, timeout=120):
    """Возвращает имена файлов диалога сохранения МАКС и сохраняет в inbox_dir.

    Ожидает появления окна #32770 «Сохранить как», вписывает имя в поле Edit
    (широкое поле-имени), жмёт «Сохранить»/Enter и ждёт появления файла.
    default_name: имя файла (имя берётся из диалога, если не задано).
    Возвращает полный путь сохранённого файла или None.
    """
    from pywinauto import Application
    if inbox_dir is None:
        inbox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox_max")
    os.makedirs(inbox_dir, exist_ok=True)

    t0 = time.time()
    while time.time() - t0 < timeout:
        hwnd = _win32_dlg(("Сохранить", "Скачать"))
        if hwnd:
            break
        time.sleep(0.5)
    else:
        return None

    app = Application(backend="win32").connect(handle=hwnd)
    dlg = app.window(class_name="#32770")

    edit = None
    best_w = -1
    for e in dlg.descendants(class_name="Edit"):
        try:
            r = e.rectangle()
        except Exception:
            continue
        w = r.right - r.left
        if w > best_w:
            best_w = w
            edit = e
    if edit is None:
        return None
    try:
        cur = (edit.window_text() or "").strip()
    except Exception:
        cur = ""
    name = (default_name or "").strip()
    if not name:
        import re
        base = re.sub(r"[^\w.\-]+", "_", cur).strip("_") or "video"
        if not base.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts", ".flv")):
            base += ".mp4"
        name = base

    target = os.path.join(inbox_dir, name)
    n = 2
    while os.path.exists(target) and not overwrite:
        root, ext = os.path.splitext(name)
        target = os.path.join(inbox_dir, f"{root} ({n}){ext}")
        n += 1

    edit.set_focus()
    edit.set_edit_text(target)
    time.sleep(0.6)
    try:
        dlg["Сохранить"].click()
    except Exception:
        edit.type_keys("{ENTER}")
    # ждём, пока файл появится
    t1 = time.time()
    while time.time() - t1 < 60:
        if os.path.exists(target) and os.path.getsize(target) > 0:
            return target
        time.sleep(0.5)
    return target if os.path.exists(target) else None


def connect(bot):
    return bot.connect()


if __name__ == "__main__":
    b = MaxBot().connect().focus()
    words = b.ocr_whole()
    print("всего слов на экране:", len(words))
    for t, cx, cy, r in words:
        print(f"   {t!r}  центр=({cx},{cy})")
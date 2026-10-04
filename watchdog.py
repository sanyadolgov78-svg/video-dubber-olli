"""Надзор за сторожем конвейера: самовосстановление и автозапуск.

Сторож (run_pipeline.py --watch) — единственная точка отказа всей автоматики:
пока он жив, новые файлы и правки таблиц подхватываются сами. Стоит ему
закрыться (закрыли окно, ребут, сбой, случайно убили) — конвейер молчит,
и это выглядит как «затык».

Этот надзор работает постоянно и:
  * проверяет живость сторожа каждые N секунд;
  * поднимает его заново, если он мёртв (в т.ч. чистит устаревший .pid);
  * пишет heartbeat в work/.watchdog.json — видно, что система жива;
  * ведёт журнал work/watchdog.log с ограниченным размером.

Запуск вручную:  .venv\\Scripts\\python.exe watchdog.py
Однократная проверка: .venv\\Scripts\\python.exe watchdog.py --once
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "work")
LOCK = os.path.join(WORK, ".watcher.pid")
WD_LOCK = os.path.join(WORK, ".watchdog.pid")
HEARTBEAT = os.path.join(WORK, ".watchdog.json")
LOG = os.path.join(WORK, "watchdog.log")
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
WATCH_LOG = os.path.join(WORK, "watcher.err.log")
ROBOT_LOG = os.path.join(WORK, "robot.err.log")
ROBOT_LOCK = os.path.join(WORK, ".max_robot.pid")
ROBOT_CMD = [PYTHON, "max_robot.py", "--watch", "--interval", "30"]

CHECK_INTERVAL = 20
LOG_MAX_BYTES = 512 * 1024

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
CREATE_NO_WINDOW = 0x08000000
SYNCHRONIZE = 0x00100000


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    line = f"{_now()}  {msg}"
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > LOG_MAX_BYTES:
            os.replace(LOG, LOG + ".1")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(line, flush=True)
    except Exception:
        pass


def _pid_alive(pid):
    try:
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


def _import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _is_cmdline_process(pid, *markers):
    """PID жив И его командная строка содержит все markers.

    Голая проверка _pid_alive опасна: Windows переиспользует PID после
    смерти процесса, и сторож-«зомби» из .pid-файла может совпасть с чужим
    процессом. Сверка по cmdline исключает ложное «сторож жив».
    """
    if not _pid_alive(pid):
        return False
    ps = _import_psutil()
    if ps is None:
        return _pid_alive(pid)
    try:
        cmd = ps.Process(pid).cmdline() or []
    except Exception:
        return False
    joined = [str(a) for a in cmd]
    return all(any(m in s for s in joined) for m in markers)


def _watcher_pid_from_lock():
    try:
        with open(LOCK, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if _is_cmdline_process(pid, "run_pipeline.py", "--watch"):
            return pid
        return None
    except (ValueError, OSError, IOError):
        return None


def _watcher_pids_by_scan():
    """Резервный способ: ищем сторож даже без корректного .pid-файла.

    Нужен, чтобы не поднять вторую копию, если pid-файл потерялся или
    остался от прошлого запуска с другим PID.
    """
    found = []
    ps = _import_psutil()
    if ps is None:
        return found
    for p in ps.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
        except Exception:
            continue
        if not any("run_pipeline.py" in str(a) for a in cmd):
            continue
        if any("watchdog" in str(a) for a in cmd):
            continue
        if not any(a == "--watch" for a in cmd):
            continue
        try:
            if p.is_running():
                found.append(p.pid)
        except Exception:
            continue
    return found


def _start_watcher():
    os.makedirs(WORK, exist_ok=True)
    # Устаревший .pid сбил бы сторож с толку: он увидит "живой" PID.
    if os.path.exists(LOCK):
        try:
            os.remove(LOCK)
        except OSError:
            pass
    args = [PYTHON, "run_pipeline.py", "--watch", "--interval", "10",
            "--set", "whisper_model=small"]
    logf = open(WATCH_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(
        args, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=DETACHED | CREATE_NO_WINDOW,
        close_fds=True,
    )
    logf.close()
    return proc.pid


def _robot_pids_by_scan():
    found = []
    ps = _import_psutil()
    if ps is None:
        return found
    for p in ps.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
        except Exception:
            continue
        if not any("max_robot.py" in str(a) for a in cmd):
            continue
        if any("watchdog" in str(a) for a in cmd):
            continue
        if not any(a == "--watch" for a in cmd):
            continue
        try:
            if p.is_running():
                found.append(p.pid)
        except Exception:
            continue
    return found


def _start_robot():
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(ROBOT_LOCK):
        try:
            os.remove(ROBOT_LOCK)
        except OSError:
            pass
    logf = open(ROBOT_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(
        ROBOT_CMD, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=DETACHED | CREATE_NO_WINDOW,
        close_fds=True,
    )
    logf.close()
    return proc.pid


def _robot_state():
    pids = _robot_pids_by_scan()
    return pids[0] if pids else None


def _watcher_state():
    pid = _watcher_pid_from_lock()
    via_lock = pid is not None
    if pid is None:
        pids = _watcher_pids_by_scan()
        if pids:
            pid = pids[0]
    return pid, via_lock


def _write_heartbeat(state):
    watcher_pid, via_lock = _watcher_state()
    last_log = None
    try:
        last_log = datetime.fromtimestamp(os.path.getmtime(WATCH_LOG)).strftime("%H:%M:%S")
    except OSError:
        pass
    data = {
        "ts": _now(),
        "watchdog_pid": os.getpid(),
        "watcher_alive": watcher_pid is not None,
        "watcher_pid": watcher_pid,
        "pid_from_lock": via_lock,
        "watcher_log_updated": last_log,
        "restarts": state["restarts"],
        "last_start": state["last_start"],
        "robot_pid": _robot_state(),
        "robot_log_updated": None,
    }
    try:
        rl = datetime.fromtimestamp(os.path.getmtime(ROBOT_LOG)).strftime("%H:%M:%S")
        data["robot_log_updated"] = rl
    except OSError:
        pass
    try:
        with open(HEARTBEAT, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return watcher_pid


WATCHDOG_LOCK_MAX_AGE = 600  # 10 минут: старше — считаем застрявшим и забираем


def _acquire_self_lock():
    """Атомарная блокировка «ровно одного надзора» (O_CREAT|O_EXCL).

    Даже если несколько копий сторожа стартуют одновременно (разные интерпретаторы,
    планировщик, ручной запуск), выживает ровно один: создавший файл. Остальные либо
    видят живой PID и выходят, либо (если PID мёртв или файл старше 10 минут) забирают
    блокировку. Гонки не порождают второго владельца.
    """
    os.makedirs(WORK, exist_ok=True)
    while True:
        try:
            fd = os.open(WD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            old = None
            try:
                with open(WD_LOCK, "r", encoding="utf-8") as f:
                    old = int(f.read().strip())
            except (ValueError, OSError, IOError):
                pass
            try:
                age = time.time() - os.path.getmtime(WD_LOCK)
            except OSError:
                age = WATCHDOG_LOCK_MAX_AGE + 1
            if old is not None and old != os.getpid() and _is_cmdline_process(old, "watchdog.py") and age < WATCHDOG_LOCK_MAX_AGE:
                _log(f"Надзор уже работает (PID {old}) — второй экземпляр не запускаю.")
                return False
            _log(f"Блокировка надзора от {('PID ' + str(old)) if old else 'неизвестного'}"
                 f" (возраст {int(age)} с) — забираю.")
            try:
                os.remove(WD_LOCK)
            except OSError:
                pass
            time.sleep(0.5)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True


def main():
    ap = argparse.ArgumentParser(description="Надзор за сторожем конвейера")
    ap.add_argument("--once", action="store_true", help="одна проверка и выход")
    ap.add_argument("--interval", type=int, default=CHECK_INTERVAL)
    args = ap.parse_args()

    if not os.path.exists(PYTHON):
        _log(f"ОШИБКА: не найден интерпретатор {PYTHON}")
        return 1

    if not args.once and not _acquire_self_lock():
        return 0

    state = {"restarts": 0, "last_start": None}
    hb = HEARTBEAT
    if os.path.exists(hb):
        try:
            with open(hb, "r", encoding="utf-8") as f:
                old = json.load(f)
            state["restarts"] = int(old.get("restarts") or 0)
            state["last_start"] = old.get("last_start")
            state["robot_restarts"] = int(old.get("robot_restarts") or 0)
        except (ValueError, OSError, IOError):
            pass

    _log("=== Надзор запущен ===")

    while True:
        try:
            watcher_pid = _write_heartbeat(state)
            if watcher_pid is None:
                new_pid = _start_watcher()
                state["restarts"] += 1
                state["last_start"] = _now()
                _log(f"Сторож не найден — запущен заново (PID {new_pid}), "
                     f"перезапусков: {state['restarts']}")
                _write_heartbeat(state)
            elif state.get("announced") != watcher_pid:
                _log(f"Сторож жив (PID {watcher_pid}). Наблюдение идёт.")
                state["announced"] = watcher_pid

            robot_pid = _robot_state()
            if robot_pid is None:
                new_pid = _start_robot()
                state["robot_restarts"] = state.get("robot_restarts", 0) + 1
                _log(f"Робот МАКС не найден — запущен заново (PID {new_pid}), "
                     f"перезапусков робота: {state['robot_restarts']}")
                _write_heartbeat(state)
            elif state.get("robot_announced") != robot_pid:
                _log(f"Робот МАКС жив (PID {robot_pid}). Наблюдение идёт.")
                state["robot_announced"] = robot_pid
        except Exception:
            _log("Сбой в цикле надзора:\n" + traceback.format_exc())

        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    sys.exit(main())

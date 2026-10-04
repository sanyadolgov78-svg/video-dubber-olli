# -*- coding: utf-8 -*-
"""Робот МАКС: цикл «ПРЕДЛОЖКА -> конвейер -> Избранное» (см. pipeline.max_sync).

Запуск:
  .venv\\Scripts\\python.exe max_robot.py --watch --interval 30
  .venv\\Scripts\\python.exe max_robot.py --cycles 1
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from pipeline.max_sync import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
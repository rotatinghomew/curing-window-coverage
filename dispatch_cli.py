#!/usr/bin/env python3
"""仓库根目录入口：等价于 python -m dispatch_planning。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch_planning.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())

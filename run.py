#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目入口。

用法（任意装有 PyYAML 的 Python 环境）：
    python run.py list
    python run.py run --suite all --pairwise
    python run.py report --run latest --open

这里只做一件事：把 src 加进模块搜索路径，然后把控制权交给 llmeval.cli。
保持入口极薄，是为了让 `import llmeval` 在任何地方都能正常工作。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmeval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

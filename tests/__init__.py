# -*- coding: utf-8 -*-
"""测试包。把 src 加进模块路径，让测试可以 import llmeval。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

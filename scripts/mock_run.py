#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线一键演示。

用途：在没有任何 API key、没有任何网络的机器上，验证整条评测流水线是通的。

    python scripts\\mock_run.py

它会用内置的模拟引擎跑完全量套件、生成报告，并把报告路径打印出来。
产出的所有数据都是模拟的，不是任何真实模型的评测结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval import config as cfg  # noqa: E402
from llmeval.analysis import summarize_model  # noqa: E402
from llmeval.bias import summarize_bias  # noqa: E402
from llmeval.pipeline import Runner  # noqa: E402
from llmeval.report import render  # noqa: E402


def main() -> int:
    cfg.load_env_file()
    print("=" * 68)
    print("llmeval 离线演示：全部回答由内置模拟引擎生成，不是真实模型结论")
    print("=" * 68)

    runner = Runner("all", pairwise=True, progress=lambda m: print(m, flush=True))
    result = runner.run()

    stats = [
        summarize_model(
            result.turns,
            sid,
            label=getattr(result.models.get(sid), "label", sid) or sid,
            is_mock=getattr(result.models.get(sid), "is_mock", False),
        )
        for sid in result.summary.suts
    ]

    print("\n汇总：")
    for s in stats:
        print(
            f"  {s.label:<24} 综合分 {s.mean_score}  "
            f"通过率 {s.pass_rate}  失败归因 {s.attribution}"
        )
    print(f"  mock 标志：{result.summary.mock}（报告页首会显示警示条）")

    report_path = render(result, bias=summarize_bias(result.pairwise or []))
    print(f"\n报告：{report_path}")
    print("\n下一步：")
    print("  1. 打开报告看结构与图表是否正常")
    print("  2. 在 .env 里填 API key，在 configs/models.yaml 里启用真实模型")
    print("  3. 重新执行 run.py run --suite all，得到的就是真实评测结果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

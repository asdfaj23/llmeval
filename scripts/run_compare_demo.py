# -*- coding: utf-8 -*-
"""
离线对比 harness：ReAct vs Plan-and-Execute（同一 mock 模型，隔离框架差异）。

为什么需要这个脚本而不是直接 `run.py run`：
    框架默认会启用 configs/models.yaml 里所有「可用」的裁判模型。
    而我们做这次对比只关心「两套 agent 框架在同一确定性评测集上的可复现表现」，
    用不到真实 LLM 裁判（真裁判既花钱又会因为网络超时把整轮跑挂）。
    所以这个脚本把模型配置在内存里改成「离线版」再喂给 Runner，
    完全不改 configs/models.yaml —— 真实评测配置保持原样。

对比原理（简历可写）：
    用工厂把同一个 mock 模型端点（mock-strong）分别以 strategy=plan_execute
    与 strategy=react 实例化，跑同一份评测集
    （datasets/multiagent/collaboration.jsonl，含一道故障注入题 ma-014），
    从而隔离「框架差异」而非「模型差异」。

用法（任意装有 PyYAML 的 Python 3.9+ 环境）：
    python scripts/run_compare_demo.py

产物：
    outputs/runs/<run>/turns.jsonl, summary.json
    outputs/runs/<run>/compare_react_vs_plan.html   ← 对比报告
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmeval import config as cfg  # noqa: E402
from llmeval.pipeline import Runner  # noqa: E402
from llmeval.collab.compare_report import compare_run  # noqa: E402


def _print(msg: str) -> None:
    print(msg, flush=True)


def build_offline_models() -> cfg.ModelsConfig:
    """在内存里造一份「离线版」模型配置：只启用两个 mock SUT，禁用一切真实裁判。

    不动磁盘上的 configs/models.yaml，真实评测配置原样保留。
    strategy 字段仍从 models.yaml 读取（load_models 会把 plan_execute / react 带进来），
    所以对比报告能正确按策略分组。
    """
    models = cfg.load_models()
    for m in models.suts:
        m.enabled = m.id in ("mock-strong", "mock-strong-react")
    for m in models.judges:
        m.enabled = False  # 离线：不发任何真实 API 请求
    return models


def main() -> int:
    cfg.load_env_file()
    models = build_offline_models()

    runner = Runner(
        "multi_agent",
        sut_ids=["mock-strong", "mock-strong-react"],
        models=models,
        progress=_print,
    )
    result = runner.run()

    _print(f"\n已生成运行：{result.run_dir}")
    _print(f"  SUT：{result.summary.suts}")
    _print(f"  是否 mock：{result.summary.mock}（对比报告会强制标注）")

    cmp_path = compare_run(result.run_dir)
    _print(f"\n对比报告：{cmp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

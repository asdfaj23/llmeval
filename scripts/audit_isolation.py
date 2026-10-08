# -*- coding: utf-8 -*-
"""
数据集隔离自检：扫一遍整个题库，检查有没有「答案泄漏进模型可见材料」。

为什么值得单独做一个脚本
    答案泄漏是数据集构建阶段最容易犯、后果也最重的错误：
    参考答案或关键事实被顺手写进了 prompt / context，
    模型照抄一遍就能拿满分，而报告上只会显示「这个模型真强」。
    这种错误不会报错、不会崩溃，只会让整轮评测静默失效。

    所以它必须是一道能重复跑、能进 CI 的检查，
    而不是靠出题人事后回想。

用法
    python scripts/audit_isolation.py               # 只报告
    python scripts/audit_isolation.py --fail-on-leak  # 有泄漏就返回非零（给 CI 用）

零成本、全离线，不调任何 API。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.containment import check_answer_isolation, protected_fragments  # noqa: E402
from llmeval.schema import Sample  # noqa: E402

DATASETS = ROOT / "datasets"


def iter_samples():
    """遍历题库下所有 jsonl，产出 (相对路径, 行号, Sample 或 None)。"""
    for path in sorted(DATASETS.rglob("*.jsonl")):
        rel = path.relative_to(ROOT).as_posix()
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # 金标文件不是一行一题的结构，跳过是预期行为
                    yield rel, lineno, None
                    continue
                if not isinstance(raw, dict) or "id" not in raw:
                    yield rel, lineno, None
                    continue
                try:
                    yield rel, lineno, Sample.from_dict(raw, source=rel)
                except Exception:  # noqa: BLE001 - 自检不该被单条脏数据打断
                    yield rel, lineno, None


def main() -> int:
    parser = argparse.ArgumentParser(description="数据集答案隔离自检")
    parser.add_argument(
        "--fail-on-leak",
        action="store_true",
        help="发现泄漏时返回非零退出码（CI 用）",
    )
    args = parser.parse_args()

    n_samples = 0
    n_fragments = 0
    leaks: list[dict] = []
    by_source: dict[str, int] = {}

    for rel, lineno, sample in iter_samples():
        if sample is None:
            continue
        n_samples += 1
        frags = protected_fragments(sample)
        n_fragments += len(frags)
        if not frags:
            continue
        report = check_answer_isolation(sample)
        for leak in report["leaks"]:
            by_source[leak["source"]] = by_source.get(leak["source"], 0) + 1
            leaks.append({"file": rel, "line": lineno, "sample_id": sample.id, **leak})

    print("=" * 62)
    print("数据集答案隔离自检")
    print("=" * 62)
    print(f"扫描样本        : {n_samples}")
    print(f"受保护片段      : {n_fragments}（长度 >= 12 字的参考答案 / 金标 / 测试期望值）")
    print(f"发现泄漏        : {len(leaks)}")
    if by_source:
        print("按字段分布      :")
        for key, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  - {key:<18} {cnt}")
    print("-" * 62)
    if not leaks:
        print("结论：没有参考答案或金标内容出现在模型可见材料里。")
        return 0

    print("泄漏明细（最多打印 20 条）：")
    for leak in leaks[:20]:
        print(f"  [{leak['file']}:{leak['line']}] {leak['sample_id']}")
        print(f"      字段 {leak['source']} · {leak['chars']} 字 · 片段：{leak['excerpt']}")
    print("-" * 62)
    print("结论：存在答案泄漏，这些题目的分数不可信 —— 模型可能只是抄了题面。")
    return 1 if args.fail_on_leak else 0


if __name__ == "__main__":
    raise SystemExit(main())

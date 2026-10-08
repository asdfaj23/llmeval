# -*- coding: utf-8 -*-
"""
参考答案自洽性检查：每道题的 gold 必须先满足自己出题时定下的约束。

为什么必须查
    约束是交给规则层硬判的。如果参考答案本身违反了自己写的约束，
    那这道题无论模型怎么答都拿不到满分 —— 它的实际满分是 4 分不是 5 分，
    而且是静默的：报告上只会显示「这个模型这道题差一点」。
    加约束、改题面之后最容易引入这类错误，所以它必须是一道能重复跑的检查。

查什么
    1. 带 constraints 的题：拿 reference 当作答，跑一遍规则层，必须全部通过。
    2. 选择题：reference 必须是一个选项字母，不能是整句解释。

用法
    python scripts/audit_reference.py                  # 只报告
    python scripts/audit_reference.py --fail-on-bad    # 有问题就返回非零（CI 用）

零成本、全离线，不调任何 API。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.metrics import evaluate_constraints  # noqa: E402
from llmeval.schema import Response, Sample  # noqa: E402

DATASETS = ROOT / "datasets"


def iter_samples():
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
                    continue
                if not isinstance(raw, dict) or "id" not in raw:
                    continue
                try:
                    yield rel, lineno, Sample.from_dict(raw, source=rel)
                except Exception:  # noqa: BLE001
                    continue


def main() -> int:
    parser = argparse.ArgumentParser(description="参考答案自洽性检查")
    parser.add_argument("--fail-on-bad", action="store_true", help="有问题时返回非零（CI 用）")
    args = parser.parse_args()

    n_with_constraints = 0
    n_mcq = 0
    problems: list[dict] = []

    for rel, lineno, sample in iter_samples():
        reference = (sample.reference or "").strip()

        if sample.task_type == "mcq":
            n_mcq += 1
            if not re.fullmatch(r"[A-Za-z]", reference):
                problems.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "id": sample.id,
                        "kind": "mcq_gold_not_letter",
                        "detail": f"reference={reference[:40]!r} 不是单个选项字母",
                    }
                )

        if not sample.constraints:
            continue
        n_with_constraints += 1
        if not reference:
            problems.append(
                {
                    "file": rel,
                    "line": lineno,
                    "id": sample.id,
                    "kind": "missing_reference",
                    "detail": "题目带 constraints 但没有 reference，无法校验规则层",
                }
            )
            continue

        response = Response(sample_id=sample.id, sut_id="__reference__", text=reference)
        verdict = evaluate_constraints(sample, response)
        if verdict is None or verdict.passed:
            continue
        detail = verdict.detail or {}
        problems.append(
            {
                "file": rel,
                "line": lineno,
                "id": sample.id,
                "kind": "reference_violates_constraints",
                "detail": json.dumps(
                    {k: v for k, v in detail.items() if k in ("violations", "hard", "soft", "score")},
                    ensure_ascii=False,
                )[:220],
            }
        )

    print("=" * 62)
    print("参考答案自洽性检查")
    print("=" * 62)
    print(f"带约束的题      : {n_with_constraints}")
    print(f"选择题          : {n_mcq}")
    print(f"发现问题        : {len(problems)}")
    print("-" * 62)
    if not problems:
        print("结论：所有参考答案都满足自己出题时的约束，规则层满分是可达的。")
        return 0

    for p in problems[:20]:
        print(f"  [{p['file']}:{p['line']}] {p['id']} · {p['kind']}")
        print(f"      {p['detail']}")
    print("-" * 62)
    print("结论：存在参考答案与约束不一致的题，这些题的满分不可达。")
    return 1 if args.fail_on_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

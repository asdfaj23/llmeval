#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测集构建流水线。

思路借鉴 Arena-Hard 的 BenchBuilder（LMSYS）：
它从约 20 万条真实用户查询里，按 7 个质量指标打分、聚类、每簇取样，
最终挑出 500 道高区分度的题 —— 关键在那 500 道题与人工排名的
一致率能到 89.1%，而 MT-Bench 的模型区分度只有 22.6%。

流程（四步）：

    1. 质量打分   7 个维度：具体性 / 领域知识 / 复杂度 / 问题解决 /
                 创造性 / 技术准确性 / 真实应用价值
    2. 相似度去重 字符 bigram 的 Jaccard 相似度，贪心剔除近重复
    3. 分层抽样   按维度配额挑，避免某个维度把题库占满
    4. 输出       写成标准 JSONL，可直接被套件引用

用法：

    # 离线跑（启发式打分，不调任何模型）
    python scripts\\build_eval_set.py \\
        --inputs datasets/general/*.jsonl --out datasets/generated/curated.jsonl

    # 输出打分明细，用于复查选题是否合理
    ... --report datasets/generated/curated_scores.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.schema import DIMENSIONS, Sample  # noqa: E402
from llmeval.textutil import normalize  # noqa: E402

# 7 个质量指标，权重体现"什么题更有诊断价值"
QUALITY_DIMS: dict[str, float] = {
    "specificity": 0.18,          # 具体性：有没有具体约束、数字、对象
    "domain_knowledge": 0.16,     # 领域知识：是否需要专业知识
    "complexity": 0.16,           # 复杂度：多步、多条件
    "problem_solving": 0.18,      # 问题解决：是否需要推理与推导
    "creativity": 0.08,           # 创造性：是否开放
    "technical_accuracy": 0.12,   # 技术准确性：是否涉及精确技术判断
    "real_world_application": 0.12,  # 真实应用价值
}

_TECH_TERMS = (
    "算法", "复杂度", "并发", "缓存", "索引", "事务", "接口", "协议", "正则",
    "容器", "部署", "监控", "权限", "加密", "序列化", "延迟", "吞吐", "哈希",
    "sql", "api", "json", "http", "python", "git", "cpu", "gpu", "内存",
)
_REASONING_MARKERS = ("为什么", "推导", "证明", "计算", "求", "判断", "解释", "分析", "比较")
_REALWORLD_MARKERS = ("客户", "预算", "工单", "投诉", "会议", "行程", "方案", "决策", "项目", "团队")
_CONSTRAINT_MARKERS = ("不超过", "以内", "必须", "不要", "禁止", "恰好", "只输出", "保留")


def _bigrams(text: str) -> set[str]:
    text = normalize(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def similarity(a: str, b: str) -> float:
    """字符 bigram 的 Jaccard 相似度。够快，且对中文有效。"""
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ------------------------------------------------------------------ 打分
def heuristic_scores(sample: Sample) -> dict[str, float]:
    """启发式质量打分。刻意不调用模型 —— 建集这一步应该能离线复现。"""
    text = f"{sample.prompt} {sample.context or ''}"
    lowered = text.lower()
    n = len(text)

    has_number = bool(re.search(r"\d", text))
    has_constraint = any(m in text for m in _CONSTRAINT_MARKERS)
    has_tech = any(t in lowered for t in _TECH_TERMS)
    has_reason = any(m in text for m in _REASONING_MARKERS)
    has_real = any(m in text for m in _REALWORLD_MARKERS)
    has_reference = bool(sample.reference)
    has_context = bool(sample.context)

    # 具体性：有具体对象、数字、约束
    specificity = 0.3 + 0.25 * has_number + 0.3 * has_constraint + 0.15 * has_reference

    # 领域知识：涉及专业术语或提供了材料
    domain_knowledge = 0.25 + 0.4 * has_tech + 0.2 * has_context + 0.15 * (sample.difficulty == "hard")

    # 复杂度：题目长度 + 多子问题
    sub_questions = text.count("？") + text.count("?") + text.count("分别") + text.count("以及")
    complexity = min(1.0, 0.2 + n / 900 + 0.12 * sub_questions)

    # 问题解决：需要推理或计算
    problem_solving = 0.25 + 0.4 * has_reason + 0.2 * has_number + 0.15 * has_context

    # 创造性：开放式、无唯一答案
    creativity = 0.3 + (0.3 if sample.meta.get("open_ended") else 0.0)
    creativity += 0.3 if sample.task_type in ("coding", "realworld") else 0.1

    # 技术准确性：涉及精确技术判断
    technical_accuracy = 0.2 + 0.5 * has_tech + 0.3 * (sample.dimension in ("coding", "reasoning", "agent"))

    # 真实应用价值
    real_world = 0.2 + 0.45 * has_real + 0.25 * (sample.dimension == "realworld")

    raw = {
        "specificity": specificity,
        "domain_knowledge": domain_knowledge,
        "complexity": complexity,
        "problem_solving": problem_solving,
        "creativity": creativity,
        "technical_accuracy": technical_accuracy,
        "real_world_application": real_world,
    }
    return {k: round(min(1.0, max(0.0, v)), 4) for k, v in raw.items()}


def llm_scores(sample: Sample, client, judge_spec) -> dict[str, float]:
    """让 LLM 按 7 个指标打分。

    与启发式版本的区别：它真的理解题意，代价是要花钱、且需要校准。
    建集阶段建议先用启发式筛一轮，再对候选集用 LLM 精筛。
    """
    from llmeval.textutil import extract_json

    dims_text = "\n".join(f"- {k}：{v:.0%} 权重" for k, v in QUALITY_DIMS.items())
    messages = [
        {
            "role": "system",
            "content": (
                "你是评测集质量评审员。请对给定题目按照下面的指标逐一打分（0 到 1 的小数），"
                "只输出 JSON，不要输出任何其他内容。\n" + dims_text
            ),
        },
        {
            "role": "user",
            "content": f"题目：\n{sample.prompt}\n\n参考答案：\n{sample.reference or '（无）'}",
        },
    ]
    result = client.chat(messages, hint={"kind": "judge", "sample": sample, "answer": sample.prompt})
    payload = extract_json(result.text) or {}
    return {k: float(payload.get(k, 0.5)) for k in QUALITY_DIMS}


def weighted_total(scores: dict[str, float]) -> float:
    return round(sum(scores.get(k, 0.0) * w for k, w in QUALITY_DIMS.items()), 4)


# ------------------------------------------------------------------ 去重
def dedupe(items: list[tuple[Sample, dict[str, float]]], threshold: float = 0.82):
    """贪心去重：按总分从高到低扫，与已保留的题过于相似就丢弃。

    顺序很关键 —— 先扫高分的，所以留下的是"这一簇里最好的那道"。
    """
    kept: list[tuple[Sample, dict[str, float]]] = []
    dropped: list[tuple[str, str, float]] = []

    for sample, scores in sorted(items, key=lambda x: -x[1]["_total"]):
        conflict = None
        for k_sample, _ in kept:
            sim = similarity(sample.prompt, k_sample.prompt)
            if sim >= threshold:
                conflict = (k_sample.id, sim)
                break
        if conflict:
            dropped.append((sample.id, conflict[0], round(conflict[1], 4)))
        else:
            kept.append((sample, scores))
    return kept, dropped


def stratified_select(
    items: list[tuple[Sample, dict[str, float]]],
    per_dimension: int,
    max_total: int = 0,
) -> list[tuple[Sample, dict[str, float]]]:
    """按维度配额挑选，避免某一个维度把题库占满。

    每个维度内部按总分排序取前 N。
    只有显式给了 max_total 时才用其他维度的余额去补足 ——
    否则「每维度 5 题」会被"补到总量"这一步反过来抵消掉。
    """
    buckets: dict[str, list[tuple[Sample, dict[str, float]]]] = defaultdict(list)
    for item in items:
        buckets[item[0].dimension].append(item)

    picked: list[tuple[Sample, dict[str, float]]] = []
    for dim in DIMENSIONS:
        group = sorted(buckets.get(dim, []), key=lambda x: -x[1]["_total"])
        picked.extend(group[:per_dimension])

    if max_total > 0 and len(picked) < max_total:
        picked_ids = {s.id for s, _ in picked}
        rest = sorted(
            (it for it in items if it[0].id not in picked_ids),
            key=lambda x: -x[1]["_total"],
        )
        picked.extend(rest[: max_total - len(picked)])

    return picked[:max_total] if max_total > 0 else picked


# ------------------------------------------------------------------ 主流程
def load_samples(paths: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for pattern in paths:
        p = Path(pattern)
        files = sorted(p.parent.glob(p.name)) if any(c in pattern for c in "*?[") else [p]
        for f in files:
            if not f.exists():
                print(f"  ! 跳过（不存在）：{f}")
                continue
            for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    samples.append(Sample.from_dict(json.loads(line), source=f.name))
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {f.name}:{lineno} 解析失败：{exc}")
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测集构建流水线（打分 → 去重 → 分层抽样）")
    parser.add_argument("--inputs", nargs="+", required=True, help="种子 JSONL，支持通配符")
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    parser.add_argument("--per-dimension", type=int, default=12, help="每个维度最多保留几题")
    parser.add_argument("--max-total", type=int, default=0, help="总题数上限，0 表示不限制")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="去重相似度阈值")
    parser.add_argument("--report", default="", help="把打分明细写成 CSV，便于复查")
    parser.add_argument("--llm", action="store_true", help="用 LLM 打分（需要可用模型），默认用启发式")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("评测集构建流水线")
    print("=" * 68)

    samples = load_samples(args.inputs)
    if not samples:
        print("没有读到任何样本，退出。")
        return 1
    print(f"读入种子题 {len(samples)} 道")

    # ---- 1. 打分
    items: list[tuple[Sample, dict[str, float]]] = []
    if args.llm:
        from llmeval import config as cfg
        from llmeval.client import LLMClient

        cfg.load_env_file()
        models = cfg.load_models()
        judges = models.active_judges()
        if not judges:
            print("没有可用的裁判模型，回退到启发式打分")
            args.llm = False
        else:
            judge = judges[0]
            client = LLMClient(judge, models.runtime)
            print(f"使用 {judge.id} 打分")
            for s in samples:
                items.append((s, llm_scores(s, client, judge)))

    if not args.llm:
        print("使用启发式打分（离线，不调用任何模型）")
        for s in samples:
            items.append((s, heuristic_scores(s)))

    for _, sc in items:
        sc["_total"] = weighted_total(sc)

    avg = mean(sc["_total"] for _, sc in items)
    print(f"平均质量分 {avg:.4f}")

    # ---- 2. 去重
    kept, dropped = dedupe(items, args.similarity_threshold)
    print(f"去重：保留 {len(kept)} 道，剔除近重复 {len(dropped)} 道")
    for sid, dup_of, sim in dropped[:5]:
        print(f"    {sid} ≈ {dup_of}（相似度 {sim}）")

    # ---- 3. 分层抽样
    picked = stratified_select(kept, args.per_dimension, args.max_total)
    pick_counts = Counter(s.dimension for s, _ in picked)
    print("分层抽样后维度分布：")
    for dim, cnt in pick_counts.most_common():
        print(f"    {DIMENSIONS[dim]['name']:<14} {cnt} 题")

    # ---- 4. 输出
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample, _ in sorted(picked, key=lambda x: (x[0].dimension, x[0].id)):
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    print(f"\n已写出 {len(picked)} 道题 → {out_path}")

    if args.report:
        rp = Path(args.report)
        if not rp.is_absolute():
            rp = ROOT / args.report
        rp.parent.mkdir(parents=True, exist_ok=True)
        header = "id,dimension,task_type,difficulty," + ",".join(QUALITY_DIMS) + ",total,selected"
        selected_ids = {s.id for s, _ in picked}
        lines = [header]
        for sample, sc in sorted(items, key=lambda x: -x[1]["_total"]):
            row = [
                sample.id,
                sample.dimension,
                sample.task_type,
                sample.difficulty,
                *[f"{sc.get(k, 0):.4f}" for k in QUALITY_DIMS],
                f"{sc['_total']:.4f}",
                "1" if sample.id in selected_ids else "0",
            ]
            lines.append(",".join(row))
        rp.write_text("\n".join(lines), encoding="utf-8")
        print(f"打分明细 → {rp}")

    print("\n下一步：在 configs/suites/*.yaml 的 datasets 里引用这个文件即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

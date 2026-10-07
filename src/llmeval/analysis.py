# -*- coding: utf-8 -*-
"""
结果分析与归因。

评测报告最容易犯的毛病是：只给一张分数表就结束了。
"3.82 分"没有任何行动价值 —— 你需要知道的是「差在哪、为什么差、下一步补什么数据」。

所以这个模块做三件事：
    1. 聚合   把逐条记录压成模型 × 维度的统计量，并给出置信区间
    2. 归因   把失败记录分类，指出是格式违规、事实错误、工具选择错，还是安全越界
    3. 策略   把归因结果转成可执行的数据建议 —— 这对应 Seed 说的「数据飞轮」：
              badcase → 归因 → 定向补数据 → 再评测
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .calibration import bootstrap_ci
from .metrics.agent_metrics import pass_at_k, pass_hat_k
from .schema import DIMENSIONS, Turn, stable_rand
from .textutil import make_snippet

_JUDGE_PREFIX = "judge:"

# 归因类别 → 面向数据侧的动作建议
_STRATEGY_HINTS: dict[str, dict[str, str]] = {
    "instruction_violation": {
        "label": "指令约束违规",
        "data_action": "定向合成多约束组合样本：否定约束（不要做什么）、格式约束（JSON/表格/行数）、长度上限同时出现，逼模型学会取舍而不是全部堆上。",
    },
    "factuality": {
        "label": "事实性不足",
        "data_action": "补长尾知识的事实核验数据：同一问题提供正例与常见错误说法，要求模型区分「材料支持」与「似是而非」。",
    },
    "tool_misuse": {
        "label": "工具使用不当",
        "data_action": "补工具调用轨迹数据，重点覆盖参数完整性（缺参时应先追问而不是硬调）与工具选择歧义场景（多个工具都能沾边时选哪个）。",
    },
    "safety": {
        "label": "安全边界失守",
        "data_action": "补提示注入与越权样本：把违规指令藏在文档、表格、工具返回值里，测模型是否会被间接指令带偏。",
    },
    "format": {
        "label": "输出格式不合规",
        "data_action": "把结构化输出作为硬约束写进训练与评测：JSON schema 校验、禁止包裹解释性文字、字段类型严格匹配。",
    },
    "quality": {
        "label": "内容质量偏低",
        "data_action": "按失败子维度（正确性/完整性/相关性/清晰度）分别取样，先定位是哪一维拖后腿，再决定补哪类数据。",
    },
    "no_answer": {
        "label": "未产出有效回答",
        "data_action": "优先排查调用链路（超时、限流、解析失败），这类问题不是模型能力问题，混进能力统计会污染结论。",
    },
}


@dataclass
class ModelStats:
    """单个被测模型的聚合结果。"""

    sut_id: str
    label: str = ""
    is_mock: bool = False
    n: int = 0
    n_error: int = 0
    mean_score: float | None = None
    score_ci: tuple[float | None, float | None] = (None, None)
    pass_rate: float | None = None
    by_dimension: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_metric: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    attribution: dict[str, int] = field(default_factory=dict)
    reliability: dict[str, Any] | None = None
    avg_latency_s: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sut_id": self.sut_id,
            "label": self.label,
            "is_mock": self.is_mock,
            "n": self.n,
            "n_error": self.n_error,
            "mean_score": self.mean_score,
            "score_ci": list(self.score_ci),
            "pass_rate": self.pass_rate,
            "by_dimension": self.by_dimension,
            "by_metric": self.by_metric,
            "attribution": self.attribution,
            "reliability": self.reliability,
            "avg_latency_s": self.avg_latency_s,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
        }


# ------------------------------------------------------------------ 单条口径
def final_score(turn: Turn) -> float | None:
    """一条记录的综合分。

    两个口径，都刻意写死：

    1. 同一个指标下有多个裁判（评审团）时，**先取中位数**，再跨指标平均。
       用中位数而不是均值：评审团的意义在于抗离群 ——
       一个裁判抽风打了 1 分，均值会被直接拉下去，中位数不会。

    2. 有裁判分就用裁判分，没有才退回确定性判定的均值。
       两条路径不混着平均 —— 把规则分和裁判分平均，
       会让「格式合规」这种确定性结论稀释掉质量信号。
    """
    by_metric: dict[str, list[float]] = {}
    for v in turn.verdicts:
        if v.tier == "llm_judge" and v.score is not None:
            by_metric.setdefault(v.metric, []).append(float(v.score))
    if by_metric:
        per_metric = [median(vals) for vals in by_metric.values()]
        return sum(per_metric) / len(per_metric)

    # 只认 decisive 的确定性判定。开放式回答的字面重合分只是参考信号（decisive=False），
    # 拿它当综合分，会把「换个说法表达同一个意思」判成低分 ——
    # 实测开放题维度均分被压到 1.8 上下，那个数不代表模型能力。
    det_scores = [v.score for v in turn.verdicts if v.score is not None and v.decisive]
    if det_scores:
        return sum(det_scores) / len(det_scores)
    return None


def is_pass(turn: Turn) -> bool | None:
    """严格口径：所有**有结论的判定**都通过，才算这条通过。

    两个限定条件都是有意加的：

    1. 只看 decisive 的判定。像开放式回答的 bigram 比对这种参考性指标，
       不该决定一条记录过没过 —— 它会因为"意思对但换了个说法"而误判失败。
    2. 一处不合规就是不合规。评测不该给「大部分不错」发安慰奖。

    口径的准确性直接决定报告里那个通过率有没有意义。
    """
    flags = [v.passed for v in turn.verdicts if v.passed is not None and v.decisive]
    if not flags:
        return None
    return all(flags)


# ------------------------------------------------------------------ 归因
def attribute_failure(turn: Turn) -> str:
    """给一条失败记录归类。顺序即优先级：越靠前的越先被认定。"""
    if not turn.response.ok:
        return "no_answer"

    by_metric = {v.metric: v for v in turn.verdicts}

    rules = by_metric.get("instruction_rules")
    if rules and not rules.passed:
        return "instruction_violation"

    safety = by_metric.get("safety_heuristic") or by_metric.get("agent_safety")
    if safety and not safety.passed:
        return "safety"

    tools = by_metric.get("agent_tools")
    if tools and not tools.passed:
        return "tool_misuse"

    faith = by_metric.get(f"{_JUDGE_PREFIX}faithfulness")
    if faith and not faith.passed:
        return "factuality"

    for metric, v in by_metric.items():
        if metric.startswith(_JUDGE_PREFIX) and not v.passed:
            detail = v.detail or {}
            if "json" in str(detail).lower() or "format" in str(detail).lower():
                return "format"
            return "quality"

    return "quality"


# ------------------------------------------------------------------ 聚合
def summarize_model(
    turns: list[Turn],
    sut_id: str,
    label: str = "",
    is_mock: bool = False,
    max_failures: int = 12,
) -> ModelStats:
    mine = [t for t in turns if t.response.sut_id == sut_id]
    stats = ModelStats(sut_id=sut_id, label=label, is_mock=is_mock, n=len(mine))
    if not mine:
        return stats

    stats.n_error = sum(1 for t in mine if not t.response.ok)

    scores = [s for s in (final_score(t) for t in mine) if s is not None]
    if scores:
        stats.mean_score = round(sum(scores) / len(scores), 4)
        stats.score_ci = bootstrap_ci(scores, stat="mean", seed=sum(ord(c) for c in sut_id))

    flags = [f for f in (is_pass(t) for t in mine) if f is not None]
    if flags:
        stats.pass_rate = round(sum(1 for f in flags if f) / len(flags), 4)

    stats.by_dimension = _group_stats(mine, lambda t: t.sample.dimension)
    stats.by_metric = _metric_stats(mine)

    latencies = [t.response.latency_s for t in mine if t.response.latency_s]
    stats.avg_latency_s = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    stats.total_cost_usd = round(sum(t.response.usage.cost_usd for t in mine), 6)
    stats.total_tokens = sum(t.response.usage.total_tokens for t in mine)

    # 失败样本与归因
    failed = [t for t in mine if is_pass(t) is False]
    stats.attribution = dict(Counter(attribute_failure(t) for t in failed))
    for t in sorted(failed, key=lambda x: (final_score(x) if final_score(x) is not None else 0))[:max_failures]:
        stats.failures.append(
            {
                "sample_id": t.sample.id,
                "dimension": t.sample.dimension,
                "dimension_name": t.sample.dimension_name,
                "task_type": t.sample.task_type,
                "score": final_score(t),
                "category": attribute_failure(t),
                "category_label": _STRATEGY_HINTS.get(attribute_failure(t), {}).get("label", "其他"),
                "prompt": make_snippet(t.sample.prompt, 140),
                "answer": make_snippet(t.response.text, 220),
                "reference": make_snippet(t.sample.reference or "", 140),
                "reasons": [
                    {
                        "metric": v.metric,
                        "score": v.score,
                        "passed": v.passed,
                        "detail": _trim_detail(v.detail),
                    }
                    for v in t.verdicts
                    if v.passed is False
                ],
            }
        )

    stats.reliability = _reliability(mine)
    return stats


def _group_stats(turns: list[Turn], key) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        buckets[key(t)].append(t)

    out: dict[str, dict[str, Any]] = {}
    for name, group in buckets.items():
        scores = [s for s in (final_score(t) for t in group) if s is not None]
        flags = [f for f in (is_pass(t) for t in group) if f is not None]
        lo, hi = bootstrap_ci(scores, stat="mean", seed=sum(ord(c) for c in name))
        out[name] = {
            "name": DIMENSIONS.get(name, {}).get("name", name),
            "n": len(group),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "ci": [lo, hi],
            "pass_rate": round(sum(1 for f in flags if f) / len(flags), 4) if flags else None,
        }
    return out


def _metric_stats(turns: list[Turn]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    pass_buckets: dict[str, list[bool]] = defaultdict(list)
    for t in turns:
        for v in t.verdicts:
            if v.score is not None:
                buckets[v.metric].append(float(v.score))
            if v.passed is not None:
                pass_buckets[v.metric].append(bool(v.passed))

    out: dict[str, dict[str, Any]] = {}
    for metric, vals in buckets.items():
        flags = pass_buckets.get(metric) or []
        lo, hi = bootstrap_ci(vals, stat="mean", seed=sum(ord(c) for c in metric))
        out[metric] = {
            "n": len(vals),
            "mean_score": round(sum(vals) / len(vals), 4),
            "ci": [lo, hi],
            "pass_rate": round(sum(1 for f in flags if f) / len(flags), 4) if flags else None,
        }
    return out


def _reliability(turns: list[Turn]) -> dict[str, Any] | None:
    """多次重复跑同一题时，给出 pass@k / pass^k。"""
    groups: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        groups[t.sample.id].append(t)
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeated:
        return None

    per_sample: dict[str, list[bool]] = {}
    for sid, group in repeated.items():
        flags = [is_pass(t) for t in group]
        flags = [bool(f) for f in flags if f is not None]
        if flags:
            per_sample[sid] = flags

    if not per_sample:
        return None
    k = min(len(v) for v in per_sample.values())
    n = len(per_sample)
    all_pass = sum(1 for f in per_sample.values() if all(f))
    any_pass = sum(1 for f in per_sample.values() if any(f))
    avg = sum(sum(f) / len(f) for f in per_sample.values()) / n
    flaky = [sid for sid, f in per_sample.items() if 0 < sum(f) < len(f)]

    return {
        "k": k,
        "n_samples": n,
        "avg_success_rate": round(avg, 4),
        "pass_at_k": round(any_pass / n, 4),
        "pass_hat_k": round(all_pass / n, 4),
        "flaky_count": len(flaky),
        "flaky_examples": flaky[:5],
        "note": (
            "pass@k 衡量能力上限（跑 k 次至少成功一次），pass^k 衡量可靠性（k 次全成功）。"
            "两者差距越大，说明模型越『不稳定』—— 这类模型上线风险最高，"
            "优先要解决的是稳定性而不是把平均分再抬高一点。"
        ),
    }


# ------------------------------------------------------------------ 数据策略
def build_strategy(stats_list: list[ModelStats]) -> list[dict[str, Any]]:
    """把归因结果翻译成可执行的数据策略。这一步是整个闭环的出口。"""
    if not stats_list:
        return []

    overall_attr: Counter[str] = Counter()
    for s in stats_list:
        overall_attr.update(s.attribution)

    total = sum(overall_attr.values()) or 1
    items: list[dict[str, Any]] = []
    for category, count in overall_attr.most_common():
        hint = _STRATEGY_HINTS.get(category, {"label": category, "data_action": "补充该类失败场景的针对性样本。",})
        items.append(
            {
                "category": category,
                "label": hint["label"],
                "count": count,
                "share": round(count / total, 4),
                "data_action": hint["data_action"],
            }
        )

    # 维度短板：均分最低的维度优先补
    dim_scores: dict[str, list[float]] = defaultdict(list)
    for s in stats_list:
        for dim, info in s.by_dimension.items():
            if info.get("mean_score") is not None:
                dim_scores[dim].append(info["mean_score"])

    weakest = sorted(
        ((dim, sum(v) / len(v)) for dim, v in dim_scores.items() if v),
        key=lambda x: x[1],
    )[:3]

    return [
        {
            "kind": "failure_attribution",
            "items": items,
        },
        {
            "kind": "dimension_priority",
            "items": [
                {
                    "dimension": dim,
                    "name": DIMENSIONS.get(dim, {}).get("name", dim),
                    "avg_score": round(avg, 4),
                    "seed_anchor": DIMENSIONS.get(dim, {}).get("seed_anchor", ""),
                }
                for dim, avg in weakest
            ],
        },
    ]


def _trim_detail(detail: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    """详情可能很长，报告里只留关键字段，避免 JSON 膨胀。"""
    if not isinstance(detail, dict):
        return {}
    keep = {}
    for k, v in detail.items():
        text = str(v)
        keep[k] = v if len(text) <= limit else text[:limit] + "…"
    return keep


def dimension_matrix(stats_list: list[ModelStats]) -> dict[str, Any]:
    """模型 × 维度 的矩阵数据，给报告画雷达图和热力表。"""
    dims = [d for d in DIMENSIONS if any(d in s.by_dimension for s in stats_list)]
    rows = []
    for s in stats_list:
        rows.append(
            {
                "sut_id": s.sut_id,
                "label": s.label or s.sut_id,
                "values": {
                    d: (s.by_dimension.get(d) or {}).get("mean_score") for d in dims
                },
            }
        )
    return {
        "dimensions": [
            {"key": d, "name": DIMENSIONS[d]["name"], "seed_anchor": DIMENSIONS[d]["seed_anchor"]}
            for d in dims
        ],
        "rows": rows,
    }


def win_matrix(pairwise: list[dict[str, Any]], sut_ids: list[str]) -> dict[str, Any]:
    """由 pairwise 记录汇总出胜率矩阵。"""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    position_records = []

    for rec in pairwise:
        a, b, w = rec.get("sut_a"), rec.get("sut_b"), rec.get("winner")
        if not a or not b or w == "invalid":
            continue
        position_records.append(rec)
        if w == a:
            cells[(a, b)].append(1.0)
            cells[(b, a)].append(0.0)
        elif w == b:
            cells[(a, b)].append(0.0)
            cells[(b, a)].append(1.0)
        else:
            cells[(a, b)].append(0.5)
            cells[(b, a)].append(0.5)

    matrix = []
    for a in sut_ids:
        row = []
        for b in sut_ids:
            if a == b:
                row.append(None)
                continue
            vals = cells.get((a, b))
            row.append(round(sum(vals) / len(vals), 4) if vals else None)
        matrix.append({"sut_id": a, "values": row})

    return {"sut_ids": sut_ids, "rows": matrix, "n_pairs": len(position_records)}

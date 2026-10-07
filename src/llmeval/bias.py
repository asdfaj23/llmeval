# -*- coding: utf-8 -*-
"""
偏差控制与度量。

裁判不是中立仪器，它有系统性偏好。已知且必须处理的四类：

    位置偏差    偏向先出现的那份回答
    冗长偏差    篇幅越长分越高，哪怕长出来的都是废话
    自偏好      给自己模型族的回答打高分
    风格偏差    偏爱自信、结构化、排版漂亮的回答，哪怕内容是错的

本项目的处理态度是：这些偏差不可能被"消除"，只能被「度量 → 缓解 → 如实披露」。
所以这个模块的重点是**度量**。
报告会把度量结果原样列出，而不是假装评测结果没有偏差 ——
一个不肯公布自己裁判一致率的评测报告，它的分数没有意义。
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable


def position_bias_report(pairwise_records: list[dict[str, Any]]) -> dict[str, Any]:
    """位置偏差的量化：换位后结论仍然一致的比例。

    一致性越高，说明结论越不受"谁先出现"影响。
    如果一致性只有五六成，那这个 pairwise 结果基本等于抛硬币，不能用来排名。
    """
    decided = [r for r in pairwise_records if r.get("winner") not in (None, "invalid")]
    if not decided:
        return {"n_pairs": 0, "note": "没有可用的 pairwise 记录"}

    consistent = sum(1 for r in decided if r.get("consistent"))
    rate = consistent / len(decided)

    if rate >= 0.9:
        verdict = "位置偏差很小，pairwise 结论可用"
    elif rate >= 0.75:
        verdict = "存在一定位置敏感性，建议结合 pointwise 结果一起看"
    else:
        verdict = "位置偏差显著，单独的 pairwise 结论不可信，需扩大样本或换裁判"

    return {
        "n_pairs": len(decided),
        "position_consistent": consistent,
        "position_consistency_rate": round(rate, 4),
        "verdict": verdict,
    }


def length_bias_report(pairwise_records: list[dict[str, Any]]) -> dict[str, Any]:
    """冗长偏差的量化：赢的一方是不是系统性地更长。

    如果"更长的回答获胜率"明显高于 50%，说明裁判在被篇幅牵着走。
    """
    decided = [
        r
        for r in pairwise_records
        if r.get("winner") not in (None, "invalid", "tie") and r.get("sut_a") and r.get("sut_b")
    ]
    if not decided:
        return {"n_decided": 0, "note": "没有决出胜负的 pairwise 记录"}

    longer_wins = 0
    diffs: list[int] = []
    for r in decided:
        len_a, len_b = int(r.get("len_a", 0)), int(r.get("len_b", 0))
        diffs.append(len_a - len_b)
        if len_a == len_b:
            continue
        winner_is_a = r["winner"] == r["sut_a"]
        longer_is_a = len_a > len_b
        if winner_is_a == longer_is_a:
            longer_wins += 1

    n_diff = sum(1 for d in diffs if d != 0)
    rate = longer_wins / n_diff if n_diff else 0.0

    if rate <= 0.55:
        verdict = "未观察到明显的冗长偏好"
    elif rate <= 0.65:
        verdict = "存在轻度冗长偏好，建议对长度做控制后再比较"
    else:
        verdict = "冗长偏好明显，直接比较长度差异大的回答会失真，应做长度去偏"

    return {
        "n_decided": len(decided),
        "n_length_diff": n_diff,
        "longer_wins": longer_wins,
        "longer_win_rate": round(rate, 4),
        "avg_len_gap": round(sum(diffs) / len(diffs), 1) if diffs else 0.0,
        "verdict": verdict,
    }


def judge_panel(scores: Iterable[float | None], rule: str = "median") -> float | None:
    """裁判委员会：用多个裁判的聚合结果替代单个裁判。

    单个大裁判的偏差是稳定的；几个不同家族的小裁判投票，
    各自的家族偏好会互相抵消 —— 这是"用陪审团代替权威"的思路。
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    if rule == "mean":
        return round(sum(valid) / len(valid), 4)
    if rule == "min":       # 保守口径：取最严格的裁判
        return float(min(valid))
    return float(median(valid))


def summarize_bias(
    pairwise_records: list[dict[str, Any]] | None = None,
    cross_family_warnings: list[str] | None = None,
    panel_used: bool = False,
    length_controlled: bool = False,
) -> dict[str, Any]:
    """把各路偏差度量汇总成报告附录里的一节。"""
    pairwise_records = pairwise_records or []
    return {
        "position": position_bias_report(pairwise_records),
        "length": length_bias_report(pairwise_records),
        "self_preference": {
            "cross_family_warnings": cross_family_warnings or [],
            "status": "已告警" if cross_family_warnings else "未发现同族风险",
        },
        "applied_controls": {
            "position_swap": bool(pairwise_records),
            "judge_panel": panel_used,
            "length_control": length_controlled,
        },
        "disclosure": (
            "以上为自动化度量结果。裁判与人工标注的一致性（Cohen's κ）"
            "单独见「裁判可信度」一节 —— 该项需要人工标注样本才能计算，"
            "未提供人工标注时不给出数值，也不用其他指标顶替。"
        ),
    }

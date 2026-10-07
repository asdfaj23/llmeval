# -*- coding: utf-8 -*-
"""
模型对比的统计检验。

存在的理由只有一条：**两个模型的分数差，可能纯粹是噪声。**

NeurIPS 2025 的《Measuring what Matters: Construct Validity in LLM Benchmarks》
系统审阅了 445 篇 benchmark 论文，发现只有 16% 在比较模型时用了统计检验或
不确定度估计 —— 绝大多数只报一个单点准确率。这意味着大量「A 比 B 强 2 个百分点」
的结论，其实样本量根本撑不起。

所以这里做两件事：

    mcnemar_test        配对通过/不通过 → 通过率差异是否显著
    bootstrap_diff_ci   配对分数差 → 差值的置信区间

两个都是**配对**检验：同一道题分别给两个模型做，比较它们在同一批题上的表现。
配对比独立两样本检验敏感得多，因为它消掉了题目难度带来的方差 ——
难题两个都错、简单题两个都对，这些对「谁更强」没有信息量。

不引入 scipy：正态近似的 p 值用 math.erf 手算，bootstrap 用固定种子的纯 Python 实现。
评测结果必须可复现，所以随机数种子写死。
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

DEFAULT_BOOTSTRAP = 4000
DEFAULT_SEED = 20260914


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mcnemar_test(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """McNemar 检验：同一批题上，两个模型的通过率差异是否显著。

    pairs 里每项是 (模型A是否通过, 模型B是否通过)。
    只看两者结论不一致的题：
        b = A 过 B 不过
        c = A 不过 B 过
    两者都过 / 都不过的题不含信息，直接丢弃 —— 这是配对检验的关键。
    """
    b = sum(1 for a, bb in pairs if a and not bb)
    c = sum(1 for a, bb in pairs if not a and bb)
    n_disc = b + c

    if n_disc == 0:
        return {
            "b": 0,
            "c": 0,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note": "两个模型在所有题上结论完全一致，无法区分",
        }

    # 连续性校正的 McNemar：不校正在小样本下会高估显著性
    stat = (abs(b - c) - 1) ** 2 / n_disc
    z = (abs(b - c) - 1) / math.sqrt(n_disc) if n_disc else 0.0
    p = 2 * (1 - _normal_cdf(abs(z)))
    p = max(0.0, min(1.0, p))

    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "statistic": round(stat, 4),
        "z": round(z, 4),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "note": "",
    }


def bootstrap_diff_ci(
    diffs: Sequence[float],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """配对分数差的 bootstrap 置信区间。

    diffs 是「同一道题上 A 的得分 - B 的得分」。对这批差值做有放回重采样，
    看均值差的分布里是否跨过 0 —— 跨过 0 就意味着「谁更强」说不准。
    """
    vals = [d for d in diffs if d is not None]
    if not vals:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "significant": False, "note": "没有可用的配对样本"}

    n = len(vals)
    observed = sum(vals) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()

    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = min(n_boot - 1, int(n_boot * (1 - alpha / 2)))
    lo, hi = means[lo_idx], means[hi_idx]

    return {
        "n": n,
        "mean": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "significant": not (lo <= 0 <= hi),
        "note": "",
    }


def compare_models(
    per_sample: dict[str, tuple[bool | None, float | None, float | None]],
    a: str,
    b: str,
) -> dict[str, Any]:
    """把一个模型的对比结果汇总成可直接进报告的字典。

    per_sample: {题目 id: (是否通过, A 的分数, B 的分数)}
    传入的是「同一个模型自己」的那份数据，通过率与分数都从里面取。
    """
    pairs: list[tuple[bool, bool]] = []
    diffs: list[float] = []
    wins_a = wins_b = ties = 0

    for _sid, (passed, sa, sb) in per_sample.items():
        if passed is not None:
            pairs.append((passed, passed))  # 占位，真正配对在外层构造
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            else:
                ties += 1

    return {
        "a": a,
        "b": b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "score_diff": bootstrap_diff_ci(diffs),
        "pass_diff": mcnemar_test(pairs) if pairs else {},
    }


def significance_table(
    turns: Sequence[Any],
    is_pass_fn,
    final_score_fn,
    suts: Sequence[str],
) -> list[dict[str, Any]]:
    """算出所有模型两两之间的显著性对比。

    turns 是全部记录；两个取分函数由调用方注入，避免这里反向依赖 analysis 模块
    （analysis 依赖 schema，stats 不该依赖 analysis，否则会出现循环导入）。
    """
    # {题目 id: {模型: (是否通过, 得分)}}
    by_sample: dict[str, dict[str, tuple[bool | None, float | None]]] = {}
    for turn in turns:
        entry = by_sample.setdefault(turn.sample.id, {})
        entry[turn.response.sut_id] = (is_pass_fn(turn), final_score_fn(turn))

    out: list[dict[str, Any]] = []
    for i in range(len(suts)):
        for j in range(i + 1, len(suts)):
            a, b = suts[i], suts[j]
            pairs: list[tuple[bool, bool]] = []
            diffs: list[float] = []
            wins_a = wins_b = ties = 0

            for _sid, models in by_sample.items():
                if a not in models or b not in models:
                    continue
                pa, sa = models[a]
                pb, sb = models[b]
                if pa is not None and pb is not None:
                    pairs.append((pa, pb))
                if sa is not None and sb is not None:
                    diffs.append(sa - sb)
                    if sa > sb:
                        wins_a += 1
                    elif sb > sa:
                        wins_b += 1
                    else:
                        ties += 1

            out.append(
                {
                    "a": a,
                    "b": b,
                    "n_paired": len(pairs),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "pass_test": mcnemar_test(pairs) if pairs else {},
                    "score_diff": bootstrap_diff_ci(diffs),
                }
            )
    return out

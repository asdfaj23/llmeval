# -*- coding: utf-8 -*-
"""
裁判可信度校准。

这个模块回答一个绕不开的问题：
    「你的 LLM 裁判打出来的分，凭什么可以信？」

答案是：不能凭感觉信，得拿人工标注去量。
流程是标准的元评测（meta-evaluation）：

    采一批样本 → 人工按同一张评分表打分 → 裁判打同一批 → 算一致性

一致性用 Cohen's κ 而不是原始一致率，因为原始一致率会被"多数类"灌水：
两个都爱打 5 分的裁判，一致率天然很高，但毫无信息量。
κ 扣掉了随机一致的那部分，所以它才是能拿去决策的数。

参考区间（Landis & Koch 1977，是惯例不是物理常数）：
    < 0.00  无一致性      0.41-0.60  中等
    0.00-0.20  极弱       0.61-0.80  强        ← 生产警戒线通常取 0.6
    0.21-0.40  一般       0.81-1.00  几乎完全一致

另一条实践共识：裁判与专家的一致率达到 85%-90%，才适合用来做发布门禁。
达不到就必须说明"分数只作参考"，不能拿它当上线依据。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

# 生产警戒线：低于这个值，裁判分数不该直接用于决策
KAPPA_GATE = 0.6
KAPPA_STRONG = 0.8


# ------------------------------------------------------------------ 一致性
def kappa_band(kappa: float | None) -> str:
    if kappa is None:
        return "样本不足，无法评估"
    if kappa < 0:
        return "低于随机水平"
    if kappa < 0.21:
        return "极弱"
    if kappa < 0.41:
        return "一般"
    if kappa < 0.61:
        return "中等"
    if kappa < 0.81:
        return "强"
    return "几乎完全一致"


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """未加权 Cohen's κ，用于离散标签（如 pass/fail）。"""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None

    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs}, key=str)
    idx = {lab: i for i, lab in enumerate(labels)}
    size = len(labels)

    observed = [[0] * size for _ in range(size)]
    for x, y in pairs:
        observed[idx[x]][idx[y]] += 1

    row_tot = [sum(row) for row in observed]
    col_tot = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    expected = sum(row_tot[i] * col_tot[i] for i in range(size)) / n

    agree = sum(observed[i][i] for i in range(size))
    if n == expected:
        return 1.0
    return round((agree - expected) / (n - expected), 4)


def quadratic_weighted_kappa(
    a: Sequence[float], b: Sequence[float], min_rating: int = 1, max_rating: int = 5
) -> float | None:
    """二次加权 κ，用于 1-5 这种有序评分。

    有序评分里"把 3 打成 5"和"把 3 打成 4"的错误程度不同，
    未加权 κ 会把两者一视同仁，加权 κ 才是这类数据的正确度量。
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None

    size = max_rating - min_rating + 1
    observed = [[0.0] * size for _ in range(size)]
    for x, y in pairs:
        xi = int(round(x)) - min_rating
        yi = int(round(y)) - min_rating
        if 0 <= xi < size and 0 <= yi < size:
            observed[xi][yi] += 1

    hist_a = [sum(observed[i]) for i in range(size)]
    hist_b = [sum(observed[i][j] for i in range(size)) for j in range(size)]

    span = (max_rating - min_rating) ** 2 or 1
    num = den = 0.0
    for i in range(size):
        for j in range(size):
            w = ((i - j) ** 2) / span
            num += w * observed[i][j]
            den += w * (hist_a[i] * hist_b[j] / n)

    if den == 0:
        return 1.0
    return round(1 - num / den, 4)


# ------------------------------------------------------------------ 相关性
def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return round(num / den, 4) if den else None


# ------------------------------------------------------------------ 区间估计
def bootstrap_ci(
    values: Sequence[float],
    stat: str = "mean",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260914,
) -> tuple[float | None, float | None]:
    """自助法置信区间。

    为什么必须给区间：两个模型平均分 3.82 和 3.79，如果没有区间，
    你会以为前者更好；有了区间你可能会发现两者完全重叠 —— 那这个差距就不存在。
    """
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return None, None

    rnd = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rnd.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n if stat == "mean" else sum(1 for v in sample if v) / n)
    stats.sort()
    lo = stats[max(0, int(n_boot * alpha / 2))]
    hi = stats[min(n_boot - 1, int(n_boot * (1 - alpha / 2)) - 1)]
    return round(lo, 4), round(hi, 4)


# ------------------------------------------------------------------ 校准入口
@dataclass
class CalibrationReport:
    n: int
    agreement_rate: float | None
    cohen_kappa: float | None
    weighted_kappa: float | None
    spearman: float | None
    mae: float | None
    band: str
    ready_for_gate: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement_rate": self.agreement_rate,
            "cohen_kappa": self.cohen_kappa,
            "quadratic_weighted_kappa": self.weighted_kappa,
            "spearman": self.spearman,
            "mae": self.mae,
            "kappa_band": self.band,
            "ready_for_gate": self.ready_for_gate,
            "note": self.note,
        }


def calibrate(
    judge_scores: Sequence[float | None],
    human_scores: Sequence[float | None],
) -> CalibrationReport:
    """把裁判分与人工分对齐，算出一致性全套指标。"""
    pairs = [
        (float(j), float(h))
        for j, h in zip(judge_scores, human_scores)
        if j is not None and h is not None
    ]
    n = len(pairs)
    if n < 10:
        return CalibrationReport(
            n=n,
            agreement_rate=None,
            cohen_kappa=None,
            weighted_kappa=None,
            spearman=None,
            mae=None,
            band="样本不足，无法评估",
            ready_for_gate=False,
            note=(
                f"仅有 {n} 条人工标注，样本量不足以给出可信的一致性结论。"
                "建议至少 100 条；样本太少时算出来的 κ 波动极大，容易误导决策。"
            ),
        )

    js = [round(p[0]) for p in pairs]
    hs = [round(p[1]) for p in pairs]
    exact = sum(1 for x, y in zip(js, hs) if x == y) / n
    close = sum(1 for x, y in zip(js, hs) if abs(x - y) <= 1) / n

    ck = cohen_kappa(js, hs)
    qwk = quadratic_weighted_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    mae = round(sum(abs(p[0] - p[1]) for p in pairs) / n, 4)

    kappa_for_gate = qwk if qwk is not None else ck
    ready = bool(kappa_for_gate is not None and kappa_for_gate >= KAPPA_GATE)

    if ready and kappa_for_gate is not None and kappa_for_gate >= KAPPA_STRONG:
        note = "裁判与人工标注高度一致，可用于批量打分，仍建议定期抽检。"
    elif ready:
        note = "裁判与人工标注一致性达标，可用于批量打分，但需保留人工抽检。"
    else:
        note = (
            "裁判与人工标注一致性未达 0.6 警戒线，此时的裁判分数只能作为参考，"
            "不应直接用于发布门禁或能力结论。"
        )

    return CalibrationReport(
        n=n,
        agreement_rate=round(exact, 4),
        cohen_kappa=ck,
        weighted_kappa=qwk,
        spearman=rho,
        mae=mae,
        band=f"{kappa_band(kappa_for_gate)}（相邻 1 分内 {close:.1%}）",
        ready_for_gate=ready,
        note=note,
    )


# ------------------------------------------------------------------ 人工标注
def load_gold(path: str | Path) -> list[dict[str, Any]]:
    """读人工标注文件。每行一个 JSON：{"sample_id","sut_id","score","notes"}。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"人工标注文件不存在：{p}")
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{p.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return out


def extract_judge_scores(
    turns: Iterable[Any], metric: str | None = None
) -> dict[tuple[str, str], float]:
    """从评测记录里抽出裁判分数，键为 (sample_id, sut_id)。"""
    scores: dict[tuple[str, str], float] = {}
    for turn in turns:
        for v in getattr(turn, "verdicts", []):
            if v.tier != "llm_judge" or v.score is None:
                continue
            if metric and v.metric != metric:
                continue
            scores[(turn.sample.id, turn.response.sut_id)] = float(v.score)
    return scores


def run_calibration(turns: Iterable[Any], gold_path: str | Path) -> CalibrationReport:
    """用人工标注校准裁判。匹配不上的标注会被明确报出来，不静默丢弃。"""
    gold = load_gold(gold_path)
    judge_scores = extract_judge_scores(turns)

    js: list[float | None] = []
    hs: list[float | None] = []
    matched, missing = 0, 0
    for item in gold:
        key = (str(item.get("sample_id")), str(item.get("sut_id")))
        j = judge_scores.get(key)
        h = item.get("score")
        if j is None or h is None:
            missing += 1
            js.append(None)
            hs.append(h if h is None else float(h))
            continue
        matched += 1
        js.append(j)
        hs.append(float(h))

    report = calibrate(js, hs)
    if missing:
        report.note += f"（另有 {missing} 条标注未能与评测结果匹配，已计入分母但未参与计算）"
        setattr(report, "unmatched", missing)
    return report


# ------------------------------------------------------------------ 评审团一致性
def inter_judge_agreement(
    per_judge_scores: dict[str, list[float | None]],
) -> dict[str, Any]:
    """评审团内部一致性 —— 只在「多裁判」时才谈得上的一个量。

    它回答的是：这几位裁判到底在各自独立判断，还是在复读同一个结论？

        κ 接近 1    评审团没有增加信息量，只是把同一份分数算了几遍
        κ 中等偏高  健康的区间：方向一致，但各自留有独立判断
        κ 偏低      评分表本身有歧义，该先回去改标准，而不是再加裁判

    参数映射里的列表必须等长且顺序对齐 —— 顺序错了算出来的是一堆无意义的数。
    """
    judges = list(per_judge_scores)
    if len(judges) < 2:
        return {
            "n_judges": len(judges),
            "judge_ids": judges,
            "note": "只有 1 个裁判，不构成评审团。配置 configs/suites 的 judges 列表可启用多裁判。",
        }

    lengths = {len(v) for v in per_judge_scores.values()}
    if len(lengths) != 1:
        raise ValueError("各裁判的分数序列长度必须一致且顺序对齐")

    n = lengths.pop()
    valid = [
        i for i in range(n) if all(per_judge_scores[j][i] is not None for j in judges)
    ]
    if not valid:
        return {
            "n_judges": len(judges),
            "judge_ids": judges,
            "n_records": 0,
            "note": "没有任何一条记录是所有裁判都成功打分的，无法评估一致性。",
        }

    ranges = []
    for i in valid:
        vals = [float(per_judge_scores[j][i]) for j in judges]
        ranges.append(max(vals) - min(vals))

    exact = near = total = 0
    kappas: list[float] = []
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            sa = [float(per_judge_scores[judges[a]][i]) for i in valid]
            sb = [float(per_judge_scores[judges[b]][i]) for i in valid]
            total += len(sa)
            exact += sum(1 for x, y in zip(sa, sb) if round(x) == round(y))
            near += sum(1 for x, y in zip(sa, sb) if abs(x - y) <= 1)
            k = quadratic_weighted_kappa(sa, sb)
            if k is not None:
                kappas.append(k)

    mean_kappa = round(sum(kappas) / len(kappas), 4) if kappas else None
    mean_range = round(sum(ranges) / len(ranges), 4)

    if mean_kappa is None:
        verdict = "样本不足，无法评估评审团一致性。"
    elif mean_kappa >= 0.8:
        verdict = (
            "评审团高度一致。注意这时要反过来检查：如果几个裁判几乎从不分歧，"
            "说明它们可能高度同质，评审团并没有比单裁判多提供信息。"
        )
    elif mean_kappa >= 0.5:
        verdict = "评审团一致性适中，多裁判起到了互相校正的作用，属于健康区间。"
    else:
        verdict = (
            "评审团一致性偏低。这通常不是裁判的错，而是评分表本身有歧义 —— "
            "应先回去把 criterion 与锚点写清楚，而不是再加裁判。"
        )

    return {
        "n_judges": len(judges),
        "judge_ids": judges,
        "n_records": len(valid),
        "mean_range": mean_range,
        "max_range": round(max(ranges), 4) if ranges else None,
        "pairwise_exact_rate": round(exact / total, 4) if total else None,
        "pairwise_near_rate": round(near / total, 4) if total else None,
        "pairwise_weighted_kappa": mean_kappa,
        "kappa_band": kappa_band(mean_kappa),
        "verdict": verdict,
    }


def collect_judge_scores_by_judge(
    turns: Iterable[Any], metric: str | None = None
) -> dict[str, list[float | None]]:
    """把逐条记录整理成 {judge_id: [分数]}，顺序按记录出现次序对齐。

    这是 inter_judge_agreement 的输入格式。对齐由本函数统一保证，
    调用方不需要自己拼装，避免顺序错位这种很难发现的 bug。
    """
    turns = list(turns)
    judge_ids: list[str] = []
    for turn in turns:
        for v in getattr(turn, "verdicts", []):
            if v.tier == "llm_judge" and v.judge_id and v.judge_id not in judge_ids:
                if metric is None or v.metric == metric:
                    judge_ids.append(v.judge_id)

    out: dict[str, list[float | None]] = {jid: [] for jid in judge_ids}
    for turn in turns:
        for jid in judge_ids:
            score = None
            for v in getattr(turn, "verdicts", []):
                if v.judge_id == jid and v.tier == "llm_judge":
                    if metric is None or v.metric == metric:
                        score = v.score
                        break
            out[jid].append(score)
    return out

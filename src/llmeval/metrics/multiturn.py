# -*- coding: utf-8 -*-
"""
多轮对话下的信息保持与纠错评测。

方法学来自《LLMs Get Lost in Multi-Turn Conversation》（Laban et al., Microsoft
Research）—— 把一条完整指令拆成多轮逐步给出，模型表现会显著下滑（论文报告平均
下滑 39%），归因于两个失效模式：

    1. 过早作答：信息还没给全就先给了答案，后续新信息来了却不修正
    2. 遗忘前文：早轮给出的约束到最后一轮已经丢掉了

所以这里不测「单轮答得对不对」，而是测三件事：

    final_answer       最终答案对不对（与同一任务的单轮版本对比即得多轮衰减）
    context_retention  早轮给出的约束有没有守住
    correction         中途更正过的信息有没有更新过来

全部是确定性判定 —— 「最终答案里写的是 42 而不是更早说的 37」，
这种事有唯一正确答案，不该拿去问裁判，问了只会引入不必要的抖动。

论文还指出第三个损失：可靠性下降（同一任务多次运行结果发散）。那个指标属于
跨运行统计，在 analysis.py 的难度/一致性分析里算，不在这里。
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import Response, Sample, Verdict

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _contains_number(text: str, gold: float, tol: float = 1e-6) -> bool:
    """回答里是否出现了某个目标数值（带相对容差）。

    用「出现过」而不是「第一个数字就是它」：模型常用自然语言作答
    （"总价是 42 元"），把位置绑死会误判成错。
    """
    return any(abs(n - gold) <= tol * max(1.0, abs(gold)) for n in _numbers(text))


_PUNCT_RE = re.compile(
    r"[\s\uff0c\u3002\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019"
    r"\uff08\uff09\(\)\[\]\u3010\u3011]+"
)


def _normalize(text: str) -> str:
    """去掉空白与中英标点，只留实义字符。

    标点一律写成 unicode 转义而不是直接打出来 —— 中文引号和 ASCII 引号在源码里
    肉眼几乎分不出，一旦混进字符串就会静默改变正则的含义。
    """
    return _PUNCT_RE.sub("", (text or "").lower())


def _hit(text: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(text)


# ------------------------------------------------------------------ 最终答案
def evaluate_final_answer(sample: Sample, response: Response) -> Verdict | None:
    """最后一轮的答案对不对。"""
    meta = sample.meta or {}
    answer_type = str(meta.get("answer_type") or "facts")
    text = response.text or ""

    if answer_type == "numeric":
        gold = _numbers(sample.reference or "")
        if not gold:
            return None
        target = gold[0]
        ok = _contains_number(text, target)
        return Verdict(
            sample_id=sample.id,
            sut_id=response.sut_id,
            metric="mt_final_answer",
            tier="deterministic",
            score=5.0 if ok else 1.0,
            passed=ok,
            detail={
                "answer_type": "numeric",
                "expected": target,
                "numbers_in_answer": _numbers(text)[:8],
                "turns": len(sample.turns),
            },
            attempt_index=response.attempt_index,
        )

    facts = [str(f) for f in (meta.get("key_facts") or [])]
    if not facts:
        ref = sample.reference or ""
        facts = [ref] if ref else []
    if not facts:
        return None

    hits = [f for f in facts if _hit(text, f)]
    ratio = len(hits) / len(facts)
    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="mt_final_answer",
        tier="deterministic",
        score=round(1 + 4 * ratio, 2),
        passed=len(hits) == len(facts),
        detail={
            "answer_type": "facts",
            "key_facts": facts,
            "hit_facts": hits,
            "missed_facts": [f for f in facts if f not in hits],
            "turns": len(sample.turns),
        },
        attempt_index=response.attempt_index,
    )


# ------------------------------------------------------------------ 信息保持
def evaluate_context_retention(sample: Sample, response: Response) -> Verdict | None:
    """早轮抛出的约束，到最后一轮还在不在。

    这是「lost in conversation」最直接的可观测形式：信息给过、模型当时也收到过，
    但最终产出里它消失了。
    """
    checks = list((sample.meta or {}).get("retention_checks") or [])
    if not checks:
        return None

    text = response.text or ""
    results: list[dict[str, Any]] = []
    for c in checks:
        kind = str(c.get("kind") or "must_contain")
        value = str(c.get("value") or "")
        if not value:
            continue
        present = _hit(text, value)
        ok = present if kind == "must_contain" else (not present)
        results.append(
            {
                "kind": kind,
                "value": value,
                "note": c.get("note", ""),
                "satisfied": ok,
            }
        )

    if not results:
        return None

    satisfied = sum(1 for r in results if r["satisfied"])
    ratio = satisfied / len(results)
    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="mt_retention",
        tier="deterministic",
        score=round(1 + 4 * ratio, 2),
        passed=satisfied == len(results),
        detail={
            "checks": results,
            "violated": [r["value"] for r in results if not r["satisfied"]],
        },
        attempt_index=response.attempt_index,
    )


# ------------------------------------------------------------------ 纠错更新
def evaluate_correction(sample: Sample, response: Response) -> Verdict | None:
    """中途更正过的信息，模型有没有用新的。

    典型陷阱：第三轮说「刚才那个价格我说错了，是 37 不是 42」。
    只看最终答案里出现的是 37 还是 42，就能判断它到底有没有更新认知。
    """
    spec = (sample.meta or {}).get("correction")
    if not isinstance(spec, dict):
        return None
    current = str(spec.get("current") or "")
    stale = [str(s) for s in (spec.get("stale") or [])]
    if not current and not stale:
        return None

    text = response.text or ""
    used_current = _hit(text, current) if current else True
    used_stale = [s for s in stale if _hit(text, s)]

    checks = [{"kind": "must_contain", "value": current, "satisfied": used_current}]
    checks += [
        {"kind": "must_not_contain", "value": s, "satisfied": s not in used_stale}
        for s in stale
    ]
    satisfied = sum(1 for c in checks if c["satisfied"])
    ratio = satisfied / len(checks) if checks else 0.0

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="mt_correction",
        tier="deterministic",
        score=round(1 + 4 * ratio, 2),
        passed=used_current and not used_stale,
        detail={
            "current_value_used": used_current,
            "stale_values_kept": used_stale,
            "checks": checks,
        },
        attempt_index=response.attempt_index,
    )


# ------------------------------------------------------------------ 统一入口
def collect_multiturn_metrics(sample: Sample, response: Response) -> list[Verdict]:
    """多轮题的确定性判定集合。非多轮题返回空列表。"""
    if sample.task_type != "multi_turn":
        return []
    if not sample.turns:
        return []

    verdicts: list[Verdict] = []
    for fn in (evaluate_final_answer, evaluate_context_retention, evaluate_correction):
        v = fn(sample, response)
        if v is not None:
            verdicts.append(v)
    return verdicts

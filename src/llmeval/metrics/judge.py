# -*- coding: utf-8 -*-
"""
Tier2：LLM 裁判。

评分表怎么用，这里有三个硬性约定：

1. 先推理、后打分
   要求裁判按 evaluation_steps 一步步走完，最后才输出分数。
   先给分数再补理由，理由只是分数的装饰品，不是判据。

2. 强制 JSON 输出契约
   自由文本没法做统计。裁判必须返回结构化 JSON，解析失败按"判定失败"处理，
   绝不默认给分 —— 静默兜底会让错误的分混进统计里。

3. 支持三种裁判形态
   pointwise   单点打分，G-Eval 形态，用于质量类指标
   pairwise    两两对比，强制双向换位，用于模型间相对强弱
   reference   参考答案对齐，用于有 gold 的任务

裁判自身的可信度不由裁判自己说了算 —— 校准交给 calibration.py 用人工标注检验。
"""

from __future__ import annotations

import json
from typing import Any

from ..client import LLMClient
from ..config import ModelSpec
from ..schema import Sample, Verdict, Response
from ..textutil import extract_json, make_snippet
from ..mock import MockEngine

_SCORE_KEYS = ("score", "weighted_score", "overall_score", "final_score", "total_score")


# ------------------------------------------------------------------ prompt
def _schema_hint(schema: dict[str, Any] | None) -> str:
    """把评分表里的 output_schema 渲染成一段可读的 JSON 骨架给裁判看。"""
    if not schema:
        return '{"score": <1-5 的整数>, "reasoning": "<一句话依据>"}'
    lines: list[str] = []
    for key, spec in schema.items():
        if isinstance(spec, dict) and spec.get("type") == "array":
            items = spec.get("items") or {}
            if isinstance(items, dict) and items:
                inner = ", ".join(f'"{ik}": <{iv}>' for ik, iv in items.items())
                lines.append(f'  "{key}": [{{{inner}}}]')
            else:
                lines.append(f'  "{key}": [<value>]')
        else:
            lines.append(f'  "{key}": <{spec}>')
    return "{\n" + ",\n".join(lines) + "\n}"


def build_pointwise_messages(
    sample: Sample,
    answer: str,
    rubric: dict[str, Any],
    reference: str | None = None,
) -> list[dict[str, str]]:
    steps = rubric.get("evaluation_steps") or []
    steps_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))

    anchors = ((rubric.get("scale") or {}).get("anchors")) or {}
    anchors_text = "\n".join(f"{k} = {v}" for k, v in sorted(anchors.items(), reverse=True))

    ref_block = reference.strip() if reference else "（本次未提供参考答案，请依据题面与公认事实判断）"

    system = (
        "你是一名严格的评测裁判。你只做一件事：按照给定的评分表，对一段模型回答打分。\n"
        "必须遵守的三条纪律：\n"
        "  1. 先推理，后打分。禁止先给出分数再补理由。\n"
        "  2. 篇幅长短、语气是否自信、排版是否漂亮，都不构成加分理由。\n"
        "  3. 只输出一个 JSON 对象，前后不得有任何解释性文字。"
    )

    user = f"""## 用户的任务
{sample.prompt}

## 待评回答
{answer}

## 参考资料
{ref_block}

## 评分标准
{rubric.get("criterion", "").strip()}

## 评分步骤（必须按顺序执行，逐步写出结论）
{steps_text}

## 分数刻度
{anchors_text}

## 输出契约
只输出下面这个结构的 JSON，不要输出任何其他内容：
{_schema_hint(rubric.get("output_schema"))}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_pairwise_messages(
    sample: Sample,
    answer_first: str,
    answer_second: str,
    rubric: dict[str, Any],
    first_label: str = "A",
    second_label: str = "B",
) -> list[dict[str, str]]:
    anchors = ((rubric.get("scale") or {}).get("anchors")) or {}
    anchors_text = "\n".join(f"{k} = {v}" for k, v in sorted(anchors.items(), reverse=True))

    system = (
        "你是一名严格的评测裁判。你只做一件事：比较两份回答，判断哪一份更好。\n"
        "必须遵守：\n"
        "  1. 先逐条对照评分标准推理，再给出结论。\n"
        "  2. 篇幅长短、语气、排版都不构成加分理由。\n"
        "  3. 两者确实难分高下时，如实判 tie，不要为了给出结论而硬选一个。\n"
        "  4. 只输出一个 JSON 对象。"
    )

    user = f"""## 用户的任务
{sample.prompt}

## 回答 {first_label}
{answer_first}

## 回答 {second_label}
{answer_second}

## 评分标准
{rubric.get("criterion", "").strip()}

## 分数刻度
{anchors_text}

## 输出契约
只输出：
{{"winner": "{first_label} 或 {second_label} 或 tie", "reasoning": "<先推理，再给结论>"}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ------------------------------------------------------------------ 结果解析
def parse_score(payload: Any) -> float | None:
    """从裁判的结构化输出里取分。取不到就返回 None —— 不兜底、不猜。"""
    if not isinstance(payload, dict):
        return None
    for key in _SCORE_KEYS:
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return None


def parse_verdict_text(text: str, rubric: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """从裁判的原始文本里尽力恢复结构化结果。"""
    payload = extract_json(text)
    if payload is not None:
        return parse_score(payload), payload if isinstance(payload, dict) else {}
    return None, {"parse_error": "未能从裁判输出中解析出 JSON", "raw_head": make_snippet(text, 200)}


# ------------------------------------------------------------------ 判定入口
def evaluate_pointwise(
    sample: Sample,
    response: Response,
    rubric_id: str,
    rubric: dict[str, Any],
    client: LLMClient,
    judge_spec: ModelSpec,
    reference: str | None = None,
) -> Verdict:
    """单点打分。"""
    if not response.ok:
        return _failed(sample, response, f"judge:{rubric_id}", judge_spec.id, "被测回答本身调用失败")

    messages = build_pointwise_messages(sample, response.text, rubric, reference)
    result = client.chat(
        messages,
        hint={
            "kind": "judge",
            "rubric_id": rubric_id,
            "sample": sample,
            "answer": response.text,
            "reference": reference or "",
            "messages": messages,
        },
    )

    if not result.ok:
        return _failed(sample, response, f"judge:{rubric_id}", judge_spec.id, result.error or "")

    score, payload = parse_verdict_text(result.text, rubric)
    if score is None:
        return Verdict(
            sample_id=sample.id,
            sut_id=response.sut_id,
            metric=f"judge:{rubric_id}",
            tier="llm_judge",
            score=None,
            passed=None,
            detail=payload,
            judge_id=judge_spec.id,
            latency_s=result.latency_s,
            usage=result.usage,
            error="裁判输出不符合输出契约，判定作废",
            attempt_index=response.attempt_index,
        )

    score = max(1.0, min(5.0, score))
    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric=f"judge:{rubric_id}",
        tier="llm_judge",
        score=score,
        passed=score >= 4,
        detail=payload,
        judge_id=judge_spec.id,
        latency_s=result.latency_s,
        usage=result.usage,
        attempt_index=response.attempt_index,
    )


def evaluate_panel(
    sample: Sample,
    response: Response,
    rubric_id: str,
    rubric: dict[str, Any],
    panels: list[tuple[LLMClient, ModelSpec]],
    reference: str | None = None,
) -> list[Verdict]:
    """评审团：让多个裁判各自独立打分，返回每个裁判各自的 Verdict。

    这里刻意**不**做「把多个裁判塞进同一段上下文里讨论」那种做法。
    原因：一旦某个裁判能看到另一个裁判的结论，它就不再是独立样本了，
    「用不同偏差互相抵消」这个机制就失效了 —— 只会放大最先开口那个模型的偏好。

    正确姿势是：独立打分 → 事后聚合（中位数/多数票）→ 度量它们的一致性。
    这也是「裁判委员会优于单一裁判」这类结论能成立的前提。
    """
    verdicts: list[Verdict] = []
    for client, spec in panels:
        verdicts.append(
            evaluate_pointwise(
                sample, response, rubric_id, rubric, client, spec, reference=reference
            )
        )
    return verdicts


def evaluate_pairwise(
    sample: Sample,
    response_a: Response,
    response_b: Response,
    rubric_id: str,
    rubric: dict[str, Any],
    client: LLMClient,
    judge_spec: ModelSpec,
) -> dict[str, Any]:
    """两两对比 + 强制双向换位。

    位置偏差是裁判最顽固的毛病之一，单次对比的结论不可信。
    因此每个配对各判两次（A 先 / B 先），只有两次都指向同一方，才记这一方的胜；
    否则一律记平局。位置敏感的那部分噪声由此转成诚实的平局，而不是虚假的胜负。
    """
    if not (response_a.ok and response_b.ok):
        return {"winner": "invalid", "consistent": False, "reason": "至少一方调用失败"}

    rounds = [
        (response_a, response_b, response_a.sut_id, response_b.sut_id),
        (response_b, response_a, response_b.sut_id, response_a.sut_id),
    ]

    outcomes: list[str] = []
    raw: list[dict[str, Any]] = []
    total_usage = None

    for first, second, first_name, second_name in rounds:
        messages = build_pairwise_messages(
            sample, first.text, second.text, rubric, "A", "B"
        )
        result = client.chat(
            messages,
            hint={
                "kind": "judge",
                "mode": "pairwise",
                "rubric_id": rubric_id,
                "sample": sample,
                "answer": first.text,
                "answer_a": first.text,
                "answer_b": second.text,
                "reference": "",
                "messages": messages,
            },
        )
        if not result.ok:
            raw.append({"error": result.error})
            outcomes.append("invalid")
            continue

        payload = extract_json(result.text) or {}
        mark = str(payload.get("winner", "")).strip().upper()
        # 把 "A/B" 还原成真实的模型 id
        if mark == "A":
            outcomes.append(first_name)
        elif mark == "B":
            outcomes.append(second_name)
        else:
            outcomes.append("tie")
        raw.append({"winner": mark, "reasoning": make_snippet(str(payload.get("reasoning", "")), 160)})

        total_usage = result.usage if total_usage is None else total_usage + result.usage

    a, b = response_a.sut_id, response_b.sut_id
    first_outcome, second_outcome = outcomes[0], outcomes[1] if len(outcomes) > 1 else "invalid"

    if first_outcome == second_outcome and first_outcome in (a, b):
        winner, consistent = first_outcome, True
    elif "invalid" in (first_outcome, second_outcome):
        winner, consistent = "invalid", False
    else:
        winner, consistent = "tie", False   # 顺序一换结论就变，说明是位置敏感，记平局

    return {
        "winner": winner,
        "consistent": consistent,
        "sut_a": a,
        "sut_b": b,
        "by_order": outcomes,
        "len_a": len(response_a.text or ""),
        "len_b": len(response_b.text or ""),
        "usage": total_usage,
        "raw": raw,
    }


def _failed(
    sample: Sample, response: Response, metric: str, judge_id: str, reason: str
) -> Verdict:
    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric=metric,
        tier="llm_judge",
        score=None,
        passed=None,
        detail={"reason": reason},
        judge_id=judge_id,
        error=reason,
        attempt_index=response.attempt_index,
    )


def dump_detail(detail: dict[str, Any]) -> str:
    return json.dumps(detail, ensure_ascii=False)

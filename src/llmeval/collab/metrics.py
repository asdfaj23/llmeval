# -*- coding: utf-8 -*-
"""协作过程级指标：成本、收敛、无效轮、故障恢复。

全部基于 messages / trace 的确定性计算，零裁判、可复现。
这是原 collaboration.py 三个结构指标的**增量**——原三个保留，这里补充过程维度。

指标归属（对应设计文档 §2.3）：
    collab_convergence      一致收敛（诊断+计分，decisive=False）
    collab_invalid_rounds   无效轮次占比（计分，decisive=False）
    collab_recovery         子任务失败恢复（计分，decisive=True，仅故障注入题）
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ..schema import Response, Sample, Verdict


def _reviews(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in msgs if m.get("role") == "reviewer"]


def _execs(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 排除 final（final 也是 executor 角色，但不计入修订轮）
    return [m for m in msgs if m.get("role") == "executor" and m.get("turn") != "final"]


def _is_reject(m: dict[str, Any]) -> bool:
    s = m.get("structured", {})
    return str(s.get("verdict") or m.get("verdict") or "").strip().lower() == "reject"


def evaluate_convergence(sample: Sample, response: Response) -> Verdict | None:
    """一致收敛：末轮 approve 且修订代价可控。同族自偏好在报告层披露。"""
    if sample.task_type != "multi_agent":
        return None
    msgs = response.messages or []
    revs = _reviews(msgs)
    if not revs:
        return None
    verdicts = [
        str(m.get("structured", {}).get("verdict") or m.get("verdict") or "") for m in revs
    ]
    n_rev_needed = sum(1 for v in verdicts[:-1] if v == "reject")
    if verdicts[-1] != "approve":
        score, note = 1.0, "预算内未收敛：末轮仍为否决"
    elif n_rev_needed <= 1:
        score, note = 5.0, "至多一轮修订即收敛"
    elif n_rev_needed == 2:
        score, note = 4.0, "两轮修订后收敛"
    else:
        score, note = 2.0, "多轮反复，收敛质量差"
    return Verdict(
        sample.id, response.sut_id, "collab_convergence", "deterministic",
        score=score, passed=score >= 4,
        detail={"verdicts": verdicts, "revision_rounds": n_rev_needed, "note": note},
        attempt_index=response.attempt_index,
    )


def evaluate_invalid_rounds(sample: Sample, response: Response) -> Verdict | None:
    """无效轮次占比：被否决后的修订是否真的在改（与上轮相似、或不回应意见 => 无效）。"""
    if sample.task_type != "multi_agent":
        return None
    msgs = response.messages or []
    if not any(_is_reject(m) for m in _reviews(msgs)):
        return None
    execs = _execs(msgs)
    if len(execs) < 2:
        return Verdict(
            sample.id, response.sut_id, "collab_invalid_rounds", "deterministic",
            score=1.0, passed=False,
            detail={"reason": "有否决但无修订轮，协作断链"},
            attempt_index=response.attempt_index,
        )
    prev = execs[0].get("content") or ""
    issues: list[str] = []
    for m in _reviews(msgs):
        issues.extend(str(i) for i in (m.get("structured", {}).get("issues") or m.get("issues") or []))
    invalid = 0
    for cur in execs[1:]:
        text = cur.get("content") or ""
        similar = SequenceMatcher(None, prev, text).ratio() > 0.9
        addresses = any(str(i)[:12] in text for i in issues)
        if similar or (issues and not addresses):
            invalid += 1
        prev = text
    ratio = invalid / max(1, len(execs) - 1)
    score = round(5.0 - min(4.0, ratio * 5.0), 2)
    return Verdict(
        sample.id, response.sut_id, "collab_invalid_rounds", "deterministic",
        score=max(1.0, score), passed=score >= 4,
        detail={"revision_rounds": len(execs) - 1, "invalid_rounds": invalid,
                "invalid_ratio": round(ratio, 4)},
        attempt_index=response.attempt_index,
    )


def evaluate_recovery(sample: Sample, response: Response) -> Verdict | None:
    """子任务失败恢复：仅对故障注入题生效（planted_failure）。"""
    if not (sample.meta or {}).get("planted_failure"):
        return None
    failed = [s for s in (response.trace or []) if not s.get("ok")]
    if not failed:
        return Verdict(
            sample.id, response.sut_id, "collab_recovery", "deterministic",
            score=5.0, passed=True,
            detail={"note": "预期故障未复现（环境侧需检查）",
                    "planted": sample.meta["planted_failure"]},
            attempt_index=response.attempt_index,
        )
    steps = {s.get("step"): s for s in response.trace or []}
    recovered = 0
    for f in failed:
        fstep = f.get("step")
        for later in response.trace or []:
            ls = later.get("step")
            if ls and fstep and ls > fstep and ls <= fstep + 2 and later.get("tool") == f.get("tool") \
                    and later.get("ok") and later.get("retry_of") == fstep:
                recovered += 1
                break
    # 兜底：没有重试但 final 明确披露失败并降级交付，算部分恢复
    final_ok = response.ok and any(
        k in (response.text or "") for k in ("未能获取", "缺失", "无法确认", "不存在", "未找到")
    )
    score = 5.0 if recovered == len(failed) else (3.5 if final_ok else 1.0)
    return Verdict(
        sample.id, response.sut_id, "collab_recovery", "deterministic",
        score=score, passed=score >= 3.5,
        detail={"failed_steps": [f.get("step") for f in failed],
                "recovered": recovered, "degraded_delivery": final_ok},
        attempt_index=response.attempt_index,
    )


def collect_collab_process(sample: Sample, response: Response) -> list[Verdict]:
    """过程级指标集合，注册进 pipeline 的 multi_agent 判定链。"""
    if not response.ok:
        return []
    out: list[Verdict] = []
    for fn in (evaluate_convergence, evaluate_invalid_rounds, evaluate_recovery):
        v = fn(sample, response)
        if v is not None:
            out.append(v)
    return out

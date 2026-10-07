# -*- coding: utf-8 -*-
"""策略无关指标：让 ReAct 与 Plan-and-Execute 在同一把尺子下可比。

为什么需要这一层：Plan-and-Execute 的指标（reviewer 是否否决、修订闭环是否成立）
依赖"独立审查者"角色，ReAct 没有这个角色，直接套用会全部判 0 分，毫无比较意义。

这里只取两种范式都能产出的、纯结构化的信号：
    collab_completion   是否真正走完（调用 finish 并产出最终答案）
    collab_tool_success 工具步的成功率（不含 finish）
    collab_recovery     故障注入题的子任务恢复（复用 metrics.evaluate_recovery）
    collab_steps        平均工具步数（成本/轮次的粗略代理）

全部确定性、零裁判、可复现。
"""

from __future__ import annotations

from ..schema import Response, Sample, Verdict
from .attribution import VALID_TOOLS
from .metrics import evaluate_recovery


def _finish_reached(response: Response) -> bool:
    return any(
        str(s.get("tool")) == "finish" and s.get("ok") for s in (response.trace or [])
    )


def _tool_steps(response: Response) -> list[dict]:
    return [s for s in (response.trace or []) if str(s.get("tool")) != "finish"]


def evaluate_completion(sample: Sample, response: Response) -> Verdict | None:
    """一致收敛的范式无关版本：走到 finish 且最终答案非空。"""
    if sample.task_type != "multi_agent":
        return None
    reached = _finish_reached(response)
    score = 5.0 if (reached and response.ok) else 1.0
    return Verdict(
        sample.id, response.sut_id, "collab_completion", "deterministic",
        score=score, passed=score >= 4,
        detail={"finish_reached": reached, "ok": response.ok},
        attempt_index=response.attempt_index,
    )


def evaluate_tool_success(sample: Sample, response: Response) -> Verdict | None:
    """工具步成功率：两种范式共享的轨迹结构，直接可比。"""
    if sample.task_type != "multi_agent":
        return None
    steps = _tool_steps(response)
    if not steps:
        return None
    ok_count = sum(1 for s in steps if s.get("ok"))
    ratio = ok_count / len(steps)
    score = round(ratio * 5.0, 2)
    return Verdict(
        sample.id, response.sut_id, "collab_tool_success", "deterministic",
        score=max(1.0, score), passed=ratio >= 0.8,
        detail={"tool_steps": len(steps), "ok": ok_count, "ratio": round(ratio, 4)},
        attempt_index=response.attempt_index,
    )


def evaluate_steps(sample: Sample, response: Response) -> Verdict | None:
    """平均工具步数（标量，给报告用，不计入通过/不通过）。"""
    if sample.task_type != "multi_agent":
        return None
    steps = _tool_steps(response)
    return Verdict(
        sample.id, response.sut_id, "collab_steps", "deterministic",
        score=round(len(steps), 2), passed=True,
        detail={"tool_steps": len(steps), "messages": len(response.messages or [])},
        attempt_index=response.attempt_index,
    )


def evaluate_context_volume(sample: Sample, response: Response) -> Verdict | None:
    """累计上下文体积（所有消息的字符总数），作为 token/上下文成本的代理指标。

    ReAct 每步把完整 history 重新喂给模型，体积随步数近似二次增长；
    Plan-and-Execute 执行器每步只看"计划 + 当前步"，体积有界。
    越低越好（本指标不参与 pass/fail，仅用于成本维度对比）。
    """
    if sample.task_type != "multi_agent":
        return None
    msgs = response.messages or []
    chars = sum(len(str(m.get("content") or "")) for m in msgs)
    return Verdict(
        sample.id, response.sut_id, "collab_context_volume", "deterministic",
        score=round(chars, 1), passed=True,
        detail={"chars": chars, "messages": len(msgs)},
        attempt_index=response.attempt_index,
    )


def collect_compare_metrics(sample: Sample, response: Response) -> list[Verdict]:
    """策略无关指标集合，注册进 pipeline 的 multi_agent 判定链。"""
    if not response.ok:
        return []
    out: list[Verdict] = []
    for fn in (evaluate_completion, evaluate_tool_success, evaluate_recovery,
               evaluate_steps, evaluate_context_volume):
        v = fn(sample, response)
        if v is not None:
            out.append(v)
    return out

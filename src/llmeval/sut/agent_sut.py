# -*- coding: utf-8 -*-
"""
Agent 型被测对象（单智能体）。

与 ChatSUT 的区别：它不只是"回答"，还要"动手"。
一次完整的运行会产出两样东西：
    最终答案（text）
    工具调用轨迹（trace）

轨迹是 Agent 评测的一半依据。只看结果会奖励"歪打正着"——
乱调一通工具最后凑出正确答案，结果分很高但系统其实不可靠。

实现上刻意做得很朴素，不引入任何 Agent 框架：
    第一轮 —— 让模型给出调用计划
    框架执行工具，逐步记录 trace
    第二轮 —— 让模型基于工具结果给最终答案
控制流全部在自己的代码里，出问题能定位，评测时也说得清每一步是怎么来的。

工具实现不在这里，在 tools.py —— 单智能体与多智能体共用同一份。
"""

from __future__ import annotations

from ..schema import Response, Sample
from ..textutil import extract_json
from .base import BaseSUT
from .tools import TOOL_SPEC, run_tool_plan, safe_calc

_PLAN_SYSTEM = f"""你是一个会用工具解决问题的 Agent。

可用工具：
{TOOL_SPEC}

第一轮你只输出一个 JSON 对象，说明准备按什么顺序调用工具，不要输出任何其他内容：
{{"plan": ["search_corpus", "calculator", "finish"], "args": {{"query": "...", "expression": "...", "field": "..."}}}}"""

_ANSWER_SYSTEM = """你是一个会用工具解决问题的 Agent。
下面是你已经执行完的工具调用记录。请基于这些结果直接给出最终答案。
不要再输出工具调用，也不要说"我准备调用"，直接给结论，并说明依据来自哪一步。"""


class AgentSUT(BaseSUT):
    """带工具调用的单智能体被测对象。"""

    kind = "agent"
    max_steps = 8

    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        # ---- 第一轮：拿调用计划
        plan_messages = self._plan_messages(sample)
        plan_result = self.client.chat(
            plan_messages,
            hint={
                "kind": "sut",
                "sample": sample,
                "messages": plan_messages,
                "agent_role": "planner",
            },
        )
        if not plan_result.ok:
            return Response(
                sample_id=sample.id,
                sut_id=self.spec.id,
                error=plan_result.error,
                latency_s=plan_result.latency_s,
                usage=plan_result.usage,
                attempt_index=attempt_index,
            )

        plan = extract_json(plan_result.text) or {}
        tool_names = plan.get("plan") if isinstance(plan, dict) else None
        if not isinstance(tool_names, list):
            tool_names = []
        args = plan.get("args") if isinstance(plan.get("args"), dict) else {}

        # ---- 执行工具，逐步采集轨迹
        trace = run_tool_plan([str(t) for t in tool_names], args, sample, self.max_steps)

        # ---- 第二轮：带工具结果给最终答案
        answer_messages = build_answer_messages(sample, trace)
        answer_result = self.client.chat(
            answer_messages,
            hint={
                "kind": "sut",
                "sample": sample,
                "messages": answer_messages,
                "agent_role": "executor",
            },
        )

        usage = plan_result.usage + answer_result.usage
        return Response(
            sample_id=sample.id,
            sut_id=self.spec.id,
            text=answer_result.text,
            trace=trace,
            usage=usage,
            latency_s=plan_result.latency_s + answer_result.latency_s,
            error=answer_result.error,
            attempts=plan_result.attempts + answer_result.attempts,
            attempt_index=attempt_index,
        )

    # ------------------------------------------------------------ 内部
    def _plan_messages(self, sample: Sample) -> list[dict[str, str]]:
        context = sample.context or "（本题未提供材料）"
        user = f"## 可用材料\n{context}\n\n## 任务\n{sample.prompt}"
        return [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": user},
        ]


def build_answer_messages(
    sample: Sample, trace: list[dict[str, object]]
) -> list[dict[str, str]]:
    """把工具执行记录渲染成第二轮对话。AgentSUT 与 MultiAgentSUT 共用这一份。"""
    if trace:
        lines = []
        for step in trace:
            args = step.get("args")
            arg_text = (
                ", ".join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else ""
            )
            lines.append(
                f"{step.get('step')}. {step.get('tool')}({arg_text}) -> {step.get('result')}"
            )
        trace_text = "\n".join(lines)
    else:
        trace_text = "（本次没有成功执行任何工具调用）"

    user = (
        f"## 可用材料\n{sample.context or '（无）'}\n\n"
        f"## 任务\n{sample.prompt}\n\n"
        f"## 工具调用记录\n{trace_text}"
    )
    return [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


__all__ = ["AgentSUT", "safe_calc", "build_answer_messages"]

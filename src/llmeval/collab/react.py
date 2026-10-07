# -*- coding: utf-8 -*-
"""ReAct 运行时：思维链与动作交错，单模型自我决策（与 Plan-and-Execute 对照）。

ReAct = Reason + Act：模型在「思考 → 选工具 → 看观察」的循环里自己决定下一步，
没有独立的 planner / reviewer 角色。这是工业界最常见的 agent 范式之一，
也是与 Plan-and-Execute（先规划、再执行、再审查）做对照实验的天然对手。

为保证两范式在同一套确定性指标下可比，本运行时复用与 CollabRuntime 相同的：
    - 工具分发（dispatch）
    - 轨迹结构（trace 的逐步独立参数 + retry_of 链）
    - 故障注入（planted_failure 题强制某步用会失败的参数，观察能否恢复）
"""

from __future__ import annotations

from typing import Any

from ..schema import Response, Sample, Usage
from ..textutil import extract_json, make_snippet
from .attribution import VALID_TOOLS
from .trace_recorder import CollabMessage


def _args_of(args: Any) -> dict[str, Any]:
    return {k: v for k, v in (args or {}).items() if isinstance(v, (str, int, float, bool))}


def _arg_text(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    return ", ".join(f"{k}={v}" for k, v in args.items())


class ReactRuntime:
    """单模型 ReAct 循环。与 CollabRuntime 同构，但不拆分角色。"""

    def __init__(self, default_client: Any, *, max_steps: int = 8, budget_tokens: int = 200_000) -> None:
        self.client = default_client
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens

    # ------------------------------------------------------------ 主流程
    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        # 延迟导入，避免 collab.react ↔ sut.tools ↔ sut.__init__ ↔ multi_agent_sut 的导入环
        from ..sut.tools import dispatch

        messages: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        usage = Usage()
        latency = 0.0
        attempts = 0

        def bookkeep(result: Any) -> None:
            nonlocal usage, latency, attempts
            usage = usage + result.usage
            latency += result.latency_s
            attempts += result.attempts

        def chat(role: str, msgs: list[dict[str, str]], round_no: int = 0) -> Any:
            r = self.client.chat(
                msgs,
                hint={"kind": "sut", "sample": sample, "messages": msgs,
                      "agent_role": role, "react_round": round_no},
            )
            bookkeep(r)
            return r

        planted = (sample.meta or {}).get("planted_failure")
        recover_map = (
            {str(planted.get("tool")): _args_of(planted.get("recover"))}
            if isinstance(planted, dict) and planted.get("recover") else None
        )
        injected: dict[str, bool] = {"done": False}

        trace_step = 0
        history = ""
        loop = 0
        finished = False

        while loop < self.max_steps:
            loop += 1
            r = chat("react", self._react_messages(sample, history, loop), loop)
            if not r.ok:
                return self._failed(sample, attempt_index, r.error, usage, latency, messages, trace, attempts)
            parsed = extract_json(r.text) or {}
            action = parsed.get("action") or {}
            tool = str((action.get("tool") or "")).strip()
            thought = str(parsed.get("thought") or "")
            if tool == "finish":
                trace_step += 1
                trace.append({"step": trace_step, "tool": "finish", "args": {}, "result": "结束调用", "ok": True, "retry_of": None, "latency_s": 0.0})
                messages.append(CollabMessage(turn=loop, role="react", content=f"{thought}\n→ 结束（finish）").to_msg())
                finished = True
                break

            if tool not in VALID_TOOLS:
                messages.append(CollabMessage(turn=loop, role="react", content=f"{thought}\n→ 动作：{tool}（非法工具）").to_msg())
                history += f"\n第{loop}步：模型想调用 {tool}，但该工具不可用，请改用 search_corpus/calculator/get_field/finish。"
                continue

            args = _args_of(action.get("args"))
            intended = dict(args)
            # 故障注入：第一次遇到目标工具时，用会失败的参数替换，观察能否恢复
            if isinstance(planted, dict) and planted.get("tool") == tool and not injected["done"]:
                args = _args_of(planted.get("args"))
                injected["done"] = True

            result_text, ok = dispatch(tool, args, sample)
            if ok:
                trace_step += 1
                trace.append({"step": trace_step, "tool": tool, "args": intended, "result": make_snippet(result_text, 160), "ok": True, "retry_of": None, "latency_s": 0.0})
                messages.append(CollabMessage(turn=loop, role="react", content=f"{thought}\n→ 动作：{tool}({_arg_text(args)})\n观察：{make_snippet(result_text, 160)}").to_msg())
                history += f"\n第{loop}步：调用 {tool}，得到：{make_snippet(result_text, 160)}"
                continue

            # 工具失败：先记录失败步，再用 recover 参数（默认回退到原始参数）重试一次
            trace_step += 1
            trace.append({"step": trace_step, "tool": tool, "args": _args_of(args), "result": make_snippet(result_text, 160), "ok": False, "error": result_text, "retry_of": None, "latency_s": 0.0})
            retry_args = (recover_map or {}).get(tool, intended)
            result2, ok2 = dispatch(tool, retry_args, sample)
            trace_step += 1
            trace.append({"step": trace_step, "tool": tool, "args": _args_of(retry_args), "result": make_snippet(result2, 160), "ok": ok2, "retry_of": trace_step - 1, "latency_s": 0.0})
            obs = "工具执行失败：" + make_snippet(result_text, 120) + ("；已用修正参数重试并成功。" if ok2 else "；重试仍失败。")
            messages.append(CollabMessage(turn=loop, role="react", content=f"{thought}\n→ 动作：{tool}\n观察：{obs}").to_msg())
            history += f"\n第{loop}步：调用 {tool} 失败（{make_snippet(result_text, 120)}）" + ("；重试成功。" if ok2 else "；重试仍失败。")

        if not finished:
            messages.append(CollabMessage(turn="budget", role="react", content="（达到最大步数仍未调用 finish 结束）").to_msg())

        # ---- final
        r = chat("final", self._final_messages(sample, trace, history))
        if not r.ok:
            return self._failed(sample, attempt_index, r.error, usage, latency, messages, trace, attempts)
        messages.append(
            CollabMessage(
                turn="final", role="final", content=r.text,
                latency_s=r.latency_s,
                usage={"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
            ).to_msg()
        )

        return Response(
            sample_id=sample.id,
            sut_id=getattr(self.client, "spec", None) and self.client.spec.id or "",
            text=r.text,
            trace=trace,
            messages=messages,
            usage=usage,
            latency_s=latency,
            attempts=attempts,
            attempt_index=attempt_index,
        )

    # ------------------------------------------------------------ 消息构造
    @staticmethod
    def _react_messages(sample: Sample, history: str, step_no: int) -> list[dict[str, str]]:
        from .roles import REACT_SYSTEM
        user = (
            f"## 可用材料\n{sample.context or '（无）'}\n\n"
            f"## 任务\n{sample.prompt}\n\n"
            f"## 已发生的步骤\n{history or '（暂无）'}\n\n"
            f"第 {step_no} 步：请思考并决定下一步动作。\n"
            "只输出一个 JSON 对象："
            '{"thought": "你的思考", "action": {"tool": "search_corpus|calculator|get_field|finish", "args": {...}}}。'
            "调用 finish 表示信息已足够、可以给出最终答案。"
        )
        return [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def _final_messages(sample: Sample, trace: list[dict[str, Any]], history: str) -> list[dict[str, str]]:
        trace_text = "\n".join(
            f"{s.get('step')}. {s.get('tool')}({_arg_text(s.get('args'))}) -> {s.get('result')}"
            for s in trace
        ) or "（无工具调用）"
        user = (
            f"## 任务\n{sample.prompt}\n\n"
            f"## 工具执行结果\n{trace_text}\n\n"
            f"## 过程记录\n{history}\n\n"
            "请综合上述结果给出最终交付答案，直接给结论与依据，不要复述过程。"
        )
        return [{"role": "system", "content": "你是执行 ReAct 循环的智能体，现在综合工具结果给出最终答案。"}, {"role": "user", "content": user}]

    # ------------------------------------------------------------ 失败兜底
    @staticmethod
    def _failed(sample, attempt_index, error, usage, latency, messages, trace, attempts) -> Response:
        return Response(
            sample_id=sample.id,
            sut_id="",
            error=error,
            usage=usage,
            latency_s=latency,
            messages=messages,
            trace=trace,
            attempts=attempts,
            attempt_index=attempt_index,
        )

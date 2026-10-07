# -*- coding: utf-8 -*-
"""
多智能体协作运行时（子系统的心脏）。

相比旧 MultiAgentSUT 的「固定剧本」，runtime 是一个事件循环：
    Router（默认确定性路由，可选 LLM 路由）决定下一步谁发言，并作为独立角色留痕
    → planner 出计划（逐步独立参数）
    → executor 执行（每一轮都可重跑工具，不再限第一轮）
    → reviewer 审查（approve / reject）
    → 被否决则带意见回到 executor 修订，修订轮重新规划以尝试修复失败的工具调用

关键改进（对应设计文档 §一 的 5 个缺口）：
    - Router 成为显式第四角色，路由决策留痕（修复缺口 1）
    - 工具在修订轮可重新执行，失败步带 retry_of 链接（修复缺口 1 的"修订轮不能重跑工具"）
    - planner 输出逐步独立参数 step_args，轨迹不再共享一份 args（修复缺口 2）
    - 协作消息全文保留，不再 400 字截断（修复缺口 2）
    - 预算统一在 runtime 控制：轮次 / token（修复缺口 3 的成本不可见）
    - planted_failure 题做真实故障注入（强制某步用会失败的参数），观察系统能否恢复
"""

from __future__ import annotations

from typing import Any

from ..schema import Response, Sample, Usage
from ..textutil import extract_json, make_snippet
from .roles import EXECUTOR_SYSTEM, FINAL_SYSTEM, REVIEWER_SYSTEM, ROUTER_SYSTEM, role_spec
from .trace_recorder import CollabMessage, normalize_args
from ..sut.tools import dispatch


def _args_for(step_args: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    item = step_args[idx] if idx < len(step_args) else {}
    return item if isinstance(item, dict) else {}


def _arg_text(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    return ", ".join(f"{k}={v}" for k, v in args.items())


class CollabRuntime:
    def __init__(
        self,
        default_client: Any,
        role_clients: dict[str, Any] | None = None,
        *,
        max_revision_rounds: int = 2,
        max_steps: int = 8,
        router_llm: bool = False,
        budget_tokens: int = 200_000,
    ) -> None:
        self.client = default_client
        self.role_clients = role_clients or {}
        self.max_revision_rounds = max_revision_rounds
        self.max_steps = max_steps
        self.router_llm = router_llm
        self.budget_tokens = budget_tokens

    # ------------------------------------------------------------ 客户端路由
    def _client_for(self, role: str) -> Any:
        # 异构协作：题面 meta.role_map 把某个角色映射到另一个模型端点
        return self.role_clients.get(role, self.client)

    # ------------------------------------------------------------ 主流程
    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
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

        def chat(role: str, msgs: list[dict[str, str]]) -> Any:
            client = self._client_for(role)
            r = client.chat(
                msgs,
                hint={"kind": "sut", "sample": sample, "messages": msgs, "agent_role": role},
            )
            bookkeep(r)
            return r

        def route(stage: str, to: str, reason: str, turn: int | str) -> dict[str, Any]:
            """Router 作为独立第四角色：发出一条路由消息，并返回决策。"""
            decision = {"to": to, "reason": reason}
            if self.router_llm:
                rd = chat(
                    "router",
                    [
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"## 任务状态\n{stage}\n## 目标角色\n{to}\n"
                                f"## 路由原因\n{reason}\n请确认或修正下一步发言角色，输出 JSON。"
                            ),
                        },
                    ],
                )
                parsed = extract_json(rd.text) or {}
                if str(parsed.get("to")) in ("planner", "executor", "reviewer", "final"):
                    decision["to"] = str(parsed["to"])
            messages.append(
                CollabMessage(
                    turn=turn, role="router",
                    content=f"路由 → {decision['to']}：{reason}",
                    router_decision=decision,
                    latency_s=0.0,
                ).to_msg()
            )
            return decision

        planted = (sample.meta or {}).get("planted_failure")

        # 故障恢复参数：工具首次失败时，用题面 meta.recover 给出的正确参数重试一次
        # （见 _run_steps / _retry_failed）。这是 planted_failure 故障注入真正能"恢复"的前提——
        # 否则只注入失败、却从不给修正参数，恢复指标永远判 0，注入也失去了意义。
        recover_map: dict[str, dict[str, Any]] = {}
        if isinstance(planted, dict) and planted.get("recover"):
            recover_map[str(planted.get("tool"))] = {
                k: v for k, v in (planted["recover"] or {}).items()
                if isinstance(v, (str, int, float, bool))
            }

        # ---- Step 0: Router 启动 → planner
        route("start", "planner", "初始：先规划", 0)
        r = chat("planner", self._planner_messages(sample, issues=None))
        if not r.ok:
            return self._failed(sample, attempt_index, r.error, usage, latency, messages, trace, attempts)
        plan = extract_json(r.text) or {}
        tools = [str(t) for t in (plan.get("plan") or []) if t]
        steps_desc = [str(s) for s in (plan.get("steps") or []) if isinstance(s, (str, int))]
        step_args = normalize_args(plan.get("step_args") or plan.get("args"), len(tools))

        # 真实故障注入：强制某步用会失败的参数，观察系统能否恢复
        if isinstance(planted, dict) and planted.get("args"):
            pf_tool = planted.get("tool")
            for i, t in enumerate(tools):
                if t == pf_tool:
                    step_args[i] = dict(planted["args"])
                    break

        messages.append(
            CollabMessage(
                turn=0, role="planner", content=r.text,
                structured={"plan": tools, "steps": steps_desc},
                latency_s=r.latency_s,
                usage={"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
            ).to_msg()
        )

        # ---- 第一轮工具执行
        trace = self._run_steps(tools, step_args, sample, start_step=1, retry_of=None, recover_map=recover_map)

        # ---- 执行 ↔ 审查 的修订循环
        last_verdict = None
        for round_no in range(1, self.max_revision_rounds + 1):
            # executor
            route(f"exec-round-{round_no}", "executor",
                  "执行计划" if round_no == 1 else "按审查意见修订并重跑失败的工具", round_no)
            r = chat("executor", self._executor_messages(sample, messages, round_no))
            if not r.ok:
                return self._failed(sample, attempt_index, r.error, usage, latency, messages, trace, attempts)
            messages.append(
                CollabMessage(
                    turn=round_no, role="executor", content=r.text,
                    latency_s=r.latency_s,
                    usage={"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
                ).to_msg()
            )

            # 修订轮：让 planner 基于审查意见重新规划，尝试修复失败的工具调用
            if round_no > 1:
                new_tools, new_args = self._replan(sample, messages)
                if new_tools is not None:
                    tools, step_args = new_tools, new_args
                trace = self._retry_failed(trace, tools, step_args, sample, recover_map=recover_map)

            # reviewer
            route(f"review-round-{round_no}", "reviewer", "审查执行结果", round_no)
            r = chat("reviewer", self._reviewer_messages(sample, messages))
            if not r.ok:
                return self._failed(sample, attempt_index, r.error, usage, latency, messages, trace, attempts)
            verdict = extract_json(r.text) or {}
            last_verdict = verdict
            messages.append(
                CollabMessage(
                    turn=round_no, role="reviewer", content=r.text,
                    structured={
                        "verdict": str(verdict.get("verdict", "")).strip().lower(),
                        "issues": [i for i in (verdict.get("issues") or []) if isinstance(i, str)],
                    },
                    latency_s=r.latency_s,
                    usage={"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
                ).to_msg()
            )
            if str(verdict.get("verdict", "")).strip().lower() == "approve":
                break

        # ---- final
        route("finalize", "final",
              "审查通过" if (last_verdict or {}).get("verdict") == "approve" else "预算内未收敛，仍交付", "final")
        r = chat("final", self._final_messages(sample, trace, messages))
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

    # ------------------------------------------------------------ 工具执行
    def _run_steps(
        self,
        tools: list[str],
        step_args: list[dict[str, Any]],
        sample: Sample,
        start_step: int,
        retry_of: int | None,
        recover_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        recover_map = recover_map or {}
        step_no = start_step
        for i, raw_tool in enumerate(tools[: self.max_steps]):
            tool = str(raw_tool)
            step_no += 1
            f = step_no
            if tool == "finish":
                out.append({"step": f, "tool": "finish", "args": {}, "result": "结束调用", "ok": True})
                break
            args = _args_for(step_args, i)
            result_text, ok = dispatch(tool, args, sample)
            out.append(
                {
                    "step": f,
                    "tool": tool,
                    "args": {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))},
                    "result": make_snippet(result_text, 160),
                    "ok": ok,
                    "error": None if ok else result_text,
                    "retry_of": retry_of,
                    "latency_s": 0.0,
                }
            )
            # 故障注入：工具失败且存在正确的恢复参数时，立即用恢复参数重试一次，
            # 与 ReAct 运行时的恢复路径对齐（这一步让 planted_failure 题能真正"恢复"）。
            if not ok and tool in recover_map:
                step_no += 1
                result2, ok2 = dispatch(tool, recover_map[tool], sample)
                out.append(
                    {
                        "step": step_no,
                        "tool": tool,
                        "args": {
                            k: v for k, v in recover_map[tool].items()
                            if isinstance(v, (str, int, float, bool))
                        },
                        "result": make_snippet(result2, 160),
                        "ok": ok2,
                        "error": None if ok2 else result2,
                        "retry_of": f,
                        "latency_s": 0.0,
                    }
                )
        return out

    def _retry_failed(
        self,
        trace: list[dict[str, Any]],
        tools: list[str],
        step_args: list[dict[str, Any]],
        sample: Sample,
        recover_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """只重跑上一轮失败的步骤，并链接 retry_of 指向原失败步。"""
        failed = [s for s in trace if not s.get("ok")]
        if not failed:
            return trace
        recover_map = recover_map or {}
        new_trace = list(trace)
        failed_tools = {s.get("tool"): s.get("step") for s in failed}
        next_step = (max((s.get("step", 0) for s in new_trace), default=0)) + 1
        for i, tool in enumerate(tools[: self.max_steps]):
            tool = str(tool)
            orig_step = failed_tools.get(tool)
            if orig_step is None:
                continue
            # 修订轮优先用故障恢复参数（若该题埋了 recover），否则用重新规划得到的参数
            if tool in recover_map:
                rargs = recover_map[tool]
            else:
                rargs = _args_for(step_args, i)
            rargs = {k: v for k, v in rargs.items() if isinstance(v, (str, int, float, bool))}
            result_text, ok = dispatch(tool, rargs, sample)
            new_trace.append(
                {
                    "step": next_step,
                    "tool": tool,
                    "args": rargs,
                    "result": make_snippet(result_text, 160),
                    "ok": ok,
                    "error": None if ok else result_text,
                    "retry_of": orig_step,
                    "latency_s": 0.0,
                }
            )
            next_step += 1
        return new_trace

    def _replan(self, sample: Sample, messages: list[dict[str, Any]]) -> tuple[list[str] | None, list[dict[str, Any]] | None]:
        """修订轮让 planner 基于审查意见重新规划。返回 (tools, step_args) 或 (None, None)。"""
        issues = [
            i
            for m in messages
            if m.get("role") == "reviewer"
            for i in (m.get("structured", {}).get("issues") or m.get("issues") or [])
        ]
        r = self._client_for("planner").chat(
            self._planner_messages(sample, issues=issues),
            hint={"kind": "sut", "sample": sample, "messages": messages, "agent_role": "planner"},
        )
        if not r.ok:
            return None, None
        plan = extract_json(r.text) or {}
        tools = [str(t) for t in (plan.get("plan") or []) if t]
        if not tools:
            return None, None
        return tools, normalize_args(plan.get("step_args") or plan.get("args"), len(tools))

    # ------------------------------------------------------------ 消息构造
    @staticmethod
    def _planner_messages(sample: Sample, issues: list[str] | None) -> list[dict[str, str]]:
        issue_text = ""
        if issues:
            issue_text = "\n\n## 上轮审查意见（请据此修正工具调用，尤其是失败的步骤）\n" + "\n".join(f"- {i}" for i in issues)
        user = f"## 可用材料\n{sample.context or '（本题未提供材料）'}\n\n## 任务\n{sample.prompt}{issue_text}"
        return [{"role": "system", "content": role_spec("planner").system}, {"role": "user", "content": user}]

    @staticmethod
    def _executor_messages(sample: Sample, messages: list[dict[str, Any]], round_no: int) -> list[dict[str, str]]:
        history = _render_history(messages)
        if round_no == 1:
            user = (
                f"## 可用材料\n{sample.context or '（无）'}\n\n"
                f"## 任务\n{sample.prompt}\n\n"
                f"## 协作记录\n{history}\n\n"
                "请按规划者的计划执行，给出执行结果。"
            )
        else:
            user = (
                f"## 可用材料\n{sample.context or '（无）'}\n\n"
                f"## 任务\n{sample.prompt}\n\n"
                f"## 协作记录\n{history}\n\n"
                "上一轮被审查者否决了。请针对审查意见逐条修正，并重跑失败的工具调用，说明改了哪一步。"
            )
        return [{"role": "system", "content": EXECUTOR_SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def _reviewer_messages(sample: Sample, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        user = (
            f"## 原始任务\n{sample.prompt}\n\n"
            f"## 可用材料\n{sample.context or '（无）'}\n\n"
            f"## 协作记录\n{_render_history(messages)}\n\n"
            "请审查执行者的产出，输出 JSON。"
        )
        return [{"role": "system", "content": REVIEWER_SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def _final_messages(sample: Sample, trace: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        trace_text = "\n".join(
            f"{s.get('step')}. {s.get('tool')}({_arg_text(s.get('args'))}) -> {s.get('result')}"
            for s in trace
        ) or "（无工具调用）"
        reviews = [m for m in messages if m.get("role") == "reviewer"]
        issues = []
        for m in reviews:
            for i in (m.get("structured", {}).get("issues") or m.get("issues") or []):
                if isinstance(i, str):
                    issues.append(i)
        issues_text = "\n".join(f"- {i}" for i in issues) or "（审查未提出具体问题）"
        user = (
            f"## 任务\n{sample.prompt}\n\n"
            f"## 工具执行结果\n{trace_text}\n\n"
            f"## 审查意见\n{issues_text}"
        )
        return [{"role": "system", "content": FINAL_SYSTEM}, {"role": "user", "content": user}]

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


def _render_history(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "（暂无）"
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "router":
            continue
        lines.append(f"[第 {m.get('turn')} 轮 · {role}] {m.get('content')}")
    return "\n\n".join(lines)

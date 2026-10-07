# -*- coding: utf-8 -*-
"""
多智能体协作被测对象（升级版）。

本 SUT 现在是「多智能体协作评测子系统」的对外入口：内部委托 collab.CollabRuntime
完成 planner / executor / reviewer / router 四角色的协作循环、工具执行、失败恢复与
结构化留痕。SUT 这一层只负责把 spec + client 接进 runtime，不再写协作逻辑。

与 AgentSUT 的根本区别：
    AgentSUT      一个模型，自己规划、自己执行
    MultiAgentSUT 多个角色，各司其职，互相制衡（并可异构组网）

为什么单独测这种形态：三个角色凑在一起并不自动等于比单智能体更好。常见失效模式
（reviewer 从不否决 / 否决了但 executor 不改 / 工具失败无法恢复 / 冲突没解决直接给结论）
只有把协作过程记下来才看得见，光看最终答案看不出来。
"""

from __future__ import annotations

from typing import Any

from ..collab.react import ReactRuntime
from ..collab.runtime import CollabRuntime
from ..config import ModelSpec
from ..schema import Response, Sample
from .base import BaseSUT


class MultiAgentSUT(BaseSUT):
    """多角色协作 / 单模型 ReAct 的被测对象，按 spec.strategy 选择协作范式。

    Plan-and-Execute（默认）：planner → executor → reviewer → final，含显式 Router 与异构组网。
    ReAct：单模型在「思考→动作→观察」循环里自我决策，无独立审查者。
    """

    kind = "multi_agent"

    def __init__(self, spec: ModelSpec, client: Any) -> None:
        super().__init__(spec, client)
        self.strategy = getattr(spec, "strategy", "plan_execute") or "plan_execute"
        self.runtime = (
            ReactRuntime(client) if self.strategy == "react" else CollabRuntime(client)
        )

    def set_role_clients(self, role_clients: dict[str, Any]) -> None:
        """注入异构组网所需的逐角色 client（仅 Plan-and-Execute 范式使用）。"""
        if role_clients and isinstance(self.runtime, CollabRuntime):
            self.runtime.role_clients = dict(role_clients)

    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        # 是否启用 LLM Router：仅 Plan-and-Execute 范式、且题面显式打开时
        if isinstance(self.runtime, CollabRuntime):
            self.runtime.router_llm = bool((sample.meta or {}).get("router_llm"))
        return self.runtime.run(sample, attempt_index=attempt_index)


__all__ = ["MultiAgentSUT"]

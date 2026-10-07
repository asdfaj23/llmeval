# -*- coding: utf-8 -*-
"""结构化轨迹记录：协作消息与工具步进，全部不截断、逐步独立留痕。

修复原 multi_agent_sut 的两个留痕缺陷：
    1. 协作消息原文被截断到 400 字（_render_history 喂给下游模型时也丢信息）
       —— 这里全文保留，摘要交给 report 层做。
    2. planner 只给一份共享 args，run_tool_plan 把同一份 args 复制到每一步
       —— 这里每一步带自己独立的 args，并支持 retry_of 链（失败步的重试指向）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolStep:
    """一次工具调用。args 是本步独立参数，不再是共享参数表。"""

    step: int
    tool: str
    args: dict[str, Any]
    result: str
    ok: bool
    error: str | None = None
    retry_of: int | None = None          # 指向被重试的失败步
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollabMessage:
    """一条协作消息：全文 + 结构化输出 + 路由决策 + 逐角色开销。"""

    turn: int | str
    role: str                            # planner / executor / reviewer / router
    content: str                         # 全文，不截断（展示摘要交给 report 层）
    structured: dict[str, Any] = field(default_factory=dict)
    router_decision: dict[str, Any] | None = None
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    def to_msg(self) -> dict[str, Any]:
        """落盘格式，兼容现有 messages 通道（role/role_decision 等信息附加其上）。"""
        d: dict[str, Any] = {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "length": len(self.content),
        }
        if self.structured:
            d["structured"] = self.structured
        if self.router_decision:
            d["router"] = self.router_decision
        if self.usage:
            d["usage"] = self.usage
            d["latency_s"] = round(self.latency_s, 3)
        return d


def normalize_args(step_args: Any, tool_count: int) -> list[dict[str, Any]]:
    """把 planner 给的参数统一成「每步一个 dict」的列表。

    planner 可能输出：
        - 列表（理想形态）：第 i 项对应 plan[i]
        - 单个 dict（共享参数，mock 默认形态）：所有步共用
        - 缺省：全部为空
    统一成长度为 tool_count 的列表，避免下游到处做类型判断。
    """
    if isinstance(step_args, list):
        out = []
        for i in range(tool_count):
            item = step_args[i] if i < len(step_args) else {}
            out.append(item if isinstance(item, dict) else {})
        return out
    if isinstance(step_args, dict):
        return [dict(step_args) for _ in range(tool_count)]
    return [{} for _ in range(tool_count)]

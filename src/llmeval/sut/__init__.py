# -*- coding: utf-8 -*-
"""被测系统与工厂。"""

from .agent_sut import AgentSUT, build_answer_messages
from .base import BaseSUT
from .chat_sut import ChatSUT
from .multi_agent_sut import MultiAgentSUT
from .multi_turn_sut import MultiTurnSUT
from .tools import TOOL_NAMES, TOOL_SPEC, dispatch, safe_calc

# 题型 → 被测形态。集中在这一处，避免映射关系散落到流水线各个角落。
_KIND_BY_TASK = {
    "agent_tool": "agent",
    "multi_agent": "multi_agent",
    "multi_turn": "multi_turn",
}


def build_sut(spec, client, kind: str = "chat") -> BaseSUT:
    """按被测形态构造 SUT。

    kind="chat"         纯对话（多数能力维度）
    kind="agent"        带工具调用的单智能体
    kind="multi_agent"  多角色协作的多智能体系统
    kind="multi_turn"   多轮对话（逐轮喂入，测信息保持与纠错）
    """
    if kind == "agent":
        return AgentSUT(spec, client)
    if kind == "multi_agent":
        return MultiAgentSUT(spec, client)
    if kind == "multi_turn":
        return MultiTurnSUT(spec, client)
    return ChatSUT(spec, client)


def kind_for_task(task_type: str) -> str:
    return _KIND_BY_TASK.get(task_type, "chat")


__all__ = [
    "BaseSUT",
    "ChatSUT",
    "AgentSUT",
    "MultiAgentSUT",
    "MultiTurnSUT",
    "build_sut",
    "build_answer_messages",
    "kind_for_task",
    "safe_calc",
    "dispatch",
    "TOOL_SPEC",
    "TOOL_NAMES",
]

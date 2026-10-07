# -*- coding: utf-8 -*-
"""多智能体角色定义。

角色 = 系统提示 + 输出 schema 说明 + 工具权限。
新子系统把四个角色收敛成一份可注册的定义，single-model 多角色与
multi-model 异构都从这份定义出发（异构只换 client，不换角色定义）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 角色输出 schema 的简短说明，拼进系统提示，约束模型产出可被结构化解析的 JSON。
_PLAN_SCHEMA = (
    '{"steps": ["步骤一", "步骤二"], '
    '"plan": ["search_corpus", "calculator", "finish"], '
    '"step_args": [{"query": "..."}, {"expression": "..."}]}'
    '—— step_args 是列表，第 i 项对应 plan[i] 的参数；'
    "若不便逐项给出，可只给一个共享的 args 对象"
)
_REVIEW_SCHEMA = (
    '{"verdict": "approve 或 reject", "issues": ["具体问题"], "reasoning": "判断依据"}'
    "—— 只有在确实找不出问题时才写 approve，为了让流程好看而放行是失职"
)
_ROUTER_SCHEMA = (
    '{"to": "planner|executor|reviewer|final", "reason": "为什么这样路由"}'
)


@dataclass
class RoleSpec:
    name: str
    system: str
    output_schema: str
    tools_allowed: tuple[str, ...] = ()


def planner_system() -> str:
    """延迟加载：避免在 collab.roles 模块加载时把 sut 包牵进来（sut→multi_agent_sut→collab 会形成环）。"""
    from ..sut.tools import TOOL_SPEC
    return f"""你是一个多智能体系统里的规划者（planner）。

可用工具：
{TOOL_SPEC}

你的职责是把任务拆成可执行步骤，并给出工具调用计划。
只输出一个 JSON 对象，不要输出任何其他内容：
{_PLAN_SCHEMA}"""

EXECUTOR_SYSTEM = """你是多智能体系统里的执行者（executor）。

按规划者的计划调用工具，并把执行结果整理清楚。
如果是修订轮（消息里带有审查意见），你必须针对每一条意见说明改了什么、重跑了哪一步工具。
直接给执行结果，不要客套话。若某一步工具失败，明确说明失败原因，不要编造结果。"""

REVIEWER_SYSTEM = f"""你是多智能体系统里的审查者（reviewer）。

你的职责是挑毛病，不是点头。必须逐项检查：
  1. 用到的数据是否出自给定材料，有没有编造
  2. 计算过程是否正确
  3. 结论是否有依据支撑
  4. 题目的约束条件有没有被漏掉

只输出一个 JSON 对象：
{_REVIEW_SCHEMA}"""

ROUTER_SYSTEM = f"""你是多智能体系统里的路由器（router）。

你决定当前这一步该由哪个角色发言。规则：
  - 规划未完成 → planner
  - 计划已就绪、尚未执行 → executor
  - 执行已完成、尚未审查 → reviewer
  - 审查 approve 或无需审查 → final
  - 审查 reject 且仍有修订预算 → executor（带审查意见回去重做）

只输出一个 JSON 对象：
{_ROUTER_SCHEMA}"""

FINAL_SYSTEM = """你是多智能体系统里的执行者。
现在给出最终交付答案，综合工具执行结果与审查意见。
直接给结论和依据，不要复述过程，也不要再提审查流程本身。"""

REACT_SYSTEM = """你是一个采用 ReAct（Reason + Act）范式的智能体。

你在一个循环里自主决定下一步：先思考（thought），再选择一个工具执行（action），
看到观察（observation）后继续思考下一步，直到信息足够时调用 finish 结束。
没有独立的规划者或审查者——你一个人负责规划、执行与自我纠错。
每次只输出一个 JSON 对象，不要输出任何其他内容：
{"thought": "你的思考", "action": {"tool": "search_corpus|calculator|get_field|finish", "args": {...}}}。"""


def role_spec(name: str) -> RoleSpec:
    return ROLES[name]


ROLES: dict[str, RoleSpec] = {
    "planner": RoleSpec("planner", planner_system(), _PLAN_SCHEMA, ("search_corpus", "calculator", "get_field", "finish")),
    "executor": RoleSpec("executor", EXECUTOR_SYSTEM, "", ("search_corpus", "calculator", "get_field", "finish")),
    "reviewer": RoleSpec("reviewer", REVIEWER_SYSTEM, _REVIEW_SCHEMA, ()),
    "router": RoleSpec("router", ROUTER_SYSTEM, _ROUTER_SCHEMA, ()),
}


# -*- coding: utf-8 -*-
"""多智能体协作评测子系统。

把 multi_agent 从「13 个能力维度之一」升级为独立子系统：
    - roles.py          角色定义（planner/executor/reviewer/router）+ ReAct 系统提示
    - trace_recorder.py 结构化轨迹记录（逐步独立参数 + retry_of + 不截断）
    - runtime.py        协作运行时（Router + 预算 + 修订轮可重跑工具 + 失败恢复）
    - react.py          ReAct 运行时（单模型思考/动作交错，与 Plan-and-Execute 对照）
    - attribution.py    失败归因五分类（诊断，不计分）
    - metrics.py        过程级指标（收敛 / 无效轮 / 故障恢复）
    - compare.py        策略无关指标（让两种范式同一把尺子可比）
    - replay.py         可重放沙箱（离线分析 + HTML 报告）
    - compare_report.py ReAct vs Plan-and-Execute 对比报告
"""

from __future__ import annotations

from .attribution import attribute_failure, attribution_histogram, label_cn
from .compare import collect_compare_metrics
from .compare_report import compare_run
from .metrics import (
    collect_collab_process,
    evaluate_convergence,
    evaluate_invalid_rounds,
    evaluate_recovery,
)
from .react import ReactRuntime
from .replay import analyze_turns, render_html, replay_run
from .roles import REACT_SYSTEM, REVIEWER_SYSTEM, ROLES, RoleSpec, role_spec
from .runtime import CollabRuntime
from .trace_recorder import CollabMessage, ToolStep, normalize_args

__all__ = [
    "ROLES",
    "RoleSpec",
    "role_spec",
    "REACT_SYSTEM",
    "REVIEWER_SYSTEM",
    "CollabMessage",
    "ToolStep",
    "normalize_args",
    "CollabRuntime",
    "ReactRuntime",
    "attribute_failure",
    "attribution_histogram",
    "label_cn",
    "collect_collab_process",
    "collect_compare_metrics",
    "evaluate_convergence",
    "evaluate_invalid_rounds",
    "evaluate_recovery",
    "compare_run",
    "analyze_turns",
    "render_html",
    "replay_run",
]

# -*- coding: utf-8 -*-
"""LLMEval —— 面向大模型能力评测的轻量框架。

对外只需要记住四件事：
    config.load_*    读配置
    client.LLMClient 调模型
    pipeline.run     跑评测
    report.render    出报告

模块职责：
    schema        数据结构与能力维度定义
    config        配置与凭据加载
    client        统一 LLM 客户端（含离线 Mock 引擎）
    sut           被测系统封装
    metrics       三层判定：规则 / 裁判 / Agent 轨迹
    bias          偏差控制
    calibration   裁判与人工标注的一致性校准
    pipeline      流水线编排
    analysis      短板归因与数据策略建议
    report        报告生成
"""

from .schema import DIMENSIONS, Response, RunSummary, Sample, Turn, Usage, Verdict

__version__ = "1.0.0"

__all__ = [
    "DIMENSIONS",
    "Sample",
    "Response",
    "Verdict",
    "Turn",
    "Usage",
    "RunSummary",
    "__version__",
]

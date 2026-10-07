# -*- coding: utf-8 -*-
"""
核心数据结构。

整个框架里流动的对象只有五种，全部是 dataclass，职责单一：
    Sample       评测集里的一道题
    Usage        token 与费用统计
    Response     某个被测模型对某道题的一次回答
    Verdict      某个指标对某次回答的判定结果
    Turn         一条完整记录（题 + 回答 + 全部判定），落盘的基本单位
    RunSummary   一次评测运行的汇总统计

另外这里定义了「能力维度表」—— 它不是装饰，是整个评测体系的对齐依据。
每个维度都标注了它对应字节 Seed 评测体系里的哪一条，报告会原样带上这张表。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ------------------------------------------------------------------ 能力维度
# 对齐依据见 docs/评测体系.md，这里只留最关键的溯源信息
DIMENSIONS: dict[str, dict[str, str]] = {
    "knowledge": {
        "name": "知识与长尾事实",
        "seed_anchor": "Seed2.0 模型卡明确点名的两个短板之一：长尾知识缺口",
    },
    "instruction": {
        "name": "复杂指令遵循",
        "seed_anchor": "Seed2.0 模型卡明确点名的两个短板之一：复杂多步指令失败",
    },
    "reasoning": {
        "name": "推理与数学",
        "seed_anchor": "Seed 评测三原则之「推进智能前沿」，测峰值能力而非平均能力",
    },
    "coding": {
        "name": "代码与 Vibe Coding",
        "seed_anchor": "Seed2.0 四维评测框架之一：从一句话需求到可运行产物",
    },
    "long_context": {
        "name": "上下文学习",
        "seed_anchor": "Seed2.0 四维评测框架之一：长文档中提取并推理",
    },
    "agent": {
        "name": "Agent 工具调用",
        "seed_anchor": "Seed1.8 / Seed2.1 的核心能力方向，参照 τ-bench 与 GAIA 范式",
    },
    "multi_agent": {
        "name": "多智能体协作",
        "seed_anchor": "Seed2.1 提出「模型价值要与 harness、工具、产品环境结合来评估」；参照 MultiAgentBench 的协作/竞争范式",
    },
    "knowledge_graph": {
        "name": "知识图谱能力",
        "seed_anchor": "Seed 评测三原则之「转向真实场景」中的结构化知识场景：三元组抽取与多跳推理",
    },
    "realworld": {
        "name": "真实世界任务",
        "seed_anchor": "Seed2.0 四维评测框架之一：端到端任务完成",
    },
    "multimodal": {
        "name": "多模态理解",
        "seed_anchor": "AI 数据与安全团队为 Seed 提供「跨模态数据服务」，跨模态是评测标准的一环",
    },
    "multiturn": {
        "name": "多轮对话与信息保持",
        "seed_anchor": "《LLMs Get Lost in Multi-Turn Conversation》：多轮下能力与可靠性双降，是当前模型系统性短板",
    },
    "office": {
        "name": "办公与生产力任务",
        "seed_anchor": "Seed2.1 Model Card 众测任务类型分布中占比最高的一类（Office & Productivity 17.8%）",
    },
    "safety": {
        "name": "安全与鲁棒",
        "seed_anchor": "AI 数据与安全团队职责范围：拒答、幻觉、提示注入",
    },
}

# 判定的三个层级，成本与可信度依次升高
TIERS = ("deterministic", "llm_judge", "human")

TASK_TYPES = (
    "mcq",
    "open_qa",
    "instruction",
    "coding",
    "agent_tool",
    "multi_agent",
    "multi_turn",
    "refusal",
    "multimodal",
)
DIFFICULTIES = ("easy", "medium", "hard")


def stable_rand(*parts: Any) -> float:
    """确定性伪随机数，落在 [0,1)。

    评测结果必须可复现，所以不能依赖 random 全局状态。
    同一组输入永远得到同一个数，跨进程、跨机器一致。
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# ------------------------------------------------------------------ 数据结构
@dataclass
class Sample:
    """评测集里的一道题。"""

    id: str
    dimension: str
    task_type: str
    prompt: str
    system: str | None = None
    reference: str | None = None          # 参考答案 / gold
    context: str | None = None            # 长上下文材料
    constraints: list[dict[str, Any]] = field(default_factory=list)
    difficulty: str = "medium"
    images: list[str] = field(default_factory=list)   # 图片路径，相对 datasets/ 解析
    turns: list[str] = field(default_factory=list)    # 多轮题的用户消息序列，按轮次
    source: str = ""                      # 题目来源文件，便于溯源
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "") -> "Sample":
        dim = raw.get("dimension") or "knowledge"
        if dim not in DIMENSIONS:
            raise ValueError(f"未知能力维度 {dim!r}（样本 id={raw.get('id')}）")
        task_type = raw.get("task_type") or "open_qa"
        if task_type not in TASK_TYPES:
            raise ValueError(f"未知题型 {task_type!r}（样本 id={raw.get('id')}）")
        difficulty = str(raw.get("difficulty", "medium"))
        if difficulty not in DIFFICULTIES:
            raise ValueError(
                f"未知难度 {difficulty!r}（样本 id={raw.get('id')}，可选 {'/'.join(DIFFICULTIES)}）"
            )
        return cls(
            id=str(raw["id"]),
            dimension=dim,
            task_type=task_type,
            prompt=raw.get("prompt", ""),
            system=raw.get("system"),
            reference=raw.get("reference"),
            context=raw.get("context"),
            constraints=list(raw.get("constraints") or []),
            difficulty=difficulty,
            images=list(raw.get("images") or []),
            turns=[str(t) for t in (raw.get("turns") or [])],
            source=source,
            meta=dict(raw.get("meta") or {}),
        )

    @property
    def dimension_name(self) -> str:
        return DIMENSIONS.get(self.dimension, {}).get("name", self.dimension)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """token 用量与估算费用。费用按 configs/models.yaml 里的单价折算。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )


@dataclass
class Response:
    """被测模型对一道题的一次回答。

    trace 与 messages 是两个不同层面的过程记录，不要混：
        trace    Agent 的工具调用轨迹（调了什么工具、参数是什么、返回什么）
        messages 多智能体的协作消息（谁在第几轮说了什么）
    单模型任务只有 trace；多智能体任务两者都有。
    """

    sample_id: str
    sut_id: str
    text: str = ""
    latency_s: float = 0.0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    attempt_index: int = 0     # 第几次重复运行，用于 pass^k

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d


@dataclass
class Verdict:
    """某个指标对某次回答的判定。"""

    sample_id: str
    sut_id: str
    metric: str                  # 指标名，如 faithfulness / instruction_rules / agent_trajectory
    tier: str                    # deterministic | llm_judge | human
    score: float | None = None   # 统一归一到 1-5
    passed: bool | None = None   # 是否通过（阈值由指标自定）
    decisive: bool = True        # 是否作为「通过率」的判定依据
    detail: dict[str, Any] = field(default_factory=dict)
    judge_id: str | None = None
    latency_s: float = 0.0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    attempt_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d


@dataclass
class Turn:
    """一条完整记录：一道题 × 一个模型 × 一次尝试，连同全部判定。"""

    sample: Sample
    response: Response
    verdicts: list[Verdict] = field(default_factory=list)

    def verdicts_for(self, metric: str) -> Verdict | None:
        for v in self.verdicts:
            if v.metric == metric:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "response": self.response.to_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


@dataclass
class RunSummary:
    """一次评测运行的元信息与汇总。"""

    run_id: str
    suite: str
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    mock: bool = False                 # 是否包含 mock 产出（报告必须据此标注）
    n_samples: int = 0
    n_turns: int = 0
    suts: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    tier_counts: dict[str, int] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunSummary":
        raw = dict(raw)
        raw["usage"] = Usage(**raw.get("usage", {}))
        return cls(**raw)


def dumps(obj: Any, indent: int | None = None) -> str:
    """统一的 JSON 序列化，中文不转义。"""
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)

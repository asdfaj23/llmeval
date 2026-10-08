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
#
# 分类用**两条轴**，这是维度边界能划清的前提（详见 docs/维度边界定义.md）：
#     axis = "capability"  能力轴：考的是哪种能力（知识 / 推理 / 指令 / 代码 / 长文本 / 图谱 / 安全）
#     axis = "form"        形态轴：交付物是哪种形态（多模态 / 多轮 / 单智能体 / 多智能体 / 办公 / 真实任务）
#
# 为什么必须分轴：像「办公任务」这种形态维度天然会用到推理、多模态等能力，
# 如果把「用到了什么能力」当成归属依据，维度就会互相重叠、统计也失去意义。
# 所以每个维度都额外声明两件事：
#     sole_judge           这一维度**唯一站得住的判定依据**（归属看它，不看题目用到了什么能力）
#     belongs_elsewhere    不属于本维度的情形 → 应归到哪个维度
DIMENSIONS: dict[str, dict[str, str]] = {
    "knowledge": {
        "name": "知识与长尾事实",
        "axis": "capability",
        "seed_anchor": "Seed2.0 模型卡明确点名的两个短板之一：长尾知识缺口",
        "benchmark": "MMLU / MMLU-Pro（知识与推理）、SimpleQA（事实性）",
        "metrics": "accuracy（选择题程序比对）、裁判参考对齐分",
        "sole_judge": "单跳事实回忆：答案是否命中唯一事实（选择题比对字母，开放题比对要点）",
        "belongs_elsewhere": "需要沿关系做两跳以上推理 → knowledge_graph；难点是要算很久 → reasoning",
    },
    "instruction": {
        "name": "复杂指令遵循",
        "axis": "capability",
        "seed_anchor": "Seed2.0 模型卡明确点名的两个短板之一：复杂多步指令失败",
        "benchmark": "IFEval（可验证指令遵循）、MT-Bench 的指令类子集",
        "metrics": "约束满足率、硬 / 软违规数（全部规则判定，零成本）",
        "sole_judge": "显式约束是否逐条被满足（字数 / 行数 / 必含 / 禁含 / JSON 结构 / 正则 / 语言 / 无开场白）",
        "belongs_elsewhere": "难点是要想很久而不是约束多 → reasoning；约束由某个产出物形态定义（如表格列名） → office",
    },
    "reasoning": {
        "name": "推理与数学",
        "axis": "capability",
        "seed_anchor": "Seed 评测三原则之「推进智能前沿」，测峰值能力而非平均能力",
        "benchmark": "GSM8K / MATH（数学推理）、GPQA（专家级问答）",
        "metrics": "数值精确匹配（相对误差容差）、裁判质量分",
        "sole_judge": "推导链条是否成立、结论是否算对（数值题按容差精确比对）",
        "belongs_elsewhere": "难点是约束多而非要想 → instruction；必须读完整长材料才能解 → long_context；用到多角色协作 → multi_agent",
    },
    "coding": {
        "name": "代码与 Vibe Coding",
        "axis": "capability",
        "seed_anchor": "Seed2.0 四维评测框架之一：从一句话需求到可运行产物",
        "benchmark": "HumanEval / MBPP（函数级）、SWE-bench（仓库级修复）",
        "metrics": "pass@1（配 tests 时子进程真实执行）、裁判质量分",
        "sole_judge": "代码是否跑通测试用例（真实执行，只看通过率）",
        "belongs_elsewhere": "只是问代码相关的知识、不要求产出可运行代码 → knowledge",
    },
    "long_context": {
        "name": "上下文学习",
        "axis": "capability",
        "seed_anchor": "Seed2.0 四维评测框架之一：长文档中提取并推理",
        "benchmark": "LongBench、∞Bench（长上下文理解）",
        "metrics": "事实一致率、裁判参考对齐分",
        "sole_judge": "答案是否必须从给出的长材料里取（材料一撤就答不出，说明确实用了长上下文）",
        "belongs_elsewhere": "材料很短、靠常识就能答 → knowledge；材料是图表/图文 → multimodal",
    },
    "knowledge_graph": {
        "name": "知识图谱能力",
        "axis": "capability",
        "seed_anchor": "Seed 评测三原则之「转向真实场景」中的结构化知识场景：三元组抽取与多跳推理",
        "benchmark": "WebQSP、CWQ（多跳知识库问答）",
        "metrics": "三元组精确率 / 召回率 / F1、关系方向错误数、多跳路径命中率",
        "sole_judge": "抽出的三元组与关系路径是否与标准答案一致（可逐条比对，含方向）",
        "belongs_elsewhere": "拆不出三元组、单跳就能答 → knowledge",
    },
    "safety": {
        "name": "安全与鲁棒",
        "axis": "capability",
        "seed_anchor": "AI 数据与安全团队职责范围：拒答、幻觉、提示注入",
        "benchmark": "SafetyBench（安全知识）、JailbreakBench（越狱与注入）",
        "metrics": "拒答正确率、注入抵抗率、agent_safety 违规布尔（禁用工具是否被调用）",
        "sole_judge": "该拒答的是否拒答、是否被藏在材料里的指令带偏、是否调用了禁用工具",
        "belongs_elsewhere": "普通事实性错误（非安全边界问题） → knowledge 或事实性指标",
    },
    "multimodal": {
        "name": "多模态理解",
        "axis": "form",
        "seed_anchor": "AI 数据与安全团队为 Seed 提供「跨模态数据服务」，跨模态是评测标准的一环",
        "benchmark": "MMMU（学科多模态）、ChartQA（图表问答）",
        "metrics": "关键事实命中率、裁判参考对齐分",
        "sole_judge": "答案是否依赖**图像内容**（把图去掉就答不出）",
        "belongs_elsewhere": "纯文本即可作答 → 该能力对应的维度（如推理看图表数值 → reasoning）",
    },
    "multiturn": {
        "name": "多轮对话与信息保持",
        "axis": "form",
        "seed_anchor": "《LLMs Get Lost in Multi-Turn Conversation》：多轮下能力与可靠性双降，是当前模型系统性短板",
        "benchmark": "MT-Bench（多轮设置）、《LLMs Get Lost in Multi-Turn Conversation》的复现实验",
        "metrics": "最终答案正确率、早轮约束保持率、认知更新（纠正）命中率",
        "sole_judge": "早轮约束是否被保持、以及被更新的旧值是否真的被替换（逐条核验）",
        "belongs_elsewhere": "单轮就能问清的（哪怕内容很复杂）不属于这里；多轮只是形式、判定不看轮次的，也不属于这里",
    },
    "agent": {
        "name": "Agent 工具调用",
        "axis": "form",
        "seed_anchor": "Seed1.8 / Seed2.1 的核心能力方向，参照 τ-bench 与 GAIA 范式",
        "benchmark": "τ-bench（工具智能体可靠性）、GAIA（通用助手任务）",
        "metrics": "tool_precision / recall / F1、redundant_steps、missing_tools、pass@k 与 pass^k",
        "sole_judge": "工具调用序列是否选对、参数是否给全、步数是否冗余、跨重复运行的可靠性",
        "belongs_elsewhere": "不需要调工具、直接回答即可 → 该能力对应的维度；出现多角色分工/审查 → multi_agent",
    },
    "multi_agent": {
        "name": "多智能体协作",
        "axis": "form",
        "seed_anchor": "Seed2.1 提出「模型价值要与 harness、工具、产品环境结合来评估」；参照 MultiAgentBench 的协作/竞争范式",
        "benchmark": "MultiAgentBench（多智能体协作与竞争）",
        "metrics": "角色覆盖率、审查否决率、修订闭环率、无效轮占比、协作流程确定性指标",
        "sole_judge": "协作过程是否成立：角色是否覆盖、审查是否真的否決过、否决后是否真的修订",
        "belongs_elsewhere": "只有一个智能体在调工具 → agent；协作只是包装、判定只看最终答案的 → 该能力对应的维度",
    },
    "office": {
        "name": "办公与生产力任务",
        "axis": "form",
        "seed_anchor": "Seed2.1 Model Card 众测任务类型分布中占比最高的一类（Office & Productivity 17.8%）",
        "benchmark": "Seed2.1 众测任务类型分布（Office & Productivity 占比最高）",
        "metrics": "任务完成度、格式合规率、裁判质量分",
        "sole_judge": "**办公产出物的结构与格式**是否合规：表格列名列数行数、JSON 字段、字数行数、要点是否齐全",
        "belongs_elsewhere": "判定落在推理上（方案评审、数据归因分析） → reasoning；判定落在读图上 → multimodal；判定落在算得对不对上 → reasoning",
    },
    "realworld": {
        "name": "真实世界任务",
        "axis": "form",
        "seed_anchor": "Seed2.0 四维评测框架之一：端到端任务完成",
        "benchmark": "Arena-Hard（真实用户查询）、GAIA（端到端助手任务）",
        "metrics": "多维加权裁判质量分（correctness / completeness / relevance / clarity）",
        "sole_judge": "端到端任务是否被真正完成（**必须组合多种能力，拆开后测不出真实水平**）",
        "belongs_elsewhere": "单靠某一种能力就能解的题，请归到对应维度 —— 本维度是兜底，不能当「不知道归哪就丢这里」的垃圾桶",
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
    containment: dict[str, Any] = field(default_factory=dict)

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

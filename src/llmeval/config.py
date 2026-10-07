# -*- coding: utf-8 -*-
"""
配置与凭据加载。

三层配置，各管各的：
    configs/models.yaml    有哪些模型可调、用什么凭据、运行时参数
    configs/suites/*.yaml  一次评测跑什么：数据集 × 指标 × 模型矩阵
    configs/rubrics/*.yaml 评分表本身

凭据只从环境变量或 .env 读，绝不写进任何配置文件 —— 这是硬约束。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少依赖 PyYAML。请执行：pip install PyYAML\n（本项目只依赖这一个三方包）"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SUITE_DIR = CONFIG_DIR / "suites"
RUBRIC_DIR = CONFIG_DIR / "rubrics"
DATASET_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------------ .env
def load_env_file(path: Path | None = None, overwrite: bool = True) -> int:
    """极简 .env 解析。不引入 python-dotenv，因为这点正则就够用了。

    规则：KEY=VALUE，# 开头为注释，值两侧的引号会被去掉。

    overwrite 默认为 True，即**项目 .env 覆盖系统环境变量**。

    这个默认值是踩过坑之后才定的：本机环境变量里残留着早就失效的旧 key，
    如果 .env 不覆盖，就会出现「明明在项目里填了新 key，一跑就 401」——
    更糟的是报错信息里显示的是旧 key 的指纹，跟刚填的对不上，
    排查时很容易往错的方向想。对「项目自带 .env」的本地工具来说，
    项目配置优先更符合直觉。
    """
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key or not value:
            continue
        if overwrite or key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def credential_fingerprint(api_key_env: str) -> str:
    """凭据指纹：只显示末 4 位和长度，用于确认"当前实际用的是哪把 key"。

    不打印完整 key，但足够定位「填了 A 结果用了 B」这类问题。
    """
    value = os.environ.get(api_key_env, "") if api_key_env else ""
    if not value:
        return "未设置"
    return f"...{value[-4:]}（{len(value)} 字符）"


# ------------------------------------------------------------------ 模型配置
@dataclass
class Provider:
    key: str
    label: str
    base_url: str
    api_key_env: str

    @property
    def is_mock(self) -> bool:
        return self.key == "mock"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class ModelSpec:
    """一个可调用的模型端点。sut 与 judge 共用这个结构。"""

    id: str
    label: str
    provider: Provider
    model: str
    role: str = "sut"                 # sut | judge
    temperature: float = 0.0
    max_tokens: int = 2048
    enabled: bool = True
    price_in: float = 0.0             # 每百万输入 token 的美元价
    price_out: float = 0.0
    strategy: str = "plan_execute"    # 多智能体被测对象的协作范式：plan_execute | react

    @property
    def is_mock(self) -> bool:
        return self.provider.is_mock

    @property
    def usable(self) -> bool:
        """能不能真的发起调用。"""
        if not self.enabled:
            return False
        if self.is_mock:
            return True
        return bool(self.model and self.provider.api_key)

    @property
    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "配置里 enabled: false"
        if self.is_mock:
            return ""
        if not self.model:
            return "尚未填写 model"
        if not self.provider.api_key:
            return f"缺少环境变量 {self.provider.api_key_env}"
        return ""

    @property
    def family(self) -> str:
        """模型族，用于校验裁判与 SUT 是否同族（自偏好偏差控制）。"""
        return self.provider.key

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1_000_000 * self.price_in
            + completion_tokens / 1_000_000 * self.price_out,
            8,
        )


@dataclass
class RuntimeCfg:
    concurrency: int = 4
    timeout_s: int = 60
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    cache: bool = True
    cache_dir: str = "outputs/.cache"


@dataclass
class ModelsConfig:
    providers: dict[str, Provider] = field(default_factory=dict)
    suts: list[ModelSpec] = field(default_factory=list)
    judges: list[ModelSpec] = field(default_factory=list)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    def get_sut(self, model_id: str) -> ModelSpec:
        for m in self.suts:
            if m.id == model_id:
                return m
        raise KeyError(f"models.yaml 里找不到被测模型 id={model_id!r}")

    def get_judge(self, model_id: str) -> ModelSpec:
        for m in self.judges:
            if m.id == model_id:
                return m
        raise KeyError(f"models.yaml 里找不到裁判模型 id={model_id!r}")

    def active_suts(self, ids: list[str] | None = None) -> list[ModelSpec]:
        """挑出可用的 SUT。显式指定 ids 时按 ids 取，否则取全部 enabled 的。"""
        pool = [self.get_sut(i) for i in ids] if ids else list(self.suts)
        return [m for m in pool if m.usable]

    def active_judges(self, ids: list[str] | None = None) -> list[ModelSpec]:
        pool = [self.get_judge(i) for i in ids] if ids else list(self.judges)
        return [m for m in pool if m.usable]

    def check_cross_family(self, suts: list[ModelSpec], judges: list[ModelSpec]) -> list[str]:
        """自偏好偏差检查：裁判与 SUT 同族会系统性抬高自己的分数。"""
        warnings: list[str] = []
        for j in judges:
            same = [s.id for s in suts if s.family == j.family and not j.is_mock]
            if same:
                warnings.append(
                    f"裁判 {j.id} 与被测 {same} 属于同一模型族（{j.family}），"
                    "存在自偏好偏差风险，建议换用其他族的裁判"
                )
        return warnings


def load_models(path: Path | None = None) -> ModelsConfig:
    """读 configs/models.yaml，并把环境变量里的凭据绑定上去。"""
    path = path or (CONFIG_DIR / "models.yaml")
    raw = _read_yaml(path)

    providers: dict[str, Provider] = {}
    for key, cfg in (raw.get("providers") or {}).items():
        providers[key] = Provider(
            key=key,
            label=cfg.get("label", key),
            base_url=cfg.get("base_url", ""),
            api_key_env=cfg.get("api_key_env", ""),
        )

    def _build(items: list[dict[str, Any]], role: str) -> list[ModelSpec]:
        out: list[ModelSpec] = []
        for item in items or []:
            pkey = item.get("provider")
            if pkey not in providers:
                raise ValueError(f"模型 {item.get('id')} 引用了未定义的 provider: {pkey}")
            out.append(
                ModelSpec(
                    id=item["id"],
                    label=item.get("label", item["id"]),
                    provider=providers[pkey],
                    model=item.get("model", "") or "",
                    role=role,
                    temperature=float(item.get("temperature", 0.0)),
                    max_tokens=int(item.get("max_tokens", 2048)),
                    enabled=bool(item.get("enabled", True)),
                    price_in=float(item.get("price_in", 0.0) or 0.0),
                    price_out=float(item.get("price_out", 0.0) or 0.0),
                    strategy=str(item.get("strategy", "plan_execute")),
                )
            )
        return out

    rt_raw = raw.get("runtime") or {}
    runtime = RuntimeCfg(
        concurrency=int(rt_raw.get("concurrency", 4)),
        timeout_s=int(rt_raw.get("timeout_s", 60)),
        max_retries=int(rt_raw.get("max_retries", 3)),
        retry_backoff_s=float(rt_raw.get("retry_backoff_s", 2.0)),
        cache=bool(rt_raw.get("cache", True)),
        cache_dir=rt_raw.get("cache_dir", "outputs/.cache"),
    )

    return ModelsConfig(
        providers=providers,
        suts=_build(raw.get("suts"), "sut"),
        judges=_build(raw.get("judges"), "judge"),
        runtime=runtime,
    )


# ------------------------------------------------------------------ 套件配置
@dataclass
class MetricPlan:
    """某个维度上的指标编排。"""

    rubric: str = "helpfulness"
    deterministic: bool = True
    llm_judge: bool = True
    reference_aware: bool = False
    repeats: int = 1              # 重复次数，>1 时额外计算 pass^k
    pass_k: int = 1               # 一致性口径：k 次全对才算通过


@dataclass
class SuiteConfig:
    id: str
    name: str
    description: str = ""
    datasets: list[str] = field(default_factory=list)
    suts: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    default_plan: MetricPlan = field(default_factory=MetricPlan)
    per_dimension: dict[str, MetricPlan] = field(default_factory=dict)
    max_samples_per_dataset: int = 0      # 0 = 不限制
    judge_sample_ratio: float = 1.0       # LLM 裁判抽样比例；生产建议 0.1-0.2
    seed: int = 20260914

    def plan_for(self, dimension: str) -> MetricPlan:
        return self.per_dimension.get(dimension, self.default_plan)


def _parse_plan(raw: dict[str, Any] | None, base: MetricPlan | None = None) -> MetricPlan:
    raw = raw or {}
    base = base or MetricPlan()
    return MetricPlan(
        rubric=raw.get("rubric", base.rubric),
        deterministic=bool(raw.get("deterministic", base.deterministic)),
        llm_judge=bool(raw.get("llm_judge", base.llm_judge)),
        reference_aware=bool(raw.get("reference_aware", base.reference_aware)),
        repeats=int(raw.get("repeats", base.repeats)),
        pass_k=int(raw.get("pass_k", base.pass_k)),
    )


def load_suite(name_or_path: str) -> SuiteConfig:
    """按套件名或路径加载。传 general 会自动找 configs/suites/general.yaml。"""
    p = Path(name_or_path)
    if not p.exists():
        candidate = SUITE_DIR / f"{name_or_path}.yaml"
        if not candidate.exists():
            available = ", ".join(sorted(x.stem for x in SUITE_DIR.glob("*.yaml"))) or "（无）"
            raise FileNotFoundError(f"找不到套件 {name_or_path!r}。可用：{available}")
        p = candidate

    raw = _read_yaml(p)
    default_plan = _parse_plan(raw.get("metrics"))
    per_dim = {
        dim: _parse_plan(cfg, default_plan)
        for dim, cfg in (raw.get("per_dimension") or {}).items()
    }
    limits = raw.get("limits") or {}

    return SuiteConfig(
        id=raw.get("id", p.stem),
        name=raw.get("name", p.stem),
        description=raw.get("description", ""),
        datasets=list(raw.get("datasets") or []),
        suts=list(raw.get("suts") or []),
        judges=list(raw.get("judges") or []),
        default_plan=default_plan,
        per_dimension=per_dim,
        max_samples_per_dataset=int(limits.get("max_samples_per_dataset", 0)),
        judge_sample_ratio=float((raw.get("metrics") or {}).get("judge_sample_ratio", 1.0)),
        seed=int(raw.get("seed", 20260914)),
    )


def load_rubric(rubric_id: str) -> dict[str, Any]:
    """读评分表。整套 YAML 原样返回，构建 prompt 时再取需要的部分。"""
    path = RUBRIC_DIR / f"{rubric_id}.yaml"
    if not path.exists():
        avail = ", ".join(sorted(x.stem for x in RUBRIC_DIR.glob("*.yaml"))) or "（无）"
        raise FileNotFoundError(f"找不到评分表 {rubric_id!r}。可用：{avail}")
    return _read_yaml(path)


def list_suites() -> list[str]:
    return sorted(x.stem for x in SUITE_DIR.glob("*.yaml"))


def list_rubrics() -> list[str]:
    return sorted(x.stem for x in RUBRIC_DIR.glob("*.yaml"))


# ------------------------------------------------------------------ 评测集
def load_dataset(rel_path: str) -> list[Any]:
    """读一个 JSONL 评测集。每条一行，非法行直接报错而不是静默跳过 —— 静默跳过会让
    评测集悄悄缩水，而你不会知道。"""
    from .schema import Sample

    p = Path(rel_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在：{p}")

    samples: list[Sample] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = __import__("json").loads(line)
        except Exception as exc:
            raise ValueError(f"{p.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        samples.append(Sample.from_dict(raw, source=p.name))

    ids = [s.id for s in samples]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{p.name} 存在重复样本 id：{sorted(dupes)}")
    return samples


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 顶层必须是 mapping")
    return data

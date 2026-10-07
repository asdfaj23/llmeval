# -*- coding: utf-8 -*-
"""
评测流水线。

把散落的组件串成一条可复现的链路：

    读配置 → 读评测集 → 造被测对象 → 逐题推理
           → 三层判定（规则 / Agent 轨迹 / LLM 裁判）
           → 汇总统计 → 落盘

三个贯穿始终的设计要求：

1. 可复现
   同一份配置 + 同一份评测集，跑到哪里结果都该一样。
   mock 引擎是确定性的，真实模型则靠 temperature=0 + 结果缓存来逼近。

2. 不因单点失败而中断
   一次请求失败只该让这一条记录标上 error，不该让整轮评测垮掉。
   评测跑一半崩了，是最浪费时间的失败方式。

3. 结果可追溯
   每条记录都保留：题面、原始回答、每个判定的完整依据、用的哪个裁判。
   事后被质疑分数时，要能一条条翻回去看。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from .client import LLMClient, run_parallel
from .config import ModelSpec, SuiteConfig, load_rubric, load_suite
from .metrics import (
    collect_agent_metrics,
    collect_coding_exec,
    collect_collaboration,
    collect_collab_process,
    collect_deterministic,
    collect_kg_metrics,
    collect_multiturn_metrics,
    evaluate_pairwise,
    evaluate_panel,
)
from .collab.attribution import attribute_failure, attribution_histogram
from .collab.compare import collect_compare_metrics
from .schema import Response, RunSummary, Sample, Turn, Usage, Verdict, stable_rand
from .sut import build_sut, kind_for_task
from .textutil import make_snippet


@dataclass
class Job:
    """一个最小执行单元：某道题 × 某个模型 × 第几次尝试。"""

    sample: Sample
    spec: ModelSpec
    attempt_index: int = 0


@dataclass
class RunResult:
    run_id: str
    suite: SuiteConfig
    turns: list[Turn]
    summary: RunSummary
    run_dir: Path
    pairwise: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, ModelSpec] = field(default_factory=dict)

    def turns_for_sut(self, sut_id: str) -> list[Turn]:
        return [t for t in self.turns if t.response.sut_id == sut_id]


@lru_cache(maxsize=32)
def _rubric(rubric_id: str) -> dict[str, Any]:
    return load_rubric(rubric_id)


class Runner:
    """一次评测运行的编排器。"""

    def __init__(
        self,
        suite_name: str,
        *,
        sut_ids: list[str] | None = None,
        limit: int = 0,
        pairwise: bool = False,
        out_root: Path | None = None,
        progress: Callable[[str], None] | None = None,
        models: Any = None,
    ) -> None:
        cfg.load_env_file()
        self.suite = load_suite(suite_name)
        # models 允许注入，这样测试可以固定用 mock 模型，
        # 不受用户 configs/models.yaml 当前状态的影响（既不花钱，也不会因关掉 mock 而挂）
        self.models = models if models is not None else cfg.load_models()
        self.runtime = self.models.runtime
        self.limit = limit
        self.want_pairwise = pairwise
        self.out_root = out_root or cfg.OUTPUT_DIR
        self._say = progress or (lambda _msg: None)

        self.suts = self.models.active_suts(self.suite.suts or sut_ids)
        self.judges = self.models.active_judges(self.suite.judges)
        self.cross_family_warnings = self.models.check_cross_family(self.suts, self.judges)

        self._clients: dict[str, LLMClient] = {}

    # ------------------------------------------------------------ 准备
    def _client(self, spec: ModelSpec) -> LLMClient:
        if spec.id not in self._clients:
            self._clients[spec.id] = LLMClient(spec, self.runtime)
        return self._clients[spec.id]

    def _find_spec(self, model_id: str) -> ModelSpec | None:
        """在已配置的 suts + judges 里按 id 找模型端点（异构组网用）。"""
        for m in self.models.suts + self.models.judges:
            if m.id == model_id:
                return m
        return None

    def _resolve_role_clients(self, sample: Sample) -> dict[str, LLMClient]:
        """解析题面 meta.role_map，把某些角色路由到不同的客户端（多模型异构）。"""
        role_map = (sample.meta or {}).get("role_map") or {}
        if not role_map:
            return {}
        out: dict[str, LLMClient] = {}
        for role, mid in role_map.items():
            spec = self._find_spec(str(mid))
            if spec is not None and spec.usable:
                out[role] = self._client(spec)
        return out

    def _load_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for rel in self.suite.datasets:
            loaded = cfg.load_dataset(rel)
            if self.suite.max_samples_per_dataset > 0:
                loaded = loaded[: self.suite.max_samples_per_dataset]
            samples.extend(loaded)
            self._say(f"  载入 {rel}：{len(loaded)} 题")
        if self.limit > 0:
            samples = samples[: self.limit]
        if not samples:
            raise SystemExit("评测集为空，检查套件里的 datasets 配置")
        return samples

    # ------------------------------------------------------------ 预检
    def preflight(self) -> None:
        """开跑前的自检。宁可在这里报错，也不要跑到一半才发现配置写错了。"""
        lines = []
        for m in self.models.suts:
            mark = "✓" if m in self.suts else "·"
            reason = "" if m in self.suts else f"（跳过：{m.unavailable_reason}）"
            lines.append(f"  被测 {mark} {m.id}{reason}")
        for m in self.models.judges:
            mark = "✓" if m in self.judges else "·"
            reason = "" if m in self.judges else f"（跳过：{m.unavailable_reason}）"
            lines.append(f"  裁判 {mark} {m.id}{reason}")

        self._say("模型清单：")
        for line in lines:
            self._say(line)

        if not self.suts:
            raise SystemExit("没有任何可用的被测模型。请在 configs/models.yaml 里启用至少一个，或填上 API key。")
        if not self.judges:
            self._say("  ! 没有可用的裁判模型，将只做规则判定，不产生质量分")
        for w in self.cross_family_warnings:
            self._say(f"  ! {w}")

    # ------------------------------------------------------------ 单题执行
    def _run_job(self, job: Job) -> Turn:
        sample, spec = job.sample, job.spec
        plan = self.suite.plan_for(sample.dimension)

        sut = build_sut(spec, self._client(spec), kind_for_task(sample.task_type))

        # 异构组网：题面 meta.role_map 把某个角色映射到另一个模型端点
        if sample.task_type == "multi_agent" and hasattr(sut, "set_role_clients"):
            role_clients = self._resolve_role_clients(sample)
            if role_clients:
                sut.set_role_clients(role_clients)

        response = sut.run(sample, attempt_index=job.attempt_index)

        verdicts: list[Verdict] = []
        verdicts.extend(collect_deterministic(sample, response))
        verdicts.extend(collect_coding_exec(sample, response))
        verdicts.extend(collect_multiturn_metrics(sample, response))
        verdicts.extend(collect_agent_metrics(sample, response))
        verdicts.extend(collect_collaboration(sample, response))
        verdicts.extend(collect_collab_process(sample, response))
        verdicts.extend(collect_compare_metrics(sample, response))
        verdicts.extend(collect_kg_metrics(sample, response))

        if plan.llm_judge and self.judges and self._judge_selected(sample, spec.id):
            rubric = _rubric(plan.rubric)
            reference = (
                sample.reference
                if (plan.reference_aware or sample.task_type == "mcq")
                else None
            )
            # 所有启用的裁判各独立打一遍，事后按中位数聚合 ——
            # 不是让它们讨论，讨论会破坏独立性，评审团也就失去意义了
            panel = [(self._client(judge), judge) for judge in self.judges]
            verdicts.extend(
                evaluate_panel(
                    sample, response, plan.rubric, rubric, panel, reference=reference
                )
            )

        return Turn(sample=sample, response=response, verdicts=verdicts)

    def _judge_selected(self, sample: Sample, sut_id: str) -> bool:
        """裁判抽样。生产环境建议 5%-20%，全量判会贵得离谱。"""
        ratio = self.suite.judge_sample_ratio
        if ratio >= 1.0:
            return True
        return stable_rand(sample.id, sut_id, self.suite.seed, "judge_sample") < ratio

    # ------------------------------------------------------------ 主流程
    def run(self) -> RunResult:
        started = datetime.now()
        mock_used = any(m.is_mock for m in self.suts + self.judges)
        run_id = f"{self.suite.id}_{started:%Y%m%d_%H%M%S}" + ("_mock" if mock_used else "")

        self._say(f"套件：{self.suite.name}（{self.suite.id}）")
        self.preflight()

        samples = self._load_samples()
        self._say(f"评测集共 {len(samples)} 题，被测模型 {len(self.suts)} 个")

        jobs: list[Job] = []
        for sample in samples:
            repeats = max(1, self.suite.plan_for(sample.dimension).repeats)
            for spec in self.suts:
                for attempt in range(repeats):
                    jobs.append(Job(sample=sample, spec=spec, attempt_index=attempt))

        self._say(f"共 {len(jobs)} 个执行单元，并发 {self.runtime.concurrency}")

        counter = {"done": 0, "errors": 0}

        def on_done(_idx: int, result: Any) -> None:
            counter["done"] += 1
            if isinstance(result, Turn) and not result.response.ok:
                counter["errors"] += 1
            if counter["done"] % max(1, len(jobs) // 10) == 0 or counter["done"] == len(jobs):
                self._say(f"  进度 {counter['done']}/{len(jobs)}")

        results = run_parallel(
            jobs,
            lambda _i, job: self._run_job(job),
            max_workers=self.runtime.concurrency,
            on_done=on_done,
        )

        turns: list[Turn] = []
        for job, res in zip(jobs, results):
            if isinstance(res, Turn):
                turns.append(res)
            else:  # worker 抛了异常，补一条带错记录，不让它消失
                turns.append(
                    Turn(
                        sample=job.sample,
                        response=Response(
                            sample_id=job.sample.id,
                            sut_id=job.spec.id,
                            error=f"{type(res).__name__}: {res}",
                            attempt_index=job.attempt_index,
                        ),
                    )
                )

        # ---- 可选：两两对比
        pairwise_records: list[dict[str, Any]] = []
        if self.want_pairwise and len(self.suts) >= 2 and self.judges:
            pairwise_records = self._run_pairwise(samples, turns)

        # ---- 汇总
        summary = self._summarize(run_id, turns, mock_used, started)
        run_dir = self._persist(run_id, turns, summary, pairwise_records)

        self._say(
            f"完成：{len(turns)} 条记录，耗时 {summary.duration_s:.1f}s，"
            f"错误 {summary.errors} 条"
        )
        return RunResult(
            run_id=run_id,
            suite=self.suite,
            turns=turns,
            summary=summary,
            run_dir=run_dir,
            pairwise=pairwise_records,
            models={m.id: m for m in self.suts + self.judges},
        )

    # ------------------------------------------------------------ 两两对比
    def _run_pairwise(self, samples: list[Sample], turns: list[Turn]) -> list[dict[str, Any]]:
        """对每道题，把各模型两两配对做换位对比，得到胜率矩阵的依据。"""
        first_attempt = [
            t for t in turns if t.response.attempt_index == 0 and t.response.ok
        ]
        by_sample: dict[str, list[Turn]] = {}
        for t in first_attempt:
            by_sample.setdefault(t.sample.id, []).append(t)

        judge_spec = self.judges[0]
        client = self._client(judge_spec)
        tasks: list[tuple[Sample, Turn, Turn]] = []
        for sample in samples:
            group = by_sample.get(sample.id) or []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    tasks.append((sample, group[i], group[j]))

        if not tasks:
            return []

        self._say(f"两两对比：{len(tasks)} 组 × 2 次换位判定")

        def _work(_i: int, item: tuple[Sample, Turn, Turn]) -> dict[str, Any]:
            sample, ta, tb = item
            plan = self.suite.plan_for(sample.dimension)
            record = evaluate_pairwise(
                sample,
                ta.response,
                tb.response,
                plan.rubric,
                _rubric(plan.rubric),
                client,
                judge_spec,
            )
            record["sample_id"] = sample.id
            record["dimension"] = sample.dimension
            return record

        out = run_parallel(tasks, _work, max_workers=self.runtime.concurrency)
        return [r for r in out if isinstance(r, dict)]

    # ------------------------------------------------------------ 汇总落盘
    def _summarize(
        self,
        run_id: str,
        turns: list[Turn],
        mock_used: bool,
        started: datetime,
    ) -> RunSummary:
        usage = Usage()
        tier_counts: dict[str, int] = {}
        errors = 0
        for t in turns:
            usage = usage + t.response.usage
            if not t.response.ok:
                errors += 1
            for v in t.verdicts:
                tier_counts[v.tier] = tier_counts.get(v.tier, 0) + 1
                usage = usage + v.usage

        finished = datetime.now()
        return RunSummary(
            run_id=run_id,
            suite=self.suite.id,
            started_at=started.isoformat(timespec="seconds"),
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 2),
            mock=mock_used,
            n_samples=len({t.sample.id for t in turns}),
            n_turns=len(turns),
            suts=[m.id for m in self.suts],
            judges=[m.id for m in self.judges],
            tier_counts=tier_counts,
            usage=usage,
            errors=errors,
        )

    def _persist(
        self,
        run_id: str,
        turns: list[Turn],
        summary: RunSummary,
        pairwise: list[dict[str, Any]],
    ) -> Path:
        run_dir = self.out_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with (run_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")

        (run_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if pairwise:
            (run_dir / "pairwise.json").write_text(
                json.dumps(pairwise, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        # 失败归因（多智能体子系统）：逐条记录诊断标签，供可重放沙箱聚合直方图
        attr_records = self._attribution_records(turns)
        if attr_records:
            (run_dir / "attribution.json").write_text(
                json.dumps(attr_records, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        self._say(f"结果已写入 {run_dir}")
        return run_dir

    def _attribution_records(self, turns: list[Turn]) -> list[dict[str, Any]]:
        """多智能体记录逐条失败归因（诊断，不计分）。"""
        out: list[dict[str, Any]] = []
        for t in turns:
            if getattr(t.sample, "task_type", "") != "multi_agent":
                continue
            tags = attribute_failure(t.sample, t.response)
            out.append(
                {
                    "sample_id": t.sample.id,
                    "sut_id": t.response.sut_id,
                    "attempt_index": t.response.attempt_index,
                    "ok": t.response.ok,
                    "tags": tags,
                }
            )
        return out


def load_run(run_dir: str | Path) -> tuple[RunSummary, list[Turn], list[dict[str, Any]], dict[str, ModelSpec]]:
    """把一次落盘的运行读回来，供报告与校准复用。"""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"运行目录不存在：{run_dir}")

    summary = RunSummary.from_dict(
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    )

    turns: list[Turn] = []
    for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        sample = Sample.from_dict(raw["sample"], source=raw["sample"].get("source", ""))
        r = raw["response"]
        response = Response(
            sample_id=r["sample_id"],
            sut_id=r["sut_id"],
            text=r.get("text", ""),
            latency_s=r.get("latency_s", 0.0),
            usage=Usage(**r.get("usage", {})),
            error=r.get("error"),
            trace=r.get("trace", []),
            messages=r.get("messages", []),
            attempts=r.get("attempts", 0),
            attempt_index=r.get("attempt_index", 0),
        )
        verdicts = []
        for v in raw.get("verdicts", []):
            v = dict(v)
            v["usage"] = Usage(**v.get("usage", {}))
            verdicts.append(Verdict(**v))
        turns.append(Turn(sample=sample, response=response, verdicts=verdicts))

    pairwise: list[dict[str, Any]] = []
    pj = run_dir / "pairwise.json"
    if pj.exists():
        pairwise = json.loads(pj.read_text(encoding="utf-8"))

    models = cfg.load_models()
    specs = {m.id: m for m in models.suts + models.judges}
    return summary, turns, pairwise, specs


def latest_run(out_root: Path | None = None) -> Path:
    root = (out_root or cfg.OUTPUT_DIR) / "runs"
    if not root.exists():
        raise FileNotFoundError(f"还没有任何运行记录：{root}")
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"运行目录为空：{root}")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def brief(text: str, limit: int = 80) -> str:
    return make_snippet(text, limit)

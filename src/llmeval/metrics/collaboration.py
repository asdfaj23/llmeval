# -*- coding: utf-8 -*-
"""
多智能体协作质量指标。

为什么多智能体要单独一套指标：

    「三个角色凑在一起」并不自动等于「比单智能体更好」。
    多智能体最常见的失效模式，都是光看最终答案看不出来的：

        reviewer 从不否决          —— 审查形同虚设，三角色退化成单人流水线
        reviewer 否决了但没人改    —— 协作断链，流程跑了但没有任何作用
        某个角色占了绝大多数轮次   —— 名为协作，实为独角戏
        该发现的问题没发现         —— 制衡机制失灵

    这些问题必须靠对协作过程的确定性判定来抓。所以本模块不依赖任何裁判，
    全部基于 Response.messages 做结构化判定 —— 免费、可复现、零偏差。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..schema import Response, Sample, Verdict

DEFAULT_ROLES = ("planner", "executor", "reviewer")


# ------------------------------------------------------------------ 角色结构
def evaluate_roles(sample: Sample, response: Response) -> Verdict | None:
    """角色覆盖与协作结构是否成立。"""
    if sample.task_type != "multi_agent":
        return None

    msgs = response.messages or []
    required = [str(r) for r in (sample.meta.get("required_roles") or DEFAULT_ROLES)]
    present = {str(m.get("role")) for m in msgs}
    covered = [r for r in required if r in present]
    coverage = len(covered) / len(required) if required else 0.0

    # Router 是独立第四角色（路由留痕 + 异构调度），只用于覆盖判定，
    # 不计入轮次上限与独角戏检测，避免异构路由被误判为"消息过多/独角戏"
    content_msgs = [m for m in msgs if str(m.get("role")) != "router"]
    role_counts = Counter(str(m.get("role")) for m in content_msgs)
    max_allowed = int(sample.meta.get("max_messages", 8))
    overrun = max(0, len(content_msgs) - max_allowed)

    # 独角戏检测：单一角色占八成以上消息，说明协作没有真的发生
    monopoly = (max(role_counts.values()) / len(content_msgs)) if content_msgs else 0.0

    score = 1 + coverage * 4
    if msgs and monopoly >= 0.8 and len(msgs) > 2:
        score = min(score, 2.0)
    score -= min(1.0, overrun * 0.25)
    score = max(1.0, min(5.0, round(score, 2)))

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="collaboration_roles",
        tier="deterministic",
        score=score,
        passed=score >= 4,
        detail={
            "required_roles": required,
            "roles_present": sorted(present),
            "missing_roles": [r for r in required if r not in present],
            "role_coverage": round(coverage, 4),
            "message_count": len(msgs),
            "role_counts": dict(role_counts),
            "monopoly_ratio": round(monopoly, 4),
            "overrun": overrun,
        },
        attempt_index=response.attempt_index,
    )


# ------------------------------------------------------------------ 审查职能
def evaluate_review(sample: Sample, response: Response) -> Verdict | None:
    """审查者是否真的在挑毛病，而不是走过场。"""
    if sample.task_type != "multi_agent":
        return None

    msgs = response.messages or []
    reviews = [m for m in msgs if str(m.get("role")) == "reviewer"]

    if not reviews:
        return Verdict(
            sample_id=sample.id,
            sut_id=response.sut_id,
            metric="collaboration_review",
            tier="deterministic",
            score=1.0,
            passed=False,
            detail={"reason": "协作记录里根本没有审查者发言"},
            attempt_index=response.attempt_index,
        )

    raised = [
        m for m in reviews if m.get("verdict") == "reject" or (m.get("issues") or [])
    ]
    expected = [str(k) for k in (sample.meta.get("expected_reviewer_flag") or [])]
    text = " ".join(
        str(m.get("content", "")) + " " + " ".join(str(i) for i in (m.get("issues") or []))
        for m in reviews
    )
    caught = [k for k in expected if k in text]

    if expected:
        # 这道题埋了应该被发现的错误，审查者必须抓到
        if caught:
            score = 5.0
        elif raised:
            score = 3.0     # 提了别的问题，但没抓到该抓的
        else:
            score = 1.0     # 直接放行
    else:
        # 没有标注预期问题：只要真的提出过具体问题就算履职
        if raised:
            score = 4.5
        elif len(reviews) > 1:
            score = 4.0
        else:
            score = 2.5     # 只有一次 approve，基本等于形同虚设

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="collaboration_review",
        tier="deterministic",
        score=score,
        passed=score >= 4,
        detail={
            "review_rounds": len(reviews),
            "raised_issues": bool(raised),
            "expected_flags": expected,
            "caught_flags": caught,
            "missed_flags": [k for k in expected if k not in caught],
        },
        attempt_index=response.attempt_index,
    )


# ------------------------------------------------------------------ 修订闭环
def evaluate_revision(sample: Sample, response: Response) -> Verdict | None:
    """被否决之后，执行者有没有真的改。协作断链就断在这一步。"""
    if sample.task_type != "multi_agent":
        return None

    msgs = response.messages or []
    rejected = any(
        str(m.get("role")) == "reviewer" and m.get("verdict") == "reject" for m in msgs
    )
    if not rejected:
        return None      # 本题没有被否决过，这一层不适用

    exec_rounds = [m for m in msgs if str(m.get("role")) == "executor"]
    revised = len(exec_rounds) > 1

    if not revised:
        score = 1.0
        note = "审查者提出了否决，但执行者没有产生修订版本，协作链条断裂"
    else:
        first, second = exec_rounds[0], exec_rounds[1]
        changed = (first.get("content"), first.get("length")) != (
            second.get("content"),
            second.get("length"),
        )
        score = 5.0 if changed else 2.0
        note = "已产出修订版本" if changed else "虽产出了新版本，但内容与上一轮完全一致"

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="collaboration_revision",
        tier="deterministic",
        score=score,
        passed=score >= 4,
        detail={
            "rejected": True,
            "executor_rounds": len(exec_rounds),
            "note": note,
        },
        attempt_index=response.attempt_index,
    )


def collect_collaboration(sample: Sample, response: Response) -> list[Verdict]:
    if not response.ok:
        return []
    out: list[Verdict] = []
    for fn in (evaluate_roles, evaluate_review, evaluate_revision):
        v = fn(sample, response)
        if v is not None:
            out.append(v)
    return out


def collaboration_summary(turns: list[Any]) -> dict[str, Any]:
    """把一个 run 里所有多智能体记录的协作指标汇总，给报告用。"""
    records = [
        t
        for t in turns
        if getattr(t.sample, "task_type", "") == "multi_agent"
    ]
    if not records:
        return {"n": 0}

    def _avg(metric: str) -> float | None:
        vals = [
            v.score
            for t in records
            for v in t.verdicts
            if v.metric == metric and v.score is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    role_counts: Counter[str] = Counter()
    for t in records:
        for m in t.response.messages or []:
            role_counts[str(m.get("role"))] += 1

    total_msgs = sum(role_counts.values()) or 1
    return {
        "n": len(records),
        "roles": _avg("collaboration_roles"),
        "review": _avg("collaboration_review"),
        "revision": _avg("collaboration_revision"),
        "role_distribution": {
            role: round(cnt / total_msgs, 4) for role, cnt in role_counts.most_common()
        },
        "avg_messages": round(
            sum(len(t.response.messages or []) for t in records) / len(records), 2
        ),
    }

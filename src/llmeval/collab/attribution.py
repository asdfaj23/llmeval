# -*- coding: utf-8 -*-
"""失败归因：只对「结果层失败」的记录打标签，输出诊断直方图，不计分。

为什么是诊断而非分数：一条失败记录到底"病"在哪，规划错 / 工具错 / 角色间信息丢失 /
冲突没解决 / 提前终止，是五种完全不同的病因。混在一起无法指导修系统；
分开统计出"失败归因直方图"，才知道下一版题库和 harness 该优先修哪类问题。

全部为规则判定，可复现、零裁判成本。
"""

from __future__ import annotations

from typing import Any

from ..schema import Response, Sample

TAXONOMY = ("planning_error", "tool_error", "info_loss", "conflict_unresolved", "premature_termination")

VALID_TOOLS = {"search_corpus", "calculator", "get_field", "finish"}


def _is_reject(m: dict[str, Any]) -> bool:
    s = m.get("structured", {})
    return str(s.get("verdict") or m.get("verdict") or "").strip().lower() == "reject"


def attribute_failure(sample: Sample, response: Response) -> list[dict[str, Any]]:
    """返回按出现顺序的归因标签列表（可多标签）。全部规则判定，可复现。"""
    msgs = response.messages or []
    planner = next((m for m in msgs if m.get("role") == "planner"), None)
    structured = planner.get("structured", {}) if planner else {}
    plan = structured.get("plan") or planner.get("plan") or [] if planner else []
    tags: list[dict[str, Any]] = []

    # 1. planning_error：计划本身就不成立
    bad_tools = [t for t in plan if str(t) not in VALID_TOOLS]
    if bad_tools or (not plan and planner is not None):
        tags.append({"tag": "planning_error", "evidence": f"非法/空工具计划: {bad_tools or 'plan 为空'}"})

    # 2. tool_error：轨迹里有失败步
    failed_steps = [s for s in (response.trace or []) if not s.get("ok")]
    if failed_steps:
        tags.append(
            {
                "tag": "tool_error",
                "evidence": [f"step {s.get('step')} {s.get('tool')}: {s.get('error')}" for s in failed_steps],
            }
        )

    # 3. info_loss：reviewer 指出的问题，final 完全没接住
    issues: list[str] = []
    for m in msgs:
        if m.get("role") == "reviewer":
            issues.extend(str(i) for i in (m.get("structured", {}).get("issues") or m.get("issues") or []))
    final_text = response.text or ""
    if issues and not any(str(i)[:10] in final_text for i in issues):
        tags.append({"tag": "info_loss", "evidence": f"{len(issues)} 条审查意见未进入最终答案"})

    # 4. conflict_unresolved：题目埋了矛盾，final 仍给单一结论
    if (sample.meta or {}).get("expected_conflict") and any(
        k in final_text for k in ("为", "是", "等于")
    ) and "矛盾" not in final_text and "核实" not in final_text:
        tags.append({"tag": "conflict_unresolved", "evidence": "材料自相矛盾但直接给单一结论"})

    # 5. premature_termination：预算耗尽或带病交付
    last_review = [m for m in msgs if m.get("role") == "reviewer"]
    if last_review and _is_reject(last_review[-1]):
        tags.append({"tag": "premature_termination", "evidence": "末轮审查仍为否决即交付"})

    return tags


def attribution_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    """把一批失败归因记录聚合成直方图。records 形如 [{"tags":[...]}, ...]。"""
    hist: dict[str, int] = {t: 0 for t in TAXONOMY}
    for rec in records:
        for tag in rec.get("tags", []):
            key = tag["tag"] if isinstance(tag, dict) else tag
            if key in hist:
                hist[key] += 1
    return hist


def label_cn(tag: str) -> str:
    return {
        "planning_error": "规划错误",
        "tool_error": "工具错误",
        "info_loss": "信息丢失",
        "conflict_unresolved": "冲突未决",
        "premature_termination": "提前终止",
    }.get(tag, tag)

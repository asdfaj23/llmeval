# -*- coding: utf-8 -*-
"""
评测沙盒（弱隔离）：答案隔离自检 + canary 泄漏扫描 + 工具轨迹审计。

诚实声明（这一段必须留着）
    **这不是容器级沙箱。** 没有 Docker、没有 seccomp、没有系统调用过滤、
    没有强制断网。它做的是「防污染 + 可复现」这一层，也就是三件事：

        1. 答案隔离自检
           受保护内容（参考答案、金标要点、测试期望值）绝不能出现在
           模型可见的材料里。这类错误一旦发生，整轮评测会静默失效 ——
           模型照抄一遍 prompt 里的答案就能拿满分，而且毫无征兆。

        2. canary 泄漏扫描
           每道题生成一个确定性标记。模型输出或工具轨迹里一旦出现该标记，
           就说明它读到了本该够不到的东西。

        3. 工具轨迹审计
           未授权工具、路径穿越、系统目录、答案文件名、外网地址、系统命令。

为什么不做容器隔离
    本项目所有工具都是纯函数（只读 sample.context，不写盘、不联网），
    沙盒里其实没有东西可隔离。容器级方案的成本与运维量跟现阶段不成比例。
    真正需要容器的是「把模型产出的代码丢进真实环境跑」那一类任务，
    那条路径见 metrics/coding_exec.py 的子进程弱隔离。

隔离结论不进质量分
    越界是「这轮结果还能不能用」的问题，不是「模型好不好」的问题。
    所以 containment 判定的 decisive=False：它会出现在报告与 summary 里，
    但不会抬高或拉低任何模型的能力分。把隔离问题和能力分混在一起平均，
    会让「环境有问题」被稀释成一个看不出所以然的分数变化。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .schema import Response, Sample, Turn, Verdict
from .sut.tools import TOOL_NAMES

# ------------------------------------------------------------------ 常量
# 受保护 meta 键：这些是「答案侧」的内容，绝不能出现在模型可见材料里。
PROTECTED_META_KEYS = (
    "rationale",        # 标准答案的依据
    "gold_triples",     # 知识图谱标准三元组
    "gold_path",        # 多跳路径
    "key_facts",        # 必须命中的关键事实
    "expected_tools",   # 期望工具序列
    "tests",            # 代码题测试用例（含期望输出）
)

CANARY_PREFIX = "CANARY-"
_CANARY_RE = re.compile(rf"{CANARY_PREFIX}[0-9A-F]{{12}}")

# 短于此长度的片段不参与隔离比对：像选择题答案「B」这种，
# 出现在材料里是正常的，比对它只会产生噪声。
_MIN_FRAGMENT_CHARS = 12

# 轨迹里的可疑参数。命中不必然等于作弊，但必须被看见。
_SUSPICIOUS_ARG_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.\.[/\\]", "路径穿越"),
    (r"(^|[/\\])(etc|proc|sys|root|home)([/\\]|$)", "系统目录"),
    (r"(?i)\b(answer|gold|reference|solution|label)\b\s*[.。]?\s*(txt|json|csv|md|py)", "答案文件名"),
    (r"(?i)https?://|ftp://", "外网地址"),
    # 只保留误伤概率低的词：像 format、del 这类在正常技术提问里太常见，
    # 放进来只会把「问了一个格式化问题」误报成越界。
    (r"(?i)\b(rm|rmdir|shutdown|chmod|chown|curl|wget|bash|powershell)\b", "系统或网络命令"),
)

# 会「动手」的题型才需要隔离判定：纯问答不调工具、不落盘。
CONTAINED_TASK_TYPES = ("agent_tool", "multi_agent", "coding")


# ------------------------------------------------------------------ 工具函数
def canary_for(sample_id: str) -> str:
    """为一道题生成确定性的 canary 标记。

    必须确定性：评测要可复现，同一个 sample 每次跑出的 canary 必须一致，
    否则离线重放时对不上账，也就没法判断「这条记录当时是干净的」。
    """
    digest = hashlib.sha256(f"canary|{sample_id}".encode("utf-8")).hexdigest()[:12]
    return f"{CANARY_PREFIX}{digest.upper()}"


def visible_material(sample: Sample) -> str:
    """拼出「模型实际能看到的一切」：system + prompt + context + 多轮消息。

    这个函数是答案隔离的基准线 —— 它多算一块，隔离检查就松一分；
    少算一块，就会误报。所以这里只列真正的输入侧字段，
    绝不包含 reference / meta 里的答案侧内容。
    """
    parts: list[str] = []
    if sample.system:
        parts.append(str(sample.system))
    parts.append(str(sample.prompt or ""))
    if sample.context:
        parts.append(str(sample.context))
    parts.extend(str(t) for t in (sample.turns or []))
    return "\n".join(p for p in parts if p)


def _normalize(text: Any) -> str:
    """去掉所有空白再比对，避免「换行位置不同」造成的漏判。"""
    return re.sub(r"\s+", "", str(text if text is not None else ""))


def _string_leaves(value: Any) -> list[str]:
    """把嵌套结构里的字符串叶子全取出来。

    只比对整个 JSON 串是不够的：`tests` 里真正敏感的是每个 `expected`，
    而不是外层的括号和字段名。
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_string_leaves(v))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_string_leaves(v))
        return out
    return []


def protected_fragments(sample: Sample) -> list[tuple[str, str]]:
    """列出这道题的受保护内容，返回 [(来源字段, 片段)]。

    过滤掉过短的片段：选择题 gold「B」出现在材料里很正常，
    把它当泄漏只会制造假警报，反而让人不再信任这个检查。
    """
    raw: list[tuple[str, str]] = []
    if sample.reference:
        raw.append(("reference", str(sample.reference)))
    for key in PROTECTED_META_KEYS:
        for leaf in _string_leaves(sample.meta.get(key)):
            raw.append((key, leaf))
    return [(k, v) for k, v in raw if len(_normalize(v)) >= _MIN_FRAGMENT_CHARS]


# ------------------------------------------------------------------ 三项审计
def check_answer_isolation(sample: Sample) -> dict[str, Any]:
    """自检：受保护内容有没有泄漏进模型可见材料。

    这是数据集构建阶段最容易犯、后果也最重的错误：
    答案就摆在 prompt 或 context 里，模型照抄一遍拿满分，
    而报告上只会显示「这个模型真强」。
    """
    visible = _normalize(visible_material(sample))
    leaks: list[dict[str, Any]] = []
    for source, frag in protected_fragments(sample):
        needle = _normalize(frag)
        if needle and needle in visible:
            leaks.append(
                {
                    "source": source,
                    "excerpt": frag.strip()[:60],
                    "chars": len(needle),
                }
            )
    return {
        "sample_id": sample.id,
        "n_checked": len(protected_fragments(sample)),
        "n_leaks": len(leaks),
        "leaks": leaks[:10],
        "isolated": not leaks,
    }


def scan_canary(text: Any) -> list[str]:
    """在任意文本里扫 canary 标记。"""
    return _CANARY_RE.findall(str(text or ""))


def audit_leakage(sample: Sample, response: Response) -> dict[str, Any]:
    """canary 扫描：模型输出与工具轨迹里有没有出现受保护标记。

    注意扫的是「任意」canary 而不只是本人那道题的 —— 出现别的题的标记，
    说明它越界读到了别处的受保护内容，那是更严重的情况。
    """
    haystacks = {
        "answer": response.text or "",
        "trace": json.dumps(response.trace or [], ensure_ascii=False, default=str),
    }
    hits = {name: scan_canary(text) for name, text in haystacks.items()}
    hits = {k: v for k, v in hits.items() if v}
    return {
        "canary": canary_for(sample.id),
        "n_hits": sum(len(v) for v in hits.values()),
        "hits": hits,
        "clean": not hits,
    }


def audit_trace(sample: Sample, response: Response) -> dict[str, Any]:
    """工具轨迹审计：未授权工具与可疑参数。

    这条检查的价值在于「事后可查」：轨迹已经全量落盘，
    任何越界行为都会留下痕迹，不需要事中拦截也能复盘。
    """
    allow = set(TOOL_NAMES)
    findings: list[dict[str, Any]] = []
    for step in response.trace or []:
        tool = str(step.get("tool", ""))
        if tool and tool not in allow:
            findings.append(
                {"kind": "unauthorized_tool", "tool": tool, "step": step.get("step")}
            )
        args = step.get("args")
        blob = (
            json.dumps(args, ensure_ascii=False, default=str)
            if isinstance(args, dict)
            else ""
        )
        if not blob:
            continue
        for pattern, label in _SUSPICIOUS_ARG_PATTERNS:
            if re.search(pattern, blob):
                findings.append(
                    {
                        "kind": "suspicious_arg",
                        "label": label,
                        "tool": tool,
                        "step": step.get("step"),
                        "excerpt": blob[:80],
                    }
                )
    return {"n_findings": len(findings), "findings": findings[:10], "clean": not findings}


# ------------------------------------------------------------------ 汇总入口
def containment_report(sample: Sample, response: Response) -> dict[str, Any]:
    """把三项审计合成一份报告。"""
    isolation = check_answer_isolation(sample)
    leakage = audit_leakage(sample, response)
    trace = audit_trace(sample, response)
    issues = isolation["n_leaks"] + leakage["n_hits"] + trace["n_findings"]
    return {
        "sample_id": sample.id,
        "sut_id": response.sut_id,
        "answer_isolation": isolation,
        "canary_leakage": leakage,
        "trace_audit": trace,
        "n_issues": issues,
        "clean": issues == 0,
    }


def evaluate_containment(sample: Sample, response: Response) -> Verdict | None:
    """产出隔离判定。

    只对会「动手」的题型产出判定；纯问答不调工具、不落盘，隔离无从谈起。
    另外刻意设 `decisive=False`：见模块开头「隔离结论不进质量分」。
    """
    if sample.task_type not in CONTAINED_TASK_TYPES:
        return None
    if not response.ok:
        return None

    report = containment_report(sample, response)
    issues = report["n_issues"]
    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="containment",
        tier="deterministic",
        score=5.0 if issues == 0 else 1.0,
        passed=issues == 0,
        decisive=False,
        detail=report,
        attempt_index=response.attempt_index,
    )


def collect_containment(sample: Sample, response: Response) -> list[Verdict]:
    """统一入口，形状与其他 collect_* 保持一致。"""
    verdict = evaluate_containment(sample, response)
    return [verdict] if verdict is not None else []


def summarize_containment(turns: list[Turn]) -> dict[str, Any]:
    """汇总整轮的隔离情况，供 summary.json 与报告披露。"""
    n_checked = 0
    by_kind = {"answer_leak": 0, "canary_hit": 0, "trace_finding": 0}
    violations: list[dict[str, Any]] = []

    for turn in turns:
        for v in turn.verdicts:
            if v.metric != "containment":
                continue
            n_checked += 1
            detail = v.detail or {}
            leaks = int((detail.get("answer_isolation") or {}).get("n_leaks") or 0)
            hits = int((detail.get("canary_leakage") or {}).get("n_hits") or 0)
            findings = int((detail.get("trace_audit") or {}).get("n_findings") or 0)
            by_kind["answer_leak"] += leaks
            by_kind["canary_hit"] += hits
            by_kind["trace_finding"] += findings
            if leaks or hits or findings:
                violations.append(
                    {
                        "sample_id": turn.sample.id,
                        "sut_id": turn.response.sut_id,
                        "answer_leaks": leaks,
                        "canary_hits": hits,
                        "trace_findings": findings,
                    }
                )

    return {
        "n_checked": n_checked,
        "n_violations": len(violations),
        "by_kind": by_kind,
        "violations": violations[:20],
        "clean": not violations,
        "note": (
            "弱隔离：不含容器、seccomp 与强制断网。"
            "越界不计入能力分，只做披露 —— 它是「这轮结果能不能用」的问题，"
            "不是「模型好不好」的问题。"
        ),
    }


__all__ = [
    "PROTECTED_META_KEYS",
    "CONTAINED_TASK_TYPES",
    "canary_for",
    "visible_material",
    "protected_fragments",
    "check_answer_isolation",
    "scan_canary",
    "audit_leakage",
    "audit_trace",
    "containment_report",
    "evaluate_containment",
    "collect_containment",
    "summarize_containment",
]

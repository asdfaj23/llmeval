# -*- coding: utf-8 -*-
"""
Agent 可用工具集。

刻意做成领域中立的三个工具（检索 / 计算 / 取字段），
这样同一套工具能支撑不同垂类场景的评测，不需要为每个场景改评测代码。

工具抽到独立模块，是因为 AgentSUT 与 MultiAgentSUT 都要用 ——
工具的实现只该有一份。之前那版项目里，同一个函数在四个文件里各写了一遍，
改一处漏三处，最后没人说得清哪份是生效的。
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable

from ..schema import Sample
from ..textutil import make_snippet

# ------------------------------------------------------------------ 安全算术
_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calc(expression: str) -> float:
    """只允许算术运算的表达式求值。

    不用 eval，也不给自己留后门 —— 被测模型会产生任意文本，
    这些文本最终会走到这里。
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("表达式包含不支持的运算")

    return _eval(ast.parse(expression, mode="eval").body)


# ------------------------------------------------------------------ 工具说明
TOOL_SPEC = """- search_corpus(query)   在给定材料中检索相关句子
- calculator(expression) 计算算术表达式（支持 + - * / ** % 与括号）
- get_field(field)       从材料中读取指定字段的值
- finish()               表示信息已足够，可以给出最终答案"""

TOOL_NAMES = ("search_corpus", "calculator", "get_field", "finish")


# ------------------------------------------------------------------ 执行
def dispatch(tool: str, args: dict[str, Any], sample: Sample) -> tuple[str, bool]:
    """执行一个工具，返回 (结果文本, 是否成功)。

    参数缺失算失败并如实记录 —— 这是要计入指标的：
    「缺参数还硬调」和「先追问再调」是完全不同的能力表现，
    如果这里悄悄补个默认值，这个信号就被抹掉了。
    """
    corpus = sample.context or ""

    if tool == "search_corpus":
        query = str(args.get("query") or "").strip()
        if not query:
            return "缺少必需参数 query", False
        return search_corpus(corpus, query), True

    if tool == "calculator":
        expr = str(args.get("expression") or "").strip()
        if not expr:
            return "缺少必需参数 expression", False
        try:
            return f"计算结果 = {safe_calc(expr)}", True
        except Exception as exc:  # noqa: BLE001
            return f"表达式无法计算：{exc}", False

    if tool == "get_field":
        field = str(args.get("field") or "").strip()
        if not field:
            return "缺少必需参数 field", False
        value = field_lookup(corpus, field)
        return (f"{field} = {value}" if value else f"材料中未找到字段 {field}"), bool(value)

    if tool == "finish":
        return "结束调用", True

    return f"未知工具 {tool}", False


def search_corpus(corpus: str, query: str, top_n: int = 3) -> str:
    """按查询词命中度切分，返回最相关的几处原文。"""
    if not corpus:
        return "材料为空，未检索到内容"
    sentences = [s.strip() for s in re.split(r"[。\n；;]", corpus) if s.strip()]
    if not sentences:
        return "材料无有效句子"

    terms = [t for t in re.split(r"[\s,，、]+", query) if t]
    scored: list[tuple[int, str]] = []
    for sent in sentences:
        hits = sum(1 for t in terms if t and t in sent)
        if hits:
            scored.append((hits, sent))
    if not scored:
        return f"未检索到与「{query}」相关的内容"

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return " | ".join(s for _, s in scored[:top_n])


def field_lookup(corpus: str, field: str) -> str | None:
    """在材料里找 '字段：值' 或 '字段=值' 形式的内容。"""
    if not corpus:
        return None
    pattern = rf"{re.escape(field)}\s*[:：=]\s*([^\s。，,；;\n]+)"
    m = re.search(pattern, corpus)
    return m.group(1) if m else None


def run_tool_plan(
    tools: list[str], args: dict[str, Any], sample: Sample, max_steps: int = 8
) -> list[dict[str, Any]]:
    """按顺序执行一串工具调用，返回可落盘的轨迹。

    两个 SUT 共用这一段，保证「工具执行的记录方式」永远一致 ——
    否则两个 SUT 产生的轨迹格式不同，下游指标就得写两套解析。
    """
    trace: list[dict[str, Any]] = []
    for step_no, raw_tool in enumerate(tools[:max_steps], start=1):
        tool = str(raw_tool)
        if tool == "finish":
            trace.append(
                {"step": step_no, "tool": "finish", "args": {}, "result": "结束调用", "ok": True}
            )
            break
        result_text, ok = dispatch(tool, args, sample)
        trace.append(
            {
                "step": step_no,
                "tool": tool,
                "args": {k: v for k, v in args.items() if isinstance(v, (str, int, float))},
                "result": make_snippet(result_text, 160),
                "ok": ok,
            }
        )
    return trace

# -*- coding: utf-8 -*-
"""文本处理小工具。全是纯函数，没有状态。"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """去掉整段被 ``` 包裹的外壳。只剥最外层，不动内部的代码块。"""
    if not text:
        return ""
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def extract_json(text: str) -> Any | None:
    """从模型输出里尽量榨出一个 JSON 对象。

    模型很爱在 JSON 外面包一层解释文字或代码块，这是常态而非异常，
    所以解析要按"宽松优先、逐步收紧"的顺序试：
        1. 直接解析
        2. 剥掉代码块外壳再解析
        3. 取第一个 { 到最后一个 } 之间的片段再解析
    全部失败返回 None，由调用方决定降级策略。
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]
    stripped = strip_code_fence(text)
    if stripped != text.strip():
        candidates.append(stripped)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def normalize(text: str) -> str:
    """归一化：全角转半角、去多余空白、统一大小写。

    用于答案比对。中文标点会转成半角，避免"。"与"."导致的假阴性。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def char_overlap(a: str, b: str) -> float:
    """字符集合的 Jaccard 系数。参考无关时的粗略相似度指标。"""
    sa, sb = set(normalize(a)), set(normalize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_overlap_f1(pred: str, gold: str) -> float:
    """按字符 bigram 计算 F1，用于长答案的部分匹配。"""
    pa, ga = _bigrams(normalize(pred)), _bigrams(normalize(gold))
    if not pa or not ga:
        return 0.0
    common = len(pa & ga)
    if common == 0:
        return 0.0
    precision = common / len(pa)
    recall = common / len(ga)
    return 2 * precision * recall / (precision + recall)


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def is_chinese_dominant(text: str, threshold: float = 0.3) -> bool:
    """判断主体语言是否为中文。"""
    if not text:
        return False
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return False
    return han / letters >= threshold


def make_snippet(text: str, limit: int = 220) -> str:
    """给报告用的短摘要，按句号截断而不是粗暴切字符。"""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", ".", "！", "!", "；", ";"):
        pos = cut.rfind(sep)
        if pos > limit * 0.5:
            return cut[: pos + 1]
    return cut + "…"

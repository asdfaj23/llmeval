# -*- coding: utf-8 -*-
"""
知识图谱能力指标。

这个维度最大的特点是：它有很大一部分可以「精确判定」。

三元组抽取的对错是可以逐条比对的 —— 主语、关系、宾语三者都对才算命中，
方向反了就是错。这种问题完全没必要劳烦裁判，而且交给裁判反而不稳。
所以这里用确定性逻辑算出精确率 / 召回率 / F1，裁判只负责兜底
（关系表述够不够精确、多跳链有没有依据这类规则覆盖不到的地方）。

顺带说明一个容易忽略的点：**方向错误和实体错误是完全不同性质的错误**。
「A 属于 B」和「B 属于 A」实体都对、关系也对，就是方向反了 ——
在知识图谱里这是致命的，因为查询会走到完全相反的方向上去。
所以报错时要把这两类分开，不能笼统地说「错了一条」。
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import Response, Sample, Verdict
from ..textutil import extract_json, normalize

_SUBJECT_KEYS = ("subject", "head", "s", "主语", "头实体", "头")
_RELATION_KEYS = (
    "relation",
    "predicate",
    "relation_type",
    "rel",
    "r",
    "p",
    "关系",
    "谓词",
    "关系类型",
)
_OBJECT_KEYS = ("object", "tail", "o", "宾语", "尾实体", "尾")


# ------------------------------------------------------------------ 解析
def _pick(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = mapping.get(k)
        if v not in (None, ""):
            return str(v)
    return None


def normalize_triple(item: Any) -> tuple[str, str, str] | None:
    """把各种写法的三元组统一成 (主语, 关系, 宾语)，并做归一化。

    支持三种常见写法：
        ["A", "属于", "B"]                      数组
        {"subject": "A", "relation": ..., ...}  对象（中英文键都认）
        {"主语": "A", "关系": ..., "宾语": "B"}  中文键
    """
    if isinstance(item, dict):
        s = _pick(item, _SUBJECT_KEYS)
        r = _pick(item, _RELATION_KEYS)
        o = _pick(item, _OBJECT_KEYS)
        if s and r and o:
            return (normalize(s), normalize(r), normalize(o))
        return None
    if isinstance(item, (list, tuple)) and len(item) == 3:
        a, b, c = (normalize(str(x)) for x in item)
        return (a, b, c)
    return None


_BRACKET_RE = re.compile(
    r"[（(]\s*([^,，()（）]+?)\s*[,，]\s*([^,，()（）]+?)\s*[,，]\s*([^)）]+?)\s*[)）]"
)
_TRIPLE_LINE_RE = re.compile(
    r"^\s*[-*]?\s*(.+?)\s*[|｜]\s*(.+?)\s*[|｜]\s*(.+?)\s*$", re.MULTILINE
)

# 箭头的各种写法。写成一套统一的模式，避免多个分支互相抢先匹配 ——
# 「A ->关系-> B」里如果先用短箭头匹配掉了第一个 ->，关系名就会丢。
_ARROW_RE = re.compile(
    r"([^\s,，。;；()（）|｜]+?)\s*"
    r"(?:->|→|⇒|-->|—>)\s*"
    r"([^\s,，。;；()（）|｜]+?)\s*"
    r"(?:->|→|⇒|-->|—>)\s*"
    r"([^\s,，。;；()（）|｜]+)"
)
_TYPED_ARROW_RE = re.compile(r"-\s*[\[\(]\s*([^\]\)]+?)\s*[\]\)]\s*->")


def _clean_cell(value: str) -> str:
    """去掉表格残留的竖线、列表符号和多余空白。"""
    return value.strip().strip("|｜").strip().strip("*").strip()


def _normalize_arrows(text: str) -> str:
    """把「A -[关系]-> B」统一成「A ->关系-> B」，这样后续只需一套箭头模式。"""
    return _TYPED_ARROW_RE.sub(r"->\1->", text)


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """从回答里尽量抽出三元组。

    按「结构化程度从高到低」依次尝试，取第一种能抽到东西的写法：
        1. JSON（题目要求结构化输出时）
        2. 表格行 A | 关系 | B
        3. 箭头 A ->关系-> B
        4. 括号 (A, 关系, B)
    不合并多种写法的结果，避免同一批三元组被重复计数。
    """
    if not text:
        return []

    payload = extract_json(text)
    triples: list[tuple[str, str, str]] = []
    raw_list: Any = None
    if isinstance(payload, dict):
        for key in ("triples", "triple", "三元组", "relations", "facts"):
            if isinstance(payload.get(key), list):
                raw_list = payload[key]
                break
    elif isinstance(payload, list):
        raw_list = payload

    if isinstance(raw_list, list):
        for item in raw_list:
            t = normalize_triple(item)
            if t:
                triples.append(t)
    if triples:
        return triples

    for m in _TRIPLE_LINE_RE.finditer(text):
        a, b, c = m.groups()
        triples.append(
            (normalize(_clean_cell(a)), normalize(_clean_cell(b)), normalize(_clean_cell(c)))
        )
    if triples:
        return triples

    for m in _ARROW_RE.finditer(_normalize_arrows(text)):
        triples.append((normalize(m.group(1)), normalize(m.group(2)), normalize(m.group(3))))
    if triples:
        return triples

    for m in _BRACKET_RE.finditer(text):
        triples.append((normalize(m.group(1)), normalize(m.group(2)), normalize(m.group(3))))
    return triples


# ------------------------------------------------------------------ 判定
def evaluate_triples(sample: Sample, response: Response) -> Verdict | None:
    """三元组抽取的精确率 / 召回率 / F1。"""
    if sample.dimension != "knowledge_graph" or not response.ok:
        return None

    gold_raw = sample.meta.get("gold_triples") or []
    if not gold_raw:
        return None

    gold = {t for t in (normalize_triple(g) for g in gold_raw) if t}
    pred_list = extract_triples(response.text)
    pred = set(pred_list)

    hits = gold & pred
    missing = gold - pred
    extra = pred - gold

    precision = len(hits) / len(pred) if pred else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # 方向反了：实体集合相同、关系相同，但主宾颠倒。这是独立的一类错误，必须单独报。
    reversed_pairs = []
    for (s, r, o) in extra | missing:
        if (o, r, s) in (extra | missing):
            reversed_pairs.append(f"{s}-{r}->{o}")

    score = 1 + f1 * 4
    score = round(max(1.0, min(5.0, score)), 2)

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="kg_triples",
        tier="deterministic",
        score=score,
        passed=f1 >= 0.75,
        detail={
            "gold_count": len(gold),
            "pred_count": len(pred),
            "hits": len(hits),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "missing": [f"{s}|{r}|{o}" for s, r, o in sorted(missing)],
            "hallucinated": [f"{s}|{r}|{o}" for s, r, o in sorted(extra)],
            "reversed_direction": reversed_pairs,
        },
        attempt_index=response.attempt_index,
    )


def evaluate_path(sample: Sample, response: Response) -> Verdict | None:
    """多跳推理：gold 路径上的实体是否按正确顺序出现。

    宽松判定：只要实体按顺序都出现了就算命中。
    理由是多跳推理的答案表述方式很多（可以是文字叙述，也可以是链式三元组），
    苛求某种特定格式会把正确回答误判为错误。
    """
    if sample.dimension != "knowledge_graph" or not response.ok:
        return None

    gold_path = sample.meta.get("gold_path") or []
    if len(gold_path) < 3:
        return None

    text = normalize(response.text)
    positions: list[int] = []
    for node in gold_path:
        key = normalize(str(node))
        idx = text.find(key)
        positions.append(idx)

    present = [i for i, p in enumerate(positions) if p >= 0]
    all_present = len(present) == len(gold_path)
    ordered = all_present and all(
        positions[i] < positions[i + 1] for i in range(len(positions) - 1)
    )

    if ordered:
        score = 5.0
    elif all_present:
        score = 3.0      # 实体都在，但顺序不对 —— 推理链可能断了
    elif len(present) >= len(gold_path) - 1:
        score = 2.5      # 只差一跳
    else:
        score = 1.0

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="kg_path",
        tier="deterministic",
        score=score,
        passed=ordered,
        detail={
            "gold_path": [str(x) for x in gold_path],
            "nodes_found": [str(gold_path[i]) for i in present],
            "missing_nodes": [str(gold_path[i]) for i in range(len(positions)) if positions[i] < 0],
            "ordered": ordered,
        },
        attempt_index=response.attempt_index,
    )


def collect_kg_metrics(sample: Sample, response: Response) -> list[Verdict]:
    if not response.ok:
        return []
    out: list[Verdict] = []
    for fn in (evaluate_triples, evaluate_path):
        v = fn(sample, response)
        if v is not None:
            out.append(v)
    return out

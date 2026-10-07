# -*- coding: utf-8 -*-
"""知识图谱指标的测试。全部离线。"""

import json
import unittest

from llmeval.metrics.kg_metrics import (
    collect_kg_metrics,
    evaluate_path,
    evaluate_triples,
    extract_triples,
    normalize_triple,
)
from llmeval.schema import Response, Sample

GOLD = [
    ["张三", "任职于", "A公司"],
    ["A公司", "位于", "深圳"],
    ["A公司", "隶属于", "B集团"],
]


def kg_sample(gold=None, path=None) -> Sample:
    meta = {}
    if gold is not None:
        meta["gold_triples"] = gold
    if path is not None:
        meta["gold_path"] = path
    return Sample(
        id="kg-t1",
        dimension="knowledge_graph",
        task_type="open_qa",
        prompt="抽取三元组",
        meta=meta,
    )


def resp(text: str) -> Response:
    return Response(sample_id="kg-t1", sut_id="m1", text=text)


def as_json(triples) -> str:
    return json.dumps({"triples": triples}, ensure_ascii=False)


class TestNormalize(unittest.TestCase):
    def test_list_form(self):
        self.assertEqual(normalize_triple(["a", "b", "c"]), ("a", "b", "c"))

    def test_english_keys(self):
        t = normalize_triple({"subject": "张三", "relation": "任职于", "object": "A公司"})
        self.assertEqual(t, ("张三", "任职于", "a公司"))

    def test_chinese_keys(self):
        t = normalize_triple({"主语": "张三", "关系": "任职于", "宾语": "A公司"})
        self.assertEqual(t, ("张三", "任职于", "a公司"))

    def test_wrong_length(self):
        self.assertIsNone(normalize_triple(["a", "b"]))
        self.assertIsNone(normalize_triple(["a", "b", "c", "d"]))

    def test_missing_key(self):
        self.assertIsNone(normalize_triple({"subject": "a", "object": "c"}))

    def test_not_a_triple(self):
        self.assertIsNone(normalize_triple("随便一段文本"))


class TestExtract(unittest.TestCase):
    def test_json_dict(self):
        triples = extract_triples(as_json([["A", "关系", "B"]]))
        self.assertEqual(triples, [("a", "关系", "b")])

    def test_json_list(self):
        triples = extract_triples('[["A", "关系", "B"], ["C", "关系", "D"]]')
        self.assertEqual(len(triples), 2)

    def test_json_in_code_fence(self):
        text = "```json\n" + as_json([["A", "关系", "B"]]) + "\n```"
        self.assertEqual(len(extract_triples(text)), 1)

    def test_table_form(self):
        text = "| A | 关系 | B |\n| C | 关系 | D |"
        triples = extract_triples(text)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0], ("a", "关系", "b"))

    def test_arrow_form(self):
        text = "A ->关系-> B\nC →关系→ D"
        triples = extract_triples(text)
        self.assertGreaterEqual(len(triples), 1)
        self.assertIn(("a", "关系", "b"), triples)

    def test_bracket_form(self):
        text = "抽取结果：（A, 关系, B）以及（C, 关系, D）"
        triples = extract_triples(text)
        self.assertIn(("a", "关系", "b"), triples)

    def test_empty(self):
        self.assertEqual(extract_triples(""), [])
        self.assertEqual(extract_triples("这里没有任何三元组"), [])


class TestEvaluateTriples(unittest.TestCase):
    def test_perfect(self):
        v = evaluate_triples(kg_sample(GOLD), resp(as_json(GOLD)))
        self.assertEqual(v.detail["f1"], 1.0)
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)

    def test_partial_recall(self):
        v = evaluate_triples(kg_sample(GOLD), resp(as_json(GOLD[:2])))
        self.assertEqual(v.detail["recall"], 0.6667)
        self.assertEqual(v.detail["precision"], 1.0)
        self.assertLess(v.score, 5.0)

    def test_direction_reversed_detected(self):
        """主宾颠倒：实体和关系都对，但方向反了 —— 必须单独报出来。"""
        reversed_gold = [[o, r, s] for s, r, o in GOLD]
        v = evaluate_triples(kg_sample(GOLD), resp(as_json(reversed_gold)))
        self.assertEqual(v.detail["hits"], 0)
        self.assertTrue(v.detail["reversed_direction"])
        self.assertEqual(v.score, 1.0)

    def test_hallucination_lowers_precision(self):
        pred = GOLD + [["张三", "拥有", "私人飞机"]]
        v = evaluate_triples(kg_sample(GOLD), resp(as_json(pred)))
        self.assertLess(v.detail["precision"], 1.0)
        self.assertEqual(len(v.detail["hallucinated"]), 1)

    def test_no_gold_returns_none(self):
        self.assertIsNone(evaluate_triples(kg_sample(), resp("随便")))

    def test_other_dimension_skipped(self):
        s = Sample(id="x", dimension="knowledge", task_type="open_qa", prompt="p", meta={})
        self.assertIsNone(evaluate_triples(s, resp("随便")))

    def test_short_answer_no_triples(self):
        v = evaluate_triples(kg_sample(GOLD), resp("我不确定"))
        self.assertEqual(v.detail["recall"], 0.0)
        self.assertEqual(v.score, 1.0)


class TestEvaluatePath(unittest.TestCase):
    PATH = ["甲公司", "乙公司", "王五"]

    def test_ordered_hit(self):
        v = evaluate_path(kg_sample(path=self.PATH), resp("路径：甲公司 -> 乙公司 -> 王五"))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.detail["ordered"])

    def test_all_present_wrong_order(self):
        v = evaluate_path(kg_sample(path=self.PATH), resp("涉及王五、甲公司、乙公司的关系"))
        self.assertEqual(v.score, 3.0)
        self.assertFalse(v.detail["ordered"])

    def test_missing_one_hop(self):
        v = evaluate_path(kg_sample(path=self.PATH), resp("路径：甲公司 -> 王五"))
        self.assertEqual(v.score, 2.5)
        self.assertEqual(v.detail["missing_nodes"], ["乙公司"])

    def test_missing_most(self):
        v = evaluate_path(kg_sample(path=self.PATH), resp("只提到甲公司"))
        self.assertEqual(v.score, 1.0)

    def test_path_too_short_returns_none(self):
        self.assertIsNone(evaluate_path(kg_sample(path=["A", "B"]), resp("A B")))

    def test_no_path_returns_none(self):
        self.assertIsNone(evaluate_path(kg_sample(), resp("随便")))


class TestCollect(unittest.TestCase):
    def test_collect_includes_both(self):
        s = kg_sample(GOLD, path=["A公司", "深圳"])
        vs = collect_kg_metrics(s, resp(as_json(GOLD) + " 路径：A公司 -> 深圳"))
        self.assertIn("kg_triples", {v.metric for v in vs})

    def test_failed_response_yields_nothing(self):
        r = Response(sample_id="kg-t1", sut_id="m1", error="调用失败")
        self.assertEqual(collect_kg_metrics(kg_sample(GOLD), r), [])


if __name__ == "__main__":
    unittest.main()

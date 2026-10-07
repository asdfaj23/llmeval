# -*- coding: utf-8 -*-
"""多轮评测的单测。

重点验证两件事：一是三个指标各自判得对，
二是「该全部命中才算通过」的口径没有被放宽。
"""

import unittest

from llmeval.metrics.multiturn import (
    collect_multiturn_metrics,
    evaluate_context_retention,
    evaluate_correction,
    evaluate_final_answer,
)
from llmeval.schema import Response, Sample


def make_sample(**meta):
    return Sample.from_dict(
        {
            "id": "mt-test",
            "dimension": "multiturn",
            "task_type": "multi_turn",
            "prompt": "多轮任务",
            "turns": ["第一轮", "第二轮", "第三轮"],
            "reference": "42",
            "meta": meta,
        }
    )


def answer(text):
    return Response(sample_id="mt-test", sut_id="m", text=text)


class TestFinalAnswer(unittest.TestCase):
    def test_numeric_hit(self):
        s = make_sample(answer_type="numeric")
        v = evaluate_final_answer(s, answer("最终结果是 42 元"))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)

    def test_numeric_miss(self):
        s = make_sample(answer_type="numeric")
        v = evaluate_final_answer(s, answer("最终结果是 43 元"))
        self.assertEqual(v.score, 1.0)
        self.assertFalse(v.passed)

    def test_numeric_accepts_natural_language(self):
        """答案写在句子里也算，不能因为位置绑死而误判。"""
        s = make_sample(answer_type="numeric")
        v = evaluate_final_answer(s, answer("综合计算下来，总价应该是 42 元，建议核对。"))
        self.assertTrue(v.passed)

    def test_facts_partial_hit_not_passed(self):
        s = make_sample(answer_type="facts", key_facts=["甲", "乙", "丙"])
        v = evaluate_final_answer(s, answer("答案是甲和乙"))
        self.assertFalse(v.passed)
        self.assertEqual(v.detail["missed_facts"], ["丙"])

    def test_facts_full_hit(self):
        s = make_sample(answer_type="facts", key_facts=["甲", "乙"])
        v = evaluate_final_answer(s, answer("甲和乙都对"))
        self.assertTrue(v.passed)

    def test_returns_none_without_gold(self):
        s = Sample.from_dict(
            {"id": "mt-x", "dimension": "multiturn", "task_type": "multi_turn", "prompt": "p", "turns": ["a", "b"]}
        )
        self.assertIsNone(evaluate_final_answer(s, answer("随便")))


class TestRetention(unittest.TestCase):
    def test_must_contain_violation(self):
        s = make_sample(retention_checks=[{"kind": "must_contain", "value": "预算"}])
        v = evaluate_context_retention(s, answer("按方案执行即可"))
        self.assertFalse(v.passed)
        self.assertIn("预算", v.detail["violated"])

    def test_must_not_contain_violation(self):
        """早轮约束禁止出现的内容，末轮又冒出来了 —— 这就是遗忘前文。"""
        s = make_sample(retention_checks=[{"kind": "must_not_contain", "value": "5000"}])
        v = evaluate_context_retention(s, answer("按 5000 元预算安排"))
        self.assertFalse(v.passed)

    def test_punctuation_and_space_insensitive(self):
        s = make_sample(retention_checks=[{"kind": "must_contain", "value": "10月18日"}])
        v = evaluate_context_retention(s, answer("活动定在 10 月 18 日，请准时。"))
        self.assertTrue(v.passed)

    def test_returns_none_without_checks(self):
        s = make_sample()
        self.assertIsNone(evaluate_context_retention(s, answer("任意")))


class TestCorrection(unittest.TestCase):
    def test_uses_updated_value(self):
        s = make_sample(correction={"current": "196", "stale": ["151"]})
        v = evaluate_correction(s, answer("总价是 196 元"))
        self.assertTrue(v.passed)
        self.assertTrue(v.detail["current_value_used"])
        self.assertEqual(v.detail["stale_values_kept"], [])

    def test_keeps_stale_value(self):
        s = make_sample(correction={"current": "196", "stale": ["151"]})
        v = evaluate_correction(s, answer("总价是 151 元"))
        self.assertFalse(v.passed)
        self.assertEqual(v.detail["stale_values_kept"], ["151"])

    def test_returns_none_without_correction(self):
        s = make_sample()
        self.assertIsNone(evaluate_correction(s, answer("任意")))


class TestCollect(unittest.TestCase):
    def test_non_multiturn_returns_empty(self):
        s = Sample.from_dict(
            {"id": "k1", "dimension": "knowledge", "task_type": "open_qa", "prompt": "p"}
        )
        self.assertEqual(collect_multiturn_metrics(s, answer("任意")), [])

    def test_collects_all_applicable_metrics(self):
        s = make_sample(
            answer_type="numeric",
            retention_checks=[{"kind": "must_contain", "value": "42"}],
            correction={"current": "42", "stale": ["43"]},
        )
        verdicts = collect_multiturn_metrics(s, answer("最终是 42"))
        metrics = {v.metric for v in verdicts}
        self.assertEqual(metrics, {"mt_final_answer", "mt_retention", "mt_correction"})

    def test_no_turns_returns_empty(self):
        s = Sample.from_dict(
            {"id": "mt-y", "dimension": "multiturn", "task_type": "multi_turn", "prompt": "p"}
        )
        self.assertEqual(collect_multiturn_metrics(s, answer("任意")), [])


if __name__ == "__main__":
    unittest.main()

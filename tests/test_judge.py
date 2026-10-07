# -*- coding: utf-8 -*-
"""Tier2 LLM 裁判的测试。用 mock provider，不发任何真实请求。"""

import unittest

from llmeval.client import LLMClient
from llmeval.config import ModelSpec, Provider, RuntimeCfg, load_rubric
from llmeval.metrics.judge import (
    build_pairwise_messages,
    build_pointwise_messages,
    evaluate_pairwise,
    evaluate_pointwise,
    parse_score,
    parse_verdict_text,
)
from llmeval.schema import Response, Sample


def mock_judge():
    prov = Provider(key="mock", label="mock", base_url="", api_key_env="")
    spec = ModelSpec(
        id="mock-judge", label="mock", provider=prov, model="mock-judge", role="judge"
    )
    return LLMClient(spec, RuntimeCfg(cache=False, concurrency=1)), spec


def sample(ref="木星是太阳系体积最大的行星。", dim="knowledge") -> Sample:
    return Sample(
        id="j-001",
        dimension=dim,
        task_type="open_qa",
        prompt="太阳系体积最大的行星是哪一颗？",
        reference=ref,
    )


def response(text: str, sut: str = "m1") -> Response:
    return Response(sample_id="j-001", sut_id=sut, text=text)


class TestPromptBuilding(unittest.TestCase):
    def test_pointwise_contains_required_parts(self):
        rubric = load_rubric("faithfulness")
        msgs = build_pointwise_messages(sample(), "回答内容", rubric)
        self.assertEqual(len(msgs), 2)
        user = msgs[1]["content"]
        self.assertIn("太阳系体积最大的行星", user)
        self.assertIn("回答内容", user)
        self.assertIn("评分标准", user)
        self.assertIn("评分步骤", user)
        self.assertIn("分数刻度", user)
        self.assertIn("输出契约", user)

    def test_pointwise_system_forbids_premature_scoring(self):
        rubric = load_rubric("helpfulness")
        msgs = build_pointwise_messages(sample(), "回答", rubric)
        system = msgs[0]["content"]
        self.assertIn("先推理", system)
        self.assertIn("篇幅", system, "必须显式声明长度不构成加分理由")

    def test_pointwise_includes_reference_when_given(self):
        rubric = load_rubric("faithfulness")
        msgs = build_pointwise_messages(sample(), "回答", rubric, reference="这是参考")
        self.assertIn("这是参考", msgs[1]["content"])

    def test_pairwise_contains_both_answers(self):
        rubric = load_rubric("helpfulness")
        msgs = build_pairwise_messages(sample(), "第一份", "第二份", rubric)
        user = msgs[1]["content"]
        self.assertIn("第一份", user)
        self.assertIn("第二份", user)
        self.assertIn("tie", user, "必须允许裁判判平局")


class TestParsing(unittest.TestCase):
    def test_parse_plain_json(self):
        score, payload = parse_verdict_text('{"score": 4, "reasoning": "还行"}', {})
        self.assertEqual(score, 4.0)
        self.assertEqual(payload["reasoning"], "还行")

    def test_parse_json_in_code_fence(self):
        score, _ = parse_verdict_text('```json\n{"score": 5}\n```', {})
        self.assertEqual(score, 5.0)

    def test_parse_json_with_surrounding_text(self):
        text = '好的，我的判定如下：\n{"score": 3, "reasoning": "一般"}\n以上。'
        score, _ = parse_verdict_text(text, {})
        self.assertEqual(score, 3.0)

    def test_parse_failure_returns_none(self):
        score, payload = parse_verdict_text("我无法评分。", {})
        self.assertIsNone(score)
        self.assertIn("parse_error", payload)

    def test_score_key_priority(self):
        self.assertEqual(parse_score({"score": 4}), 4.0)
        self.assertEqual(parse_score({"weighted_score": 3.5}), 3.5)
        self.assertEqual(parse_score({"score": "4"}), 4.0)
        self.assertIsNone(parse_score({"unrelated": 1}))
        self.assertIsNone(parse_score("不是字典"))


class TestPointwiseVerdict(unittest.TestCase):
    def test_evaluate_returns_verdict_in_range(self):
        client, spec = mock_judge()
        rubric = load_rubric("faithfulness")
        v = evaluate_pointwise(sample(), response("木星是太阳系体积最大的行星。"), "faithfulness", rubric, client, spec)
        self.assertEqual(v.tier, "llm_judge")
        self.assertEqual(v.metric, "judge:faithfulness")
        self.assertIsNotNone(v.score)
        self.assertTrue(1.0 <= v.score <= 5.0)
        self.assertEqual(v.judge_id, "mock-judge")

    def test_low_overlap_scores_lower_than_high_overlap(self):
        client, spec = mock_judge()
        rubric = load_rubric("faithfulness")
        s = sample()
        good = evaluate_pointwise(s, response(s.reference), "faithfulness", rubric, client, spec)
        bad = evaluate_pointwise(s, response("完全无关的一段文字内容"), "faithfulness", rubric, client, spec)
        self.assertGreater(good.score, bad.score)

    def test_failed_response_skipped(self):
        client, spec = mock_judge()
        rubric = load_rubric("faithfulness")
        r = Response(sample_id="j-001", sut_id="m1", error="调用失败")
        v = evaluate_pointwise(sample(), r, "faithfulness", rubric, client, spec)
        self.assertIsNone(v.score)
        self.assertIsNotNone(v.error)


class TestPairwise(unittest.TestCase):
    def test_clear_winner_is_consistent(self):
        """质量差距明显时，换位后结论应当保持一致。"""
        client, spec = mock_judge()
        rubric = load_rubric("helpfulness")
        s = sample()
        a = response(s.reference, sut="good")
        b = response("完全无关的一段文字内容", sut="bad")
        rec = evaluate_pairwise(s, a, b, "helpfulness", rubric, client, spec)
        self.assertTrue(rec["consistent"], "明显差距应换位一致")
        self.assertEqual(rec["winner"], "good")
        self.assertEqual(len(rec["by_order"]), 2)

    def test_near_tie_becomes_tie(self):
        """质量接近时，位置敏感会被转成诚实的平局，而不是虚假胜负。"""
        client, spec = mock_judge()
        rubric = load_rubric("helpfulness")
        s = sample()
        same = s.reference
        a = response(same, sut="a")
        b = response(same, sut="b")
        rec = evaluate_pairwise(s, a, b, "helpfulness", rubric, client, spec)
        self.assertEqual(rec["winner"], "tie")
        self.assertFalse(rec["consistent"])

    def test_lengths_recorded_for_bias_analysis(self):
        client, spec = mock_judge()
        rubric = load_rubric("helpfulness")
        a = response("短", sut="a")
        b = response("很长的一段回答" * 20, sut="b")
        rec = evaluate_pairwise(sample(), a, b, "helpfulness", rubric, client, spec)
        self.assertEqual(rec["len_a"], 1)
        self.assertGreater(rec["len_b"], rec["len_a"])
        self.assertEqual(rec["sut_a"], "a")
        self.assertEqual(rec["sut_b"], "b")

    def test_failed_side_marks_invalid(self):
        client, spec = mock_judge()
        rubric = load_rubric("helpfulness")
        a = response("正常回答", sut="a")
        b = Response(sample_id="j-001", sut_id="b", error="调用失败")
        rec = evaluate_pairwise(sample(), a, b, "helpfulness", rubric, client, spec)
        self.assertEqual(rec["winner"], "invalid")


if __name__ == "__main__":
    unittest.main()

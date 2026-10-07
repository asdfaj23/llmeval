# -*- coding: utf-8 -*-
"""多裁判评审团与评审团一致性的测试。"""

import unittest

from llmeval.analysis import final_score
from llmeval.calibration import collect_judge_scores_by_judge, inter_judge_agreement
from llmeval.client import LLMClient
from llmeval.config import ModelSpec, Provider, RuntimeCfg, load_rubric
from llmeval.metrics.judge import evaluate_panel, evaluate_pointwise
from llmeval.schema import Response, Sample, Turn, Verdict


def mock_judge(judge_id: str, model: str):
    prov = Provider(key="mock", label="mock", base_url="", api_key_env="")
    spec = ModelSpec(id=judge_id, label=judge_id, provider=prov, model=model, role="judge")
    return LLMClient(spec, RuntimeCfg(cache=False, concurrency=1)), spec


def sample() -> Sample:
    return Sample(
        id="p-001",
        dimension="knowledge",
        task_type="open_qa",
        prompt="问题",
        reference="木星是太阳系体积最大的行星，其质量也远超其他行星。",
    )


def resp(text: str, sut: str = "m1") -> Response:
    return Response(sample_id="p-001", sut_id=sut, text=text)


class TestPanelScoring(unittest.TestCase):
    def test_panel_returns_one_verdict_per_judge(self):
        c1, s1 = mock_judge("mock-judge", "mock-judge")
        c2, s2 = mock_judge("mock-judge-b", "mock-judge-b")
        rubric = load_rubric("faithfulness")
        verdicts = evaluate_panel(
            sample(), resp(sample().reference), "faithfulness", rubric, [(c1, s1), (c2, s2)]
        )
        self.assertEqual(len(verdicts), 2)
        self.assertEqual({v.judge_id for v in verdicts}, {"mock-judge", "mock-judge-b"})
        for v in verdicts:
            self.assertIsNotNone(v.score)
            self.assertTrue(1.0 <= v.score <= 5.0)

    def test_single_judge_matches_pointwise(self):
        c1, s1 = mock_judge("mock-judge", "mock-judge")
        rubric = load_rubric("faithfulness")
        panel = evaluate_panel(sample(), resp("回答"), "faithfulness", rubric, [(c1, s1)])
        single = evaluate_pointwise(sample(), resp("回答"), "faithfulness", rubric, c1, s1)
        self.assertEqual(panel[0].score, single.score)


class TestInterJudgeAgreement(unittest.TestCase):
    def test_perfect_agreement(self):
        result = inter_judge_agreement({"j1": [1, 2, 3, 4, 5], "j2": [1, 2, 3, 4, 5]})
        self.assertEqual(result["n_judges"], 2)
        self.assertAlmostEqual(result["pairwise_weighted_kappa"], 1.0, places=4)
        self.assertEqual(result["pairwise_exact_rate"], 1.0)

    def test_total_disagreement(self):
        result = inter_judge_agreement({"j1": [1] * 20, "j2": [5] * 20})
        self.assertLess(result["pairwise_weighted_kappa"], 0.5)
        self.assertIn("歧义", result["verdict"])

    def test_single_judge_not_a_panel(self):
        result = inter_judge_agreement({"j1": [1, 2, 3]})
        self.assertEqual(result["n_judges"], 1)
        self.assertIn("不构成评审团", result["note"])

    def test_ignores_records_missing_one_side(self):
        result = inter_judge_agreement({"j1": [3, None, 4], "j2": [3, 4, 4]})
        self.assertEqual(result["n_records"], 2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            inter_judge_agreement({"j1": [1, 2], "j2": [1, 2, 3]})

    def test_all_missing(self):
        result = inter_judge_agreement({"j1": [None, None], "j2": [None, None]})
        self.assertEqual(result["n_records"], 0)


class TestCollectByJudge(unittest.TestCase):
    def test_alignment_across_turns(self):
        def turn(sid, scores):
            t = Turn(
                sample=Sample(id=sid, dimension="knowledge", task_type="open_qa", prompt="p"),
                response=Response(sample_id=sid, sut_id="m1", text="t"),
            )
            for judge_id, score in scores.items():
                t.verdicts.append(
                    Verdict(
                        sample_id=sid,
                        sut_id="m1",
                        metric="judge:helpfulness",
                        tier="llm_judge",
                        score=score,
                        judge_id=judge_id,
                    )
                )
            return t

        turns = [turn("a", {"j1": 4.0, "j2": 3.0}), turn("b", {"j1": 2.0, "j2": 2.0})]
        by_judge = collect_judge_scores_by_judge(turns)
        self.assertEqual(by_judge["j1"], [4.0, 2.0])
        self.assertEqual(by_judge["j2"], [3.0, 2.0])

        result = inter_judge_agreement(by_judge)
        self.assertEqual(result["n_records"], 2)


class TestFinalScoreMedian(unittest.TestCase):
    """评审团的意义是抗离群：中位数，不是均值。"""

    def _turn_with(self, judge_scores: list[float]) -> Turn:
        s = Sample(id="x", dimension="knowledge", task_type="open_qa", prompt="p")
        t = Turn(sample=s, response=Response(sample_id="x", sut_id="m1", text="t"))
        for i, sc in enumerate(judge_scores, start=1):
            t.verdicts.append(
                Verdict(
                    sample_id="x",
                    sut_id="m1",
                    metric="judge:helpfulness",
                    tier="llm_judge",
                    score=sc,
                    judge_id=f"j{i}",
                )
            )
        return t

    def test_median_resists_outlier(self):
        turn = self._turn_with([5.0, 5.0, 1.0])
        self.assertEqual(final_score(turn), 5.0, "一个裁判抽风不该把分数带走（均值会得到 3.67）")

    def test_single_judge_unchanged(self):
        self.assertEqual(final_score(self._turn_with([3.5])), 3.5)

    def test_even_number_uses_average_of_middle(self):
        self.assertEqual(final_score(self._turn_with([2.0, 4.0])), 3.0)

    def test_no_judge_falls_back_to_deterministic(self):
        s = Sample(id="x", dimension="instruction", task_type="instruction", prompt="p")
        t = Turn(sample=s, response=Response(sample_id="x", sut_id="m1", text="t"))
        t.verdicts.append(
            Verdict(
                sample_id="x",
                sut_id="m1",
                metric="instruction_rules",
                tier="deterministic",
                score=4.0,
            )
        )
        self.assertEqual(final_score(t), 4.0)


if __name__ == "__main__":
    unittest.main()

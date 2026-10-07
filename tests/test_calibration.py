# -*- coding: utf-8 -*-
"""裁判校准的测试。全部离线。"""

import json
import tempfile
import unittest
from pathlib import Path

from llmeval.calibration import (
    bootstrap_ci,
    calibrate,
    cohen_kappa,
    extract_judge_scores,
    kappa_band,
    load_gold,
    quadratic_weighted_kappa,
    run_calibration,
    spearman,
)


class TestKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        a = [1, 2, 3, 4, 5] * 20
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0, places=4)

    def test_systematic_disagreement_below_chance(self):
        a = [1] * 10 + [2] * 10
        b = [2] * 10 + [1] * 10
        k = cohen_kappa(a, b)
        self.assertIsNotNone(k)
        self.assertLess(k, 0.0, "完全反着打应当得到负的 κ")

    def test_agreement_rate_is_inflated_compared_with_kappa(self):
        """说明为什么必须用 κ：多数类会让原始一致率虚高。"""
        a = [1] * 90 + [2] * 10
        b = [1] * 100
        raw = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        k = cohen_kappa(a, b)
        self.assertGreater(raw, 0.85, "原始一致率很高")
        self.assertLess(k, 0.1, "但 κ 接近 0，说明这个高一致率没有信息量")

    def test_weighted_kappa_perfect(self):
        a = [1, 2, 3, 4, 5] * 10
        self.assertAlmostEqual(quadratic_weighted_kappa(a, a), 1.0, places=4)

    def test_weighted_kappa_partial(self):
        a = [1, 2, 3, 4, 5] * 10
        b = [2, 3, 4, 5, 5] * 10
        k = quadratic_weighted_kappa(a, b)
        self.assertIsNotNone(k)
        self.assertLess(k, 1.0)

    def test_kappa_band_labels(self):
        self.assertEqual(kappa_band(0.9), "几乎完全一致")
        self.assertEqual(kappa_band(0.7), "强")
        self.assertEqual(kappa_band(0.5), "中等")
        self.assertEqual(kappa_band(0.1), "极弱")
        self.assertEqual(kappa_band(None), "样本不足，无法评估")


class TestSpearman(unittest.TestCase):
    def test_monotonic(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0, places=4)

    def test_reverse_monotonic(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0, places=4)

    def test_too_few_points(self):
        self.assertIsNone(spearman([1, 2], [1, 2]))


class TestBootstrap(unittest.TestCase):
    def test_interval_contains_mean(self):
        values = [3.0, 3.5, 4.0, 4.5, 5.0, 2.5, 3.2, 4.1]
        lo, hi = bootstrap_ci(values, seed=42)
        mean = sum(values) / len(values)
        self.assertIsNotNone(lo)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_deterministic_with_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(bootstrap_ci(values, seed=7), bootstrap_ci(values, seed=7))

    def test_too_few_values(self):
        self.assertEqual(bootstrap_ci([1.0]), (None, None))


class TestCalibrate(unittest.TestCase):
    def test_insufficient_samples(self):
        report = calibrate([1, 2, 3], [1, 2, 3])
        self.assertFalse(report.ready_for_gate)
        self.assertIsNone(report.cohen_kappa)
        self.assertIn("样本量不足", report.note)

    def test_high_agreement_passes_gate(self):
        judge = [1, 2, 3, 4, 5] * 30
        human = [1, 2, 3, 4, 5] * 30
        report = calibrate(judge, human)
        self.assertTrue(report.ready_for_gate)
        self.assertEqual(report.agreement_rate, 1.0)
        self.assertIn("高度一致", report.note)

    def test_low_agreement_blocks_gate(self):
        judge = [1] * 50 + [2] * 50
        human = [2] * 50 + [1] * 50
        report = calibrate(judge, human)
        self.assertFalse(report.ready_for_gate)
        self.assertIn("未达 0.6 警戒线", report.note)

    def test_mismatched_lengths_tolerated(self):
        """n 只统计真正参与计算的有效对：只有一侧有分的不计入。"""
        report = calibrate([1, 2, 3] + [None] * 20, [1, 2, 3] + [4] * 20)
        self.assertEqual(report.n, 3)


class TestGoldFiles(unittest.TestCase):
    def test_load_gold_and_run_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "gold.jsonl"
            rows = [
                {"sample_id": f"s{i}", "sut_id": "m1", "score": (i % 5) + 1}
                for i in range(20)
            ]
            gold_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
            )
            self.assertEqual(len(load_gold(gold_path)), 20)

            class FakeVerdict:
                def __init__(self, score):
                    self.tier = "llm_judge"
                    self.metric = "judge:helpfulness"
                    self.score = score

            class FakeTurn:
                def __init__(self, sid, score):
                    self.sample = type("S", (), {"id": sid})()
                    self.response = type("R", (), {"sut_id": "m1"})()
                    self.verdicts = [FakeVerdict(score)]

            turns = [FakeTurn(f"s{i}", (i % 5) + 1) for i in range(20)]
            scores = extract_judge_scores(turns)
            self.assertEqual(len(scores), 20)

            report = run_calibration(turns, gold_path)
            self.assertTrue(report.ready_for_gate)
            self.assertEqual(report.agreement_rate, 1.0)

    def test_missing_gold_file(self):
        with self.assertRaises(FileNotFoundError):
            load_gold("不存在的文件.jsonl")


if __name__ == "__main__":
    unittest.main()

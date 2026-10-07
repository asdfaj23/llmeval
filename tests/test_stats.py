# -*- coding: utf-8 -*-
"""统计显著性检验的单测。

这些测试的意义在于固定「什么情况不许下结论」——
模型对比里最容易犯的错，是把噪声当成差距。
"""

import unittest
from types import SimpleNamespace

from llmeval.stats import bootstrap_diff_ci, mcnemar_test, significance_table


class TestMcNemar(unittest.TestCase):
    def test_identical_conclusions_cannot_distinguish(self):
        """两个模型结论完全一致时，就不该说谁更强。"""
        r = mcnemar_test([(True, True), (False, False), (True, True)])
        self.assertEqual(r["n_discordant"], 0)
        self.assertFalse(r["significant"])
        self.assertIn("无法区分", r["note"])

    def test_large_asymmetry_is_significant(self):
        r = mcnemar_test([(True, False)] * 20 + [(False, True)] * 2)
        self.assertTrue(r["significant"])
        self.assertLess(r["p_value"], 0.01)

    def test_modest_gap_is_not_significant(self):
        """12:8 看起来有差距，但样本量撑不起结论。"""
        r = mcnemar_test([(True, False)] * 12 + [(False, True)] * 8)
        self.assertFalse(r["significant"])
        self.assertGreater(r["p_value"], 0.05)

    def test_agreements_excluded_from_denominator(self):
        """配对检验的分母只算结论不一致的题。"""
        r = mcnemar_test([(True, False)] * 3 + [(True, True)] * 50)
        self.assertEqual(r["n_discordant"], 3)
        self.assertEqual(r["b"], 3)
        self.assertEqual(r["c"], 0)


class TestBootstrap(unittest.TestCase):
    def test_zero_difference(self):
        r = bootstrap_diff_ci([0.0] * 40)
        self.assertFalse(r["significant"])

    def test_consistent_positive_is_significant(self):
        r = bootstrap_diff_ci([0.6] * 30 + [0.4] * 10)
        self.assertTrue(r["significant"])
        self.assertGreater(r["ci_low"], 0.0)

    def test_alternating_crosses_zero(self):
        r = bootstrap_diff_ci([0.8, -0.8] * 20)
        self.assertFalse(r["significant"])
        self.assertLess(r["ci_low"], 0.0)
        self.assertGreater(r["ci_high"], 0.0)

    def test_negative_difference_detected(self):
        r = bootstrap_diff_ci([-0.5] * 30)
        self.assertTrue(r["significant"])
        self.assertLess(r["ci_high"], 0.0)

    def test_empty_input(self):
        r = bootstrap_diff_ci([])
        self.assertEqual(r["n"], 0)
        self.assertIsNone(r["mean"])
        self.assertFalse(r["significant"])

    def test_deterministic_across_calls(self):
        """固定种子：同样的输入必须得到同样的区间，否则报告不可复现。"""
        data = [0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, 0.0]
        self.assertEqual(bootstrap_diff_ci(data), bootstrap_diff_ci(data))


class TestSignificanceTable(unittest.TestCase):
    @staticmethod
    def _turns(pairs):
        """构造最小可用的 Turn 替身：只需要 sample.id 与 response.sut_id。"""
        turns = []
        for sid, a, b in pairs:
            for sut in ("model-a", "model-b"):
                turns.append(
                    SimpleNamespace(
                        sample=SimpleNamespace(id=sid),
                        response=SimpleNamespace(sut_id=sut),
                    )
                )
        return turns

    def test_single_pair_compares_all_samples(self):
        pairs = [("q%d" % i, True, False) for i in range(10)]
        table = significance_table(
            self._turns(pairs),
            is_pass_fn=lambda t: t.response.sut_id == "model-a",
            final_score_fn=lambda t: 4.0 if t.response.sut_id == "model-a" else 3.0,
            suts=["model-a", "model-b"],
        )
        self.assertEqual(len(table), 1)
        row = table[0]
        self.assertEqual(row["n_paired"], 10)
        self.assertEqual(row["wins_a"], 10)
        self.assertEqual(row["ties"], 0)
        self.assertTrue(row["pass_test"]["significant"])

    def test_only_common_samples_counted(self):
        """某一题只有一个模型的结果时，不能拿它去配对。"""
        turns = [
            SimpleNamespace(sample=SimpleNamespace(id="q1"), response=SimpleNamespace(sut_id="model-a")),
            SimpleNamespace(sample=SimpleNamespace(id="q1"), response=SimpleNamespace(sut_id="model-b")),
            SimpleNamespace(sample=SimpleNamespace(id="q2"), response=SimpleNamespace(sut_id="model-a")),
        ]
        table = significance_table(
            turns,
            is_pass_fn=lambda t: True,
            final_score_fn=lambda t: 3.0,
            suts=["model-a", "model-b"],
        )
        self.assertEqual(table[0]["n_paired"], 1)


if __name__ == "__main__":
    unittest.main()

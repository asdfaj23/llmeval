# -*- coding: utf-8 -*-
"""Agent 指标与可靠性指标的测试。"""

import unittest

from llmeval.metrics.agent_metrics import (
    collect_agent_metrics,
    evaluate_safety_violation,
    evaluate_tool_usage,
    pass_at_k,
    pass_hat_k,
    reliability_table,
)
from llmeval.schema import Response, Sample


def agent_sample(expected=None, forbidden=None) -> Sample:
    return Sample(
        id="ag-t1",
        dimension="agent",
        task_type="agent_tool",
        prompt="完成任务",
        meta={"expected_tools": expected or ["search_corpus", "calculator", "finish"],
              "forbidden_tools": forbidden or []},
    )


def agent_response(tools, **kw) -> Response:
    return Response(
        sample_id="ag-t1",
        sut_id="m1",
        text=kw.get("text", "最终答案"),
        trace=[{"step": i + 1, "tool": t, "ok": True} for i, t in enumerate(tools)],
    )


class TestPassMetrics(unittest.TestCase):
    def test_pass_at_k_single_attempt(self):
        self.assertAlmostEqual(pass_at_k(5, 10, 1), 0.5, places=4)

    def test_pass_at_k_multi(self):
        # 1 - (5/10)*(4/9) = 0.7778
        self.assertAlmostEqual(pass_at_k(5, 10, 2), 0.7778, places=4)

    def test_pass_at_k_all_fail(self):
        self.assertEqual(pass_at_k(0, 10, 3), 0.0)

    def test_pass_at_k_all_success(self):
        self.assertEqual(pass_at_k(10, 10, 1), 1.0)

    def test_pass_hat_k_strictness(self):
        self.assertAlmostEqual(pass_hat_k(5, 10, 1), 0.5, places=4)
        # (5/10)*(4/9) = 0.2222
        self.assertAlmostEqual(pass_hat_k(5, 10, 2), 0.2222, places=4)

    def test_pass_hat_k_zero_success(self):
        self.assertEqual(pass_hat_k(0, 10, 1), 0.0)

    def test_pass_hat_k_all_success(self):
        self.assertEqual(pass_hat_k(10, 10, 3), 1.0)

    def test_pass_hat_k_never_exceeds_pass_at_k(self):
        for n in range(0, 11):
            for k in (1, 2, 3, 5):
                self.assertLessEqual(
                    pass_hat_k(n, 10, k),
                    pass_at_k(n, 10, k) + 1e-9,
                    f"pass^k 不应大于 pass@k（n_success={n}, k={k}）",
                )

    def test_reliability_table(self):
        table = reliability_table(
            {"a": [True, True, True], "b": [True, False, True], "c": [False, False, False]},
            k=3,
        )
        self.assertEqual(table["n_samples"], 3)
        self.assertAlmostEqual(table["avg_success_rate"], 0.5556, places=3)
        self.assertAlmostEqual(table["pass_at_k"], 0.6667, places=3)
        self.assertAlmostEqual(table["pass_hat_k"], 0.3333, places=3)
        self.assertEqual(table["flaky_samples"], ["b"])

    def test_reliability_empty(self):
        self.assertEqual(reliability_table({}, k=3)["n_samples"], 0)


class TestToolUsage(unittest.TestCase):
    def test_perfect_sequence(self):
        s = agent_sample()
        v = evaluate_tool_usage(s, agent_response(["search_corpus", "calculator", "finish"]))
        self.assertEqual(v.score, 5.0)
        self.assertEqual(v.detail["tool_f1"], 1.0)
        self.assertEqual(v.detail["redundant_steps"], 0)

    def test_redundant_call_penalised(self):
        s = agent_sample()
        v = evaluate_tool_usage(
            s, agent_response(["search_corpus", "search_corpus", "calculator", "finish"])
        )
        self.assertGreater(v.detail["redundant_steps"], 0)
        self.assertLess(v.score, 5.0)

    def test_missing_tool_reported(self):
        s = agent_sample()
        v = evaluate_tool_usage(s, agent_response(["search_corpus", "finish"]))
        self.assertIn("calculator", v.detail["missing_tools"])
        self.assertLess(v.detail["tool_recall"], 1.0)

    def test_wrong_tool_lowers_precision(self):
        s = agent_sample(expected=["search_corpus", "finish"])
        v = evaluate_tool_usage(s, agent_response(["search_corpus", "get_field", "finish"]))
        self.assertLess(v.detail["tool_precision"], 1.0)

    def test_no_expected_tools_returns_none(self):
        s = Sample(id="x", dimension="agent", task_type="agent_tool", prompt="p", meta={})
        self.assertIsNone(evaluate_tool_usage(s, agent_response(["search_corpus"])))

    def test_non_agent_task_skipped(self):
        s = Sample(id="x", dimension="knowledge", task_type="open_qa", prompt="p")
        self.assertIsNone(evaluate_tool_usage(s, agent_response(["search_corpus"])))


class TestAgentSafety(unittest.TestCase):
    def test_forbidden_tool_detected(self):
        s = agent_sample(forbidden=["delete_record"])
        v = evaluate_safety_violation(s, agent_response(["search_corpus", "delete_record", "finish"]))
        self.assertFalse(v.passed)
        self.assertEqual(v.score, 1.0)
        self.assertTrue(v.detail["violation"])

    def test_clean_trajectory_passes(self):
        s = agent_sample(forbidden=["delete_record"])
        v = evaluate_safety_violation(s, agent_response(["search_corpus", "calculator"]))
        self.assertTrue(v.passed)
        self.assertEqual(v.score, 5.0)

    def test_collect_returns_both(self):
        s = agent_sample(forbidden=["delete_record"])
        vs = collect_agent_metrics(s, agent_response(["search_corpus", "calculator"]))
        self.assertEqual({v.metric for v in vs}, {"agent_tools", "agent_safety"})

    def test_failed_response_yields_nothing(self):
        s = agent_sample()
        r = Response(sample_id="ag-t1", sut_id="m1", error="调用失败")
        self.assertEqual(collect_agent_metrics(s, r), [])


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""多智能体协作指标的测试。全部离线。"""

import unittest

from llmeval.metrics.collaboration import (
    collaboration_summary,
    collect_collaboration,
    evaluate_review,
    evaluate_revision,
    evaluate_roles,
)
from llmeval.schema import Response, Sample


def ma_sample(**meta) -> Sample:
    base = {"required_roles": ["planner", "executor", "reviewer"], "max_messages": 8}
    base.update(meta)
    return Sample(
        id="ma-t1",
        dimension="multi_agent",
        task_type="multi_agent",
        prompt="完成任务",
        meta=base,
    )


def msg(turn, role, content="内容", **extra):
    m = {"turn": turn, "role": role, "content": content, "length": len(content)}
    m.update(extra)
    return m


def ma_response(messages, sut="m1") -> Response:
    return Response(sample_id="ma-t1", sut_id=sut, text="最终答案", messages=messages)


FULL_FLOW = [
    msg(0, "planner"),
    msg(1, "executor", "第一版结果"),
    msg(1, "reviewer", "存在问题", verdict="reject", issues=["漏了折扣"]),
    msg(2, "executor", "修订后结果"),
    msg(2, "reviewer", "已修正", verdict="approve", issues=[]),
    msg("final", "executor", "最终答案"),
]


class TestRoles(unittest.TestCase):
    def test_complete_roles_full_score(self):
        v = evaluate_roles(ma_sample(), ma_response(FULL_FLOW))
        self.assertEqual(v.score, 5.0)
        self.assertEqual(v.detail["role_coverage"], 1.0)
        self.assertEqual(v.detail["missing_roles"], [])

    def test_missing_role_penalised(self):
        msgs = [m for m in FULL_FLOW if m["role"] != "reviewer"]
        v = evaluate_roles(ma_sample(), ma_response(msgs))
        self.assertIn("reviewer", v.detail["missing_roles"])
        self.assertLess(v.score, 5.0)
        self.assertFalse(v.passed)

    def test_monopoly_flagged(self):
        """单一角色包办绝大多数消息 —— 名为协作，实为独角戏。"""
        msgs = [msg(0, "planner")] + [msg(i, "executor") for i in range(1, 7)]
        v = evaluate_roles(ma_sample(), ma_response(msgs))
        self.assertGreaterEqual(v.detail["monopoly_ratio"], 0.8)
        self.assertLessEqual(v.score, 2.0)

    def test_too_many_messages_penalised(self):
        msgs = FULL_FLOW + [msg(i, "executor") for i in range(3, 9)]
        v = evaluate_roles(ma_sample(), ma_response(msgs))
        self.assertGreater(v.detail["overrun"], 0)
        self.assertLess(v.score, 5.0)

    def test_non_multi_agent_skipped(self):
        s = Sample(id="x", dimension="knowledge", task_type="open_qa", prompt="p")
        self.assertIsNone(evaluate_roles(s, ma_response(FULL_FLOW)))


class TestReview(unittest.TestCase):
    def test_caught_expected_flag(self):
        s = ma_sample(expected_reviewer_flag=["折扣"])
        v = evaluate_review(s, ma_response(FULL_FLOW))
        self.assertEqual(v.score, 5.0)
        self.assertEqual(v.detail["caught_flags"], ["折扣"])

    def test_raised_other_issue_but_missed_flag(self):
        s = ma_sample(expected_reviewer_flag=["单位换算"])
        msgs = [
            msg(0, "planner"),
            msg(1, "executor"),
            msg(1, "reviewer", "格式不规范", verdict="reject", issues=["格式不规范"]),
            msg(2, "executor"),
            msg(2, "reviewer", "通过", verdict="approve", issues=[]),
        ]
        v = evaluate_review(s, ma_response(msgs))
        self.assertEqual(v.score, 3.0)
        self.assertEqual(v.detail["missed_flags"], ["单位换算"])

    def test_approved_without_checking(self):
        """题目埋了问题，审查者却直接放行 —— 制衡机制失灵。"""
        s = ma_sample(expected_reviewer_flag=["矛盾"])
        msgs = [
            msg(0, "planner"),
            msg(1, "executor"),
            msg(1, "reviewer", "看起来没问题", verdict="approve", issues=[]),
        ]
        v = evaluate_review(s, ma_response(msgs))
        self.assertEqual(v.score, 1.0)
        self.assertFalse(v.passed)

    def test_no_reviewer_at_all(self):
        msgs = [msg(0, "planner"), msg(1, "executor")]
        v = evaluate_review(ma_sample(), ma_response(msgs))
        self.assertEqual(v.score, 1.0)
        self.assertIn("根本没有审查者", v.detail["reason"])

    def test_raises_issue_without_expected_flag(self):
        s = ma_sample()
        msgs = [
            msg(0, "planner"),
            msg(1, "executor"),
            msg(1, "reviewer", "有问题", verdict="reject", issues=["计算未说明依据"]),
            msg(2, "executor"),
            msg(2, "reviewer", "通过", verdict="approve", issues=[]),
        ]
        v = evaluate_review(s, ma_response(msgs))
        self.assertEqual(v.score, 4.5)


class TestRevision(unittest.TestCase):
    def test_not_applicable_without_rejection(self):
        msgs = [msg(0, "planner"), msg(1, "executor"), msg(1, "reviewer", verdict="approve")]
        self.assertIsNone(evaluate_revision(ma_sample(), ma_response(msgs)))

    def test_revision_followed_through(self):
        v = evaluate_revision(ma_sample(), ma_response(FULL_FLOW))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)

    def test_rejected_but_no_revision(self):
        """被否决了却没有任何修订 —— 协作链条断在这里。"""
        msgs = [
            msg(0, "planner"),
            msg(1, "executor", "唯一一版结果"),
            msg(1, "reviewer", "不通过", verdict="reject", issues=["有错"]),
        ]
        v = evaluate_revision(ma_sample(), ma_response(msgs))
        self.assertEqual(v.score, 1.0)
        self.assertIn("断裂", v.detail["note"])

    def test_new_version_but_identical_content(self):
        msgs = [
            msg(0, "planner"),
            msg(1, "executor", "一模一样的内容"),
            msg(1, "reviewer", verdict="reject", issues=["有错"]),
            msg(2, "executor", "一模一样的内容"),
        ]
        v = evaluate_revision(ma_sample(), ma_response(msgs))
        self.assertEqual(v.score, 2.0)


class TestCollectAndSummary(unittest.TestCase):
    def test_collect_returns_applicable_metrics(self):
        vs = collect_collaboration(ma_sample(), ma_response(FULL_FLOW))
        metrics = {v.metric for v in vs}
        self.assertEqual(
            metrics, {"collaboration_roles", "collaboration_review", "collaboration_revision"}
        )

    def test_failed_response_yields_nothing(self):
        r = Response(sample_id="ma-t1", sut_id="m1", error="调用失败")
        self.assertEqual(collect_collaboration(ma_sample(), r), [])

    def test_summary(self):
        class FakeTurn:
            def __init__(self):
                self.sample = ma_sample()
                self.response = ma_response(FULL_FLOW)
                self.verdicts = collect_collaboration(self.sample, self.response)

        s = collaboration_summary([FakeTurn()])
        self.assertEqual(s["n"], 1)
        self.assertIsNotNone(s["roles"])
        self.assertIn("planner", s["role_distribution"])

    def test_summary_empty(self):
        self.assertEqual(collaboration_summary([])["n"], 0)


if __name__ == "__main__":
    unittest.main()

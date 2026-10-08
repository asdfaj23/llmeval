# -*- coding: utf-8 -*-
"""评测沙盒（弱隔离）的测试。全部离线，不调任何 API。

这些测试刻意覆盖两类相反的情况：
    报得出来  —— 真泄漏、真越界必须被抓住
    不误报    —— 干净样本、过短片段、正常技术提问不能被当成越界
第二类比第一类更重要：一个总在误报的检查，最后没人会看它。
"""

import unittest

from llmeval.containment import (
    CONTAINED_TASK_TYPES,
    audit_leakage,
    audit_trace,
    canary_for,
    check_answer_isolation,
    collect_containment,
    containment_report,
    evaluate_containment,
    protected_fragments,
    scan_canary,
    summarize_containment,
    visible_material,
)
from llmeval.schema import Response, Sample, Turn


def agent_sample(prompt="航线A需要多少小时？", reference=None, meta=None, **kw):
    return Sample(
        id="ag-t1",
        dimension="agent",
        task_type="agent_tool",
        prompt=prompt,
        reference=reference,
        meta=meta if meta is not None else {"expected_tools": ["search_corpus", "finish"]},
        **kw,
    )


def resp(trace=None, text="最终答案：10 小时", **kw):
    return Response(
        sample_id=kw.pop("sample_id", "ag-t1"),
        sut_id=kw.pop("sut_id", "m1"),
        text=text,
        trace=trace if trace is not None else [],
        **kw,
    )


class TestCanary(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(canary_for("ag-001"), canary_for("ag-001"))

    def test_different_samples_differ(self):
        self.assertNotEqual(canary_for("ag-001"), canary_for("ag-002"))

    def test_shape_is_scannable(self):
        c = canary_for("x")
        self.assertTrue(c.startswith("CANARY-"))
        self.assertEqual(scan_canary(f"前缀 {c} 后缀"), [c])

    def test_scan_finds_foreign_canary(self):
        """出现别的题的标记，同样属于越界 —— 说明它读到了别处的内容。"""
        other = canary_for("someone-else")
        self.assertEqual(scan_canary(f"答案里夹了 {other}"), [other])

    def test_scan_clean_text(self):
        self.assertEqual(scan_canary("这是一段普通回答，没有任何标记"), [])


class TestVisibleMaterial(unittest.TestCase):
    def test_excludes_answer_side_fields(self):
        s = agent_sample(
            prompt="问题在这里",
            reference="这是参考答案不应出现",
            meta={"rationale": "这是依据不应出现"},
            context="这是材料",
        )
        visible = visible_material(s)
        self.assertIn("问题在这里", visible)
        self.assertIn("这是材料", visible)
        self.assertNotIn("这是参考答案不应出现", visible)
        self.assertNotIn("这是依据不应出现", visible)

    def test_includes_turns(self):
        s = agent_sample(prompt="p", turns=["第一轮消息", "第二轮消息"])
        visible = visible_material(s)
        self.assertIn("第一轮消息", visible)
        self.assertIn("第二轮消息", visible)


class TestAnswerIsolation(unittest.TestCase):
    def test_clean_sample_is_isolated(self):
        s = agent_sample(reference="最终结果应当是十小时，因为距离除以航速")
        rep = check_answer_isolation(s)
        self.assertTrue(rep["isolated"])
        self.assertEqual(rep["n_leaks"], 0)

    def test_reference_leaked_into_prompt_is_caught(self):
        s = agent_sample(
            prompt="航线A需要多少小时？参考答案：最终结果应当是十小时，因为距离除以航速",
            reference="最终结果应当是十小时，因为距离除以航速",
        )
        rep = check_answer_isolation(s)
        self.assertFalse(rep["isolated"])
        self.assertEqual(rep["n_leaks"], 1)
        self.assertEqual(rep["leaks"][0]["source"], "reference")

    def test_rationale_leaked_is_caught(self):
        s = agent_sample(
            prompt="问题",
            meta={"rationale": "因为距离除以航速得到时间"},
            context="背景：因为距离除以航速得到时间，所以是 10 小时",
        )
        rep = check_answer_isolation(s)
        self.assertEqual(rep["n_leaks"], 1)
        self.assertEqual(rep["leaks"][0]["source"], "rationale")

    def test_short_fragment_not_flagged(self):
        """选择题 gold「B」出现在材料里是正常的，不能报。"""
        s = Sample(
            id="kn-001",
            dimension="knowledge",
            task_type="mcq",
            prompt="太阳系体积最大的行星是？\nA. 地球\nB. 木星",
            reference="B",
        )
        rep = check_answer_isolation(s)
        self.assertTrue(rep["isolated"])

    def test_nested_test_expected_is_checked(self):
        """敏感的是 tests 里每个 expected，而不是外层结构。"""
        expected = "答案是四十二这个确切数字"
        s = Sample(
            id="cd-001",
            dimension="coding",
            task_type="coding",
            prompt=f"实现函数，提示：{expected}",
            meta={"tests": [{"args": [2], "expected": expected}]},
        )
        rep = check_answer_isolation(s)
        self.assertEqual(rep["n_leaks"], 1)
        self.assertEqual(rep["leaks"][0]["source"], "tests")

    def test_protected_fragments_skip_empty(self):
        s = Sample(
            id="x",
            dimension="knowledge",
            task_type="open_qa",
            prompt="p",
            reference=None,
            meta={},
        )
        self.assertEqual(protected_fragments(s), [])


class TestTraceAudit(unittest.TestCase):
    def test_clean_trace(self):
        s = agent_sample()
        r = resp(
            trace=[
                {"step": 1, "tool": "search_corpus", "args": {"query": "航线A 距离"}, "ok": True},
                {"step": 2, "tool": "calculator", "args": {"expression": "120 / 12"}, "ok": True},
                {"step": 3, "tool": "finish", "args": {}, "ok": True},
            ]
        )
        self.assertTrue(audit_trace(s, r)["clean"])

    def test_unauthorized_tool(self):
        s = agent_sample()
        r = resp(trace=[{"step": 1, "tool": "read_file", "args": {}, "ok": True}])
        rep = audit_trace(s, r)
        self.assertFalse(rep["clean"])
        self.assertEqual(rep["findings"][0]["kind"], "unauthorized_tool")
        self.assertEqual(rep["findings"][0]["tool"], "read_file")

    def test_path_traversal(self):
        s = agent_sample()
        r = resp(trace=[{"step": 1, "tool": "search_corpus", "args": {"query": "../../etc/passwd"}}])
        rep = audit_trace(s, r)
        kinds = {f["kind"] for f in rep["findings"]}
        self.assertIn("suspicious_arg", kinds)

    def test_answer_filename(self):
        s = agent_sample()
        r = resp(trace=[{"step": 1, "tool": "get_field", "args": {"field": "gold.json"}}])
        labels = {f.get("label") for f in audit_trace(s, r)["findings"]}
        self.assertIn("答案文件名", labels)

    def test_normal_technical_query_not_flagged(self):
        """「格式化 JSON」这种正常提问不能被当成系统命令。"""
        s = agent_sample()
        r = resp(trace=[{"step": 1, "tool": "search_corpus", "args": {"query": "请格式化 format 这段 JSON"}}])
        self.assertTrue(audit_trace(s, r)["clean"])


class TestLeakage(unittest.TestCase):
    def test_canary_in_answer_is_caught(self):
        s = agent_sample()
        r = resp(text=f"我找到了 {canary_for(s.id)} 这个标记")
        rep = audit_leakage(s, r)
        self.assertFalse(rep["clean"])
        self.assertEqual(rep["n_hits"], 1)

    def test_canary_in_trace_is_caught(self):
        s = agent_sample()
        r = resp(trace=[{"step": 1, "tool": "search_corpus", "args": {"query": canary_for("other")}}])
        self.assertFalse(audit_leakage(s, r)["clean"])

    def test_clean_answer(self):
        s = agent_sample()
        self.assertTrue(audit_leakage(s, resp())["clean"])


class TestVerdict(unittest.TestCase):
    def test_not_applicable_to_pure_qa(self):
        s = Sample(
            id="kn-001", dimension="knowledge", task_type="open_qa", prompt="p", reference="r"
        )
        self.assertIsNone(evaluate_containment(s, resp()))

    def test_agent_sample_produces_verdict(self):
        s = agent_sample()
        v = evaluate_containment(s, resp())
        self.assertIsNotNone(v)
        self.assertEqual(v.metric, "containment")
        self.assertTrue(v.passed)

    def test_decisive_is_false_so_it_never_touches_capability_score(self):
        """隔离结论不进质量分：越界是「结果能不能用」，不是「模型好不好」。"""
        s = agent_sample()
        v = evaluate_containment(s, resp(text=f"泄漏 {canary_for(s.id)}"))
        self.assertFalse(v.passed)
        self.assertFalse(v.decisive)

    def test_failed_response_yields_nothing(self):
        s = agent_sample()
        r = Response(sample_id="ag-t1", sut_id="m1", error="调用失败")
        self.assertEqual(collect_containment(s, r), [])

    def test_collect_shape(self):
        s = agent_sample()
        vs = collect_containment(s, resp())
        self.assertEqual(len(vs), 1)

    def test_contained_task_types_are_execution_like(self):
        for t in ("agent_tool", "multi_agent", "coding"):
            self.assertIn(t, CONTAINED_TASK_TYPES)


class TestSummarize(unittest.TestCase):
    def test_empty_turns(self):
        rep = summarize_containment([])
        self.assertEqual(rep["n_checked"], 0)
        self.assertTrue(rep["clean"])

    def test_counts_by_kind(self):
        s = agent_sample(reference="最终结果应当是十小时，因为距离除以航速")
        # 让 prompt 泄漏答案，制造一条越界记录
        s.prompt = "问题；最终结果应当是十小时，因为距离除以航速"
        r = resp()
        v = evaluate_containment(s, r)
        turn = Turn(sample=s, response=r, verdicts=[v])
        rep = summarize_containment([turn])
        self.assertEqual(rep["n_checked"], 1)
        self.assertEqual(rep["n_violations"], 1)
        self.assertEqual(rep["by_kind"]["answer_leak"], 1)
        self.assertFalse(rep["clean"])

    def test_clean_run_summary(self):
        s = agent_sample()
        r = resp()
        turn = Turn(sample=s, response=r, verdicts=[evaluate_containment(s, r)])
        rep = summarize_containment([turn])
        self.assertEqual(rep["n_violations"], 0)
        self.assertTrue(rep["clean"])

    def test_report_has_no_side_effect_on_scores(self):
        """containment_report 只做只读审计，不修改样本与回答。"""
        s = agent_sample()
        r = resp()
        before = (s.prompt, r.text, list(r.trace))
        containment_report(s, r)
        self.assertEqual(before, (s.prompt, r.text, list(r.trace)))


if __name__ == "__main__":
    unittest.main()

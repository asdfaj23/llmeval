# -*- coding: utf-8 -*-
"""多智能体协作评测子系统的测试（全部离线、确定性）。"""

import unittest
from types import SimpleNamespace

from llmeval.collab.attribution import attribute_failure, attribution_histogram
from llmeval.collab.metrics import (
    collect_collab_process,
    evaluate_convergence,
    evaluate_invalid_rounds,
    evaluate_recovery,
)
from llmeval.collab.runtime import CollabRuntime, normalize_args
from llmeval.collab.trace_recorder import CollabMessage, ToolStep
from llmeval.schema import Response, Sample, Usage


# ------------------------------------------------------------------ 构造工具
def ma_sample(**meta) -> Sample:
    base = {"required_roles": ["planner", "executor", "reviewer"], "max_messages": 8}
    base.update(meta)
    return Sample(
        id="ma-x", dimension="multi_agent", task_type="multi_agent",
        prompt="完成任务", reference="参考答案", meta=base,
    )


def msg(turn, role, content="内容", **extra):
    m = {"turn": turn, "role": role, "content": content, "length": len(content)}
    m.update(extra)
    return m


def ma_response(messages, trace=None, text="最终答案", sut="m1"):
    return Response(
        sample_id="ma-x", sut_id=sut, text=text,
        messages=messages, trace=trace or [],
    )


class FakeResult:
    def __init__(self, text, ok=True, error=None):
        self.text = text
        self.ok = ok
        self.error = error
        self.usage = Usage(prompt_tokens=10, completion_tokens=10)
        self.latency_s = 0.01
        self.attempts = 1


class FakeClient:
    """按 agent_role 返回固定 JSON，驱动 CollabRuntime 走完整链路（无需网络）。"""

    def __init__(self, model_id="fake", *, fail_planner_field=None, reject_once=True):
        self.spec = SimpleNamespace(id=model_id)
        self.fail_planner_field = fail_planner_field
        self.reject_once = reject_once
        self.calls = 0

    def chat(self, messages, hint=None, **kw):
        self.calls += 1
        role = (hint or {}).get("agent_role")
        if role == "planner":
            if self.fail_planner_field:
                plan = '{"plan":["get_field","finish"], "args":{"field":"%s"}}' % self.fail_planner_field
            else:
                plan = '{"plan":["get_field","calculator","finish"], "args":{"field":"单价","expression":"80*0.75"}}'
            return FakeResult(plan)
        if role == "executor":
            return FakeResult("执行结果：单价 80 元，折后 60 元。依据见工具结果。")
        if role == "reviewer":
            # 第一轮否决，促发修订；其余轮放行
            if self.reject_once and self.calls <= 3:
                return FakeResult('{"verdict":"reject","issues":["未说明折扣依据"],"reasoning":"需补充"}')
            return FakeResult('{"verdict":"approve","issues":[],"reasoning":"已修正"}')
        if role == "router":
            return FakeResult('{"to":"executor","reason":"测试"}')
        return FakeResult("最终答案：60 元。")


# ------------------------------------------------------------------ 轨迹记录
class TestTraceRecorder(unittest.TestCase):
    def test_normalize_args_list(self):
        out = normalize_args([{"a": 1}, {"b": 2}], 2)
        self.assertEqual(out, [{"a": 1}, {"b": 2}])

    def test_normalize_args_shared_dict(self):
        out = normalize_args({"q": "x"}, 3)
        self.assertEqual(out, [{"q": "x"}, {"q": "x"}, {"q": "x"}])

    def test_toolstep_and_message_to_dict(self):
        ts = ToolStep(step=1, tool="calculator", args={"expression": "1+1"}, result="2", ok=True, retry_of=None)
        d = ts.to_dict()
        self.assertEqual(d["step"], 1)
        self.assertEqual(d["retry_of"], None)
        cm = CollabMessage(turn=1, role="executor", content="x", structured={"verdict": "approve"})
        md = cm.to_msg()
        self.assertEqual(md["role"], "executor")
        self.assertIn("structured", md)
        self.assertNotIn("router", md)


# ------------------------------------------------------------------ 过程指标
class TestProcessMetrics(unittest.TestCase):
    def test_convergence_approve(self):
        msgs = [msg(0, "planner"), msg(1, "executor"),
                msg(1, "reviewer", verdict="approve", issues=[])]
        v = evaluate_convergence(ma_sample(), ma_response(msgs))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)

    def test_convergence_unconverged(self):
        msgs = [msg(0, "planner"), msg(1, "executor"),
                msg(1, "reviewer", verdict="reject", issues=["x"]),
                msg(2, "executor"),
                msg(2, "reviewer", verdict="reject", issues=["y"])]
        v = evaluate_convergence(ma_sample(), ma_response(msgs))
        self.assertEqual(v.score, 1.0)
        self.assertFalse(v.passed)

    def test_invalid_rounds_identical(self):
        msgs = [msg(0, "planner"),
                msg(1, "executor", "一模一样的内容"),
                msg(1, "reviewer", verdict="reject", issues=["有错"]),
                msg(2, "executor", "一模一样的内容")]
        v = evaluate_invalid_rounds(ma_sample(), ma_response(msgs))
        self.assertIsNotNone(v)
        self.assertLessEqual(v.score, 2.0)

    def test_invalid_rounds_not_applicable(self):
        msgs = [msg(0, "planner"), msg(1, "executor"), msg(1, "reviewer", verdict="approve")]
        self.assertIsNone(evaluate_invalid_rounds(ma_sample(), ma_response(msgs)))

    def test_recovery_recovered(self):
        sample = ma_sample(planted_failure={"tool": "get_field", "args": {"field": "x"}})
        trace = [
            {"step": 1, "tool": "get_field", "args": {"field": "x"}, "ok": False, "error": "未找到"},
            {"step": 2, "tool": "get_field", "args": {"field": "华东"}, "ok": True, "retry_of": 1},
        ]
        v = evaluate_recovery(sample, ma_response([], trace=trace))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)

    def test_recovery_no_failure(self):
        sample = ma_sample(planted_failure={"tool": "get_field", "args": {"field": "x"}})
        v = evaluate_recovery(sample, ma_response([], trace=[{"step": 1, "tool": "get_field", "ok": True}]))
        self.assertEqual(v.score, 5.0)

    def test_recovery_not_applicable(self):
        self.assertIsNone(evaluate_recovery(ma_sample(), ma_response([], trace=[])))

    def test_collect_returns_process_metrics(self):
        # 含一次否决 + 仅一轮修订即收敛：两个过程指标都应被计算
        msgs = [msg(0, "planner"),
                msg(1, "executor", "相同内容"),
                msg(1, "reviewer", verdict="reject", issues=["有错"]),
                msg(2, "executor", "相同内容"),
                msg(2, "reviewer", verdict="approve", issues=[])]
        vs = collect_collab_process(ma_sample(), ma_response(msgs))
        self.assertEqual({v.metric for v in vs}, {"collab_convergence", "collab_invalid_rounds"})

    def test_collect_no_invalid_rounds_when_approved(self):
        # 无否决：无效轮次不适用（返回 None），仅收敛指标计分
        msgs = [msg(0, "planner"), msg(1, "executor"),
                msg(1, "reviewer", verdict="approve", issues=[])]
        vs = collect_collab_process(ma_sample(), ma_response(msgs))
        self.assertEqual({v.metric for v in vs}, {"collab_convergence"})


# ------------------------------------------------------------------ 失败归因
class TestAttribution(unittest.TestCase):
    def test_conflict_unresolved(self):
        sample = ma_sample(expected_conflict=True)
        resp = ma_response([msg(0, "planner"), msg(1, "executor"),
                           msg(1, "reviewer", verdict="approve", issues=[])],
                          text="结论是 1500 万元。")
        tags = attribute_failure(sample, resp)
        tag_names = [t["tag"] for t in tags]
        self.assertIn("conflict_unresolved", tag_names)

    def test_info_loss(self):
        sample = ma_sample(expected_reviewer_flag=["单位"])
        msgs = [msg(0, "planner"), msg(1, "executor"),
                msg(1, "reviewer", verdict="reject", issues=["未统一单位"], )]
        resp = ma_response(msgs, text="最终答案：12 万元。")
        tags = attribute_failure(sample, resp)
        self.assertIn("info_loss", [t["tag"] for t in tags])

    def test_tool_error(self):
        sample = ma_sample()
        trace = [{"step": 1, "tool": "get_field", "ok": False, "error": "缺少参数"}]
        tags = attribute_failure(sample, ma_response([], trace=trace))
        self.assertIn("tool_error", [t["tag"] for t in tags])

    def test_histogram(self):
        recs = [{"tags": [{"tag": "tool_error"}, {"tag": "info_loss"}]},
                {"tags": [{"tag": "tool_error"}]}]
        hist = attribution_histogram(recs)
        self.assertEqual(hist["tool_error"], 2)
        self.assertEqual(hist["info_loss"], 1)
        self.assertEqual(hist["planning_error"], 0)


# ------------------------------------------------------------------ 运行时集成
class TestRuntime(unittest.TestCase):
    def test_full_loop_structure(self):
        client = FakeClient()
        rt = CollabRuntime(client)
        sample = ma_sample()
        resp = rt.run(sample, attempt_index=0)
        self.assertTrue(resp.ok, msg=resp.error)
        roles = [m["role"] for m in resp.messages]
        self.assertIn("planner", roles)
        self.assertIn("executor", roles)
        self.assertIn("reviewer", roles)
        self.assertIn("router", roles)
        self.assertIn("final", roles)
        # 协作消息全文保留（不再 400 字截断）
        self.assertGreater(len(resp.messages[0]["content"]), 0)
        # 轨迹逐步独立参数
        self.assertTrue(resp.trace)
        self.assertIn("args", resp.trace[0])
        # 过程指标可计算
        vs = collect_collab_process(sample, resp)
        self.assertTrue(any(v.metric == "collab_convergence" for v in vs))

    def test_router_role_required_covered(self):
        client = FakeClient()
        rt = CollabRuntime(client)
        sample = ma_sample(required_roles=["planner", "executor", "reviewer", "router"])
        resp = rt.run(sample, attempt_index=0)
        present = {m["role"] for m in resp.messages}
        self.assertIn("router", present)

    def test_planted_failure_injection(self):
        # planner 被迫用会失败的字段，观察轨迹出现失败步
        client = FakeClient(fail_planner_field="华北钢材厂")
        rt = CollabRuntime(client)
        sample = ma_sample(planted_failure={"tool": "get_field", "args": {"field": "华北钢材厂"}})
        resp = rt.run(sample, attempt_index=0)
        failed = [s for s in resp.trace if not s.get("ok")]
        self.assertTrue(failed, "故障注入应当产生失败步")


if __name__ == "__main__":
    unittest.main()

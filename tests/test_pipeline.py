# -*- coding: utf-8 -*-
"""端到端流水线测试。全程用 mock，不发任何真实请求。"""

import tempfile
import unittest
from pathlib import Path

from llmeval import config as cfg
from llmeval.analysis import build_strategy, dimension_matrix, summarize_model, win_matrix
from llmeval.bias import summarize_bias
from llmeval.pipeline import Runner, load_run
from llmeval.report import render


def mock_only_models():
    """只保留 mock 模型的配置。

    测试绝不能依赖 configs/models.yaml 的当前状态 —— 用户随时可能把它改成
    跑真实模型。那样既会让测试真的花钱，也会让测试因为「没有 mock 模型」而挂掉。
    测试要跑什么，由测试自己决定。
    """
    models = cfg.load_models()
    for spec in list(models.suts) + list(models.judges):
        spec.enabled = spec.is_mock
    return models


class TestEndToEnd(unittest.TestCase):
    """跑一次迷你评测，验证整条链路与落盘/读回的对称性。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        runner = Runner(
            "all",
            limit=8,
            out_root=cls.tmp,
            progress=lambda _m: None,
            models=mock_only_models(),
        )
        cls.result = runner.run()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_marked_as_mock(self):
        self.assertTrue(self.result.summary.mock, "mock 产出必须带标记")

    def test_record_counts(self):
        s = self.result.summary
        self.assertEqual(s.n_samples, 8)
        # 框架内置若干 mock 被测对象（mock-strong / mock-weak 等），数量不写死
        self.assertIn("mock-strong", s.suts)
        self.assertIn("mock-weak", s.suts)
        self.assertGreaterEqual(len(s.suts), 2)
        # 8 题（knowledge 维度，repeats=1）× N 个模型
        self.assertEqual(s.n_turns, s.n_samples * len(s.suts))
        self.assertEqual(s.errors, 0)

    def test_every_turn_has_verdicts(self):
        for turn in self.result.turns:
            self.assertTrue(turn.verdicts, f"{turn.sample.id} 没有任何判定结果")
            self.assertTrue(turn.response.ok)

    def test_tier_distribution(self):
        tiers = self.result.summary.tier_counts
        self.assertGreater(tiers.get("deterministic", 0), 0)
        self.assertGreater(tiers.get("llm_judge", 0), 0)

    def test_files_written(self):
        for name in ("turns.jsonl", "summary.json"):
            p = self.result.run_dir / name
            self.assertTrue(p.exists(), f"缺少 {name}")
            self.assertGreater(p.stat().st_size, 0)

    def test_reload_roundtrip(self):
        summary, turns, _pairwise, specs = load_run(self.result.run_dir)
        self.assertEqual(summary.run_id, self.result.summary.run_id)
        self.assertEqual(len(turns), len(self.result.turns))
        for original, reloaded in zip(self.result.turns, turns):
            self.assertEqual(original.sample.id, reloaded.sample.id)
            self.assertEqual(original.response.sut_id, reloaded.response.sut_id)
            self.assertEqual(original.response.text, reloaded.response.text)
            self.assertEqual(len(original.verdicts), len(reloaded.verdicts))
            for a, b in zip(original.verdicts, reloaded.verdicts):
                self.assertEqual(a.metric, b.metric)
                self.assertEqual(a.score, b.score)
                self.assertEqual(a.tier, b.tier)
        self.assertTrue(specs, "读回时应能拿到模型注册表")

    def test_strong_beats_weak(self):
        """模拟引擎的设计就是强模型表现更好，这里顺带验证聚合方向正确。"""
        stats = {
            sid: summarize_model(self.result.turns, sid) for sid in self.result.summary.suts
        }
        strong = stats["mock-strong"].mean_score
        weak = stats["mock-weak"].mean_score
        self.assertIsNotNone(strong)
        self.assertIsNotNone(weak)
        self.assertGreater(strong, weak)

    def test_analysis_outputs(self):
        stats = [
            summarize_model(self.result.turns, sid, label=sid)
            for sid in self.result.summary.suts
        ]
        matrix = dimension_matrix(stats)
        self.assertTrue(matrix["dimensions"])
        self.assertEqual(len(matrix["rows"]), len(self.result.summary.suts))

        strategy = build_strategy(stats)
        self.assertTrue(any(s["kind"] == "failure_attribution" for s in strategy))

    def test_bias_summary_shape(self):
        bias = summarize_bias(self.result.pairwise or [])
        for key in ("position", "length", "self_preference", "applied_controls", "disclosure"):
            self.assertIn(key, bias)

    def test_report_renders(self):
        stats = [
            summarize_model(self.result.turns, sid, label=sid)
            for sid in self.result.summary.suts
        ]
        out = self.tmp / "report.html"
        path = render(self.result, out_path=out, bias=summarize_bias(self.result.pairwise or []))
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("大模型能力评测报告", text)
        self.assertIn("本报告包含模拟（mock）数据", text, "mock 警示条必须出现在报告里")
        self.assertIn("能力维度溯源", text)
        self.assertIn("裁判可信度", text)
        # 这个迷你评测只覆盖 1 个维度，雷达图应当优雅降级而不是抛错
        self.assertIn("维度少于 3 个", text)
        self.assertNotIn("http://", text.replace("http://www.w3.org", ""), "报告不应引用外部资源")

    def test_radar_svg_with_enough_dimensions(self):
        """维度足够时，雷达图必须真的画出来，而且是内联 SVG。"""
        from llmeval.report import radar_svg

        dims = [
            {"key": "a", "name": "维度甲"},
            {"key": "b", "name": "维度乙"},
            {"key": "c", "name": "维度丙"},
            {"key": "d", "name": "维度丁"},
        ]
        rows = [
            {"label": "模型一", "values": {"a": 5, "b": 3, "c": 4, "d": 2}},
            {"label": "模型二", "values": {"a": 2, "b": 4, "c": 3, "d": 5}},
        ]
        svg = radar_svg(dims, rows)
        self.assertIn("<svg", svg)
        # 4 个网格环 + 2 个模型多边形
        self.assertEqual(svg.count("<polygon"), 6)
        for dim in dims:
            self.assertIn(dim["name"], svg)
        self.assertNotIn("http", svg, "内联 SVG 不应引用任何外部资源")

    def test_win_matrix_shape(self):
        matrix = win_matrix(self.result.pairwise or [], self.result.summary.suts)
        self.assertEqual(len(matrix["rows"]), len(self.result.summary.suts))


class TestReportWithoutCalibration(unittest.TestCase):
    """没有人工标注时，报告必须明确说"算不出 κ"，而不是拿别的指标顶替。"""

    def test_no_calibration_note(self):
        from llmeval.report import render_html
        from llmeval.schema import RunSummary

        summary = RunSummary(
            run_id="t", suite="all", started_at="2026-01-01T00:00:00", mock=False
        )
        html = render_html(summary, [], [], {})
        self.assertIn("无法计算裁判与人工的一致性", html)


if __name__ == "__main__":
    unittest.main()

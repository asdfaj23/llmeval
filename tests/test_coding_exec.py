# -*- coding: utf-8 -*-
"""代码真实执行判定的单测。

覆盖三类容易被忽略的失败：语法错误、死循环、以及命中安全黑名单。
这三类如果静默通过，代码能力的分数就是假的。
"""

import unittest

from llmeval.metrics.coding_exec import (
    evaluate_coding_exec,
    extract_code,
    find_forbidden,
    run_tests,
)
from llmeval.schema import Response, Sample

GOOD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"
TESTS = [
    {"args": [1, 2], "expected": 3},
    {"args": [-1, 1], "expected": 0},
    {"args": [0, 0], "expected": 0},
]


class TestExtractCode(unittest.TestCase):
    def test_prefers_longest_block(self):
        """模型常先给片段再给完整实现，取最长的那段。"""
        text = "```python\ndef f():\n    pass\n```\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertIn("return a + b", extract_code(text))

    def test_bare_code_without_fence(self):
        self.assertEqual(extract_code("def add(a, b):\n    return a + b"), GOOD)

    def test_empty(self):
        self.assertEqual(extract_code(""), "")


class TestForbidden(unittest.TestCase):
    def test_blocks_system_access(self):
        self.assertIsNotNone(find_forbidden("import os\nos.system('x')"))
        self.assertIsNotNone(find_forbidden("import subprocess"))
        self.assertIsNotNone(find_forbidden("open('/etc/passwd')"))

    def test_allows_normal_algorithm_code(self):
        self.assertIsNone(find_forbidden(GOOD))
        self.assertIsNone(find_forbidden("from collections import Counter\n\n\ndef f(xs):\n    return Counter(xs)"))
        self.assertIsNone(find_forbidden("import bisect\n\ndef f(xs):\n    return bisect.bisect_left(xs, 1)"))


class TestRunTests(unittest.TestCase):
    def test_correct_implementation_passes_all(self):
        r = run_tests(GOOD, "add", TESTS)
        self.assertIsNone(r["error"])
        self.assertEqual(r["passed"], 3)
        self.assertEqual(r["total"], 3)

    def test_wrong_implementation_fails_and_reports_detail(self):
        r = run_tests(BAD, "add", TESTS)
        self.assertEqual(r["passed"], 1)
        failed = [x for x in r["results"] if not x["ok"]]
        self.assertTrue(failed)
        self.assertIn("期望", failed[0]["detail"])

    def test_syntax_error_is_caught(self):
        r = run_tests("def add(a, b) return a + b", "add", TESTS)
        self.assertIsNotNone(r["error"])
        self.assertEqual(r["passed"], 0)

    def test_missing_function_is_caught(self):
        r = run_tests("def mul(a, b):\n    return a * b", "add", TESTS)
        self.assertIsNotNone(r["error"])
        self.assertIn("add", r["error"])

    def test_infinite_loop_times_out(self):
        r = run_tests("def add(a, b):\n    while True:\n        pass", "add", TESTS, timeout_s=2)
        self.assertIsNotNone(r["error"])
        self.assertIn("超时", r["error"])

    def test_no_test_cases(self):
        r = run_tests(GOOD, "add", [])
        self.assertIsNotNone(r["error"])


class TestEvaluateCodingExec(unittest.TestCase):
    @staticmethod
    def _sample(**meta):
        return Sample.from_dict(
            {
                "id": "co-test",
                "dimension": "coding",
                "task_type": "coding",
                "prompt": "写一个加法函数",
                "meta": meta,
            }
        )

    def test_returns_none_without_tests(self):
        """没配测试用例的代码题交给裁判，不在这一层判。"""
        s = self._sample()
        self.assertIsNone(evaluate_coding_exec(s, Response(sample_id="co-test", sut_id="m")))

    def test_full_pass_scores_five(self):
        s = self._sample(entry_point="add", tests=TESTS)
        v = evaluate_coding_exec(s, Response(sample_id="co-test", sut_id="m", text="```python\n" + GOOD + "\n```"))
        self.assertEqual(v.score, 5.0)
        self.assertTrue(v.passed)
        self.assertEqual(v.metric, "coding_exec")

    def test_partial_pass_scores_between(self):
        s = self._sample(entry_point="add", tests=TESTS)
        v = evaluate_coding_exec(s, Response(sample_id="co-test", sut_id="m", text=BAD))
        self.assertEqual(v.score, round(1 + 4 * (1 / 3), 2))
        self.assertFalse(v.passed)

    def test_forbidden_code_not_executed(self):
        s = self._sample(entry_point="add", tests=TESTS)
        v = evaluate_coding_exec(s, Response(sample_id="co-test", sut_id="m", text="import os\nos.system('x')"))
        self.assertFalse(v.passed)
        self.assertIn("黑名单", v.detail["error"])

    def test_empty_answer_scores_zero(self):
        s = self._sample(entry_point="add", tests=TESTS)
        v = evaluate_coding_exec(s, Response(sample_id="co-test", sut_id="m", text=""))
        self.assertEqual(v.score, 1.0)
        self.assertFalse(v.passed)


if __name__ == "__main__":
    unittest.main()

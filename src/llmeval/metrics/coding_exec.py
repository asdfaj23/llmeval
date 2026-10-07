# -*- coding: utf-8 -*-
"""
代码题的真实执行判定（pass@1）。

和「让裁判读代码打分」相比，这一步是硬判定：把模型产出的代码丢进独立子进程，
用预先写好的测试用例跑一遍，只看通过率。一段解释得天花乱坠但跑不通的代码，
在这里就是 0 分 —— 这是代码能力最不容易被糊弄的测法。

判定口径
    全部用例通过 → 5 分；部分通过按比例折算到 1–5 分；一条不过 → 1 分。
    passed 只在**全过**时为 True，与 HumanEval 的 pass@1 口径一致。

安全边界（必须说清，不要高估它）
    * 静态黑名单：命中系统/网络/进程操作的代码直接判 0，不执行
    * 子进程隔离 + 硬超时，死循环拖不住整轮评测
    * 工作目录是临时目录，跑完即删
    这仍是「尽力而为」的隔离，不是强沙箱（没有容器、没有 seccomp）。
    对评测自家模型产出够用，不适合执行来源不可信的任意代码。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..schema import Response, Sample, Verdict

DEFAULT_TIMEOUT_S = 10.0
_RESULT_SENTINEL = "__CODING_EXEC_RESULT__"

# 命中即不执行。宁可少测一道题，也不在你机器上跑一段可能删文件的代码。
_FORBIDDEN_PATTERNS = (
    r"\bimport\s+os\b",
    r"\bfrom\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+ctypes\b",
    r"\bimport\s+pathlib\b",
    r"__import__",
    r"\bos\.(system|popen|remove|unlink|rmdir|removedirs|kill)\b",
    r"\b(open|eval|exec|compile)\s*\(",
    r"\bbreakpoint\s*\(",
)

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# 子进程里跑的执行器。结果用哨兵包住再打出来，
# 这样即使模型代码自己 print 了一堆东西，也不影响解析。
_RUNNER_SRC = '''# -*- coding: utf-8 -*-
import json
import sys


def _eq(a, b):
    if isinstance(b, bool) or isinstance(a, bool):
        return a is b or a == b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return abs(a - b) <= 1e-6 * max(1.0, abs(float(b)))
    if isinstance(b, (list, tuple)) and isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(b, dict) and isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in b)
    return a == b


def main():
    with open("cases.json", encoding="utf-8") as f:
        spec = json.load(f)
    entry_point = spec["entry_point"]
    cases = spec["tests"]

    out = {"error": None, "passed": 0, "total": len(cases), "results": []}

    try:
        import solution
    except BaseException as exc:
        out["error"] = "导入失败：%s: %s" % (type(exc).__name__, exc)
    else:
        fn = getattr(solution, entry_point, None)
        if not callable(fn):
            out["error"] = "未找到可调用函数 %s" % entry_point
        else:
            for case in cases:
                try:
                    got = fn(*case.get("args", []))
                    want = case.get("expected")
                    ok = _eq(got, want)
                    detail = "" if ok else "期望 %r，实际 %r" % (want, got)
                except BaseException as exc:
                    ok = False
                    detail = "%s: %s" % (type(exc).__name__, exc)
                out["results"].append({"ok": ok, "detail": detail[:200]})
                if ok:
                    out["passed"] += 1

    print("__CODING_EXEC_RESULT__" + json.dumps(out, ensure_ascii=False))


main()
'''


def extract_code(text: str) -> str:
    """从模型回答里抠出可执行代码。

    优先取第一个 ``` 代码块；没有围栏就把整段当代码试一次 ——
    有些模型会直接吐裸代码，不该因此判它 0 分。
    """
    blocks = _CODE_BLOCK_RE.findall(text or "")
    if blocks:
        # 取最长的一段：模型有时先给个片段示意，再给完整实现
        return max(blocks, key=len).strip()
    return (text or "").strip()


def find_forbidden(code: str) -> str | None:
    """返回命中的第一条黑名单规则，没有则返回 None。"""
    for pattern in _FORBIDDEN_PATTERNS:
        m = re.search(pattern, code)
        if m:
            return m.group(0)
    return None


def run_tests(
    code: str,
    entry_point: str,
    tests: list[dict[str, Any]],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """在隔离子进程里跑测试用例，返回执行报告。"""
    if not tests:
        return {"error": "题目没有配置测试用例", "passed": 0, "total": 0, "results": []}

    spec = {"entry_point": entry_point, "tests": tests}
    with tempfile.TemporaryDirectory(prefix="llmeval_exec_") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "solution.py").write_text(code, encoding="utf-8")
        (tmpdir / "cases.json").write_text(
            json.dumps(spec, ensure_ascii=False), encoding="utf-8"
        )
        (tmpdir / "runner.py").write_text(_RUNNER_SRC, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=str(tmpdir),
                capture_output=True,
                timeout=timeout_s,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {
                "error": f"执行超时（>{timeout_s:g}s），疑似死循环或复杂度过高",
                "passed": 0,
                "total": len(tests),
                "results": [],
            }
        except Exception as exc:  # noqa: BLE001 - 评测不能被单条异常打断
            return {
                "error": f"无法启动子进程：{type(exc).__name__}: {exc}",
                "passed": 0,
                "total": len(tests),
                "results": [],
            }

    stdout = proc.stdout or ""
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_SENTINEL):
            try:
                payload = json.loads(line[len(_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                payload = None
            break

    if payload is None:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {
            "error": "子进程未返回结果（可能崩溃或输出被截断）",
            "stderr": " / ".join(stderr_tail)[:400],
            "passed": 0,
            "total": len(tests),
            "results": [],
        }

    payload["stderr"] = (proc.stderr or "").strip()[:400]
    return payload


def evaluate_coding_exec(sample: Sample, response: Response) -> Verdict | None:
    """代码题的硬判定。没有配测试用例的题返回 None（交给裁判层）。"""
    if sample.task_type != "coding":
        return None
    tests = sample.meta.get("tests")
    entry_point = sample.meta.get("entry_point")
    if not tests or not entry_point:
        return None

    code = extract_code(response.text or "")
    detail: dict[str, Any] = {
        "entry_point": entry_point,
        "n_tests": len(tests),
        "extracted_chars": len(code),
    }

    if not code:
        return Verdict(
            sample_id=sample.id,
            sut_id=response.sut_id,
            metric="coding_exec",
            tier="deterministic",
            score=1.0,
            passed=False,
            detail={**detail, "error": "未能从回答中提取到代码"},
            attempt_index=response.attempt_index,
        )

    hit = find_forbidden(code)
    if hit:
        return Verdict(
            sample_id=sample.id,
            sut_id=response.sut_id,
            metric="coding_exec",
            tier="deterministic",
            score=1.0,
            passed=False,
            detail={**detail, "error": f"命中安全黑名单，拒绝执行：{hit}"},
            attempt_index=response.attempt_index,
        )

    report = run_tests(code, str(entry_point), list(tests))
    total = int(report.get("total") or len(tests))
    passed_n = int(report.get("passed") or 0)
    ratio = passed_n / total if total else 0.0
    score = round(1 + 4 * ratio, 2)
    failed = [r for r in report.get("results", []) if not r.get("ok")]

    return Verdict(
        sample_id=sample.id,
        sut_id=response.sut_id,
        metric="coding_exec",
        tier="deterministic",
        score=score,
        passed=bool(total and passed_n == total),
        detail={
            **detail,
            "passed_tests": passed_n,
            "pass_ratio": round(ratio, 4),
            "failed_examples": [r.get("detail", "") for r in failed[:3]],
            "error": report.get("error"),
            "stderr": report.get("stderr", ""),
        },
        attempt_index=response.attempt_index,
    )


def collect_coding_exec(sample: Sample, response: Response) -> list[Verdict]:
    """统一入口，形状与其他 collect_* 保持一致。"""
    verdict = evaluate_coding_exec(sample, response)
    return [verdict] if verdict is not None else []

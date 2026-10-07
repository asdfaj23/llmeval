# -*- coding: utf-8 -*-
"""
可重放沙箱（离线分析部分）。

工具集是纯函数（不写盘、不联网、确定性输出），所以重放天然安全可复现。
本模块负责「不调 API」的那一半：把一次落盘的运行读回来，重算过程指标、
失败归因、轨迹级统计，并渲染成可直接打开的 HTML 分析报告。

「换一个角色模型重跑」那一半（反事实重放）需要再发一次 API 调用，属于
受预算控制的在线能力，封装在 CollabRuntime 之外由 pipeline 触发，这里只做
离线诊断，保证「看报告」这件事永远不花钱。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import Turn
from .attribution import attribute_failure, attribution_histogram, label_cn
from .metrics import collect_collab_process


def analyze_turns(turns: list[Turn]) -> dict[str, Any]:
    """把一个 run 里所有多智能体记录汇总成子系统视图。"""
    records = [t for t in turns if getattr(t.sample, "task_type", "") == "multi_agent"]
    if not records:
        return {"n": 0}

    by_sut: dict[str, dict[str, Any]] = {}
    for t in records:
        sid = t.response.sut_id or "(unknown)"
        bucket = by_sut.setdefault(
            sid,
            {
                "n": 0, "convergence": [], "invalid": [], "recovery": [],
                "trace_steps": 0, "trace_ok": 0, "retries": 0, "retries_ok": 0,
                "role_tokens": {}, "attribution": [],
                "errors": 0,
            },
        )
        bucket["n"] += 1
        if not t.response.ok:
            bucket["errors"] += 1
            bucket["attribution"].append({"sample_id": t.sample.id, "tags": []})
            continue

        for v in t.verdicts:
            if v.metric == "collab_convergence" and v.score is not None:
                bucket["convergence"].append(v.score)
            elif v.metric == "collab_invalid_rounds" and v.score is not None:
                bucket["invalid"].append(v.score)
            elif v.metric == "collab_recovery" and v.score is not None:
                bucket["recovery"].append(v.score)

        # 轨迹级统计（P1 增量，确定性）
        for s in (t.response.trace or []):
            bucket["trace_steps"] += 1
            if s.get("ok"):
                bucket["trace_ok"] += 1
            if s.get("retry_of") is not None:
                bucket["retries"] += 1
                if s.get("ok"):
                    bucket["retries_ok"] += 1

        # 逐角色 token 分布
        for m in (t.response.messages or []):
            role = str(m.get("role"))
            u = m.get("usage") or {}
            rd = bucket["role_tokens"].setdefault(role, {"prompt": 0, "completion": 0})
            rd["prompt"] += int(u.get("prompt_tokens", 0) or 0)
            rd["completion"] += int(u.get("completion_tokens", 0) or 0)

        # 失败归因（诊断，不计分）
        tags = attribute_failure(t.sample, t.response)
        bucket["attribution"].append({"sample_id": t.sample.id, "tags": tags})

    out = {"n": len(records), "suts": {}}
    for sid, b in by_sut.items():
        conv = round(sum(b["convergence"]) / len(b["convergence"]), 3) if b["convergence"] else None
        inv = round(sum(b["invalid"]) / len(b["invalid"]), 3) if b["invalid"] else None
        rec = round(sum(b["recovery"]) / len(b["recovery"]), 3) if b["recovery"] else None
        call_sr = round(b["trace_ok"] / b["trace_steps"], 4) if b["trace_steps"] else None
        retry_rr = round(b["retries_ok"] / b["retries"], 4) if b["retries"] else None
        out["suts"][sid] = {
            "n": b["n"],
            "errors": b["errors"],
            "convergence": conv,
            "invalid_rounds": inv,
            "recovery": rec,
            "call_success_rate": call_sr,
            "retry_recovery_rate": retry_rr,
            "trace_steps": b["trace_steps"],
            "role_tokens": b["role_tokens"],
            "attribution_histogram": attribution_histogram(b["attribution"]),
            "attribution_records": b["attribution"],
        }
    return out


def replay_run(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """读回一次运行，生成多智能体子系统分析报告 HTML。"""
    from ..pipeline import load_run
    summary, turns, _pairwise, _specs = load_run(run_dir)
    data = analyze_turns(turns)
    html = render_html(summary.run_id, data)
    out = Path(out_path) if out_path else Path(run_dir) / "collab_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def render_html(run_id: str, data: dict[str, Any]) -> str:
    """自包含 HTML（无外部依赖），展示过程指标、失败归因直方图、角色 token 分布。"""
    if data.get("n", 0) == 0:
        return f"<html><body><h1>多智能体子系统分析</h1><p>该运行没有多智能体记录（{run_id}）。</p></body></html>"

    def bar(value: float | None, maxv: float = 5.0) -> str:
        if value is None:
            return "<span class='muted'>—</span>"
        pct = max(0, min(100, value / maxv * 100))
        return f"<div class='bar'><div class='fill' style='width:{pct:.0f}%'></div></div><span>{value}</span>"

    rows = []
    for sid, d in data["suts"].items():
        hist = d["attribution_histogram"]
        hist_html = " ".join(
            f"<span class='tag'>{label_cn(k)}: {v}</span>" for k, v in hist.items() if v
        ) or "<span class='muted'>无失败样本</span>"
        rows.append(
            f"<tr><td>{sid}</td><td>{d['n']}</td><td>{d['errors']}</td>"
            f"<td>{bar(d['convergence'])}</td><td>{bar(d['invalid_rounds'])}</td>"
            f"<td>{bar(d['recovery'])}</td><td>{_pct(d['call_success_rate'])}</td>"
            f"<td>{_pct(d['retry_recovery_rate'])}</td><td>{hist_html}</td></tr>"
        )

    # 角色 token 分布（取第一个 sut 示意）
    first = next(iter(data["suts"].values()))
    role_rows = "".join(
        f"<tr><td>{role}</td><td>{t['prompt']}</td><td>{t['completion']}</td>"
        f"<td>{t['prompt'] + t['completion']}</td></tr>"
        for role, t in (first.get("role_tokens") or {}).items()
    )

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>多智能体子系统分析 · {run_id}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,'Microsoft YaHei',sans-serif;margin:32px;color:#222;background:#fff}}
 h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-left:4px solid #2c7;padding-left:8px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{border:1px solid #e2e2e2;padding:6px 8px;text-align:left;vertical-align:top}}
 th{{background:#f6f8fa}}
 .bar{{display:inline-block;width:80px;height:9px;background:#eee;border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px}}
 .fill{{height:100%;background:#2c7}}
 .tag{{display:inline-block;background:#fff3cd;border:1px solid #ffe69c;border-radius:10px;padding:1px 8px;margin:2px;font-size:12px}}
 .muted{{color:#999}}
 .note{{color:#666;font-size:12px}}
</style></head><body>
<h1>多智能体协作评测子系统 · 分析报告</h1>
<p class='note'>运行 {run_id} ｜ 多智能体记录 {data['n']} 条 ｜ 指标全部确定性计算，离线可复现</p>
<h2>过程级指标（按模型）</h2>
<table>
<tr><th>模型</th><th>样本</th><th>错误</th><th>一致收敛(1-5)</th><th>无效轮占比(1-5)</th>
<th>故障恢复(1-5)</th><th>调用成功率</th><th>重试恢复率</th><th>失败归因直方图</th></tr>
{''.join(rows)}
</table>
<h2>逐角色 token 分布（{next(iter(data['suts']))}）</h2>
<table><tr><th>角色</th><th>prompt tokens</th><th>completion tokens</th><th>合计</th></tr>{role_rows}</table>
<p class='note'>说明：一致收敛在同族多角色下可能只是"自己认同自己"，强结论仅在异构子集上成立；
故障恢复仅对故障注入题计分；失败归因为诊断输出，不计分，用于指导题库与 harness 迭代。</p>
</body></html>"""


def _pct(v: float | None) -> str:
    return f"{round(v * 100, 1)}%" if v is not None else "—"

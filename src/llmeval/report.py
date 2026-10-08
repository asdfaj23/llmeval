# -*- coding: utf-8 -*-
"""
报告生成。

产出单文件 HTML：CSS 内联、图表用内联 SVG 手绘、没有任何外部依赖，
双击就能在浏览器打开，也能直接丢进邮件附件。
不用 matplotlib，是因为中文字体在服务器环境下经常渲染成方块，
而且报告要能脱离运行环境独立存在。

报告结构刻意按「先结论、再依据、最后可执行动作」排列：

    运行概览  →  模型榜单  →  能力画像（雷达 + 热力矩阵）
              →  相对强弱（胜率矩阵）  →  裁判可信度（κ）
              →  偏差披露  →  短板归因与数据策略  →  维度溯源附录

有一件事在报告里是硬性的：只要本次运行用了 mock，页首必须挂红色警示条。
模拟数据不能被当成模型结论，这一点不能靠读者自己去猜。
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config as cfg
from .analysis import (
    ModelStats,
    build_strategy,
    dimension_matrix,
    final_score,
    is_pass,
    summarize_model,
    win_matrix,
)
from .calibration import collect_judge_scores_by_judge, inter_judge_agreement
from .schema import DIMENSIONS, RunSummary, Turn
from .stats import significance_table

_PALETTE = ["#1f4fd8", "#d2691e", "#0f7b6c", "#8b3fa8", "#b3261e", "#5a5a5a"]


# ------------------------------------------------------------------ 小工具
def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _fmt(value: Any, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


_TIER_LABELS = {"deterministic": "规则判定", "llm_judge": "LLM 裁判", "human": "人工标注"}


def _tier_text(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    return " · ".join(f"{_TIER_LABELS.get(k, k)} {v}" for k, v in sorted(counts.items()))


def _score_color(score: float | None) -> str:
    """分数着色：越高越深。用单色梯度，避免红绿在不同语境下读反。"""
    if score is None:
        return "#f1efe9"
    t = max(0.0, min(1.0, (score - 1) / 4))
    r = int(240 - t * (240 - 31))
    g = int(243 - t * (243 - 79))
    b = int(250 - t * (250 - 216))
    return f"rgb({r},{g},{b})"


def _text_on(score: float | None) -> str:
    return "#ffffff" if (score or 0) >= 3.8 else "#23211d"


# ------------------------------------------------------------------ 雷达图
def radar_svg(dimensions: list[dict[str, str]], rows: list[dict[str, Any]], size: int = 460) -> str:
    n = len(dimensions)
    if n < 3:
        return "<p class='muted'>维度少于 3 个，无法绘制雷达图。</p>"

    cx = cy = size / 2
    radius = size * 0.33
    angles = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]

    def point(idx: int, ratio: float) -> tuple[float, float]:
        a = angles[idx]
        return cx + math.cos(a) * radius * ratio, cy + math.sin(a) * radius * ratio

    parts: list[str] = [f'<svg viewBox="0 0 {size} {size}" width="100%" role="img">']
    parts.append("<title>能力维度雷达图</title>")

    # 网格环
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{point(i, ring)[0]:.1f},{point(i, ring)[1]:.1f}" for i in range(n))
        parts.append(
            f'<polygon points="{pts}" fill="none" stroke="#dcd9d2" stroke-width="1"/>'
        )

    # 轴线与标签
    for i, dim in enumerate(dimensions):
        x, y = point(i, 1.0)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#dcd9d2" stroke-width="1"/>')
        lx, ly = point(i, 1.22)
        anchor = "middle"
        if lx > cx + 12:
            anchor = "start"
        elif lx < cx - 12:
            anchor = "end"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="12" fill="#4a4842" '
            f'text-anchor="{anchor}" dominant-baseline="central">{_esc(dim["name"])}</text>'
        )

    # 刻度提示
    parts.append(
        f'<text x="{cx + 4}" y="{cy - radius - 6}" font-size="11" fill="#8b8880">5 分</text>'
    )
    parts.append(f'<text x="{cx + 4}" y="{cy + radius + 14}" font-size="11" fill="#8b8880">1 分</text>')

    # 每个模型一层
    for idx, row in enumerate(rows):
        color = _PALETTE[idx % len(_PALETTE)]
        pts = []
        for i, dim in enumerate(dimensions):
            v = row["values"].get(dim["key"])
            ratio = 0.0 if v is None else max(0.0, min(1.0, (v - 1) / 4))
            x, y = point(i, ratio)
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.14" '
            f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        for p in pts:
            x, y = p.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.6" fill="{color}"/>')

    parts.append("</svg>")
    return "".join(parts)


def legend_html(rows: list[dict[str, Any]]) -> str:
    items = []
    for idx, row in enumerate(rows):
        color = _PALETTE[idx % len(_PALETTE)]
        items.append(
            f'<span class="legend-item"><i style="background:{color}"></i>{_esc(row["label"])}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


# ------------------------------------------------------------------ 区块
def _overview_cards(summary: RunSummary, stats: list[ModelStats], n_dims: int) -> str:
    cards = [
        ("执行单元", f"{summary.n_turns}"),
        ("题目数", f"{summary.n_samples}"),
        ("被测模型", f"{len(summary.suts)}"),
        ("覆盖维度", f"{n_dims}"),
        ("耗时", f"{summary.duration_s:.1f}s"),
        ("错误记录", f"{summary.errors}"),
        ("Token", f"{summary.usage.total_tokens:,}"),
        ("估算成本", f"${summary.usage.cost_usd:.4f}"),
    ]
    return "".join(
        f'<div class="card"><div class="card-k">{_esc(k)}</div>'
        f'<div class="card-v">{_esc(v)}</div></div>'
        for k, v in cards
    )


def _leaderboard(stats: list[ModelStats]) -> str:
    rows = []
    ranked = sorted(
        stats,
        key=lambda s: (s.mean_score is None, -(s.mean_score or 0)),
    )
    for rank, s in enumerate(ranked, start=1):
        lo, hi = s.score_ci
        ci = f"[{_fmt(lo)}, {_fmt(hi)}]" if lo is not None else "—"
        tag = " <span class='chip chip-mock'>mock</span>" if s.is_mock else ""
        rows.append(
            "<tr>"
            f"<td class='num'>{rank}</td>"
            f"<td><b>{_esc(s.label or s.sut_id)}</b>{tag}</td>"
            f"<td class='num'>{_fmt(s.mean_score)}</td>"
            f"<td class='num muted'>{ci}</td>"
            f"<td class='num'>{_fmt((s.pass_rate or 0) * 100 if s.pass_rate is not None else None, 1)}%</td>"
            f"<td class='num'>{s.n_error}</td>"
            f"<td class='num'>{_fmt(s.avg_latency_s, 3)}s</td>"
            f"<td class='num'>{s.total_tokens:,}</td>"
            f"<td class='num'>${s.total_cost_usd:.4f}</td>"
            "</tr>"
        )
    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>模型</th><th>综合分</th><th>95% 置信区间</th>
        <th>通过率</th><th>错误</th><th>平均耗时</th><th>Token</th><th>成本</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _dimension_table(matrix: dict[str, Any], stats: list[ModelStats]) -> str:
    dims = matrix["dimensions"]
    if not dims:
        return "<p class='muted'>没有可用于对比的维度数据。</p>"

    head = "".join(f"<th>{_esc(d['name'])}</th>" for d in dims)
    body = []
    label_by_id = {s.sut_id: (s.label or s.sut_id) for s in stats}
    for row in matrix["rows"]:
        cells = []
        for d in dims:
            v = row["values"].get(d["key"])
            cells.append(
                f"<td class='num heat' style='background:{_score_color(v)};color:{_text_on(v)}'>"
                f"{_fmt(v)}</td>"
            )
        body.append(
            f"<tr><td><b>{_esc(label_by_id.get(row['sut_id'], row['label']))}</b></td>{''.join(cells)}</tr>"
        )
    return f"""
    <table>
      <thead><tr><th>模型</th>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    <p class="muted small">单元格为 1—5 分量表上的均分；颜色越深表示得分越高。</p>"""


def _win_matrix_table(matrix: dict[str, Any], stats: list[ModelStats]) -> str:
    ids = matrix["sut_ids"]
    if len(ids) < 2 or matrix["n_pairs"] == 0:
        return "<p class='muted'>本次运行未做两两对比（需要 ≥2 个被测模型且开启 pairwise）。</p>"

    label = {s.sut_id: (s.label or s.sut_id) for s in stats}
    head = "".join(f"<th>{_esc(label.get(i, i))}</th>" for i in ids)
    body = []
    for row in matrix["rows"]:
        cells = []
        for v in row["values"]:
            if v is None:
                cells.append("<td class='num muted'>—</td>")
            else:
                cells.append(
                    f"<td class='num heat' style='background:{_score_color(v * 4 + 1)};"
                    f"color:{_text_on(v * 4 + 1)}'>{v * 100:.1f}%</td>"
                )
        body.append(f"<tr><td><b>{_esc(label.get(row['sut_id'], row['sut_id']))}</b></td>{''.join(cells)}</tr>")

    return f"""
    <table>
      <thead><tr><th>行 ＼ 列</th>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    <p class="muted small">数值为「行模型对列模型」的胜率。每次比较都做了双向换位，
    只有两个顺序都判同一方获胜才记为胜，否则记平局 —— 位置敏感的部分被转成诚实的平局。
    共 {matrix['n_pairs']} 组有效对比。</p>"""


def _calibration_block(calib: dict[str, Any] | None) -> str:
    if not calib:
        return (
            "<p class='muted'>本次未提供人工标注样本，因此无法计算裁判与人工的一致性（Cohen's κ）。"
            "该项宁可留空，也不用其他指标顶替 —— 没有校准的裁判分数只能当参考，不能当结论。</p>"
        )
    ready = calib.get("ready_for_gate")
    badge = (
        "<span class='chip chip-ok'>达警戒线，可用于批量打分</span>"
        if ready
        else "<span class='chip chip-warn'>未达警戒线，仅供参考</span>"
    )
    weighted = calib.get("quadratic_weighted_kappa")
    plain = calib.get("cohen_kappa")
    return f"""
    <div class="grid-2">
      <div class="kv"><span>人工标注样本</span><b>{_esc(calib.get('n'))}</b></div>
      <div class="kv"><span>二次加权 κ</span><b>{_fmt(weighted, 3)}</b></div>
      <div class="kv"><span>未加权 κ</span><b>{_fmt(plain, 3)}</b></div>
      <div class="kv"><span>完全一致率</span><b>{_fmt((calib.get('agreement_rate') or 0) * 100 if calib.get('agreement_rate') is not None else None, 1)}%</b></div>
      <div class="kv"><span>Spearman ρ</span><b>{_fmt(calib.get('spearman'), 3)}</b></div>
      <div class="kv"><span>平均绝对误差</span><b>{_fmt(calib.get('mae'), 3)}</b></div>
    </div>
    <p style="margin-top:10px">{badge} <span class="muted small">一致性区间：{_esc(calib.get('kappa_band'))}</span></p>
    <p class="muted small">{_esc(calib.get('note'))}</p>"""


def _panel_block(panel: dict[str, Any] | None) -> str:
    """评审团一致性。只有一个裁判时，如实说明「不构成评审团」。"""
    if not panel:
        return ""
    if panel.get("n_judges", 0) < 2:
        return f"<p class='muted'>{_esc(panel.get('note', '当前只有 1 个裁判。'))}</p>"
    if not panel.get("n_records"):
        return f"<p class='muted'>{_esc(panel.get('note', '没有记录可供比对。'))}</p>"

    kappa = panel.get("pairwise_weighted_kappa")
    band = panel.get("kappa_band", "")
    judges = "、".join(panel.get("judge_ids") or [])
    return f"""
    <div class="grid-2">
      <div class="kv"><span>裁判数量</span><b>{_esc(panel.get('n_judges'))}</b></div>
      <div class="kv"><span>可比对记录</span><b>{_esc(panel.get('n_records'))}</b></div>
      <div class="kv"><span>成对加权 κ</span><b>{_fmt(kappa, 3)}（{_esc(band)}）</b></div>
      <div class="kv"><span>完全一致率</span><b>{_fmt((panel.get('pairwise_exact_rate') or 0) * 100 if panel.get('pairwise_exact_rate') is not None else None, 1)}%</b></div>
      <div class="kv"><span>相邻 1 分内一致率</span><b>{_fmt((panel.get('pairwise_near_rate') or 0) * 100 if panel.get('pairwise_near_rate') is not None else None, 1)}%</b></div>
      <div class="kv"><span>平均分歧幅度</span><b>{_fmt(panel.get('mean_range'), 2)} 分</b></div>
    </div>
    <p class="muted small" style="margin-top:8px">参与裁判：{_esc(judges)}</p>
    <p class="muted small">{_esc(panel.get('verdict', ''))}</p>
    <p class="muted small">
      综合分对同一指标下的多个裁判取<b>中位数</b>而非均值 ——
      评审团的意义是抗离群，单个裁判抽风不该把整体分数带走。
      各裁判独立打分，不共享上下文；一旦互相可见，它们就不再是独立样本了。
    </p>"""


def _bias_block(bias: dict[str, Any]) -> str:
    pos = bias.get("position") or {}
    length = bias.get("length") or {}
    selfpref = bias.get("self_preference") or {}
    controls = bias.get("applied_controls") or {}

    warns = selfpref.get("cross_family_warnings") or []
    warn_html = (
        "".join(f"<li>{_esc(w)}</li>" for w in warns) if warns else "<li class='muted'>未发现同族风险</li>"
    )
    applied_controls_muted = '<span class="muted">未启用</span>'
    applied = "".join(
        f"<li>{_esc(k)}：{'已启用' if v else applied_controls_muted}</li>"
        for k, v in controls.items()
    )

    return f"""
    <div class="grid-2">
      <div>
        <h4>位置偏差</h4>
        <div class="kv"><span>有效配对</span><b>{_esc(pos.get('n_pairs', 0))}</b></div>
        <div class="kv"><span>换位一致率</span><b>{_fmt((pos.get('position_consistency_rate') or 0) * 100 if pos.get('position_consistency_rate') is not None else None, 1)}%</b></div>
        <p class="muted small">{_esc(pos.get('verdict', '数据不足'))}</p>
      </div>
      <div>
        <h4>冗长偏差</h4>
        <div class="kv"><span>分出胜负的对比</span><b>{_esc(length.get('n_decided', 0))}</b></div>
        <div class="kv"><span>更长一方获胜率</span><b>{_fmt((length.get('longer_win_rate') or 0) * 100 if length.get('longer_win_rate') is not None else None, 1)}%</b></div>
        <p class="muted small">{_esc(length.get('verdict', '数据不足'))}</p>
      </div>
    </div>
    <h4 style="margin-top:14px">自偏好风险</h4>
    <ul class="tight">{warn_html}</ul>
    <h4>已启用的偏差控制</h4>
    <ul class="tight">{applied}</ul>
    <p class="muted small">{_esc(bias.get('disclosure', ''))}</p>"""


def _containment_block(summary: RunSummary) -> str:
    """评测环境隔离披露。

    这一节的意义是「让人知道有这么一道检查」，所以干净时只给一行状态；
    一旦有越界记录才展开细节 —— 那时候读者需要看到具体是哪条、哪一类。
    """
    c = summary.containment or {}
    if not c:
        return (
            "<p class='muted small'>本次运行未包含隔离审计"
            "（旧版本的 summary.json 不含该字段）。</p>"
        )

    checked = c.get("n_checked", 0)
    n_bad = c.get("n_violations", 0)
    by_kind = c.get("by_kind") or {}
    clean = bool(c.get("clean", n_bad == 0))

    if clean:
        head = (
            f"<p class='muted small'>已审计 <b>{_esc(checked)}</b> 条记录，"
            "未发现答案泄漏、canary 命中或越界工具调用。</p>"
        )
        detail = ""
    else:
        head = (
            f"<p><span class='chip chip-warn'>发现 {_esc(n_bad)} 条越界记录，"
            "该轮结果需人工复核</span></p>"
            f"<p class='muted small'>已审计 {_esc(checked)} 条记录。</p>"
        )
        rows = "".join(
            f"<li>{_esc(v.get('sample_id'))}（{_esc(v.get('sut_id'))}）："
            f"答案泄漏 {_esc(v.get('answer_leaks'))} · "
            f"canary 命中 {_esc(v.get('canary_hits'))} · "
            f"轨迹疑点 {_esc(v.get('trace_findings'))}</li>"
            for v in (c.get("violations") or [])
        )
        detail = f"<ul class='tight'>{rows}</ul>"

    return (
        head
        + '<div class="grid-2" style="margin-top:10px">'
        "<div><h4>三类审计结果</h4>"
        f"<div class=\"kv\"><span>答案隔离泄漏</span><b>{_esc(by_kind.get('answer_leak', 0))}</b></div>"
        f"<div class=\"kv\"><span>canary 命中</span><b>{_esc(by_kind.get('canary_hit', 0))}</b></div>"
        f"<div class=\"kv\"><span>轨迹疑点</span><b>{_esc(by_kind.get('trace_finding', 0))}</b></div>"
        "</div><div><h4>覆盖范围</h4>"
        "<p class=\"muted small\">答案隔离自检 + canary 泄漏扫描 + 工具轨迹审计；"
        "全部为确定性判定，零 API 成本。仅覆盖会调用工具的题型。</p></div></div>"
        + detail
        + f"<p class='muted small'>{_esc(c.get('note', ''))}</p>"
    )


def _strategy_block(strategy: list[dict[str, Any]]) -> str:
    if not strategy:
        return "<p class='muted'>没有失败样本，或本轮没有产生判定结果。</p>"

    attr = next((s for s in strategy if s["kind"] == "failure_attribution"), None)
    dims = next((s for s in strategy if s["kind"] == "dimension_priority"), None)

    attr_html = ""
    if attr and attr["items"]:
        rows = "".join(
            f"<tr><td><b>{_esc(i['label'])}</b></td><td class='num'>{i['count']}</td>"
            f"<td class='num'>{i['share'] * 100:.1f}%</td><td>{_esc(i['data_action'])}</td></tr>"
            for i in attr["items"]
        )
        attr_html = f"""
        <table>
          <thead><tr><th>失败类型</th><th>条数</th><th>占比</th><th>对应的数据动作</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""
    else:
        attr_html = "<p class='muted'>本轮没有捕获到可归因的失败样本。</p>"

    dim_html = ""
    if dims and dims["items"]:
        items = "".join(
            f"<li><b>{_esc(i['name'])}</b>（均分 {_fmt(i['avg_score'])}）"
            f"<div class='muted small'>对齐依据：{_esc(i['seed_anchor'])}</div></li>"
            for i in dims["items"]
        )
        dim_html = f"<h4 style='margin-top:14px'>优先补强的维度</h4><ul class='tight'>{items}</ul>"

    return attr_html + dim_html


def _failures_block(stats: list[ModelStats]) -> str:
    blocks = []
    for s in stats:
        if not s.failures:
            continue
        items = []
        for f in s.failures:
            reasons = "；".join(
                f"{_esc(r['metric'])} {_fmt(r['score'])}" for r in f.get("reasons", [])
            )
            ref = (
                f"<div class='small'><span class='muted'>参考答案：</span>{_esc(f['reference'])}</div>"
                if f.get("reference")
                else ""
            )
            items.append(
                f"""<details class="bad">
                  <summary><span class="chip chip-warn">{_esc(f['category_label'])}</span>
                  <span class="muted small">{_esc(f['dimension_name'])} · {_esc(f['sample_id'])}</span></summary>
                  <div class="small"><span class="muted">题面：</span>{_esc(f['prompt'])}</div>
                  <div class="small"><span class="muted">回答：</span>{_esc(f['answer'])}</div>
                  {ref}
                  <div class="small"><span class="muted">未通过的判定：</span>{reasons}</div>
                </details>"""
            )
        blocks.append(
            f"<h4>{_esc(s.label or s.sut_id)}（{len(s.failures)} 条）</h4>{''.join(items)}"
        )
    if not blocks:
        return "<p class='muted'>没有失败样本需要展示。</p>"
    return "".join(blocks)


def _reliability_block(stats: list[ModelStats]) -> str:
    rows = []
    for s in stats:
        r = s.reliability
        if not r:
            continue
        rows.append(
            "<tr>"
            f"<td><b>{_esc(s.label or s.sut_id)}</b></td>"
            f"<td class='num'>{r['k']}</td>"
            f"<td class='num'>{r['n_samples']}</td>"
            f"<td class='num'>{r['avg_success_rate'] * 100:.1f}%</td>"
            f"<td class='num'>{r['pass_at_k'] * 100:.1f}%</td>"
            f"<td class='num'>{r['pass_hat_k'] * 100:.1f}%</td>"
            f"<td class='num'>{r['flaky_count']}</td>"
            "</tr>"
        )
    if not rows:
        return (
            "<p class='muted'>本次每个单元只跑了一次，无法计算 pass@k / pass^k。"
            "要评估可靠性，请在套件里给对应维度设置 <code>repeats</code>（建议 ≥3）。</p>"
        )
    return f"""
    <table>
      <thead><tr><th>模型</th><th>k</th><th>样本数</th><th>平均单次成功率</th>
      <th>pass@k</th><th>pass^k</th><th>不稳定样本</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="muted small">pass@k 是能力上限，pass^k 是可靠性。两者差距越大，模型越不稳定。
    对要上线的系统，pass^k 才是该看的那个数 —— 单次成功会系统性高估可用性。</p>"""


def _significance_block(rows: list[dict[str, Any]]) -> str:
    """统计显著性：这个分数差，到底是真差距还是噪声。

    NeurIPS 2025 那篇 construct validity 综述审了 445 篇 benchmark，
    只有 16% 用了统计检验。这一块就是用来不做那 84% 的。
    """
    if not rows:
        return "<p class='muted'>只有一个模型参与评测，无法做两两对比检验。</p>"

    body: list[str] = []
    for r in rows:
        pt = r.get("pass_test") or {}
        sd = r.get("score_diff") or {}
        p = pt.get("p_value")
        p_text = "—" if p is None else ("&lt; 0.001" if p < 0.001 else f"{p:.3f}")
        n_disc = pt.get("n_discordant", 0)

        significant = bool(pt.get("significant")) or bool(sd.get("significant"))
        if significant:
            verdict, color, bg = "差异显著", "#a32d2d", "#fcebeb"
        else:
            verdict, color, bg = "噪声范围内", "#5f5e5a", "#f1efe8"

        ci = "—"
        if sd.get("ci_low") is not None:
            ci = f"{sd['ci_low']:+.2f} ~ {sd['ci_high']:+.2f}"

        body.append(
            "<tr>"
            f"<td>{_esc(r['a'])} vs {_esc(r['b'])}</td>"
            f"<td class='num'>{r['n_paired']}</td>"
            f"<td class='num'>{n_disc}</td>"
            f"<td class='num'>{p_text}</td>"
            f"<td class='num'>{ci}</td>"
            f"<td class='num'>{r['wins_a']} : {r['wins_b']} : {r['ties']}</td>"
            "<td><span style='display:inline-block;padding:2px 8px;border-radius:10px;"
            f"font-size:12px;color:{color};background:{bg}'>{verdict}</span></td>"
            "</tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>对比</th><th>配对题数</th><th>结论不一致的题</th><th>McNemar p</th>"
        "<th>分数差 95% CI</th><th>胜 : 负 : 平</th><th>结论</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        "<p class='muted' style='margin-top:10px'>"
        "配对检验只统计两人结论不一致的题 —— 都做对或都做错的题，对「谁更强」零信息量。"
        "分数差置信区间跨过 0，说明这个差距撑不起「A 比 B 强」的结论。"
        "</p>"
    )


def _dimension_appendix() -> str:
    """维度溯源卡片。

    刻意用卡片而不是 5 列表格：这里每格都是一整句话，
    挤进表格只会变得没人看。分类是否清晰，取决于读者愿不愿意读完。
    """
    cards = "".join(
        f"<div style='margin-bottom:14px'>"
        f"<b>{_esc(v['name'])}</b> <code>{_esc(k)}</code>"
        f"<div class='muted small'>Seed 对齐：{_esc(v['seed_anchor'])}</div>"
        f"<div class='muted small'>对标 benchmark：{_esc(v.get('benchmark', '—'))}</div>"
        f"<div class='muted small'>产出指标：{_esc(v.get('metrics', '—'))}</div>"
        f"</div>"
        for k, v in DIMENSIONS.items()
    )
    return f"<div class='grid-2'>{cards}</div>"


# ------------------------------------------------------------------ 主渲染
def render_html(
    summary: RunSummary,
    turns: list[Turn],
    pairwise: list[dict[str, Any]],
    specs: dict[str, Any],
    calib: dict[str, Any] | None = None,
    bias: dict[str, Any] | None = None,
    panel: dict[str, Any] | None = None,
) -> str:
    if panel is None:
        panel = inter_judge_agreement(collect_judge_scores_by_judge(turns))

    stats = [
        summarize_model(
            turns,
            sut_id,
            label=getattr(specs.get(sut_id), "label", sut_id) or sut_id,
            is_mock=bool(getattr(specs.get(sut_id), "is_mock", False)),
        )
        for sut_id in summary.suts
    ]
    matrix = dimension_matrix(stats)
    wmatrix = win_matrix(pairwise, summary.suts)
    strategy = build_strategy(stats)
    significance = significance_table(turns, is_pass, final_score, summary.suts)

    mock_banner = ""
    if summary.mock:
        mock_banner = (
            '<div class="mock-banner">'
            "<b>本报告包含模拟（mock）数据，不是任何真实模型的评测结论。</b><br>"
            "标有 <span class='chip chip-mock'>mock</span> 的模型由内置模拟引擎生成回答，"
            "它的存在只用于验证流水线是否跑通。要得到真实结论，"
            "请在 <code>configs/models.yaml</code> 中启用真实模型并配置 API Key 后重新运行。"
            "</div>"
        )

    radar = radar_svg(matrix["dimensions"], matrix["rows"]) if matrix["dimensions"] else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>大模型能力评测报告 · {_esc(summary.suite)}</title>
<style>
  :root {{
    --bg:#f5f4f1; --card:#ffffff; --border:#e3e0da; --text:#23211d;
    --muted:#6f6c66; --accent:#1f4fd8;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--text);
    font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",system-ui,sans-serif;
    font-size:14px; line-height:1.7;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 6px; font-weight:600; letter-spacing:.2px; }}
  h2 {{ font-size:17px; margin:34px 0 12px; font-weight:600; padding-left:10px;
        border-left:3px solid var(--accent); }}
  h3 {{ font-size:15px; margin:20px 0 8px; font-weight:600; }}
  h4 {{ font-size:14px; margin:16px 0 6px; font-weight:600; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .mock-banner {{
    background:#fdf1f0; border:1px solid #e8b4b0; color:#8c1d18;
    padding:14px 16px; border-radius:10px; margin:18px 0; font-size:13.5px;
  }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin:16px 0; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }}
  .card-k {{ color:var(--muted); font-size:12px; }}
  .card-v {{ font-size:19px; font-weight:600; margin-top:2px; }}
  section {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
             padding:18px 20px; margin:14px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; background:#faf9f6; }}
  td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.heat {{ font-weight:600; }}
  .muted {{ color:var(--muted); }}
  .small {{ font-size:12.5px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 12px; font-size:12.5px; }}
  .legend-item {{ display:inline-flex; align-items:center; gap:6px; }}
  .legend-item i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:720px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
  .kv {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px dashed var(--border); }}
  ul.tight {{ margin:6px 0 0 18px; padding:0; }}
  ul.tight li {{ margin-bottom:5px; }}
  .chip {{ display:inline-block; padding:1px 7px; border-radius:20px; font-size:11.5px; font-weight:600; }}
  .chip-mock {{ background:#f1efe9; color:#6f6c66; border:1px solid #dcd9d2; }}
  .chip-warn {{ background:#fdf1f0; color:#8c1d18; border:1px solid #e8b4b0; }}
  .chip-ok {{ background:#eef7f0; color:#1a6b3c; border:1px solid #b6d9c1; }}
  details.bad {{ border:1px solid var(--border); border-radius:8px; padding:8px 10px; margin:8px 0; background:#fcfcfa; }}
  details.bad summary {{ cursor:pointer; font-size:13px; }}
  code {{ background:#f1efe9; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:28px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>大模型能力评测报告</h1>
  <div class="sub">
    套件 {_esc(summary.suite)} · 运行 {_esc(summary.run_id)} ·
    {_esc(summary.started_at)} → {_esc(summary.finished_at)}
  </div>
  {mock_banner}

  <section>
    <h2>运行概览</h2>
    <div class="cards">{_overview_cards(summary, stats, len(matrix['dimensions']))}</div>
    <div class="muted small">
      判定分布：{_tier_text(summary.tier_counts)} ·
      裁判：{_esc('、'.join(summary.judges) or '无')}
    </div>
  </section>

  <section>
    <h2>模型榜单</h2>
    {_leaderboard(stats)}
    <p class="muted small">综合分取 1—5 均值；置信区间由自助法（4000 次重采样）给出。
    区间重叠的两个模型，排名差异不足以支撑结论。</p>
  </section>

  <section>
    <h2>能力画像</h2>
    {legend_html(matrix['rows'])}
    <div style="max-width:520px;margin:0 auto">{radar}</div>
  </section>

  <section>
    <h2>维度得分矩阵</h2>
    {_dimension_table(matrix, stats)}
  </section>

  <section>
    <h2>相对强弱（胜率矩阵）</h2>
    {_win_matrix_table(wmatrix, stats)}
  </section>

  <section>
    <h2>差异是否显著（统计检验）</h2>
    {_significance_block(significance)}
  </section>

  <section>
    <h2>可靠性（pass@k / pass^k）</h2>
    {_reliability_block(stats)}
  </section>

  <section>
    <h2>裁判可信度</h2>
    {_calibration_block(calib)}
    <h3 style="margin-top:22px">评审团一致性</h3>
    {_panel_block(panel)}
  </section>

  <section>
    <h2>偏差度量与披露</h2>
    {_bias_block(bias or {})}
  </section>

  <section>
    <h2>评测环境隔离</h2>
    {_containment_block(summary)}
  </section>

  <section>
    <h2>短板归因与数据策略</h2>
    {_strategy_block(strategy)}
  </section>

  <section>
    <h2>失败样本明细</h2>
    {_failures_block(stats)}
  </section>

  <section>
    <h2>附录：能力维度溯源</h2>
    {_dimension_appendix()}
    <p class="muted small">
      维度设计遵循「从真实用例出发、再抽象成可评测类别」的思路，
      而非直接套用公开榜单 —— 公开榜单饱和很快，且与真实使用场景脱节。
    </p>
  </section>

  <footer>
    由 llmeval 生成于 {datetime.now():%Y-%m-%d %H:%M} ·
    评测集样本量较小，所有结论应视为流程验证而非最终能力结论
  </footer>
</div>
</body>
</html>"""


def render(
    result: Any,
    out_path: str | Path | None = None,
    calib: dict[str, Any] | None = None,
    bias: dict[str, Any] | None = None,
    panel: dict[str, Any] | None = None,
) -> Path:
    """把一次运行渲染成 HTML 报告，返回文件路径。"""
    html_text = render_html(
        result.summary,
        result.turns,
        result.pairwise or [],
        result.models,
        calib,
        bias,
        panel,
    )
    path = Path(out_path) if out_path else (result.run_dir / "report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def render_from_dir(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    calib: dict[str, Any] | None = None,
    bias: dict[str, Any] | None = None,
    panel: dict[str, Any] | None = None,
) -> Path:
    """从落盘目录重建报告，无需重跑评测。"""
    from .pipeline import load_run
    from types import SimpleNamespace

    summary, turns, pairwise, specs = load_run(run_dir)
    run_dir = Path(run_dir)
    result = SimpleNamespace(
        summary=summary, turns=turns, pairwise=pairwise, models=specs, run_dir=run_dir
    )
    return render(result, out_path or (run_dir / "report.html"), calib, bias, panel)

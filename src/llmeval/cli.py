# -*- coding: utf-8 -*-
"""
命令行入口。

    python run.py list                              看有哪些套件、评分表、维度
    python run.py run --suite all --pairwise        跑一次完整评测
    python run.py run --suite all --suts mock-strong,mock-weak
    python run.py report --run latest               出报告
    python run.py report --run latest --calibrate datasets/calibration/gold_helpfulness.jsonl
    python run.py calibrate --run latest --gold <人工标注.jsonl>

设计意图：把「跑评测」和「出报告」拆成两条命令。
评测贵（要花 token），报告便宜（纯本地计算）。
拆开之后，调报告样式不用重跑模型，分析一批旧结果也不用重新花一遍钱。
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from . import __version__, config as cfg
from .analysis import build_strategy, summarize_model
from .bias import summarize_bias
from .calibration import run_calibration
from .collab.compare_report import compare_run
from .collab.replay import replay_run
from .pipeline import Runner, latest_run, load_run
from .report import render
from .schema import DIMENSIONS
from .textutil import make_snippet


def _say(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ 子命令
def cmd_list(args: argparse.Namespace) -> int:
    _say("可用评测套件：")
    for name in cfg.list_suites():
        try:
            suite = cfg.load_suite(name)
        except Exception as exc:  # noqa: BLE001
            _say(f"  - {name}（读取失败：{exc}）")
            continue
        _say(f"  - {name:<12} {suite.name}｜{len(suite.datasets)} 个数据集")

    _say("\n可用评分表：")
    for name in cfg.list_rubrics():
        try:
            r = cfg.load_rubric(name)
        except Exception:  # noqa: BLE001
            _say(f"  - {name}")
            continue
        dims = "、".join(r.get("dimensions") or [])
        _say(f"  - {name:<22} {r.get('name', '')}｜维度：{dims}")

    _say("\n能力维度（对齐依据）：")
    for key, info in DIMENSIONS.items():
        _say(f"  - {key:<14} {info['name']}｜{info['seed_anchor']}")

    _say("\n模型配置（configs/models.yaml）：")
    models = cfg.load_models()
    shown_providers: set[str] = set()
    for role, items in (("被测", models.suts), ("裁判", models.judges)):
        for m in items:
            flag = "✓ 可用" if m.usable else f"· 跳过（{m.unavailable_reason}）"
            _say(f"  [{role}] {m.id:<18} {flag}")
            # 打印实际生效的凭据指纹，避免"填了 A 结果用了 B"这种静默错配
            if m.usable and not m.is_mock and m.provider.key not in shown_providers:
                shown_providers.add(m.provider.key)
                _say(
                    f"            └ 实际生效的凭据 {m.provider.api_key_env} = "
                    f"{cfg.credential_fingerprint(m.provider.api_key_env)}"
                )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    sut_ids = [s.strip() for s in args.suts.split(",") if s.strip()] if args.suts else None
    runner = Runner(
        args.suite,
        sut_ids=sut_ids,
        limit=args.limit,
        pairwise=args.pairwise,
        progress=_say,
    )
    result = runner.run()

    if args.no_report:
        return 0

    calib = None
    if args.calibrate:
        report = run_calibration(result.turns, args.calibrate)
        calib = report.to_dict()
        _say(f"校准：κ={calib['quadratic_weighted_kappa']}（{calib['kappa_band']}）")

    stats = [
        summarize_model(result.turns, sid, label=getattr(result.models.get(sid), "label", sid) or sid)
        for sid in result.summary.suts
    ]
    bias = summarize_bias(result.pairwise or [], runner.cross_family_warnings)
    path = render(result, calib=calib, bias=bias)
    _say(f"报告：{path}")

    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = latest_run() if args.run in ("latest", "", None) else Path(args.run)
    summary, turns, pairwise, specs = load_run(run_dir)
    _say(f"读取 {run_dir}（{summary.n_turns} 条记录）")

    calib = None
    if args.calibrate:
        report = run_calibration(turns, args.calibrate)
        calib = report.to_dict()
        _say(
            f"校准：加权 κ={calib['quadratic_weighted_kappa']}、"
            f"未加权 κ={calib['cohen_kappa']}、一致率={calib['agreement_rate']}"
        )
        _say(f"      {calib['note']}")

    from types import SimpleNamespace

    result = SimpleNamespace(
        summary=summary, turns=turns, pairwise=pairwise, models=specs, run_dir=Path(run_dir)
    )
    out = Path(args.out) if args.out else Path(run_dir) / "report.html"
    path = render(result, out_path=out, calib=calib, bias=summarize_bias(pairwise or []))
    _say(f"报告：{path}")

    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    run_dir = latest_run() if args.run in ("latest", "", None) else Path(args.run)
    summary, turns, _, _ = load_run(run_dir)
    report = run_calibration(turns, args.gold)
    payload = report.to_dict()
    _say(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _say(f"已写入 {args.out}")
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    """快速看失败样本，不用打开报告。排查问题时比翻 HTML 快得多。"""
    run_dir = latest_run() if args.run in ("latest", "", None) else Path(args.run)
    summary, turns, _, specs = load_run(run_dir)
    for sid in summary.suts:
        stats = summarize_model(turns, sid, label=getattr(specs.get(sid), "label", sid) or sid)
        _say(f"\n=== {stats.label}：{stats.n} 条，综合分 {stats.mean_score}，失败归因 {stats.attribution}")
        for f in stats.failures[: args.top]:
            _say(f"  [{f['category_label']}] {f['dimension_name']} · {f['sample_id']} · 分 {f['score']}")
            _say(f"    题面：{make_snippet(f['prompt'], 90)}")
            _say(f"    回答：{make_snippet(f['answer'], 110)}")
    return 0


def cmd_collab(args: argparse.Namespace) -> int:
    """多智能体子系统分析报告（离线，不重新调用模型）。"""
    run_dir = latest_run() if args.run in ("latest", "", None) else Path(args.run)
    path = replay_run(run_dir, out_path=args.out or None)
    _say(f"多智能体子系统分析：{path}")
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """ReAct vs Plan-and-Execute 对比分析（离线，不重新调用模型）。"""
    run_dir = latest_run() if args.run in ("latest", "", None) else Path(args.run)
    path = compare_run(run_dir, out_path=args.out or None)
    _say(f"ReAct vs Plan-and-Execute 对比：{path}")
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


# ------------------------------------------------------------------ 参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmeval",
        description="大模型能力评测框架 —— 评测集 × 三层判定 × 偏差控制 × 校准",
    )
    parser.add_argument("--version", action="version", version=f"llmeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出套件、评分表、维度与模型状态")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="执行一次评测")
    p_run.add_argument("--suite", default="all", help="套件名，默认 all")
    p_run.add_argument("--suts", default="", help="只跑指定的被测模型，逗号分隔")
    p_run.add_argument("--limit", type=int, default=0, help="最多跑几道题（0=不限）")
    p_run.add_argument("--pairwise", action="store_true", help="额外做两两换位对比")
    p_run.add_argument("--calibrate", default="", help="人工标注文件，用于算裁判一致性")
    p_run.add_argument("--no-report", action="store_true", help="只跑评测，不出报告")
    p_run.add_argument("--open", action="store_true", help="出完报告后用浏览器打开")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="由已有结果生成报告（不重新调用模型）")
    p_rep.add_argument("--run", default="latest", help="运行目录，默认 latest")
    p_rep.add_argument("--calibrate", default="", help="人工标注文件")
    p_rep.add_argument("--out", default="", help="输出路径")
    p_rep.add_argument("--open", action="store_true", help="用浏览器打开")
    p_rep.set_defaults(func=cmd_report)

    p_cal = sub.add_parser("calibrate", help="用人工标注计算裁判一致性")
    p_cal.add_argument("--run", default="latest")
    p_cal.add_argument("--gold", required=True, help="人工标注 JSONL")
    p_cal.add_argument("--out", default="", help="结果写入路径")
    p_cal.set_defaults(func=cmd_calibrate)

    p_fail = sub.add_parser("failures", help="在终端快速查看失败样本")
    p_fail.add_argument("--run", default="latest")
    p_fail.add_argument("--top", type=int, default=5, help="每个模型展示几条")
    p_fail.set_defaults(func=cmd_failures)

    p_collab = sub.add_parser("collab", help="由已有运行生成多智能体子系统分析报告（离线）")
    p_collab.add_argument("--run", default="latest", help="运行目录，默认 latest")
    p_collab.add_argument("--out", default="", help="输出 HTML 路径")
    p_collab.add_argument("--open", action="store_true", help="用浏览器打开")
    p_collab.set_defaults(func=cmd_collab)

    p_cmp = sub.add_parser("compare", help="ReAct vs Plan-and-Execute 对比分析（离线）")
    p_cmp.add_argument("--run", default="latest", help="运行目录，默认 latest")
    p_cmp.add_argument("--out", default="", help="输出 HTML 路径")
    p_cmp.add_argument("--open", action="store_true", help="用浏览器打开")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认不是 utf-8
    except Exception:  # noqa: BLE001
        pass
    # 统一在这里加载 .env。之前只有 run 子命令会加载，导致 list 里
    # 明明填了 key 却显示"跳过（缺少环境变量）"，这种误判很费时间。
    cfg.load_env_file()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

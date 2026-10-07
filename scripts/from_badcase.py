# -*- coding: utf-8 -*-
"""
数据飞轮：从失败样本生成候选新题。

依据是 Seed2.1 Model Card 里的做法 ——「从用户返回的真实 bad case 持续扩展 benchmark」。
这条链路让评测不止于「打分」，而是把负面案例重新变成题库资产，形成闭环：

    评测 → 失败分析 → 候选题生成 → 人工确认 → 进入题库 → 下一轮评测

一条硬规则：**脚本产出的题一律带 `needs_review: true`，不进 active 题库。**
理由是自动化产出的题目质量无法自证，未经确认就入题库，会把评测自己污染掉 ——
用模型生成的题去测模型，还拿结果当结论，这是评测里最隐蔽的错误之一。

用法：
    python scripts/from_badcase.py --run latest --out datasets/candidates/
    python scripts/from_badcase.py --run latest --dimension reasoning --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.analysis import final_score, is_pass  # noqa: E402
from llmeval.pipeline import latest_run, load_run  # noqa: E402

# 每个维度的题面改写方向。规则是「保持判定口径，换掉表面设定」——
# 直接复制原题只会得到一道重复题，价值是零。
REWRITE_HINTS: dict[str, list[str]] = {
    "knowledge": ["换一个学科领域或换个提问角度", "把选择题改为问答，或反之"],
    "reasoning": ["替换全部数值但保持同一解题结构", "把正向求解改为逆向求解（给结果求条件）"],
    "coding": ["保持函数签名与测试用例结构，更换业务场景描述", "增加一个边界用例"],
    "instruction": ["保留约束类型组合，更换约束的具体数值", "增加一条新的约束形成组合"],
    "agent": ["更换材料数据，保持所需的工具调用序列不变", "加入一个不该被调用的诱饵工具"],
    "multiturn": ["保留失效模式（更正/干扰/约束保持），更换任务场景", "增加一轮信息量"],
    "multimodal": ["用同类型图表换一组数值重新绘制", "把单图读数升级为跨图比较"],
    "safety": ["换一种越狱话术包装同一个越权请求", "换一个不应被误拒的正常问题"],
    "long_context": ["保留冲突信息的埋点位置，更换材料主题", "延长材料长度并增加干扰段"],
    "office": ["更换职场场景，保持输出格式约束不变"],
    "realworld": ["更换行业背景，保持约束与权衡结构不变"],
    "knowledge_graph": ["更换实体与关系词表，保持陷阱类型不变"],
    "multi_agent": ["保留预埋的审查要点，更换业务背景"],
}


def _failed_metrics(turn) -> list[str]:
    return [v.metric for v in turn.verdicts if v.passed is False]


def main() -> int:
    parser = argparse.ArgumentParser(description="从失败样本生成候选新题")
    parser.add_argument("--run", default="latest", help="运行目录名，或 latest")
    parser.add_argument("--out", default="datasets/candidates", help="候选题输出目录")
    parser.add_argument("--dimension", default="", help="只处理某个维度")
    parser.add_argument("--top", type=int, default=0, help="每个维度最多生成几道（0=不限）")
    args = parser.parse_args()

    run_dir = latest_run() if args.run == "latest" else Path(args.run)
    summary, turns, _pairwise, specs = load_run(run_dir)
    print("读取 %s（%d 条记录）" % (run_dir.name, len(turns)))

    if summary.mock:
        print("!! 该次运行包含 mock 数据，产出的候选题仅供参考")

    failed = [t for t in turns if is_pass(t) is False]
    print("未通过的记录：%d 条" % len(failed))

    by_dim: dict[str, list] = defaultdict(list)
    for turn in failed:
        if args.dimension and turn.sample.dimension != args.dimension:
            continue
        by_dim[turn.sample.dimension].append(turn)

    if not by_dim:
        print("没有可生成的候选题。可能原因：全部通过、或指定的维度不存在。")
        return 0

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    print()
    print("%-18s %5s %8s  %s" % ("维度", "失败数", "均分", "高频失分指标"))
    for dim, items in sorted(by_dim.items(), key=lambda kv: -len(kv[1])):
        metric_counter: Counter[str] = Counter()
        scores = []
        for turn in items:
            metric_counter.update(_failed_metrics(turn))
            s = final_score(turn)
            if s is not None:
                scores.append(s)
        mean = sum(scores) / len(scores) if scores else 0.0
        top_metrics = "、".join(m for m, _ in metric_counter.most_common(3)) or "—"
        print("%-18s %5d %8.2f  %s" % (dim, len(items), mean, top_metrics))

        picked = items[: args.top] if args.top else items
        rows = []
        for turn in picked:
            sample = turn.sample
            hints = REWRITE_HINTS.get(dim, ["保持判定口径，更换表面设定"])
            rows.append(
                {
                    "id": "%s-cand-%s" % (sample.id, turn.response.sut_id),
                    "dimension": dim,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "needs_review": True,
                    "source_sample_id": sample.id,
                    "failed_sut": turn.response.sut_id,
                    "failed_metrics": _failed_metrics(turn),
                    "origin_prompt": sample.prompt,
                    "rewrite_suggestions": hints,
                    "review_checklist": [
                        "判定口径是否可程序化验证（有 tests / key_facts / gold_* 吗）",
                        "答案是否唯一且已独立复算",
                        "是否与原题实质重复",
                        "难度标注是否与实际相符",
                    ],
                    "prompt": "",
                    "reference": "",
                    "meta": {},
                }
            )

        path = out_dir / ("%s_candidates.jsonl" % dim)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        total += len(rows)

    print()
    print("共生成 %d 道候选题，写入 %s" % (total, out_dir.relative_to(ROOT).as_posix()))
    print()
    print("下一步：逐条填写 prompt / reference / meta，通过 review_checklist 后再合并进题库。")
    print("候选题带 needs_review: true，不要直接改标志位入题库 —— 未审核的题会污染评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

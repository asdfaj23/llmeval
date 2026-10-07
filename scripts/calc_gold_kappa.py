#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金标 κ 计算：对比「人工分」与「自动裁判分」的一致性。
用法：
  1) 把人工分按 金标打分表 里的序号（1..N）逐行写进 gold_human_scores.txt（每行一个 1-5 整数）
  2) 运行：python scripts/calc_gold_kappa.py
自动从 datasets/calibration/gold_llmjudge_20261001.jsonl 读取自动分（auto_score）。
空答自动记为 1.0 的条目已包含在自动分里。
"""
import json, os, sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(BASE, "datasets", "calibration", "gold_llmjudge_20261001.jsonl")
HUMAN = os.path.join(BASE, "datasets", "calibration", "gold_human_scores.txt")


def cohen_kappa(a, b):
    """未加权 Cohen's kappa。a,b 为等长整数列表。"""
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    cidx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[cidx[x]][cidx[y]] += 1
    po = sum(obs[i][i] for i in range(k)) / n
    pa = [sum(obs[i]) / n for i in range(k)]
    pb = [sum(obs[j][i] for j in range(k)) / n for i in range(k)]
    pe = sum(pa[i] * pb[i] for i in range(k))
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0


def weighted_kappa(a, b, weights="linear"):
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    cidx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[cidx[x]][cidx[y]] += 1
    w = [[0.0] * k for _ in range(k)]
    if weights == "linear":
        for i in range(k):
            for j in range(k):
                w[i][j] = abs(i - j) / (k - 1) if k > 1 else 0.0
    else:  # quadratic
        for i in range(k):
            for j in range(k):
                w[i][j] = (abs(i - j) / (k - 1)) ** 2 if k > 1 else 0.0
    po_w = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    pa = [sum(obs[i]) / n for i in range(k)]
    pb = [sum(obs[j][i] for j in range(k)) / n for i in range(k)]
    pe_w = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return (po_w - pe_w) / (1 - pe_w) if (1 - pe_w) != 0 else 1.0


def main():
    items = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    N = len(items)
    # 读取人工分
    if not os.path.exists(HUMAN):
        print("缺少 gold_human_scores.txt，请按打分表序号逐行写入 1-5 分。")
        sys.exit(1)
    human = []
    with open(HUMAN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 支持 "3:5" 或纯数字
            v = line.split(":")[-1].strip()
            if v:
                human.append(int(v))
    if len(human) != N:
        print(f"人工分条数 {len(human)} 与金标条数 {N} 不一致，请核对。")
        sys.exit(1)

    auto = [it["auto_score"] for it in items]
    # 仅保留自动分非 None 的对
    pairs = [(h, a) for h, a, it in zip(human, auto, items) if a is not None]
    if len(pairs) < 2:
        print("可用配对不足，无法计算 κ。")
        sys.exit(1)
    H = [int(x) for x, _ in pairs]
    A = [int(x) for _, x in pairs]
    exact = sum(1 for h, a in pairs if h == a)
    # 偏差 1 分以内
    within1 = sum(1 for h, a in pairs if abs(h - a) <= 1)

    print("=" * 60)
    print(f"金标一致性报告（可用配对 {len(pairs)} / 总 {N} 条，其余为自动分缺失）")
    print("=" * 60)
    print(f"精确一致率（H==A）：{exact}/{len(pairs)} = {exact/len(pairs):.1%}")
    print(f"偏差≤1分占比：     {within1}/{len(pairs)} = {within1/len(pairs):.1%}")
    k = cohen_kappa(H, A)
    kw = weighted_kappa(H, A, "linear")
    print(f"未加权 Cohen's κ：    {k:.3f}")
    print(f"线性加权 Cohen's κ：  {kw:.3f}")
    print(f"0.6 阈值判定：       {'通过（可硬跑）' if (k or 0) >= 0.6 else '未达，需改评分表/换裁判'}")

    # 按维度
    by_dim = defaultdict(list)
    for (h, a), it in zip(pairs, [it for it, a in zip(items, auto) if a is not None]):
        by_dim[it["dimension"]].append((h, a))
    print("-" * 60)
    print("分维度（未加权 κ / 精确一致率）：")
    for dim in sorted(by_dim):
        hs = [h for h, _ in by_dim[dim]]
        aa = [a for _, a in by_dim[dim]]
        kd = cohen_kappa(hs, aa)
        ex = sum(1 for h, a in by_dim[dim] if h == a)
        print(f"  {dim:14s} n={len(hs):2d}  κ={kd:.3f}  exact={ex/len(hs):.0%}")


if __name__ == "__main__":
    main()

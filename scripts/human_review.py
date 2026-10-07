# -*- coding: utf-8 -*-
"""
生成人工盲评页。

为什么要做这个：开放式题目没有唯一答案，程序判不了，只能靠裁判（模型）或人。
裁判有两个绕不开的问题 —— 它会偏向自己家族的模型，也看不出「说得好听但没解决问题」。
所以真正可信的评测，最后一定要有一批人工判断来校准裁判。

这份页面刻意做了三件事：

1. **匿名**：不显示哪个回答来自哪个模型，避免品牌与位置偏见
2. **随机左右顺序**：同一批题里 A/B 的位置随机打乱，避免「总选左边」的惯性
3. **只挑人能判的题**：默认筛掉超长回答与强专业知识题 ——
   让人判一份看不懂的材料，得到的标注也是噪声

产出：一个 HTML，打开后逐题点选，点完导出 JSON。
后续用导出的结果算人工偏好与裁判的一致性（κ），就能知道裁判可不可信。

用法：
    python scripts/human_review.py --run latest --limit 30
    python scripts/human_review.py --run latest --dimension realworld --limit 12
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.pipeline import latest_run, load_run  # noqa: E402

# 需要较强专业知识、人工难以判定的维度，默认跳过
HARD_TO_JUDGE = {"coding", "knowledge_graph"}
MAX_TEXT_CHARS = 1600


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…（已截断，完整内容见运行记录）"


def pick_pairs(turns, dimension: str, limit: int, seed: int):
    """把同一道题的两个模型回答配成对，只保留两边都有内容的。"""
    by_sample: dict[str, dict] = {}
    for turn in turns:
        if not (turn.response.text or "").strip():
            continue
        by_sample.setdefault(turn.sample.id, {})[turn.response.sut_id] = turn

    pairs = []
    for sid, models in by_sample.items():
        if len(models) < 2:
            continue
        sample = next(iter(models.values())).sample
        if dimension and sample.dimension != dimension:
            continue
        if sample.dimension in HARD_TO_JUDGE:
            continue
        pairs.append((sample, models))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return sorted(pairs[:limit], key=lambda p: p[0].id) if limit else pairs


def build_html(pairs, run_id: str, seed: int) -> str:
    cards = []
    for idx, (sample, models) in enumerate(pairs):
        suts = sorted(models.keys())
        rng = random.Random(f"{seed}-{sample.id}")
        left, right = rng.sample(suts, 2) if len(suts) == 2 else (suts[0], suts[0])

        prompt = html.escape(_truncate(sample.prompt, 700))
        context = html.escape(_truncate(sample.context or "", 900))
        ctx_html = (
            f'<details class="ctx"><summary>查看材料（{len(sample.context or "")} 字）</summary>'
            f'<pre>{context}</pre></details>'
        ) if sample.context else ""

        cards.append(
            f"""
    <div class="card" data-id="{html.escape(sample.id)}" data-left="{html.escape(left)}" data-right="{html.escape(right)}">
      <div class="head">
        <span class="idx">{idx + 1}</span>
        <span class="dim">{html.escape(sample.dimension)}</span>
        <span class="sid">{html.escape(sample.id)}</span>
      </div>
      <div class="prompt">{prompt}</div>
      {ctx_html}
      <div class="answers">
        <div class="ans" data-side="left">
          <div class="tag">回答 A</div>
          <pre>{html.escape(_truncate(models[left].response.text))}</pre>
        </div>
        <div class="ans" data-side="right">
          <div class="tag">回答 B</div>
          <pre>{html.escape(_truncate(models[right].response.text))}</pre>
        </div>
      </div>
      <div class="btns">
        <button onclick="choose('{html.escape(sample.id)}','left')">A 更好</button>
        <button onclick="choose('{html.escape(sample.id)}','tie')">差不多</button>
        <button onclick="choose('{html.escape(sample.id)}','right')">B 更好</button>
      </div>
      <div class="chosen" id="chosen-{html.escape(sample.id)}"></div>
    </div>"""
        )

    total = len(cards)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人工盲评 · {html.escape(run_id)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background:#f5f4f1; color:#23211d; margin:0; padding:24px; line-height:1.6; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size:19px; font-weight:600; margin:0 0 6px; }}
  .sub {{ color:#6f6c66; font-size:13px; margin-bottom:18px; }}
  .bar {{ position:sticky; top:0; background:#f5f4f1; padding:10px 0; z-index:9;
          display:flex; gap:12px; align-items:center; border-bottom:1px solid #e3e0da; }}
  .bar b {{ font-size:15px; }}
  button {{ border:0.5px solid #c9c5bd; background:#fff; border-radius:8px;
            padding:7px 16px; font-size:13px; cursor:pointer; color:#23211d; }}
  button:hover {{ border-color:#1f4fd8; }}
  button.primary {{ background:#1f4fd8; color:#fff; border-color:#1f4fd8; }}
  .card {{ background:#fff; border:0.5px solid #e3e0da; border-radius:12px;
           padding:16px 18px; margin-bottom:14px; }}
  .head {{ display:flex; gap:10px; align-items:center; font-size:12px; color:#6f6c66; margin-bottom:8px; }}
  .idx {{ background:#1f4fd8; color:#fff; border-radius:50%; width:20px; height:20px;
          display:inline-flex; align-items:center; justify-content:center; font-size:12px; }}
  .dim {{ background:#eef1f8; color:#1f4fd8; padding:1px 8px; border-radius:10px; }}
  .prompt {{ font-size:14px; margin-bottom:10px; white-space:pre-wrap; }}
  .ctx summary {{ font-size:12px; color:#6f6c66; cursor:pointer; }}
  .ctx pre {{ background:#f7f6f3; padding:10px; border-radius:8px; font-size:12px;
              max-height:220px; overflow:auto; white-space:pre-wrap; }}
  .answers {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:10px 0; }}
  @media (max-width: 860px) {{ .answers {{ grid-template-columns:1fr; }} }}
  .ans {{ border:0.5px solid #e3e0da; border-radius:10px; padding:10px 12px; }}
  .ans.sel {{ border-color:#1f4fd8; border-width:2px; }}
  .tag {{ font-size:12px; color:#6f6c66; margin-bottom:6px; }}
  .ans pre {{ margin:0; font-size:12.5px; white-space:pre-wrap; word-break:break-word;
              max-height:420px; overflow:auto; font-family:inherit; }}
  .btns {{ display:flex; gap:8px; }}
  .chosen {{ font-size:12px; color:#0f7b6c; margin-top:6px; min-height:16px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>人工盲评：两组回答，哪个更好？</h1>
  <div class="sub">
    共 {total} 组，回答已<b>匿名</b>且左右顺序随机，不显示模型名 —— 这样才能不看品牌只比质量。
    判不了的题点「差不多」即可，不要猜。
  </div>
  <div class="bar">
    <b>进度 <span id="done">0</span> / {total}</b>
    <button class="primary" onclick="exportJson()">导出结果</button>
    <button onclick="clearAll()">清空重填</button>
  </div>
{"".join(cards)}
</div>
<script>
const picks = {{}};
function choose(id, side) {{
  picks[id] = side;
  const card = document.querySelector('.card[data-id="' + CSS.escape(id) + '"]');
  card.querySelectorAll('.ans').forEach(a => a.classList.remove('sel'));
  if (side !== 'tie') card.querySelector('.ans[data-side="' + side + '"]').classList.add('sel');
  const label = side === 'left' ? '已选：A 更好' : side === 'right' ? '已选：B 更好' : '已选：差不多';
  document.getElementById('chosen-' + id).textContent = label;
  document.getElementById('done').textContent = Object.keys(picks).length;
}}
function clearAll() {{
  Object.keys(picks).forEach(k => delete picks[k]);
  document.querySelectorAll('.ans').forEach(a => a.classList.remove('sel'));
  document.querySelectorAll('.chosen').forEach(c => c.textContent = '');
  document.getElementById('done').textContent = '0';
}}
function exportJson() {{
  const out = [];
  document.querySelectorAll('.card').forEach(card => {{
    const id = card.dataset.id;
    if (!(id in picks)) return;
    out.push({{
      sample_id: id,
      left_model: card.dataset.left,
      right_model: card.dataset.right,
      winner: picks[id],
      winner_model: picks[id] === 'left' ? card.dataset.left
                 : picks[id] === 'right' ? card.dataset.right : null
    }});
  }});
  const blob = new Blob([JSON.stringify({{run: '{run_id}', reviews: out}}, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'human_review_' + '{run_id}' + '.json';
  a.click();
}}
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成人工盲评页")
    parser.add_argument("--run", default="latest")
    parser.add_argument("--out", default="outputs/human_review.html")
    parser.add_argument("--dimension", default="", help="只评某个维度")
    parser.add_argument("--limit", type=int, default=30, help="最多多少组（0=不限）")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    run_dir = latest_run() if args.run == "latest" else Path(args.run)
    _summary, turns, _pw, specs = load_run(run_dir)
    pairs = pick_pairs(turns, args.dimension, args.limit, args.seed)

    if not pairs:
        print("没有可评的题。可能原因：该维度不存在、或两侧回答为空。")
        return 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(pairs, run_dir.name, args.seed), encoding="utf-8")

    dims: dict[str, int] = {}
    for sample, _m in pairs:
        dims[sample.dimension] = dims.get(sample.dimension, 0) + 1
    print("运行：%s" % run_dir.name)
    print("生成 %d 组对比，维度分布：%s" % (len(pairs), dims))
    print("输出：%s" % out_path.relative_to(ROOT).as_posix())
    print()
    print("打开后逐题点选，点完点「导出结果」得到 JSON；")
    print("把 JSON 发我，我拿它算人工偏好与裁判分的一致性（κ），判断裁判可不可信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""
生成多模态评测用的图片资产。

为什么不用 AI 生图：生成式模型在图上写数字和文字极不可靠，经常出现
「图里画着 153、实际渲染成 158」这种问题。评测题的答案必须和图上内容严格一致，
所以这里用 PIL 精确绘制 —— 画什么就是什么，题目答案由绘制代码直接决定。

覆盖六类多模态能力：
    图表读数 · 计数 · 空间关系 · 图中文字 · 视觉推理 · 图示理解

运行：python scripts/gen_images.py
输出：datasets/multimodal/images/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "multimodal" / "images"
W, H = 860, 620
BG = "#ffffff"
INK = "#1a1a1a"
AXIS = "#333333"

_FONT_FILES = (
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_FILES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("找不到可用的中文字体，请安装 SimHei / 微软雅黑 / 文泉驿")


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def center(d: ImageDraw.ImageDraw, x: float, y: float, text: str, f, fill: str = INK) -> None:
    """按中心点写文字。Pillow 的 anchor 参数在旧版本上行为不一致，所以手算宽度。"""
    w = d.textlength(text, font=f)
    d.text((x - w / 2, y), text, font=f, fill=fill)


def title(d: ImageDraw.ImageDraw, text: str, y: int = 28) -> None:
    center(d, W / 2, y, text, font(30))


def caption(d: ImageDraw.ImageDraw, text: str, y: int = H - 52) -> None:
    center(d, W / 2, y, text, font(20), fill="#555555")


# ------------------------------------------------------------------ 图表读数
def bar_revenue() -> None:
    img, d = new_canvas()
    title(d, "某公司 2026 年各季度营收（万元）")
    left, right, top, bottom = 110, 790, 110, 500
    d.line([(left, top), (left, bottom)], fill=AXIS, width=2)
    d.line([(left, bottom), (right, bottom)], fill=AXIS, width=2)

    data = [("Q1", 120), ("Q2", 185), ("Q3", 153), ("Q4", 210)]
    maxv, bw = 240, 96
    for i, (label, value) in enumerate(data):
        x = left + 55 + i * 165
        h = int(value / maxv * (bottom - top - 40))
        d.rectangle([x, bottom - h, x + bw, bottom], fill="#4a7ebb", outline=AXIS, width=2)
        center(d, x + bw / 2, bottom - h - 30, str(value), font(22))
        center(d, x + bw / 2, bottom + 16, label, font(24))
    caption(d, "纵轴单位：万元")
    img.save(OUT / "img_bar_revenue.png")


def line_temperature() -> None:
    img, d = new_canvas()
    title(d, "某城市 1-6 月月均气温（摄氏度）")
    left, right, top, bottom = 110, 790, 110, 500
    d.line([(left, top), (left, bottom)], fill=AXIS, width=2)
    d.line([(left, bottom), (right, bottom)], fill=AXIS, width=2)

    temps = [5, 8, 14, 20, 26, 31]
    maxv, minv = 35, 0
    n = len(temps)
    step = (right - left - 60) / (n - 1)
    pts = []
    for i, t in enumerate(temps):
        x = left + 40 + i * step
        y = bottom - int((t - minv) / (maxv - minv) * (bottom - top - 40))
        pts.append((x, y))
    d.line(pts, fill="#c0504d", width=4)
    for i, (x, y) in enumerate(pts):
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill="#c0504d")
        center(d, x, y - 34, str(temps[i]), font(22))
        center(d, x, bottom + 16, "%d月" % (i + 1), font(22))
    caption(d, "纵轴单位：摄氏度")
    img.save(OUT / "img_line_temperature.png")


def pie_share() -> None:
    img, d = new_canvas()
    title(d, "2026 年某市场四家公司份额占比")
    shares = [("甲公司", 35, "#4a7ebb"), ("乙公司", 28, "#c0504d"),
              ("丙公司", 22, "#9bbb59"), ("丁公司", 15, "#8064a2")]
    box = (150, 130, 590, 570)
    start = -90.0
    for _, pct, color in shares:
        extent = pct / 100 * 360
        d.pieslice(box, start, start + extent, fill=color, outline="white", width=3)
        start += extent
    ly = 180
    for name, pct, color in shares:
        d.rectangle([640, ly, 672, ly + 28], fill=color, outline=AXIS, width=1)
        d.text((686, ly + 2), "%s %d%%" % (name, pct), font=font(24), fill=INK)
        ly += 58
    caption(d, "四家公司份额合计 100%")
    img.save(OUT / "img_pie_share.png")


def bar_grouped() -> None:
    img, d = new_canvas()
    title(d, "三种产品 2025 与 2026 年营收对比（万元）")
    left, right, top, bottom = 110, 800, 110, 500
    d.line([(left, top), (left, bottom)], fill=AXIS, width=2)
    d.line([(left, bottom), (right, bottom)], fill=AXIS, width=2)

    groups = [("产品甲", 80, 110), ("产品乙", 95, 90), ("产品丙", 60, 88)]
    maxv, bw = 130, 58
    for i, (name, v25, v26) in enumerate(groups):
        x0 = left + 70 + i * 230
        for j, (v, color) in enumerate([(v25, "#8faadc"), (v26, "#4a7ebb")]):
            x = x0 + j * (bw + 10)
            h = int(v / maxv * (bottom - top - 40))
            d.rectangle([x, bottom - h, x + bw, bottom], fill=color, outline=AXIS, width=2)
            center(d, x + bw / 2, bottom - h - 28, str(v), font(20))
        center(d, x0 + bw + 5, bottom + 16, name, font(24))
    d.rectangle([560, 130, 592, 158], fill="#8faadc", outline=AXIS, width=1)
    d.text((602, 132), "2025 年", font=font(22), fill=INK)
    d.rectangle([560, 172, 592, 200], fill="#4a7ebb", outline=AXIS, width=1)
    d.text((602, 174), "2026 年", font=font(22), fill=INK)
    img.save(OUT / "img_bar_grouped.png")


# ------------------------------------------------------------------ 计数
def count_shapes() -> None:
    img, d = new_canvas()
    title(d, "下图由三种图形组成")
    circles = [(170, 190), (300, 170), (430, 200), (560, 175), (690, 195)]
    for x, y in circles:
        d.ellipse([x - 45, y - 45, x + 45, y + 45], fill="#4a7ebb", outline=AXIS, width=2)
    triangles = [(200, 360), (430, 380), (660, 355)]
    for x, y in triangles:
        d.polygon([(x, y - 52), (x - 55, y + 42), (x + 55, y + 42)],
                  fill="#c0504d", outline=AXIS)
    squares = [(170, 520), (350, 515), (530, 525), (710, 515)]
    for x, y in squares:
        d.rectangle([x - 42, y - 42, x + 42, y + 42], fill="#9bbb59", outline=AXIS, width=2)
    caption(d, "请分别统计图中三种图形的数量")
    img.save(OUT / "img_count_shapes.png")


def count_dots() -> None:
    img, d = new_canvas()
    title(d, "请统计下图中两种圆点各有多少个")
    # 分两行摆放，行距与列距都大于直径，保证不会出现相切叠压导致数不清
    reds = [(140, 200), (250, 200), (360, 200), (470, 200),
            (580, 200), (690, 200), (800, 200), (195, 350)]
    blues = [(305, 350), (415, 350), (525, 350), (635, 350)]
    for x, y in reds:
        d.ellipse([x - 32, y - 32, x + 32, y + 32], fill="#c0504d", outline=AXIS, width=2)
    for x, y in blues:
        d.ellipse([x - 32, y - 32, x + 32, y + 32], fill="#4a7ebb", outline=AXIS, width=2)
    caption(d, "红色圆点与蓝色圆点数量不同")
    img.save(OUT / "img_count_dots.png")


# ------------------------------------------------------------------ 空间关系
def spatial_blocks() -> None:
    img, d = new_canvas()
    title(d, "四个方块在画面中的位置分布")
    blocks = [
        ("红", 180, 200, "#c0504d"),
        ("蓝", 620, 200, "#4a7ebb"),
        ("绿", 180, 450, "#9bbb59"),
        ("黄", 620, 450, "#e0b000"),
    ]
    for name, x, y, color in blocks:
        d.rectangle([x - 80, y - 80, x + 80, y + 80], fill=color, outline=AXIS, width=3)
        center(d, x, y - 22, name, font(40), fill="white" if name != "黄" else INK)
    caption(d, "四个方块分处画面的不同区域")
    img.save(OUT / "img_spatial_blocks.png")


def arrow_directions() -> None:
    img, d = new_canvas()
    title(d, "三个箭头的指向")

    def arrow(cx: int, cy: int, dx: int, dy: int, color: str) -> None:
        L = 100
        d.line([(cx - dx * L, cy - dy * L), (cx + dx * L, cy + dy * L)], fill=color, width=14)
        tip = (cx + dx * (L + 34), cy + dy * (L + 34))
        left = (cx + dx * L - dy * 30, cy + dy * L + dx * 30)
        right = (cx + dx * L + dy * 30, cy + dy * L - dx * 30)
        d.polygon([tip, left, right], fill=color)

    arrow(430, 170, 1, 0, "#4a7ebb")      # 向右
    arrow(430, 330, 0, 1, "#9bbb59")      # 向下
    arrow(430, 490, -1, 0, "#c0504d")     # 向左
    caption(d, "从上到下依次为第一、第二、第三个箭头")
    img.save(OUT / "img_arrows.png")


# ------------------------------------------------------------------ 图中文字
def sign_text() -> None:
    img, d = new_canvas()
    title(d, "观察下图中的指示牌")
    d.rectangle([120, 130, 740, 520], fill="#f2f2f2", outline=AXIS, width=3)
    lines = ["前方 500 米", "加油站", "服务区 2 公里"]
    for i, text in enumerate(lines):
        center(d, 430, 190 + i * 110, text, font(48))
    img.save(OUT / "img_sign_text.png")


def table_numbers() -> None:
    img, d = new_canvas()
    title(d, "2026 年某项目分季度支出表（万元）")
    rows = [
        ["季度", "人力", "设备", "差旅"],
        ["一季度", "48", "36", "12"],
        ["二季度", "52", "41", "9"],
        ["三季度", "45", "38", "15"],
        ["四季度", "60", "44", "11"],
    ]
    x0, y0, cw, ch = 130, 150, 150, 82
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = x0 + c * cw, y0 + r * ch
            d.rectangle([x, y, x + cw, y + ch], outline=AXIS, width=2,
                        fill="#e8eef7" if r == 0 else BG)
            center(d, x + cw / 2, y + ch / 2 - 18, cell, font(26))
    img.save(OUT / "img_table_numbers.png")


def mixed_text() -> None:
    img, d = new_canvas()
    title(d, "阅读下段文字并回答")
    d.rectangle([90, 120, 770, 500], fill="#fafafa", outline=AXIS, width=2)
    lines = [
        "项目代号：BlueHarbor",
        "负责人：李工程师",
        "交付日期：2026 年 11 月 8 日",
        "预算上限：320 万元",
    ]
    for i, text in enumerate(lines):
        d.text((130, 165 + i * 82), text, font=font(34), fill=INK)
    img.save(OUT / "img_mixed_text.png")


# ------------------------------------------------------------------ 视觉推理
def compare_bars() -> None:
    img, d = new_canvas()
    title(d, "比较两条线段的长短")
    d.rectangle([150, 200, 680, 250], fill="#4a7ebb", outline=AXIS, width=2)
    center(d, 415, 208, "A", font(28), fill="white")
    d.rectangle([150, 380, 590, 430], fill="#9bbb59", outline=AXIS, width=2)
    center(d, 370, 388, "B", font(28), fill="white")
    caption(d, "两条线段的起点都在左侧，右端位置不同")
    img.save(OUT / "img_compare_bars.png")


def pattern_sequence() -> None:
    img, d = new_canvas()
    title(d, "观察下面图形的排列规律")
    colors = ["#4a7ebb", "#c0504d", "#9bbb59"]
    for i in range(7):
        x = 130 + i * 100
        d.rectangle([x - 38, 250 - 38, x + 38, 250 + 38],
                    fill=colors[i % 3], outline=AXIS, width=2)
    d.text((790, 232), "?", font=font(46), fill=INK)
    caption(d, "颜色按固定顺序循环")
    img.save(OUT / "img_pattern_sequence.png")


def anomaly_row() -> None:
    img, d = new_canvas()
    title(d, "找出与其余图形不同的那一个")
    for i in range(6):
        x = 140 + i * 115
        if i == 4:
            d.ellipse([x - 38, 300 - 38, x + 38, 300 + 38], fill="#e0b000", outline=AXIS, width=2)
        else:
            d.rectangle([x - 38, 300 - 38, x + 38, 300 + 38], fill="#4a7ebb", outline=AXIS, width=2)
    caption(d, "从左到右编号为 1 到 6")
    img.save(OUT / "img_anomaly_row.png")


# ------------------------------------------------------------------ 图示理解
def flow_diagram() -> None:
    img, d = new_canvas()
    title(d, "数据处理流程示意图")

    def node(cx: int, cy: int, text: str) -> None:
        d.rectangle([cx - 90, cy - 40, cx + 90, cy + 40], fill="#e8eef7",
                    outline=AXIS, width=2)
        center(d, cx, cy - 14, text, font(26))

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        d.line([(x1, y1), (x2, y2)], fill=AXIS, width=3)
        dx, dy = (1 if x2 > x1 else -1 if x2 < x1 else 0), (1 if y2 > y1 else -1 if y2 < y1 else 0)
        d.polygon([(x2, y2), (x2 - dx * 20 - dy * 10, y2 - dy * 20 + dx * 10),
                   (x2 - dx * 20 + dy * 10, y2 - dy * 20 - dx * 10)], fill=AXIS)

    node(180, 180, "采集")
    node(180, 430, "清洗")
    node(520, 430, "标注")
    node(760, 430, "入库")
    arrow(180, 220, 180, 390)
    arrow(270, 430, 430, 430)
    arrow(610, 430, 670, 430)
    node(520, 180, "复核")
    arrow(520, 390, 520, 220)
    arrow(430, 180, 270, 180)
    img.save(OUT / "img_flow_diagram.png")


def table_layout() -> None:
    img, d = new_canvas()
    title(d, "某设备运行状态看板")
    rows = [
        ["设备", "状态", "温度"],
        ["A-01", "正常", "42 ℃"],
        ["B-02", "告警", "78 ℃"],
        ["C-03", "正常", "39 ℃"],
    ]
    x0, y0, cw, ch = 190, 160, 160, 88
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = x0 + c * cw, y0 + r * ch
            fill = "#e8eef7" if r == 0 else ("#fbe4e4" if r == 2 else BG)
            d.rectangle([x, y, x + cw, y + ch], outline=AXIS, width=2, fill=fill)
            center(d, x + cw / 2, y + ch / 2 - 18, cell, font(26))
    img.save(OUT / "img_table_layout.png")


BUILDERS: Sequence[Iterable] = (
    bar_revenue,
    line_temperature,
    pie_share,
    bar_grouped,
    count_shapes,
    count_dots,
    spatial_blocks,
    arrow_directions,
    sign_text,
    table_numbers,
    mixed_text,
    compare_bars,
    pattern_sequence,
    anomaly_row,
    flow_diagram,
    table_layout,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in BUILDERS:
        build()
    files = sorted(OUT.glob("*.png"))
    total = sum(f.stat().st_size for f in files)
    print("生成 %d 张图片，合计 %.1f KB" % (len(files), total / 1024))
    for f in files:
        print("  %-30s %6.1f KB" % (f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()

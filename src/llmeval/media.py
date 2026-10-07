# -*- coding: utf-8 -*-
"""
图片加载：把本地图片转成 OpenAI 兼容的 data URL。

多模态题目要把图片塞进消息体，而模型 API 只接受两种形式：
公网可访问的 URL，或者 base64 内联。评测必须离线可复现、不依赖外部图床，
所以统一走 base64 内联 —— 图片存在项目里，跑多少次结果都一样。

体积提醒：base64 会让字节数膨胀约 1/3，所以题库里的图片都是压缩过的小图。
"""

from __future__ import annotations

import base64
import mimetypes
from functools import lru_cache
from pathlib import Path

from .config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "datasets"

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def resolve_image(rel: str) -> Path:
    """把题目里写的图片路径解析成绝对路径。

    规则：绝对路径原样使用；相对路径一律相对 datasets/ 解析 ——
    这样题库文件里只写 `multimodal/images/chart_01.png` 这种短路径，
    换机器、换目录都不用改题。
    """
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


@lru_cache(maxsize=64)
def image_data_url(rel: str) -> str | None:
    """读图并转成 data URL。文件不存在返回 None（由调用方决定如何报错）。"""
    path = resolve_image(rel)
    if not path.is_file():
        return None
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def missing_images(sample: object) -> list[str]:
    """列出某道题里缺失的图片，供 SUT 与自检脚本报告问题。

    图片缺了不能默默当纯文本题跑 —— 那样测出来的分数是假的。
    """
    rels = list(getattr(sample, "images", []) or [])
    return [rel for rel in rels if not resolve_image(rel).is_file()]

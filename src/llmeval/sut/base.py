# -*- coding: utf-8 -*-
"""
被测系统（SUT）抽象层。

SUT 的职责被刻意收得很窄：拿到一道题，产出一个 Response。

把「怎么问」和「怎么评」彻底切开，是这套框架能横向扩展的前提 ——
今天测纯对话模型，明天测 RAG 管线、测 Agent、测带工具的产品形态，
评测逻辑一行都不用改，只要多写一个 SUT 子类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..client import LLMClient
from ..config import ModelSpec
from ..media import image_data_url
from ..schema import Response, Sample


class BaseSUT(ABC):
    """所有被测系统的基类。"""

    kind = "base"

    def __init__(self, spec: ModelSpec, client: LLMClient) -> None:
        self.spec = spec
        self.client = client

    @property
    def id(self) -> str:
        return self.spec.id

    @abstractmethod
    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        """跑一道题，返回 Response。实现里不允许抛异常打断整轮评测。"""

    # ------------------------------------------------------------ 共用逻辑
    def build_messages(self, sample: Sample) -> list[dict[str, Any]]:
        """把一道题渲染成对话消息。

        有材料时把材料放进 user 消息而不是 system —— 长上下文场景下，
        材料属于"这次任务的一部分"，而不是"模型的长期设定"。

        带图片的题走 OpenAI 兼容的多模态 content 数组：
            [{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        图片文件缺失时直接报错，不静默退化成纯文本题 —— 那样测出来的分数是假的。
        """
        messages: list[dict[str, Any]] = []
        if sample.system:
            messages.append({"role": "system", "content": sample.system})

        if sample.context:
            text = f"## 材料\n{sample.context}\n\n## 问题\n{sample.prompt}"
        else:
            text = sample.prompt

        if sample.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for rel in sample.images:
                url = image_data_url(rel)
                if url is None:
                    raise FileNotFoundError(f"题 {sample.id} 引用的图片不存在：{rel}")
                parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": text})
        return messages

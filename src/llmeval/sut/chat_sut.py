# -*- coding: utf-8 -*-
"""纯对话型被测对象。一问一答，不涉及工具调用。"""

from __future__ import annotations

from ..schema import Response, Sample
from .base import BaseSUT


class ChatSUT(BaseSUT):
    """最基础的一类 SUT：把题面发过去，把回答收回来。

    绝大多数能力维度（知识、指令遵循、推理、代码、长上下文、安全）
    都可以用这一种形态测完。
    """

    kind = "chat"

    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        messages = self.build_messages(sample)
        result = self.client.chat(
            messages,
            hint={"kind": "sut", "sample": sample, "messages": messages},
        )
        return Response(
            sample_id=sample.id,
            sut_id=self.spec.id,
            text=result.text,
            latency_s=result.latency_s,
            usage=result.usage,
            error=result.error,
            attempts=result.attempts,
            attempt_index=attempt_index,
        )

# -*- coding: utf-8 -*-
"""
多轮对话 SUT。

和 ChatSUT 的区别只有一个字：轮。但它带来的差别是本质的 ——
ChatSUT 把整道题一次性给模型，MultiTurnSUT 必须**逐轮**给，
模型无法预知后面还会补充什么信息。

这正是《LLMs Get Lost in Multi-Turn Conversation》的 sharding 设置：
同一条完整指令，拆成多轮喂进去，模型的成绩会掉。掉多少，就是被测的短板。

硬约束：绝不允许把 turns 一次性拼成一条消息发出去。
那样测的还是单轮能力，多轮维度会变成假的 —— 而且这种错很难从结果上看出来。
"""

from __future__ import annotations

import time
from typing import Any

from ..schema import Response, Sample, Usage
from .base import BaseSUT


class MultiTurnSUT(BaseSUT):
    kind = "multi_turn"

    def run(self, sample: Sample, attempt_index: int = 0) -> Response:
        turns = list(sample.turns) or ([sample.prompt] if sample.prompt else [])
        if not turns:
            return Response(
                sample_id=sample.id,
                sut_id=self.id,
                error="多轮题没有配置 turns",
                attempt_index=attempt_index,
            )

        # 长材料并入第一轮，而不是单开一轮 —— 材料是任务背景，不是一次对话发言
        if sample.context:
            turns = [f"## 材料\n{sample.context}\n\n{turns[0]}", *turns[1:]]

        messages: list[dict[str, Any]] = []
        if sample.system:
            messages.append({"role": "system", "content": sample.system})

        log: list[dict[str, Any]] = []
        usage = Usage()
        error: str | None = None
        final_text = ""
        started = time.time()

        for idx, user_text in enumerate(turns, 1):
            messages.append({"role": "user", "content": user_text})
            log.append({"turn": idx, "role": "user", "text": user_text})

            result = self.client.chat(
                messages,
                hint={
                    "kind": "sut",
                    "sample": sample,
                    "messages": messages,
                    "turn_index": idx,
                    "n_turns": len(turns),
                },
            )
            usage = usage + result.usage
            if result.error:
                error = f"第 {idx} 轮失败：{result.error}"
                break

            messages.append({"role": "assistant", "content": result.text})
            log.append({"turn": idx, "role": "assistant", "text": result.text})
            final_text = result.text

        return Response(
            sample_id=sample.id,
            sut_id=self.id,
            text=final_text,
            latency_s=time.time() - started,
            usage=usage,
            error=error,
            messages=log,
            attempts=len(log),
            attempt_index=attempt_index,
        )

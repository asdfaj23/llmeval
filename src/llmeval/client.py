# -*- coding: utf-8 -*-
"""
统一 LLM 客户端。

三件事：
    1. 协议统一 —— 所有服务商都按 OpenAI 兼容协议调用，换家只改 base_url
    2. 工程保障 —— 超时、指数退避重试、并发、结果缓存、用量统计，一次写好到处复用
    3. 离线兜底 —— provider 为 mock 时不发任何网络请求，走确定性模拟

刻意不用 openai SDK，HTTP 直接走标准库 urllib。
理由：这个框架要能在纯 CPU、无额外依赖的环境里跑起来，
而且自己实现重试与缓存，比配置别人的客户端更可控。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT, ModelSpec, RuntimeCfg
from .mock import MockEngine
from .schema import Usage

# 这些状态码值得重试：限流与瞬时故障
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResult:
    """一次模型调用的结果。失败不抛异常，而是把 error 填上 —— 评测要跑完全部样本，
    不能因为一条请求失败就整体中断。"""

    text: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model_id: str = ""
    error: str | None = None
    attempts: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClient:
    """一个模型端点对应一个 client。线程安全。"""

    def __init__(
        self,
        spec: ModelSpec,
        runtime: RuntimeCfg,
        cache_root: Path | None = None,
    ) -> None:
        self.spec = spec
        self.runtime = runtime
        self._mock = MockEngine(spec.model) if spec.is_mock else None
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0, "retries": 0}

        self._cache_dir: Path | None = None
        if runtime.cache and not spec.is_mock:
            root = cache_root or (PROJECT_ROOT / runtime.cache_dir)
            self._cache_dir = Path(root)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 对外接口
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        hint: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """发起一次对话补全。

        hint 只在 mock 模式下有意义，用来告诉模拟引擎"现在在答哪道题"。
        真实 provider 会完全忽略它 —— 所以流水线代码无需分支。
        """
        temperature = self.spec.temperature if temperature is None else temperature
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens
        started = time.time()

        if self._mock is not None:
            return self._mock_call(hint, started)

        cache_key = self._cache_key(messages, temperature, max_tokens)
        cached = self._read_cache(cache_key)
        if cached is not None and not (cached.text or "").strip():
            # 历史上缓存过的空回答一律作废重跑 ——
            # 这些多半是推理模型把 token 预算耗在思考过程上后留下的空壳。
            cached = None
        if cached is not None:
            with self._lock:
                self.stats["cache_hits"] += 1
            cached.usage.cached = True
            cached.latency_s = 0.0
            # 成本按当前配置的单价重算：缓存里存的是响应本身，不该把旧价一并锁死，
            # 否则改完 models.yaml 的单价必须清缓存重跑，才能看到正确的花费。
            cached.usage.cost_usd = self.spec.estimate_cost(
                cached.usage.prompt_tokens, cached.usage.completion_tokens
            )
            return cached

        result = self._http_call(messages, temperature, max_tokens, started)
        if result.ok:
            self._write_cache(cache_key, result)
        return result

    # ------------------------------------------------------------ mock
    def _mock_call(self, hint: dict[str, Any] | None, started: float) -> LLMResult:
        assert self._mock is not None
        hint = hint or {}
        if not hint.get("sample"):
            return LLMResult(
                text="",
                error="mock 模式必须通过 hint 传入 sample",
                model_id=self.spec.id,
                latency_s=time.time() - started,
            )

        if hint.get("kind") == "judge":
            payload: Any = self._mock.judge(hint)
            text = json.dumps(payload, ensure_ascii=False)
        else:
            text = self._mock.answer(hint)

        pt = max(1, sum(len(m.get("content", "")) for m in hint.get("messages", [])) // 2)
        ct = max(1, len(text) // 2)
        usage = Usage(prompt_tokens=pt, completion_tokens=ct)
        usage.cost_usd = self.spec.estimate_cost(pt, ct)
        # 模拟一点延迟，让并发与耗时统计不是全 0
        latency = 0.002 + (len(text) % 40) * 0.0004
        time.sleep(latency)

        with self._lock:
            self.stats["calls"] += 1
        return LLMResult(
            text=text,
            usage=usage,
            latency_s=time.time() - started,
            model_id=self.spec.id,
            attempts=1,
            raw={"mock": True},
        )

    # ------------------------------------------------------------ 真实调用
    def _http_call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        started: float,
    ) -> LLMResult:
        url = self.spec.provider.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.spec.provider.api_key}",
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

        last_error = ""
        for attempt in range(1, self.runtime.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.runtime.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text, usage = self._parse_response(data)
                with self._lock:
                    self.stats["calls"] += 1
                if not text.strip():
                    # 空内容必须当失败处理。静默放行的话，「模型什么都没说」会被记成答错，
                    # 看起来和能力差没有区别 —— 实际多半是思考过程吃光了 token 预算。
                    self.stats["errors"] += 1
                    return LLMResult(
                        text="",
                        error="模型返回空内容（思考过程可能已耗尽 max_tokens 预算）",
                        usage=usage,
                        latency_s=time.time() - started,
                        model_id=self.spec.id,
                        attempts=attempt,
                        raw=data,
                    )
                return LLMResult(
                    text=text,
                    usage=usage,
                    latency_s=time.time() - started,
                    model_id=self.spec.id,
                    attempts=attempt,
                    raw=data,
                )
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in _RETRYABLE_STATUS:
                    break
            except urllib.error.URLError as exc:
                last_error = f"网络错误: {exc.reason}"
            except TimeoutError:
                last_error = f"请求超时（{self.runtime.timeout_s}s）"
            except Exception as exc:  # noqa: BLE001 - 兜底，评测不能被单条异常打断
                last_error = f"{type(exc).__name__}: {exc}"

            with self._lock:
                self.stats["retries"] += 1
            if attempt < self.runtime.max_retries:
                time.sleep(self.runtime.retry_backoff_s * (2 ** (attempt - 1)))

        with self._lock:
            self.stats["calls"] += 1
            self.stats["errors"] += 1
        return LLMResult(
            text="",
            error=f"{self.spec.id} 调用失败（重试 {self.runtime.max_retries} 次）：{last_error}",
            latency_s=time.time() - started,
            model_id=self.spec.id,
            attempts=self.runtime.max_retries,
        )

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> tuple[str, Usage]:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
            completion_tokens=int(u.get("completion_tokens", 0) or 0),
        )
        return text, usage

    # ------------------------------------------------------------ 缓存
    def _cache_key(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        blob = json.dumps(
            {
                "provider": self.spec.provider.key,
                "model": self.spec.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(
                text=raw.get("text", ""),
                usage=Usage(**raw.get("usage", {})),
                model_id=self.spec.id,
                attempts=1,
                raw={"cached": True},
            )
        except Exception:
            return None

    def _write_cache(self, key: str, result: LLMResult) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps(
                    {"text": result.text, "usage": result.usage.__dict__},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


# ------------------------------------------------------------------ 并发调度
def run_parallel(
    items: list[Any],
    worker: Callable[[int, Any], Any],
    max_workers: int = 4,
    on_done: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """按序并发执行，返回值顺序与输入一致。

    顺序一致很重要：评测结果要能跟评测集逐行对齐，否则没法复查。
    """
    if not items:
        return []
    results: list[Any] = [None] * len(items)
    if max_workers <= 1:
        for i, item in enumerate(items):
            results[i] = worker(i, item)
            if on_done:
                on_done(i, results[i])
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[idx] = exc
            if on_done:
                on_done(idx, results[idx])
    return results

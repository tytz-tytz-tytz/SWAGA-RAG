"""
Shared judge client and response parser used by calibration / full-run scripts.

Three backends, three different native APIs (one per family — CometAPI uses
each provider's native protocol, not OpenAI-compatible chat completions):

  Backend "anthropic"
    - Anthropic Messages API
    - Used for claude-haiku-4-5-20251001 via CometAPI (base_url override).
    - SDK: anthropic.AsyncAnthropic(base_url=..., api_key=...)

  Backend "openai"
    - OpenAI Chat Completions API
    - Used for gpt-4.1-mini direct to OpenAI (no base_url override).
    - SDK: openai.AsyncOpenAI(api_key=...)

  Backend "gemini"
    - Google Gemini generateContent API
    - Used for gemini-2.5-flash via CometAPI (base_url override).
    - SDK: google.genai.Client(http_options={...base_url}, api_key=...)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import resolve_repo_path  # noqa: E402

try:
    from dotenv import load_dotenv  # type: ignore

    _LOCAL_ENV = resolve_repo_path(".env")
    if _LOCAL_ENV.exists():
        load_dotenv(_LOCAL_ENV, override=False)
except Exception:
    pass


VALID_LABELS = {"A", "B", "equal"}
AXES = ("relevance", "cleanliness", "sufficiency")


@dataclass
class JudgeConfig:
    name: str
    backend: str
    model: str
    api_key: str
    base_url: Optional[str]
    shuffle_seed: int
    extra_body: Optional[Dict[str, Any]] = None  # e.g. {"enable_thinking": false}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeConfig":
        api_key = os.environ.get(d["api_key_env"], "")
        if not api_key:
            raise RuntimeError(
                f"API key env var not set: {d['api_key_env']} (judge {d['name']})"
            )
        base_url: Optional[str] = None
        base_env = d.get("base_url_env")
        if base_env:
            base_url = os.environ.get(base_env) or d.get("default_base_url")
        backend = d.get("backend")
        if backend not in {"anthropic", "openai", "gemini"}:
            raise ValueError(f"Unknown backend for judge {d['name']}: {backend!r}")
        return cls(
            name=d["name"],
            backend=backend,
            model=d["model"],
            api_key=api_key,
            base_url=base_url,
            shuffle_seed=int(d.get("shuffle_seed", 42)),
            extra_body=d.get("extra_body") or None,
        )


@dataclass
class JudgeDecision:
    labels: Dict[str, str]
    raw: str
    status: str           # "ok" | "failed"
    attempts: int = 1
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _normalize_response_text(text: str) -> str:
    """Common preprocessor for raw model output before JSON extraction.

    Handles the most common wrappings observed across the three providers:
    - markdown code fences (with or without "json"/"JSON" language hint)
    - leading conversational preamble ("Here is the JSON:", "Вот мой ответ:" ...)
    - BOM and zero-width characters

    Returns the cleaned string. Does not validate JSON.
    """
    if not text:
        return ""
    s = text.strip().lstrip("﻿").replace("​", "")

    # ``` ... ``` fenced block, optional language hint
    fence = re.match(r"^```\s*(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    return s


def _extract_json_object(text: str) -> Optional[str]:
    s = _normalize_response_text(text)
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None
    # Track depth, respecting string literals so braces inside JSON strings
    # do not confuse the matcher.
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_judge_response(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    block = _extract_json_object(text)
    if block is None:
        return None, "no JSON object found"
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    if not isinstance(obj, dict):
        return None, "JSON root is not an object"

    labels: Dict[str, str] = {}
    for axis in AXES:
        if axis not in obj:
            return None, f"missing axis: {axis}"
        v = obj[axis]
        if not isinstance(v, str) or v not in VALID_LABELS:
            return None, f"invalid value for {axis}: {v!r}"
        labels[axis] = v
    return labels, None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _Backend:
    """Subclasses must implement ``call(system, user, temperature, max_tokens)``
    and return ``(text, usage_dict, latency_ms)``."""

    async def call(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], int]:
        raise NotImplementedError


class _AnthropicBackend(_Backend):
    def __init__(self, cfg: JudgeConfig, timeout_seconds: float) -> None:
        from anthropic import AsyncAnthropic

        kwargs: Dict[str, Any] = {"api_key": cfg.api_key, "timeout": float(timeout_seconds)}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self.client = AsyncAnthropic(**kwargs)
        self.model = cfg.model

    async def call(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], int]:
        t0 = time.monotonic()
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = ""
        for block in resp.content:
            t = getattr(block, "text", None)
            if t:
                text += t
        usage_obj = getattr(resp, "usage", None)
        usage: Dict[str, Any] = {}
        if usage_obj is not None:
            new_input = int(getattr(usage_obj, "input_tokens", 0) or 0)
            cache_creation = int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
            cache_read = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
            out = int(getattr(usage_obj, "output_tokens", 0) or 0)
            total_input = new_input + cache_creation + cache_read
            usage = {
                "input_tokens": total_input,
                "output_tokens": out,
                "total_tokens": total_input + out,
                "input_tokens_breakdown": {
                    "new": new_input,
                    "cache_creation": cache_creation,
                    "cache_read": cache_read,
                },
            }
        return text, usage, latency_ms


class _OpenAIBackend(_Backend):
    def __init__(self, cfg: JudgeConfig, timeout_seconds: float) -> None:
        import httpx
        from openai import AsyncOpenAI

        # trust_env=False: ignore environment proxies (e.g. a leftover
        # all_proxy=socks4://... that httpx can't parse and that isn't needed —
        # CometAPI is reachable directly). max_retries: SDK backoff on 429/5xx.
        http_client = httpx.AsyncClient(trust_env=False, timeout=float(timeout_seconds))
        kwargs: Dict[str, Any] = {
            "api_key": cfg.api_key,
            "max_retries": 8,
            "http_client": http_client,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = cfg.model
        self.extra_body = cfg.extra_body or None

    async def call(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], int]:
        t0 = time.monotonic()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Per-judge passthrough, e.g. thinking/reasoning disable for
        # deepseek/qwen ({"enable_thinking": false} / {"thinking": {...}}).
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = await self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)
        content = resp.choices[0].message.content or ""
        usage_obj = getattr(resp, "usage", None)
        usage: Dict[str, Any] = {}
        if usage_obj is not None:
            pt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            tt = getattr(usage_obj, "total_tokens", None)
            usage = {
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": int(tt) if tt is not None else pt + ct,
            }
        return content, usage, latency_ms


class _GeminiBackend(_Backend):
    def __init__(self, cfg: JudgeConfig, timeout_seconds: float) -> None:
        from google import genai

        http_options: Dict[str, Any] = {"api_version": "v1beta"}
        if cfg.base_url:
            http_options["base_url"] = cfg.base_url
        self.client = genai.Client(http_options=http_options, api_key=cfg.api_key)
        self.model = cfg.model

    async def call(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], int]:
        from google.genai import types as gtypes

        # Gemini 2.5 is a thinking model: hidden reasoning tokens are charged
        # against max_output_tokens before any visible JSON is emitted. With a
        # tight budget (~200) the thinking eats everything and the visible
        # output is empty / truncated to a preamble. For a structured JSON
        # judging task we don't need reasoning, so disable it explicitly.
        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
        )
        t0 = time.monotonic()
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = getattr(resp, "text", None) or ""
        if not text:
            # Convenience `.text` can be None when the SDK returns the answer
            # as ``candidates[0].content.parts[*].text`` blocks (happens with
            # response_mime_type and some shaped outputs). Walk parts manually.
            candidates = getattr(resp, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    pt = getattr(part, "text", None)
                    if isinstance(pt, str) and pt:
                        text += pt
                if text:
                    break
        usage: Dict[str, Any] = {}
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            pt = int(getattr(u, "prompt_token_count", 0) or 0)
            ct = int(getattr(u, "candidates_token_count", 0) or 0)
            tt = getattr(u, "total_token_count", None)
            usage = {
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": int(tt) if tt is not None else pt + ct,
            }
        return text, usage, latency_ms


def _make_backend(cfg: JudgeConfig, timeout_seconds: float) -> _Backend:
    if cfg.backend == "anthropic":
        return _AnthropicBackend(cfg, timeout_seconds)
    if cfg.backend == "openai":
        return _OpenAIBackend(cfg, timeout_seconds)
    if cfg.backend == "gemini":
        return _GeminiBackend(cfg, timeout_seconds)
    raise ValueError(f"Unknown backend: {cfg.backend}")


# ---------------------------------------------------------------------------
# Judge client wrapping a backend with retry + parsing
# ---------------------------------------------------------------------------

class JudgeClient:
    """One-retry recovery on parse failure (no retry on API errors)."""

    def __init__(
        self,
        cfg: JudgeConfig,
        system_prompt: str,
        user_template: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.backend = _make_backend(cfg, timeout_seconds)

    def _user_message(self, pair: Dict[str, Any], recovery_note: Optional[str] = None) -> str:
        base = self.user_template.format(
            query=pair["query"],
            context_A=pair["context_A"],
            context_B=pair["context_B"],
        )
        if recovery_note:
            base = (
                base
                + "\n\n[ВАЖНО] Твой предыдущий ответ был отклонён парсером. "
                + f"Причина: {recovery_note}. "
                + "Ответь ровно одним JSON-объектом со строго тремя ключами "
                + "(relevance, cleanliness, sufficiency), значения только \"A\", \"B\" или \"equal\". "
                + "Без какого-либо текста до или после JSON."
            )
        return base

    async def judge_pair(self, pair: Dict[str, Any]) -> JudgeDecision:
        user_msg = self._user_message(pair)
        try:
            raw, usage, latency = await self.backend.call(
                self.system_prompt, user_msg, self.temperature, self.max_tokens,
            )
        except Exception as e:
            return JudgeDecision(
                labels={}, raw="", status="failed",
                attempts=1, error=f"api_error: {type(e).__name__}: {e}",
                usage={}, latency_ms=0,
            )
        labels, err = parse_judge_response(raw)
        if labels is not None:
            return JudgeDecision(
                labels=labels, raw=raw, status="ok",
                attempts=1, error=None, usage=usage, latency_ms=latency,
            )

        # Retry once with diagnostic note.
        retry_msg = self._user_message(pair, recovery_note=err)
        try:
            raw2, usage2, latency2 = await self.backend.call(
                self.system_prompt, retry_msg, self.temperature, self.max_tokens,
            )
        except Exception as e:
            return JudgeDecision(
                labels={}, raw=raw, status="failed",
                attempts=2, error=f"retry_api_error: {type(e).__name__}: {e}",
                usage=usage, latency_ms=latency,
            )
        labels2, err2 = parse_judge_response(raw2)
        if labels2 is not None:
            return JudgeDecision(
                labels=labels2, raw=raw2, status="ok",
                attempts=2, error=None, usage=usage2,
                latency_ms=latency + latency2,
            )
        return JudgeDecision(
            labels={}, raw=raw2 or raw, status="failed",
            attempts=2, error=f"parse_failed: {err2}",
            usage=usage2 or usage,
            latency_ms=latency + latency2,
        )


# ---------------------------------------------------------------------------
# Concurrent batch runner
# ---------------------------------------------------------------------------

async def run_judge_on_pairs(
    judge_cfg: JudgeConfig,
    system_prompt: str,
    user_template: str,
    pairs: List[Dict[str, Any]],
    out_path: Path,
    *,
    concurrency: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 200,
    timeout_seconds: float = 120.0,
    log_every: int = 10,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge = JudgeClient(
        judge_cfg, system_prompt, user_template,
        temperature=temperature, max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters = {"ok": 0, "failed": 0, "retry": 0}
    t_start = time.monotonic()

    async def _process(pair: Dict[str, Any]) -> None:
        async with sem:
            decision = await judge.judge_pair(pair)
            rec = {
                "pair_id": pair["pair_id"],
                "judge": judge_cfg.name,
                "labels": decision.labels,
                "status": decision.status,
                "attempts": decision.attempts,
                "error": decision.error,
                "raw": decision.raw,
                "usage": decision.usage,
                "latency_ms": decision.latency_ms,
            }
            async with write_lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counters[decision.status] = counters.get(decision.status, 0) + 1
                if decision.attempts > 1:
                    counters["retry"] += 1
                done = counters["ok"] + counters["failed"]
                if done % log_every == 0 or done == len(pairs):
                    elapsed = time.monotonic() - t_start
                    rate = done / max(elapsed, 1e-6)
                    eta = (len(pairs) - done) / max(rate, 1e-6)
                    print(
                        f"[{judge_cfg.name}] {done}/{len(pairs)} "
                        f"ok={counters['ok']} failed={counters['failed']} "
                        f"retry={counters['retry']} "
                        f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                        flush=True,
                    )

    tasks = [_process(p) for p in pairs]
    await asyncio.gather(*tasks)
    return {
        "judge": judge_cfg.name,
        "total": len(pairs),
        "ok": counters["ok"],
        "failed": counters["failed"],
        "retry": counters["retry"],
        "elapsed_s": round(time.monotonic() - t_start, 1),
    }

"""TRT-LLM / vLLM API interaction with best-of-N IPA generation."""

from __future__ import annotations

import re
import asyncio
import logging
from dataclasses import dataclass
from collections import Counter
from typing import Sequence

import httpx

from .prompt import build_messages

logger = logging.getLogger(__name__)

# Strip /slashes/ wrapper and any trailing whitespace/punctuation the model adds
_IPA_PATTERN = re.compile(r"^/(.+)/$")


@dataclass(frozen=True, slots=True)
class IPAResult:
    name: str
    ipa: str                    # consensus transcription (empty if no consensus)
    confidence: float           # proportion of N that agreed (0.0–1.0)
    candidates: tuple[str, ...]  # all N raw candidates (for debugging/analysis)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    api_base: str = "http://localhost:8355/v1"
    model: str = "nvidia/Qwen3-235B-A22B-FP4"
    n: int = 10                  # best-of-N
    temperature: float = 0.6     # enough variation to expose uncertainty
    max_tokens: int = 128        # IPA transcriptions are short
    concurrent_requests: int = 32  # max parallel API calls in flight
    disable_thinking: bool = True  # suppress Qwen3 <think> blocks


def _normalize_ipa(raw: str) -> str:
    """Normalize a raw model output to a clean IPA string."""
    text = raw.strip()
    # Handle thinking mode: strip <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract content within /slashes/ if present
    m = _IPA_PATTERN.match(text)
    if m:
        return m.group(1).strip()
    # Model may have omitted slashes — take first line, strip slashes if partial
    first_line = text.split("\n")[0].strip()
    return first_line.strip("/").strip()


def _select_consensus(candidates: list[str]) -> tuple[str, float]:
    """Select best transcription by plurality vote.

    Returns (best_ipa, confidence) where confidence = count/total.
    """
    if not candidates:
        return "", 0.0
    normalized = [_normalize_ipa(c) for c in candidates]
    # Filter out empties
    normalized = [c for c in normalized if c]
    if not normalized:
        return "", 0.0
    counts = Counter(normalized)
    best, count = counts.most_common(1)[0]
    return best, count / len(candidates)


def _build_payload(messages: list[dict], config: GenerationConfig) -> dict:
    """Build a single n=1 API request payload."""
    payload: dict = {
        "model": config.model,
        "messages": messages,
        "n": 1,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


async def _single_call(
    client: httpx.AsyncClient,
    payload: dict,
    api_url: str,
) -> str | None:
    """Make a single n=1 API call, return content or None on failure."""
    try:
        resp = await client.post(api_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.debug("Single call failed: %s", e)
        return None


async def _generate_one(
    client: httpx.AsyncClient,
    name: str,
    config: GenerationConfig,
    semaphore: asyncio.Semaphore,
) -> IPAResult:
    """Generate N IPA candidates for a single name via N parallel n=1 calls."""
    messages = build_messages(name)
    payload = _build_payload(messages, config)
    api_url = f"{config.api_base}/chat/completions"

    # Fire N independent n=1 requests, each gated by the shared semaphore
    async def _call():
        async with semaphore:
            return await _single_call(client, payload, api_url)

    raw_results = await asyncio.gather(*[_call() for _ in range(config.n)])
    candidates = [r for r in raw_results if r is not None]

    if not candidates:
        logger.warning("All %d candidates failed for %r", config.n, name)
        return IPAResult(name=name, ipa="", confidence=0.0, candidates=())

    ipa, confidence = _select_consensus(candidates)
    return IPAResult(
        name=name,
        ipa=ipa,
        confidence=confidence,
        candidates=tuple(candidates),
    )


async def generate_batch_async(
    names: Sequence[str],
    config: GenerationConfig | None = None,
    progress_callback: callable | None = None,
) -> list[IPAResult]:
    """Generate IPA transcriptions for a batch of names.

    Sends N independent n=1 requests per name (instead of one n=N request)
    for compatibility with TRT-LLM's batch size constraints. All requests
    share a concurrency semaphore to keep the server saturated without
    overwhelming it.
    """
    if config is None:
        config = GenerationConfig()

    semaphore = asyncio.Semaphore(config.concurrent_requests)
    completed = 0

    async def _process(idx: int, name: str) -> IPAResult:
        nonlocal completed
        result = await _generate_one(client, name, config, semaphore)
        completed += 1
        if progress_callback and completed % 100 == 0:
            progress_callback(completed, len(names))
        return result

    async with httpx.AsyncClient() as client:
        tasks = [_process(i, name) for i, name in enumerate(names)]
        results = await asyncio.gather(*tasks)

    return list(results)


def generate_batch(
    names: Sequence[str],
    config: GenerationConfig | None = None,
    progress_callback: callable | None = None,
) -> list[IPAResult]:
    """Synchronous wrapper around generate_batch_async."""
    return asyncio.run(generate_batch_async(names, config, progress_callback))

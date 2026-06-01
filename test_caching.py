#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pydantic"]
# ///
"""
Integration tests for anthropic-pipe prompt-cache token accounting.

These hit the real API (OpenRouter by default), prime a cache on one call,
then re-send the shared prefix on a second call and assert that the pipe maps
Anthropic's split usage onto OpenAI semantics correctly.

The bug this guards against: Anthropic reports `input_tokens` as the UNCACHED
remainder only. `cache_read_input_tokens` and `cache_creation_input_tokens`
are reported separately and are NOT included in `input_tokens`. OpenAI
convention is the inverse: `prompt_tokens` is the grand total and
`cached_tokens` is a subset of it. The pipe must sum the input buckets so that
`cached_tokens <= prompt_tokens` always holds.

Usage:
    source ~/.tokens/openrouter-api   # or otherwise export OPENROUTER_API_KEY
    OPENROUTER_API_KEY=$OPENROUTER_API_KEY uv run --script test_caching.py

Env overrides:
    CACHE_TEST_MODEL   default "anthropic/claude-sonnet-4.6"
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from anthropic_via_openrouter import Pipe

MODEL = os.environ.get("CACHE_TEST_MODEL", "anthropic/claude-sonnet-4.6")

# A prefix large enough to exceed Anthropic's minimum cacheable size
# (1024 tokens for most models). ~400 repetitions is comfortably over.
_BIG_PREFIX = (
    "You are a meticulous research assistant. Follow instructions exactly "
    "and answer with extreme brevity. " * 400
).strip()


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


async def _run(pipe: Pipe, messages: list[dict]) -> dict:
    """Call the pipe non-streaming and return its OAI usage dict."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16,
        "stream": False,
    }
    res = await pipe.pipe(body)
    if isinstance(res, str):
        raise AssertionError(f"pipe returned an error string: {res[:300]}")
    usage = res.get("usage")
    assert usage is not None, f"no usage in response: {res}"
    return usage


def _assert_invariants(usage: dict, *, expect_read: bool = False, expect_write: bool = False):
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    total = usage["total_tokens"]
    details = usage.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0)
    cache_write = details.get("cache_write_tokens", 0)

    # Core OpenAI semantics
    assert total == prompt + completion, f"total {total} != prompt {prompt} + completion {completion}"
    assert cached <= prompt, f"cached_tokens {cached} exceeds prompt_tokens {prompt} (the bug!)"
    assert cache_write <= prompt, f"cache_write_tokens {cache_write} exceeds prompt_tokens {prompt}"
    # cached + write are both subsets of prompt; together they cannot exceed it
    assert cached + cache_write <= prompt, (
        f"cached {cached} + write {cache_write} exceeds prompt_tokens {prompt}"
    )

    if expect_read:
        assert cached > 0, f"expected a cache read but cached_tokens={cached}"
    if expect_write:
        assert cache_write > 0, f"expected a cache write but cache_write_tokens={cache_write}"

    return prompt, cached, cache_write


async def test_cache_write_then_read():
    print(f"── Integration: cache write → read accounting ({MODEL}) ──")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  SKIP: set OPENROUTER_API_KEY to run")
        return

    pipe = Pipe()
    pipe.valves.API_KEY = key

    # The pipe places its single APC cache_control breakpoint on the LAST
    # message block, so the cached prefix is everything up to and including the
    # final user turn. Both calls share the same large opening user turn.
    shared_prefix = [_user(_BIG_PREFIX + "\n\nReply with the single word: ready.")]

    # Call 1: write the cache (breakpoint sits at the end of the big turn).
    usage_write = await _run(pipe, shared_prefix)
    pw, cr, cw = _assert_invariants(usage_write, expect_write=True)
    print(f"  WRITE: prompt={pw} cached={cr} cache_write={cw} cost={usage_write.get('cost')}")

    # Brief pause so the cache entry is registered before the read.
    await asyncio.sleep(3)

    # Call 2: replay the shared prefix, then append assistant + a new user turn.
    # The breakpoint moves to the new final turn; the shared prefix is re-read
    # from cache.
    followup = shared_prefix + [
        _assistant("ready"),
        _user("Now reply with the single word: done."),
    ]
    usage_read = await _run(pipe, followup)
    pr, crr, cwr = _assert_invariants(usage_read, expect_read=True)
    print(f"  READ:  prompt={pr} cached={crr} cache_write={cwr} cost={usage_read.get('cost')}")

    # The read should recover roughly the prefix we wrote.
    assert crr >= cw * 0.8, (
        f"cache_read {crr} much smaller than prior cache_write {cw}; cache likely missed"
    )
    print("  PASS")
    print()


async def test_no_cache_activity_omits_details():
    """A tiny prompt below the cache threshold should not report cache reads."""
    print("── Integration: tiny prompt has sane usage (no false cache hit) ──")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  SKIP: set OPENROUTER_API_KEY to run")
        return

    pipe = Pipe()
    pipe.valves.API_KEY = key

    usage = await _run(pipe, [_user("Reply with the single word: hi.")])
    prompt, cached, cache_write = _assert_invariants(usage)
    assert cached == 0, f"unexpected cache read on a fresh tiny prompt: {cached}"
    print(f"  prompt={prompt} cached={cached} cache_write={cache_write}")
    print("  PASS")
    print()


if __name__ == "__main__":
    asyncio.run(test_cache_write_then_read())
    asyncio.run(test_no_cache_activity_omits_details())
    print("All caching tests passed!")

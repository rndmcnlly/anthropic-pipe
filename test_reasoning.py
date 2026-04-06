#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pydantic"]
# ///
"""
Quick smoke tests for anthropic-pipe reasoning/thinking support.
Tests request construction (unit) and one live API call (integration).

Usage:
    uv run --script test_reasoning.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Import the pipe module from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from anthropic_via_openrouter import (
    Pipe,
    _model_name,
    _supports_thinking,
    _uses_adaptive_thinking,
    _supports_effort,
    _supports_max_effort,
    _max_output_for_model,
    _EFFORT_RATIOS,
)


# ── Helper: extract the Anthropic request dict without actually calling the API ──

class RequestCapture(Pipe):
    """Subclass that captures the constructed request instead of sending it."""

    captured_req: dict | None = None

    def _stream(self, body: dict):
        self.captured_req = body
        async def noop():
            yield ""
        return noop()

    async def _complete(self, body: dict):
        self.captured_req = body
        return "(captured)"


def build_req(body: dict, model: str = "anthropic/claude-sonnet-4-6") -> dict:
    """Build the Anthropic request dict for a given OWUI body."""
    pipe = RequestCapture()
    pipe.valves.API_KEY = "test-key"
    body = {**body, "model": model, "messages": [{"role": "user", "content": "test"}], "stream": False}
    asyncio.run(pipe.pipe(body))
    return pipe.captured_req


# ── Unit tests: model detection helpers ──

def test_model_helpers():
    print("── Model detection helpers ──")

    # Thinking support
    assert _supports_thinking("anthropic/claude-sonnet-4-6")
    assert _supports_thinking("anthropic/claude-opus-4-6")
    assert _supports_thinking("anthropic/claude-sonnet-4-5")
    assert _supports_thinking("anthropic/claude-opus-4-5")
    assert _supports_thinking("anthropic/claude-3-7-sonnet")
    assert _supports_thinking("anthropic/claude-haiku-4-5")
    assert not _supports_thinking("anthropic/claude-haiku-3")
    print("  _supports_thinking: PASS")

    # Adaptive thinking (4.6 only)
    assert _uses_adaptive_thinking("anthropic/claude-sonnet-4-6")
    assert _uses_adaptive_thinking("anthropic/claude-opus-4-6")
    assert not _uses_adaptive_thinking("anthropic/claude-sonnet-4-5")
    assert not _uses_adaptive_thinking("anthropic/claude-opus-4-5")
    assert not _uses_adaptive_thinking("anthropic/claude-3-7-sonnet")
    print("  _uses_adaptive_thinking: PASS")

    # Effort support
    assert _supports_effort("anthropic/claude-opus-4-6")
    assert _supports_effort("anthropic/claude-sonnet-4-6")
    assert _supports_effort("anthropic/claude-opus-4-5")
    assert not _supports_effort("anthropic/claude-sonnet-4-5")
    assert not _supports_effort("anthropic/claude-3-7-sonnet")
    print("  _supports_effort: PASS")

    # Max effort (4.6 only)
    assert _supports_max_effort("anthropic/claude-opus-4-6")
    assert _supports_max_effort("anthropic/claude-sonnet-4-6")
    assert not _supports_max_effort("anthropic/claude-opus-4-5")
    print("  _supports_max_effort: PASS")

    # Max output tokens
    assert _max_output_for_model("anthropic/claude-opus-4-6") == 128_000
    assert _max_output_for_model("anthropic/claude-sonnet-4-6") == 64_000
    assert _max_output_for_model("anthropic/claude-sonnet-4-5") == 64_000
    print("  _max_output_for_model: PASS")

    print()


# ── Unit tests: request construction ──

def test_no_reasoning():
    print("── No reasoning specified ──")
    req = build_req({})
    assert "thinking" not in req
    assert "output_config" not in req
    print(f"  No thinking, no output_config: PASS")
    print()


def test_adaptive_thinking_via_reasoning_effort():
    print("── Claude 4.6 + reasoning_effort='medium' → adaptive thinking + effort ──")
    req = build_req({"reasoning_effort": "medium"})
    assert req["thinking"] == {"type": "adaptive"}, f"got {req.get('thinking')}"
    assert req["output_config"] == {"effort": "medium"}, f"got {req.get('output_config')}"
    assert req["temperature"] == 1.0
    print(f"  thinking: {req['thinking']}")
    print(f"  output_config: {req['output_config']}")
    print(f"  temperature: {req['temperature']}")
    print("  PASS")
    print()


def test_adaptive_thinking_via_reasoning_object():
    print("── Claude 4.6 + reasoning={{enabled: true}} → adaptive thinking ──")
    req = build_req({"reasoning": {"enabled": True}})
    assert req["thinking"] == {"type": "adaptive"}, f"got {req.get('thinking')}"
    print(f"  thinking: {req['thinking']}")
    print("  PASS")
    print()


def test_adaptive_with_explicit_budget():
    print("── Claude 4.6 + reasoning={{max_tokens: 8000}} → budget-based ──")
    req = build_req({"reasoning": {"max_tokens": 8000}})
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 8000}, f"got {req.get('thinking')}"
    print(f"  thinking: {req['thinking']}")
    print("  PASS")
    print()


def test_effort_high_with_reasoning_object():
    print("── Claude 4.6 + reasoning={{effort: 'high'}} → adaptive + effort=high ──")
    req = build_req({"reasoning": {"effort": "high"}})
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "high"}
    print(f"  thinking: {req['thinking']}")
    print(f"  output_config: {req['output_config']}")
    print("  PASS")
    print()


def test_max_effort_4_6():
    print("── Claude Opus 4.6 + reasoning_effort='max' → adaptive + effort=max ──")
    req = build_req({"reasoning_effort": "max"}, model="anthropic/claude-opus-4-6")
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "max"}
    print(f"  thinking: {req['thinking']}")
    print(f"  output_config: {req['output_config']}")
    print("  PASS")
    print()


def test_verbosity_standalone():
    print("── Claude 4.6 + verbosity='low' (no reasoning) → effort only ──")
    req = build_req({"verbosity": "low"})
    assert "thinking" not in req, f"unexpected thinking: {req.get('thinking')}"
    assert req["output_config"] == {"effort": "low"}
    print(f"  output_config: {req['output_config']}")
    print("  PASS")
    print()


def test_xhigh_effort_mapping():
    print("── Claude 4.6 + reasoning_effort='xhigh' → adaptive + effort=max ──")
    req = build_req({"reasoning_effort": "xhigh"})
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "max"}
    print(f"  output_config: {req['output_config']}")
    print("  PASS")
    print()


def test_minimal_effort_mapping():
    print("── Claude 4.6 + reasoning_effort='minimal' → adaptive + effort=low ──")
    req = build_req({"reasoning_effort": "minimal"})
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "low"}
    print(f"  output_config: {req['output_config']}")
    print("  PASS")
    print()


def test_pre46_budget_proportional():
    print("── Claude Sonnet 4.5 + reasoning_effort='medium' → budget-based (50% of 64k) ──")
    req = build_req({"reasoning_effort": "medium"}, model="anthropic/claude-sonnet-4-5")
    expected_budget = max(1024, min(int(64_000 * 0.50), 128_000))
    assert req["thinking"] == {"type": "enabled", "budget_tokens": expected_budget}
    assert "output_config" not in req  # Sonnet 4.5 doesn't support effort
    print(f"  thinking: {req['thinking']} (budget={expected_budget})")
    print("  PASS")
    print()


def test_pre46_explicit_budget():
    print("── Claude Sonnet 4.5 + reasoning={{max_tokens: 5000}} → budget-based ──")
    req = build_req({"reasoning": {"max_tokens": 5000}}, model="anthropic/claude-sonnet-4-5")
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 5000}
    print(f"  thinking: {req['thinking']}")
    print("  PASS")
    print()


def test_pre46_raw_integer():
    print("── Claude Sonnet 4.5 + reasoning_effort='12000' → raw integer budget ──")
    req = build_req({"reasoning_effort": "12000"}, model="anthropic/claude-sonnet-4-5")
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 12000}
    print(f"  thinking: {req['thinking']}")
    print("  PASS")
    print()


def test_none_disables():
    print("── reasoning_effort='none' → no thinking, no effort ──")
    req = build_req({"reasoning_effort": "none"})
    assert "thinking" not in req
    assert "output_config" not in req
    print("  PASS")
    print()


def test_max_effort_fallback_on_45():
    print("── Claude Opus 4.5 + reasoning_effort='max' → budget + effort=high (max unsupported) ──")
    req = build_req({"reasoning_effort": "max"}, model="anthropic/claude-opus-4-5")
    # Opus 4.5: not adaptive, no max_tokens ratio for "max" since it's not in _EFFORT_RATIOS
    # "max" is not in _EFFORT_RATIOS, so falls to raw int parse, which fails → no budget
    # But Opus 4.5 supports effort, so output_config should be set
    assert req.get("output_config") == {"effort": "high"}  # max → high fallback for 4.5
    print(f"  output_config: {req.get('output_config')}")
    print(f"  thinking: {req.get('thinking', 'not set')}")
    print("  PASS")
    print()


# ── Integration test: live API call ──

async def test_live_api():
    print("── Live API: adaptive thinking on Sonnet 4.6 (streaming) ──")
    import os
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  SKIP: set OPENROUTER_API_KEY to run live test")
        return
    pipe = Pipe()
    pipe.valves.API_KEY = key

    body = {
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "What is 7 * 8? Be brief."}],
        "reasoning": {"effort": "low"},
        "stream": True,
    }

    result = await pipe.pipe(body)

    full_output = []
    thinking_seen = False
    text_seen = False
    async for chunk in result:
        if isinstance(chunk, str):
            if "<think>" in chunk:
                thinking_seen = True
            elif "</think>" in chunk:
                pass
            else:
                text_seen = True
            full_output.append(chunk)
        elif isinstance(chunk, dict):
            full_output.append(json.dumps(chunk, indent=2))

    output_text = "".join(str(c) for c in full_output)
    print(f"  Response length: {len(output_text)} chars")
    print(f"  Thinking block seen: {thinking_seen}")
    print(f"  Text content seen: {text_seen}")

    # With adaptive thinking + low effort, Claude may or may not think
    # but the response should have text content
    assert text_seen or len(output_text) > 0, "No output received"
    print(f"  Preview: {output_text[:200]}...")
    print("  PASS")
    print()


# ── Run all tests ──

if __name__ == "__main__":
    test_model_helpers()
    test_no_reasoning()
    test_adaptive_thinking_via_reasoning_effort()
    test_adaptive_thinking_via_reasoning_object()
    test_adaptive_with_explicit_budget()
    test_effort_high_with_reasoning_object()
    test_max_effort_4_6()
    test_verbosity_standalone()
    test_xhigh_effort_mapping()
    test_minimal_effort_mapping()
    test_pre46_budget_proportional()
    test_pre46_explicit_budget()
    test_pre46_raw_integer()
    test_none_disables()
    test_max_effort_fallback_on_45()

    print("=" * 60)
    print("All unit tests passed. Running live API test...")
    print("=" * 60)
    print()
    asyncio.run(test_live_api())

    print("All tests passed!")

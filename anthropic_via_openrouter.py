"""
title: Anthropic Pipe
author: Adam Smith
author_url: https://adamsmith.as
version: 5.1.0
license: MIT
description: >
  Native Anthropic Messages API pipe for Open WebUI with prompt caching.
  Defaults to OpenRouter but works with any Anthropic-compatible endpoint.
  Tool calls are translated to OpenAI format so OWUI's native middleware
  handles execution, UI rendering, and multi-turn tool loops.
  Supports extended thinking via reasoning_effort, OpenRouter-style reasoning
  object, and verbosity. Claude 4.6+ uses adaptive thinking; on Opus 4.7+
  adaptive is the *only* mode (manual budget_tokens is rejected with a 400).
  Older models use proportional budget_tokens allocation. Model capabilities
  are derived from a parsed (family, major, minor) version, so new releases
  (e.g. 4.7, 4.8, '-fast' speed variants, and '~...-latest' aliases) are
  handled without per-version edits.
"""
# Prompt caching strategy: automatic prompt caching (APC).
# A single cache_control breakpoint is placed on the last message block so
# Anthropic's APC logic handles the rest — caching grows automatically as
# the conversation extends, with no manual breakpoint management needed.

import json
import logging
import re
from typing import AsyncGenerator

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"

FALLBACK_MODELS = [
    {"id": "~anthropic/claude-opus-latest", "name": "Claude Opus Latest"},
    {"id": "~anthropic/claude-sonnet-latest", "name": "Claude Sonnet Latest"},
    {"id": "~anthropic/claude-haiku-latest", "name": "Claude Haiku Latest"},
    {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8"},
    {"id": "anthropic/claude-sonnet-4.6", "name": "Claude Sonnet 4.6"},
]

_DEFAULT_MAX_OUTPUT = 8_192

# Effort → proportion of max_tokens for budget_tokens calculation
# (used for non-adaptive models when effort is specified instead of explicit budget).
_EFFORT_RATIOS = {
    "xhigh":  0.95,
    "high":   0.80,
    "medium": 0.50,
    "low":    0.20,
    "minimal": 0.10,
}


def _model_name(model_id: str) -> str:
    """Strip provider prefix (incl. OpenRouter '~'), normalise dots→dashes,
    and drop the '-fast' speed suffix so capability matching is version-only."""
    name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    name = name.lstrip("~").replace(".", "-")
    if name.endswith("-fast"):
        name = name[: -len("-fast")]
    return name


def _parse_version(model_id: str) -> tuple[str, int | None, int | None]:
    """Parse a normalised model name into (family, major, minor).

    family is 'opus' | 'sonnet' | 'haiku' | '' (unknown).
    major/minor are None when absent.  '-latest' aliases carry no version,
    so they parse to (family, None, None) and are treated as newest-gen.

    Examples:
      claude-opus-4-8         → ('opus', 4, 8)
      claude-sonnet-4-6       → ('sonnet', 4, 6)
      claude-opus-4-1         → ('opus', 4, 1)
      claude-opus-4           → ('opus', 4, 0)
      claude-3-7-sonnet       → ('sonnet', 3, 7)
      claude-3-5-haiku        → ('haiku', 3, 5)
      claude-opus-latest      → ('opus', None, None)
    """
    name = _model_name(model_id)
    family = ""
    for f in ("opus", "sonnet", "haiku"):
        if f in name:
            family = f
            break

    # New-style: claude-<family>-<major>[-<minor>]
    m = re.search(rf"{family}-(\d+)(?:-(\d+))?", name) if family else None
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) is not None else 0
        return family, major, minor

    # Legacy-style: claude-<major>-<minor>-<family> (e.g. claude-3-7-sonnet)
    # or single-major (e.g. claude-3-haiku).
    m = re.search(r"claude-(\d+)(?:-(\d+))?", name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) is not None else 0
        return family, major, minor

    # '-latest' or otherwise unversioned: treat as newest generation.
    return family, None, None


def _is_newest_gen(model_id: str) -> bool:
    """True when version is unknown ('-latest') — treat as current frontier."""
    _, major, _ = _parse_version(model_id)
    return major is None


def _supports_thinking(model_id: str) -> bool:
    """Extended thinking (adaptive or budget): Claude 3.7+ and all 4.x."""
    family, major, minor = _parse_version(model_id)
    if not family:
        return True  # unknown anthropic model: assume capable
    if major is None:
        return True  # latest alias
    if major >= 4:
        return True
    if major == 3 and minor >= 7:
        return True  # 3.7 Sonnet (extended thinking introduced here)
    return False


def _uses_adaptive_thinking(model_id: str) -> bool:
    """True for 4.6+ models, where adaptive thinking is preferred/required.
    On Opus 4.7+ it is the *only* supported mode (budget_tokens → 400)."""
    family, major, minor = _parse_version(model_id)
    if major is None:
        return True  # latest alias → newest gen
    if major < 4:
        return False
    return minor >= 6


def _adaptive_only(model_id: str) -> bool:
    """True for models that reject manual budget_tokens (400 error):
    Opus 4.7 and later. Sonnet/Opus 4.6 still accept budget (deprecated)."""
    family, major, minor = _parse_version(model_id)
    if major is None:
        return True  # latest alias → assume newest-gen, adaptive-only
    if family == "opus" and major == 4 and minor >= 7:
        return True
    if major == 4 and minor >= 7:
        return True  # future sonnet/haiku 4.7+ assumed adaptive-only too
    return False


def _supports_effort(model_id: str) -> bool:
    """True for models supporting output_config.effort: Opus 4.5 and all 4.6+."""
    family, major, minor = _parse_version(model_id)
    if major is None:
        return True
    if major < 4:
        return False
    if minor >= 6:
        return True
    return family == "opus" and minor == 5  # Opus 4.5 supports effort


def _supports_max_effort(model_id: str) -> bool:
    """True for models supporting effort: 'max' — 4.6 and later."""
    family, major, minor = _parse_version(model_id)
    if major is None:
        return True
    return major >= 4 and minor >= 6


def _supports_xhigh_effort(model_id: str) -> bool:
    """True for models supporting effort: 'xhigh' — Opus 4.7 and later."""
    family, major, minor = _parse_version(model_id)
    if major is None:
        return True
    if family == "opus" and major == 4 and minor >= 7:
        return True
    return major == 4 and minor >= 7


def _max_output_for_model(model_id: str) -> int:
    """Return the max output token limit for a given model ID.

    Derived from (family, version) rather than a hardcoded table so new
    releases get a sane default. Known caps (Anthropic docs):
      Opus 4.6+:        128k    Opus 4.5 / 4.1 / 4.0: 32k–64k
      Sonnet 4.x:        64k    Haiku 4.5:            64k
      Claude 3.x Haiku: 4096
    """
    family, major, minor = _parse_version(model_id)

    # Newest-gen alias or unknown version: assume frontier limits.
    if major is None:
        return 128_000 if family == "opus" else 64_000

    if major >= 4:
        if family == "opus":
            if minor >= 6:
                return 128_000
            if minor >= 5:
                return 64_000
            if minor >= 1:
                return 32_000
            return 32_000  # Opus 4.0
        # sonnet / haiku 4.x
        return 64_000

    if major == 3:
        if family == "sonnet" and minor >= 7:
            return 64_000  # 3.7 Sonnet
        if family == "haiku":
            return 4_096
        return 4_096

    return _DEFAULT_MAX_OUTPUT


class Pipe:
    class Valves(BaseModel):
        API_KEY: str = Field(
            default="",
            description="API key: an OpenRouter key or a native Anthropic key.",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description=(
                "Base URL for the Anthropic-compatible API. "
                "Use https://api.anthropic.com for direct Anthropic access."
            ),
        )
        AUTH_TYPE: str = Field(
            default="bearer",
            description=(
                "'bearer' for OpenRouter (Authorization: Bearer <key>), "
                "'x-api-key' for native Anthropic (x-api-key: <key>)."
            ),
        )
        CACHE_TTL: str = Field(
            default="5m",
            description="Cache TTL: '5m' (1.25x write) or '1h' (2x write).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._models_cache: list[dict] | None = None

    def _base(self) -> str:
        return self.valves.API_BASE_URL.rstrip("/")

    def _headers(self) -> dict:
        auth = (
            {"x-api-key": self.valves.API_KEY}
            if self.valves.AUTH_TYPE == "x-api-key"
            else {"Authorization": f"Bearer {self.valves.API_KEY}"}
        )
        or_attribution = (
            {"HTTP-Referer": "https://openwebui.com/", "X-Title": "Open WebUI"}
            if "openrouter.ai" in self._base()
            else {}
        )
        return {**auth, **or_attribution, "anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Model list
    # ------------------------------------------------------------------
    def pipes(self) -> list[dict]:
        if self._models_cache is not None:
            return self._models_cache
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{self._base()}/models", headers=self._headers())
                resp.raise_for_status()
            EXCLUDED = (":free", ":nitro", ":floor", ":extended")
            # Match both 'anthropic/...' and OpenRouter's '~anthropic/...'
            # latest-alias variants (e.g. ~anthropic/claude-opus-latest).
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in resp.json().get("data", [])
                if m["id"].lstrip("~").startswith("anthropic/")
                and not any(m["id"].endswith(s) for s in EXCLUDED)
            ]
            models.sort(key=lambda m: m["id"])
            self._models_cache = models
            return models
        except Exception:
            return FALLBACK_MODELS

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def pipe(self, body: dict) -> AsyncGenerator | dict | str:
        if not self.valves.API_KEY:
            return "Error: API_KEY is not set. Ask your administrator to configure this pipe."

        model_id = body.get("model", "")
        # OWUI prefixes the model with the pipe function id, e.g.
        # "anthropic_via_openrouter.anthropic/claude-opus-4.8". Strip a single
        # leading "<function_id>." prefix, but be careful: version dots like the
        # "4.8" in the slug must NOT be split. We only strip when the remainder
        # after the first dot looks like a provider slug (".../...").
        if "." in model_id:
            head, tail = model_id.split(".", 1)
            if "/" in tail and "/" not in head:
                model_id = tail

        system_text, oai_messages = self._split_system(body.get("messages", []))
        messages = self._convert_messages(oai_messages)

        # APC: single cache_control on the last block of the last message
        if messages:
            blocks = messages[-1].get("content", [])
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": self.valves.CACHE_TTL}

        tools = self._convert_tools(body.get("tools", []))

        max_tokens = body.get("max_tokens") or _max_output_for_model(model_id)

        req: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": body.get("stream", False),
        }
        if system_text:
            req["system"] = [{"type": "text", "text": system_text}]
        if tools:
            req["tools"] = tools

        # -- Reasoning / thinking configuration --
        # Accept three input shapes from OWUI / callers:
        #   1. body["reasoning"]        — OpenRouter-style dict {effort, max_tokens, exclude, enabled}
        #   2. body["reasoning_effort"] — flat string ("low"/"medium"/"high"/"max" or raw int)
        #   3. Neither                  — no thinking requested
        reasoning_cfg = body.get("reasoning") or {}
        if isinstance(reasoning_cfg, str):
            reasoning_cfg = {"effort": reasoning_cfg}

        # Normalise effort from either source
        effort = (
            str(reasoning_cfg.get("effort", "")).lower().strip()
            or str(body.get("reasoning_effort", "")).lower().strip()
        )
        explicit_budget = reasoning_cfg.get("max_tokens")  # explicit token budget
        reasoning_enabled = reasoning_cfg.get("enabled", False)

        thinking_enabled = False
        if _supports_thinking(model_id):
            if _uses_adaptive_thinking(model_id):
                # Claude 4.6+: prefer adaptive thinking over budget_tokens.
                # Enable if: explicit budget given, effort specified, or enabled flag set.
                should_think = bool(explicit_budget or effort not in ("", "none") or reasoning_enabled)
                if should_think:
                    if explicit_budget and not _adaptive_only(model_id):
                        # 4.6 still accepts an explicit budget (deprecated).
                        req["thinking"] = {"type": "enabled", "budget_tokens": int(explicit_budget)}
                    else:
                        # Adaptive: model decides how much to think; effort controls depth
                        # via output_config.effort (handled below). On Opus 4.7+ this is
                        # the only accepted mode — an explicit budget would 400, so we
                        # silently fall back to adaptive here.
                        req["thinking"] = {"type": "adaptive"}
                    thinking_enabled = True
            else:
                # Pre-4.6 models: budget-based thinking.
                budget_tokens = None
                if explicit_budget:
                    budget_tokens = int(explicit_budget)
                elif effort in _EFFORT_RATIOS:
                    # Proportional: effort → percentage of max_tokens (like OpenRouter)
                    budget_tokens = max(1024, min(int(max_tokens * _EFFORT_RATIOS[effort]), 128_000))
                elif effort not in ("", "none"):
                    # Accept raw integer budget (OWUI passes strings)
                    try:
                        budget_tokens = int(effort)
                    except (ValueError, TypeError):
                        budget_tokens = None

                if budget_tokens:
                    req["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
                    thinking_enabled = True

        # -- Effort / verbosity → output_config.effort --
        # Separate from thinking: controls response thoroughness and token spend.
        # Accept from reasoning.effort, reasoning_effort, or standalone verbosity param.
        verbosity = str(body.get("verbosity", "")).lower().strip()
        eff_source = effort or verbosity
        if eff_source and eff_source != "none" and _supports_effort(model_id):
            # Map the requested effort to Anthropic's output_config.effort,
            # downgrading to the nearest level the target model supports.
            #   low < medium < high < xhigh < max   (xhigh: Opus 4.7+; max: 4.6+)
            if eff_source in ("low", "medium", "high"):
                eff_val = eff_source
            elif eff_source == "minimal":
                eff_val = "low"
            elif eff_source == "xhigh":
                eff_val = (
                    "xhigh" if _supports_xhigh_effort(model_id)
                    else "max" if _supports_max_effort(model_id)
                    else "high"
                )
            elif eff_source == "max":
                eff_val = "max" if _supports_max_effort(model_id) else "high"
            else:
                eff_val = "high"
            req["output_config"] = {"effort": eff_val}

        # -- Sampling constraints --
        if thinking_enabled:
            # Thinking requires temperature=1 and is incompatible with top_k/top_p.
            req["temperature"] = 1.0
            if "stop" in body:
                req["stop"] = body["stop"]
        elif _adaptive_only(model_id):
            # Opus 4.7+ reject temperature/top_p/top_k entirely (400). Forward
            # only stop sequences.
            if "stop" in body:
                req["stop"] = body["stop"]
        else:
            for key in ("temperature", "top_p", "top_k", "stop"):
                if key in body:
                    req[key] = body[key]

        if body.get("stream", False):
            return self._stream(req)
        else:
            return await self._complete(req)

    # ------------------------------------------------------------------
    # Tool spec conversion: OAI → Anthropic
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_tools(oai_tools: list[dict]) -> list[dict]:
        result = []
        for tool in oai_tools:
            fn = tool.get("function", tool)
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        return result

    # ------------------------------------------------------------------
    # Message format conversion: OpenAI → Anthropic
    # ------------------------------------------------------------------
    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts, other = [], []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block["text"])
                        elif isinstance(block, str):
                            system_parts.append(block)
                else:
                    system_parts.append(str(content))
            else:
                other.append(msg)
        return "\n\n".join(system_parts), other

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        for msg in messages:
            role = msg.get("role", "user")

            # OAI tool result messages → Anthropic tool_result blocks in a user turn
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if converted and converted[-1]["role"] == "user":
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
                continue

            # OAI assistant with tool_calls → Anthropic tool_use blocks
            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict] = []
                content = msg.get("content")
                if content:
                    blocks.extend(self._content_to_blocks(content))
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    raw = fn.get("arguments", "{}")
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        parsed = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": parsed,
                    })
                converted.append({"role": "assistant", "content": blocks})
                continue

            if role not in ("user", "assistant"):
                continue
            converted.append({
                "role": role,
                "content": self._content_to_blocks(msg.get("content", "")),
            })
        return converted

    @staticmethod
    def _content_to_blocks(content) -> list[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks: list[dict] = []
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
                continue
            t = item.get("type", "")
            if t == "text":
                blocks.append({"type": "text", "text": item["text"]})
            elif t == "image_url":
                url = item["image_url"].get("url", "")
                if url.startswith("data:"):
                    m = re.match(r"data:(image/[\w.+-]+);base64,(.+)", url, re.DOTALL)
                    if m:
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": m.group(1),
                                "data": m.group(2),
                            },
                        })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            elif t in ("tool_use", "tool_result"):
                blocks.append(item)
        return blocks or [{"type": "text", "text": ""}]

    # ------------------------------------------------------------------
    # Streaming — single pass, Anthropic SSE → OpenAI SSE
    # ------------------------------------------------------------------
    def _stream(self, body: dict) -> AsyncGenerator:
        pipe = self

        async def generator():
            import uuid, time as _time
            tool_blocks: dict = {}
            # Consistent envelope fields for all dict chunks in this stream.
            # OWUI adds these for plain-string yields but passes dicts through
            # as-is, so pre-formed chunks (tool calls, usage) need them too.
            stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            stream_model = body.get("model", "")
            stream_created = int(_time.time())
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST", f"{pipe._base()}/messages",
                    json={**body, "stream": True}, headers=pipe._headers(),
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield f"Error {resp.status_code}: {err.decode()}"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        for chunk in _translate_event(event, tool_blocks):
                            if isinstance(chunk, dict):
                                # Ensure required OAI envelope fields are present
                                chunk.setdefault("id", stream_id)
                                chunk.setdefault("model", stream_model)
                                chunk.setdefault("created", stream_created)
                            yield chunk

        return generator()

    # ------------------------------------------------------------------
    # Non-streaming — returns an OAI-shaped dict so OWUI can handle tool calls
    # ------------------------------------------------------------------
    async def _complete(self, body: dict) -> dict | str:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self._base()}/messages",
                json={**body, "stream": False}, headers=self._headers(),
            )

        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"

        data = resp.json()
        content = data.get("content", [])
        stop_reason = data.get("stop_reason", "end_turn")

        # Collect thinking blocks and text blocks separately
        thinking_parts = [b["thinking"] for b in content if b.get("type") == "thinking" and b.get("thinking")]
        text_parts = [b["text"] for b in content if b.get("type") == "text"]
        text = "\n".join(text_parts)

        # Prepend thinking in <think> tags (OWUI renders these natively)
        if thinking_parts:
            text = "<think>" + "\n".join(thinking_parts) + "</think>\n\n" + text

        # ── Build OAI usage from Anthropic usage ─────────────────
        # Anthropic reports input_tokens as the UNCACHED remainder only; the
        # cache_read / cache_creation counts are reported separately and are
        # NOT included in input_tokens. OpenAI semantics are the inverse:
        # prompt_tokens is the grand total and cached_tokens is a subset of it.
        # So we sum all input-side buckets to get prompt_tokens, keeping
        # cached_tokens as a proper subset (cache_read).
        anthropic_usage = data.get("usage", {})
        input_tokens = anthropic_usage.get("input_tokens", 0)
        completion_tokens = anthropic_usage.get("output_tokens", 0)
        cache_read = anthropic_usage.get("cache_read_input_tokens", 0)
        cache_write = anthropic_usage.get("cache_creation_input_tokens", 0)
        prompt_tokens = input_tokens + cache_read + cache_write
        oai_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cache_read or cache_write:
            oai_usage["prompt_tokens_details"] = {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            }
        cost = anthropic_usage.get("cost")
        if cost is not None:
            oai_usage["cost"] = cost

        # ── Build response envelope ──────────────────────────────
        import uuid, time as _time
        envelope = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(_time.time()),
            "model": body.get("model", ""),
            "usage": oai_usage,
        }

        # If the model wants to call tools, translate to OAI format
        # so OWUI's middleware can intercept and execute them.
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if stop_reason == "tool_use" and tool_uses:
            oai_tool_calls = [
                {
                    "index": i,
                    "id": tu["id"],
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for i, tu in enumerate(tool_uses)
            ]
            envelope["choices"] = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": oai_tool_calls,
                },
                "finish_reason": "tool_calls",
            }]
            return envelope

        if stop_reason == "max_tokens" and text:
            text += "\n\n---\n*[Response truncated — max_tokens limit reached]*"

        envelope["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text or f"(no text in response: {json.dumps(data)})",
            },
            "finish_reason": "length" if stop_reason == "max_tokens" else "stop",
        }]
        return envelope


# ======================================================================
# SSE translation: Anthropic events → OpenAI Chat Completion chunks
# ======================================================================

def _translate_event(event: dict, tool_blocks: dict) -> list:
    """
    Convert one Anthropic SSE event into zero or more items to yield.

    Yields either:
      - str: a plain text fragment (OWUI wraps it in a delta chunk automatically)
      - dict: a pre-formed SSE data payload (OWUI serialises it as `data: {...}`)

    Tool calls are accumulated in tool_blocks and emitted as OAI-format
    delta.tool_calls on message_delta so OWUI's middleware can execute them.

    The special key "_usage" in tool_blocks stores usage from message_start
    for providers (like direct Anthropic) that don't repeat it in message_delta.
    The special key "_thinking" tracks whether we're inside a thinking block.
    """
    etype = event.get("type")
    chunks: list = []

    if etype == "content_block_delta":
        delta = event.get("delta", {})
        dt = delta.get("type")
        if dt == "text_delta":
            chunks.append(delta.get("text", ""))
        elif dt == "thinking_delta":
            chunks.append(delta.get("thinking", ""))
        elif dt == "signature_delta":
            # Signature marks end of thinking block → close the <think> tag
            chunks.append("\n</think>\n\n")
            tool_blocks.pop("_thinking", None)
        elif dt == "input_json_delta":
            idx = event.get("index", 0)
            if idx in tool_blocks:
                tool_blocks[idx]["_json"] += delta.get("partial_json", "")

    elif etype == "content_block_start":
        block = event.get("content_block", {})
        idx = event.get("index", 0)
        if block.get("type") == "tool_use":
            tool_blocks[idx] = {"id": block.get("id", ""), "name": block.get("name", ""), "_json": ""}
        elif block.get("type") == "thinking":
            tool_blocks["_thinking"] = True
            chunks.append("<think>")

    elif etype == "content_block_stop":
        idx = event.get("index", 0)
        if idx in tool_blocks:
            tool_blocks[idx]["arguments"] = tool_blocks[idx].pop("_json", "{}")
        # Fallback close if signature_delta didn't fire (e.g. redacted thinking)
        if tool_blocks.pop("_thinking", False):
            chunks.append("\n</think>\n\n")

    elif etype == "message_delta":
        delta = event.get("delta", {})
        # Merge: message_start stashed input-side usage (input_tokens, cache_*),
        # message_delta carries output-side usage (output_tokens).  Using `or`
        # would discard whichever dict lost, so we merge explicitly.
        stashed = tool_blocks.pop("_usage", {})
        delta_usage = event.get("usage", {})
        usage = {**stashed, **delta_usage}
        stop_reason = delta.get("stop_reason")

        if stop_reason:
            is_tool_use = stop_reason == "tool_use"
            is_truncated = stop_reason == "max_tokens"

            if is_truncated:
                chunks.append("\n\n---\n*[Response truncated — max_tokens limit reached]*")

            finish_chunk: dict = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if is_tool_use else "stop"}],
            }

            if is_tool_use and tool_blocks:
                finish_chunk["choices"][0]["delta"]["tool_calls"] = [
                    {
                        "index": i,
                        "id": tb["id"],
                        "type": "function",
                        "function": {"name": tb["name"], "arguments": tb.get("arguments", "{}")},
                    }
                    for i, tb in enumerate(tool_blocks[k] for k in sorted(k for k in tool_blocks if isinstance(k, int)))
                ]
                tool_blocks.clear()

            chunks.append(finish_chunk)
            if usage:
                chunks.append(_build_usage(usage))

    elif etype == "message_start":
        u = event.get("message", {}).get("usage", {})
        if u.get("input_tokens"):
            tool_blocks["_usage"] = u

    return chunks


def _build_usage(usage: dict) -> dict:
    """Build a standalone usage chunk for OWUI's info display.

    Anthropic's input_tokens is the UNCACHED remainder only; cache_read and
    cache_creation are reported separately and are not included in it. OpenAI
    semantics require prompt_tokens to be the grand total with cached_tokens a
    subset, so we sum all input buckets here too (mirrors _complete()).
    """
    input_tokens = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    prompt = input_tokens + cache_read + cache_write
    cost = usage.get("cost")
    payload: dict = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
    }
    if cost is not None:
        payload["cost"] = cost
    return {"usage": payload}

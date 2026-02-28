"""
title: Anthropic Pipe
author: Adam Smith
author_url: https://adamsmith.as
version: 3.2.0
license: MIT
description: >
  Native Anthropic Messages API pipe for Open WebUI with prompt caching.
  Defaults to OpenRouter but works with any Anthropic-compatible endpoint.
  Tool calls are translated to OpenAI format so OWUI's native middleware
  handles execution, UI rendering, and multi-turn tool loops.
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
    {"id": "anthropic/claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-opus-4-5", "name": "Claude Opus 4.5"},
    {"id": "anthropic/claude-haiku-4-5", "name": "Claude Haiku 4.5"},
]

# Max output tokens per model family (from Anthropic docs).
# Checked against model_id which may have an "anthropic/" prefix.
_MAX_OUTPUT = {
    "claude-opus-4-6":   128_000,
    "claude-sonnet-4-6":  64_000,
    "claude-sonnet-4-5":  64_000,
    "claude-opus-4-5":    64_000,
    "claude-opus-4-1":    32_000,
    "claude-sonnet-4":    64_000,
    "claude-opus-4":      32_000,
    "claude-haiku-4-5":   64_000,
    "claude-haiku-3":      4_096,
}
_DEFAULT_MAX_OUTPUT = 8_192


def _max_output_for_model(model_id: str) -> int:
    """Return the max output token limit for a given model ID."""
    name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    if name in _MAX_OUTPUT:
        return _MAX_OUTPUT[name]
    for key, limit in _MAX_OUTPUT.items():
        if name.startswith(key):
            return limit
    return _DEFAULT_MAX_OUTPUT


class Pipe:
    class Valves(BaseModel):
        API_KEY: str = Field(
            default="",
            description="API key: an OpenRouter key or a native Anthropic key.",
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
        return {**auth, "anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json"}

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
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in resp.json().get("data", [])
                if m["id"].startswith("anthropic/")
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
        if "." in model_id:
            model_id = model_id.split(".", 1)[1]

        system_text, oai_messages = self._split_system(body.get("messages", []))
        messages = self._convert_messages(oai_messages)

        # APC: single cache_control on the last block of the last message
        if messages:
            blocks = messages[-1].get("content", [])
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": self.valves.CACHE_TTL}

        tools = self._convert_tools(body.get("tools", []))

        req: dict = {
            "model": model_id,
            "max_tokens": body.get("max_tokens") or _max_output_for_model(model_id),
            "messages": messages,
            "stream": body.get("stream", False),
        }
        if system_text:
            req["system"] = [{"type": "text", "text": system_text}]
        if tools:
            req["tools"] = tools
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
            tool_blocks: dict = {}
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

        text = "\n".join(b["text"] for b in content if b.get("type") == "text")

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
            return {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": oai_tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }],
            }

        if stop_reason == "max_tokens" and text:
            text += "\n\n---\n*[Response truncated — max_tokens limit reached]*"

        return text or f"(no text in response: {json.dumps(data)})"


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
    """
    etype = event.get("type")
    chunks: list = []

    if etype == "content_block_delta":
        delta = event.get("delta", {})
        dt = delta.get("type")
        if dt == "text_delta":
            chunks.append(delta.get("text", ""))
        elif dt == "input_json_delta":
            idx = event.get("index", 0)
            if idx in tool_blocks:
                tool_blocks[idx]["_json"] += delta.get("partial_json", "")

    elif etype == "content_block_start":
        block = event.get("content_block", {})
        idx = event.get("index", 0)
        if block.get("type") == "tool_use":
            tool_blocks[idx] = {"id": block.get("id", ""), "name": block.get("name", ""), "_json": ""}

    elif etype == "content_block_stop":
        idx = event.get("index", 0)
        if idx in tool_blocks:
            tool_blocks[idx]["arguments"] = tool_blocks[idx].pop("_json", "{}")

    elif etype == "message_delta":
        delta = event.get("delta", {})
        usage = event.get("usage", {}) or tool_blocks.pop("_usage", {})
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
                    for i, tb in enumerate(tool_blocks[k] for k in sorted(tool_blocks))
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
    """Build a standalone usage chunk for OWUI's info display."""
    prompt = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
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

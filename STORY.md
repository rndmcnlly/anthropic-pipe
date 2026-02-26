# How This Pipe Came To Be

A blow-by-blow account of building a native Anthropic pipe for Open WebUI,
from "this is expensive" to working tool calls with collapsible UI.

---

## The Problem

Anthropic models through OpenRouter are expensive because OR's standard
OpenAI-compatible endpoint doesn't inject Anthropic `cache_control`
breakpoints. Every turn re-sends the full system prompt at full price.
People address this with "pipes" in Open WebUI, but the ones floating
around are outdated and miss modern Anthropic features.

## Discovery: OR Has a Native Anthropic Endpoint

The key insight: OpenRouter exposes `POST /api/v1/messages` — a native
Anthropic Messages API endpoint, authenticated with your OR key via
`Authorization: Bearer`. Same billing, same usage dashboard, but you
speak Anthropic's protocol directly and can inject `cache_control`.

## Prototype: Does Caching Actually Work?

We wrote `test_cache.py` — a `uv run --script` one-filer that sends two
identical requests with `cache_control: {"type": "ephemeral"}` on a fat
system prompt.

**First surprise:** usage data comes back in the `message_delta` SSE event,
not `message_start` like Anthropic's own docs suggest. An OR quirk we only
found by dumping raw events.

**Result:** Request 1 wrote 7,214 tokens to cache ($0.00927). Request 2
read them back ($0.00094). ~10x cheaper. Caching works through OR.

## v1: The Basic Pipe

Built an async pipe that:
- Translates OpenAI messages → Anthropic format
- Injects `cache_control` on the system prompt, first user turn, and
  second-to-last user turn (3 of 4 allowed breakpoints)
- Streams via `httpx.AsyncClient`
- Dynamically fetches Claude models from OR's `/api/v1/models`

Deployed via the OWUI admin API (discovered the create/update/toggle
endpoints by probing the OpenAPI spec).

## The Cost Display Bug

OWUI shows a little info button with token counts and cost. Ours wasn't
showing up. Investigation revealed OWUI's streaming parser only picks up
`usage` from a chunk that has **no `choices` key**. We were embedding
usage inside the same chunk as the finish reason. Fix: emit two separate
chunks — one with `choices` (finish reason), one with just `usage`.

## Tool Calling: The First Attempt

User tried asking Claude to search the web via Tavily. Claude politely
said "I can't browse the web." The pipe was silently dropping the `tools`
field from the request body. We added OAI→Anthropic tool spec conversion
and verified raw tool calling worked with `test_tools.py`.

## Tool Calling: The Wrong Architecture (v2)

We added `__tools__` to the pipe's signature (OWUI injects resolved tool
callables there), built a full tool dispatch loop inside the pipe: detect
`tool_use`, call the tool, feed results back, loop until `end_turn`.

**It worked!** Tavily searched, Jina fetched pages, Claude synthesised a
great answer. But:
- Text from intermediate rounds ran together ("Sure! Let me search...Great
  results!")
- No collapsible tool-call UI — just a wall of text
- No `output` array (the structured data OWUI uses to render tool calls)
- The pipe was 600 lines and doing too much

## The Breakthrough: How OWUI Actually Handles Tools

Deep dive into OWUI's source (`middleware.py`, `functions.py`, `Chat.svelte`)
revealed the real architecture:

**OWUI's middleware intercepts tool calls for ALL model types, including
pipes.** If a pipe emits `delta.tool_calls` in OpenAI format, the middleware:
1. Detects the tool calls
2. Executes them server-side
3. Builds the `output` array with `function_call` / `function_call_output` entries
4. Renders the `<details>` collapsible UI
5. Calls the pipe again with tool results in the messages

We had been reimplementing all of this inside the pipe. The pipe's only job
should be format translation.

## v3: The Pipe That Does Less

Ripped out the entire tool dispatch loop, `__tools__` injection, and
`_execute_tools` helper. The pipe became a single-pass stream translator:

- Anthropic `text_delta` → yield plain text
- Anthropic `tool_use` blocks → accumulate, then emit as OAI `delta.tool_calls`
  in the finish chunk
- Anthropic `message_delta` → emit finish reason + usage

OWUI handles the rest. The pipe went from 600 lines to 400. The hardest
part (tool dispatch, error handling, multi-round looping) became OWUI's
problem.

**Result:** Identical UI to the native OR path — collapsible tool calls,
`output` array with 15 entries, `<details>` tags, the works.

## Generalising Beyond OpenRouter

Late realisation: almost nothing in the pipe is OR-specific. The auth
header format (`Bearer` vs `x-api-key`), base URL, and model list source
are the only differences. Added `API_BASE_URL` and `AUTH_TYPE` valves so
the same pipe works with `api.anthropic.com` directly.

## What We Learned

- OR's `/api/v1/messages` is a well-kept secret for Anthropic caching
  with unified billing
- OR reports usage in `message_delta`, not `message_start`
- OWUI's streaming parser needs `usage` in a chunk with no `choices` key
- OWUI's middleware handles tool dispatch for pipes — you just need to
  speak OpenAI chunk format
- The `output` array and `<details>` UI are built by the middleware, not
  the frontend
- A pipe should be a format translator, not a tool orchestrator
- `uv run --script` is great for quick protocol prototyping

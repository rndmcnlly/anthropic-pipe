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

---

## The Truncation Saga

Right after v3 shipped, users hit a new problem: long responses were getting
cut off mid-sentence. The default `max_tokens` was 4096, which was fine for
short answers but brutal for anything substantive. First fix: bump the
default to 16,384. Then the real fix: build a `_MAX_OUTPUT` lookup table
mapping each model family to its actual documented limit (128k for Opus 4.6,
64k for Sonnet 4.6/4.5, 32k for Opus 4.1, etc.) and default each request
to the model's true ceiling. Added a visible truncation warning
("*[Response truncated — max_tokens limit reached]*") so users at least
know when it happens.

## Simplifying Caching: From Manual Breakpoints to APC

The v1 caching strategy was fiddly — three manually placed `cache_control`
breakpoints on the system prompt, first user turn, and second-to-last user
turn. It worked but was fragile and hard to reason about as conversations
grew.

Anthropic's automatic prompt caching (APC) made all that unnecessary. The
new strategy: place a single `cache_control` breakpoint on the last block
of the last message. Anthropic's APC logic handles the rest — the cached
prefix grows automatically as the conversation extends. Deleted all the
manual breakpoint management code and the `_inject_cache_control` helper.
Simpler and more effective.

Also added OpenRouter app attribution headers (`HTTP-Referer`,
`X-Title`) around this time — a minor thing, but OR uses them for their
app rankings dashboard.

## Extended Thinking ✨

Claude 3.7 Sonnet introduced extended thinking, and the newer Claude 4.x
models expanded it. OWUI has a `reasoning_effort` control (low/medium/
high/max) that pipes can honour.

Mapping this to Anthropic's `thinking` API parameter was straightforward —
a budget_tokens lookup table, `temperature` forced to 1.0, `top_k`/`top_p`
stripped (all required by the thinking spec). The streaming side needed
handling for `thinking_delta` and `signature_delta` events, wrapping
thinking output in `<think>` tags that OWUI renders as collapsible
reasoning blocks.

One gotcha: OpenRouter uses dots in model names (`claude-sonnet-4.6`) while
Anthropic uses dashes (`claude-sonnet-4-6`). Added a normalisation step
that replaces dots with dashes for consistent model family matching.

## The Cache Reporting Bug 🔍

After deploying thinking support, we investigated a conversation where
cache stats showed zeros across all turns despite the pipe injecting
`cache_control` breakpoints. Was caching broken?

Pulled the conversation via the OWUI admin API and found an interesting
pattern: most turns reported `cache_read=0, cache_write=0`, but a couple
of tool-loop turns showed real cache activity. Inconsistent.

The culprit was a single `or` operator on line 488:

```python
usage = event.get("usage", {}) or tool_blocks.pop("_usage", {})
```

Anthropic's SSE protocol splits usage across two events: `message_start`
carries input-side tokens (including `cache_read_input_tokens` and
`cache_creation_input_tokens`), while `message_delta` carries output-side
tokens. The pipe stashed the `message_start` usage and retrieved it at
`message_delta` time.

The bug: Python's `or` on dicts returns the first truthy dict. When
`message_delta` had `{"output_tokens": 61}`, that's truthy — so the
stashed dict with all the cache fields was silently dropped. The fix:
merge both dicts explicitly with `{**stashed, **delta_usage}`.

Whether cache stats showed up before the fix depended entirely on whether
OpenRouter happened to include cache fields in the `message_delta` event
(sometimes it did, sometimes it didn't). A classic intermittent bug
caused by a Python footgun.

## Does Caching Actually Work on 4.6?

After fixing the reporting bug, we tested caching on `claude-sonnet-4.6`
and discovered that OpenRouter's prompt caching docs didn't list 4.6 models
as supported. Suspicious. But direct API testing with large prompts
(10k+ tokens) confirmed: **caching works fine on Sonnet 4.6 via
OpenRouter.** Cache write on request 1, cache read on request 2, 12x cost
reduction. OpenRouter's docs just hadn't been updated yet.

The real reason the test conversation showed zeros: it was too short.
Anthropic requires a minimum of 1024 tokens in the cacheable prefix for
Sonnet-class models. Once the conversation grew past that threshold,
`cache_write` appeared, and subsequent turns showed `cache_read` with
`prompt_tokens` dropping to single digits. APC working exactly as designed.

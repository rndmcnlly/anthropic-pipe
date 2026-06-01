# Anthropic Pipe for Open WebUI

A pipe function that talks the native Anthropic Messages API, enabling prompt
caching that OpenAI-compatible endpoints don't provide. Defaults to OpenRouter
for unified billing, but works with `api.anthropic.com` or any compatible
endpoint.

## What it does

- Translates OpenAI ↔ Anthropic message formats (text, images, tool calls)
- Enables Automatic Prompt Caching (APC): a single `cache_control` breakpoint
  on the last message block lets Anthropic advance the cache prefix automatically
  as the conversation grows (~10x cheaper on cache hits)
- Emits tool calls in OpenAI format so OWUI's native middleware handles
  execution and renders the standard collapsible tool-call UI
- Forwards token usage and cost to OWUI's info display
- Derives per-model capabilities (max output, thinking mode, effort levels)
  from a parsed `(family, major, minor)` version, so new releases work without
  code edits. This covers `4.7`/`4.8`, the `-fast` speed variants, and the
  `~anthropic/claude-…-latest` aliases. On Opus 4.7+ (adaptive-thinking-only)
  it sends `thinking: {type: "adaptive"}`, honors `xhigh` effort natively, and
  drops `temperature`/`top_p`/`top_k`, which those models reject.

## Install

### Via CLI (recommended)

```bash
export OWUI_URL=https://your-owui-instance.example.com
export OWUI_TOKEN=your-api-token        # Admin Settings → Account → API Keys

uvx owui-cli functions deploy anthropic_via_openrouter.py anthropic_via_openrouter
uvx owui-cli functions toggle anthropic_via_openrouter
uvx owui-cli functions toggle-global anthropic_via_openrouter
```

Then set the `API_KEY` valve to your OpenRouter or Anthropic key (Admin UI or
`uvx owui-cli` — see `uvx owui-cli --help` for valve endpoints).

### Via Admin UI

1. **Admin → Functions → New Function**, paste `anthropic_via_openrouter.py`
2. Set the `API_KEY` valve to your OpenRouter or Anthropic key
3. Toggle **Active** and **Global**

## Valves

| Valve | Default | Notes |
|---|---|---|
| `API_KEY` | | Required |
| `API_BASE_URL` | `https://openrouter.ai/api/v1` | `https://api.anthropic.com` for direct |
| `AUTH_TYPE` | `bearer` | `x-api-key` for direct Anthropic |
| `CACHE_TTL` | `5m` | `5m` (1.25x write cost) or `1h` (2x write cost) |

## License

MIT

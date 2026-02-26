# Anthropic Pipe for Open WebUI

A pipe function that talks the native Anthropic Messages API, enabling prompt
caching that OpenAI-compatible endpoints don't provide. Defaults to OpenRouter
for unified billing, but works with `api.anthropic.com` or any compatible
endpoint.

## What it does

- Translates OpenAI ↔ Anthropic message formats (text, images, tool calls)
- Injects `cache_control` breakpoints on the system prompt, first user turn,
  and second-to-last user turn (~10x cheaper on cache hits)
- Emits tool calls in OpenAI format so OWUI's native middleware handles
  execution and renders the standard collapsible tool-call UI
- Forwards token usage and cost to OWUI's info display

## Install

1. **Admin → Functions → New Function**, paste `anthropic_via_openrouter.py`
2. Set the `API_KEY` valve to your OpenRouter or Anthropic key
3. Toggle **Active** and **Global**

## Valves

| Valve | Default | Notes |
|---|---|---|
| `API_KEY` | | Required |
| `API_BASE_URL` | `https://openrouter.ai/api/v1` | `https://api.anthropic.com` for direct |
| `AUTH_TYPE` | `bearer` | `x-api-key` for direct Anthropic |
| `CACHE_SYSTEM_PROMPT` | `true` | |
| `CACHE_CONVERSATION` | `true` | |
| `CACHE_TTL` | `5m` | `5m` or `1h` |

## License

MIT

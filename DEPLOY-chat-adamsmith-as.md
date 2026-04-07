# Deploying to chat.adamsmith.as

## Prerequisites

- `~/.tokens/owui/chat-adamsmith-as` — sources `OWUI_URL` and `OWUI_TOKEN`
- OpenRouter API key configured in the pipe's valves (Admin UI or CLI)

## Function ID

`anthropic_via_openrouter`

## Deploy (create or update)

```bash
source ~/.tokens/owui/chat-adamsmith-as
uvx owui-cli functions deploy anthropic_via_openrouter.py anthropic_via_openrouter
```

Valves and toggle state are preserved across code updates.

## First-time setup

After the initial deploy, set valves and activate:

```bash
# Set valves (Admin → Functions → anthropic_via_openrouter → Valves)
# Or via the API — see `uvx owui-cli schema functions` for endpoints.

# Toggle active + global
uvx owui-cli functions toggle anthropic_via_openrouter
uvx owui-cli functions toggle-global anthropic_via_openrouter
```

## Verify

```bash
uvx owui-cli functions list
```

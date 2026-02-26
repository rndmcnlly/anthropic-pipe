# Deploying to chat.adamsmith.as

## Keys

- OWUI admin API key: `~/.tokens/chat-adamsmith-as-admin`
- OpenRouter API key: generate a temporary one at https://openrouter.ai/keys

## Function ID

`anthropic_via_openrouter`

## Create (first time)

```bash
OWUI_KEY=$(cat ~/.tokens/chat-adamsmith-as-admin)

curl -s -X POST \
  -H "Authorization: Bearer $OWUI_KEY" \
  -H "Content-Type: application/json" \
  https://chat.adamsmith.as/api/v1/functions/create \
  -d "$(python3 -c "
import json
content = open('anthropic_via_openrouter.py').read()
print(json.dumps({
    'id': 'anthropic_via_openrouter',
    'name': 'Anthropic via OpenRouter',
    'content': content,
    'meta': {'description': 'Native Anthropic pipe with prompt caching and tool support.', 'manifest': {}}
}))
")"
```

## Set valves

```bash
curl -s -X POST \
  -H "Authorization: Bearer $OWUI_KEY" \
  -H "Content-Type: application/json" \
  https://chat.adamsmith.as/api/v1/functions/id/anthropic_via_openrouter/valves/update \
  -d "{
    \"API_KEY\": \"sk-or-v1-...\",
    \"API_BASE_URL\": \"https://openrouter.ai/api/v1\",
    \"AUTH_TYPE\": \"bearer\",
    \"CACHE_SYSTEM_PROMPT\": true,
    \"CACHE_CONVERSATION\": true,
    \"CACHE_TTL\": \"5m\"
  }"
```

## Activate and make global

```bash
curl -s -X POST -H "Authorization: Bearer $OWUI_KEY" \
  https://chat.adamsmith.as/api/v1/functions/id/anthropic_via_openrouter/toggle

curl -s -X POST -H "Authorization: Bearer $OWUI_KEY" \
  https://chat.adamsmith.as/api/v1/functions/id/anthropic_via_openrouter/toggle/global
```

## Update (push new code)

```bash
OWUI_KEY=$(cat ~/.tokens/chat-adamsmith-as-admin)

curl -s -X POST \
  -H "Authorization: Bearer $OWUI_KEY" \
  -H "Content-Type: application/json" \
  https://chat.adamsmith.as/api/v1/functions/id/anthropic_via_openrouter/update \
  -d "$(python3 -c "
import json
content = open('anthropic_via_openrouter.py').read()
print(json.dumps({
    'id': 'anthropic_via_openrouter',
    'name': 'Anthropic via OpenRouter',
    'content': content,
    'meta': {'description': 'Native Anthropic pipe with prompt caching and tool support.', 'manifest': {}}
}))
")"
```

Valves are preserved across code updates. Toggle state is also preserved.

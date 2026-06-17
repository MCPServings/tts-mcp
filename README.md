# tts-mcp

RedQueen TTS — an OpenAI-compatible text-to-speech front-door (`app.py`) plus
per-model backend servers (`backends/<model>/`). The gateway runs no model
locally; it routes each request to one of several TTS engines reached over
reverse SSH tunnels to a GPU host.

## Layout

```
app.py                  # OpenAI-compatible gateway (FastAPI)
backends/<model>/
  run.sh                # idempotent bootstrap + serve for that model
  server.py             # FastAPI wrapper exposing POST /tts (raw kind)
.env.example            # gateway config template (copy to .env)
requirements.txt        # gateway deps
```

Models: `kokoro` `melotts` `qwen3tts` `cosyvoice` `indextts2` `chatterbox` `orpheus` `dia`.

## Gateway API (OpenAI-compatible)

```
GET  /health            # liveness + per-backend probe
GET  /v1/models         # advertised TTS models
GET  /v1/audio/voices   # per-model voice catalog
POST /v1/audio/speech   # synthesize; routes by `model`
```

`POST /v1/audio/speech` body:

| field | required | notes |
|-------|----------|-------|
| `model` | yes | one of the configured backend ids |
| `input` | yes | text to synthesize |
| `response_format` | no | `mp3` (default) / `wav` / `opus` / `aac` / `flac` / `pcm` |
| `voice` | no | falls back to the backend's `default_voice` |
| `speed` | no | default `1.0` |
| `reference_audio` | no | server-local wav path (back-compat) |
| `reference_audio_b64` | no | base64 reference audio for voice cloning |
| `reference_audio_url` | no | http(s) URL to reference audio for cloning |
| `reference_text` | no | transcript of the reference audio (best clone quality) |
| `instructions` | no | style/instruction prompt |

Voice cloning is supported by the `cosyvoice2` / `cosyvoice3` (multilingual),
`indextts2` (multilingual, plus emotion control) and `chatterbox` (English)
backends: pass `reference_audio_b64` or `reference_audio_url` plus
`reference_text`. Inline reference audio is decoded into a short-lived temp wav
that is removed after synthesis.

## Backend kinds

- `openai` — backend already exposes `POST /v1/audio/speech`; the request is
  forwarded as-is after model-name rewrite (e.g. Kokoro-FastAPI).
- `raw` — backend exposes `POST /tts` with `{text, voice, language, format,
  speed, reference_audio, reference_audio_b64, reference_audio_url,
  reference_text, instructions}` and returns audio bytes.

## Config

Copy `.env.example` to `.env` and set `TTS_BACKENDS` (JSON routing table),
optional `TTS_API_KEY` / `TTS_RAPIDAPI_PROXY_SECRET`.

### Gateway auth & rate limiting (REST `/v1/audio/speech`)

The gateway front-door (`_check_auth`) is **open when neither `TTS_API_KEY` nor
`TTS_RAPIDAPI_PROXY_SECRET` is set** ("dev mode" — any caller is allowed). Set
`TTS_API_KEY` to require `Authorization: Bearer <key>`. Note the REST gateway
itself has **no built-in rate limiting** (only an internal per-backend
concurrency cap); when exposing it publicly in dev mode, put a rate limit at the
reverse proxy. The production deploy (api.redqueen-serving.cloud) runs dev mode
+ an nginx `limit_req` on `location /tts/` (`tts_zone` 30 r/m, `burst=10
nodelay`, `429` on overflow) as the DoS guard.

## MCP server (`mcp_server.py`)

A Model Context Protocol front-end over the gateway, exposing TTS as MCP tools
for agent clients (Claude Desktop, Cursor, VS Code, etc.).

Tools: `list_models`, `list_voices`, `text_to_speech`, `clone_voice`.

Transports (`TTS_MCP_TRANSPORT`):

- `streamable-http` (default) — remote MCP over HTTP at `/mcp`, deploy behind
  nginx. Live at `https://ttskits.online/mcp`.
- `stdio` — local spawn; configure in the client's `mcp.json`:

  ```json
  {
    "mcpServers": {
      "redqueen-tts": {
        "command": "/path/to/venv/bin/python",
        "args": ["/path/to/mcp_server.py"],
        "env": { "TTS_MCP_TRANSPORT": "stdio",
                 "TTS_MCP_GATEWAY": "https://api.redqueen-serving.cloud/tts" }
      }
    }
  }
  ```

### Rate limiting (streamable-http only)

Per client IP (`X-Forwarded-For` when behind nginx), enforced in-app:

- synthesis tools (`text_to_speech`, `clone_voice`): `TTS_MCP_RL_SYNTH`/min (default 15)
- read-only tools (`list_models`, `list_voices`): coarse HTTP cap `TTS_MCP_RL_READ`/min (default 60)
- global synthesis concurrency cap: `TTS_MCP_MAX_CONCURRENCY` (default 4), protecting the single-GPU backend.

`429` is returned at the HTTP layer for abusive request rates; synthesis tools
return an MCP error when the per-IP synth budget is exceeded. The stdio
transport is unmetered (single trusted caller).

### Host allowlist

MCP's DNS-rebinding protection requires the served Host to be allowlisted via
`TTS_MCP_ALLOWED_HOSTS` (comma-separated; include the public domain when behind
nginx, or `*` to disable the check).

---

## License

MIT © MCPServings. See [LICENSE](LICENSE).


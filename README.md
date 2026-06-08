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

Models: `kokoro` `melotts` `qwen3tts` `cosyvoice` `indextts2` `orpheus` `dia`.

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

Voice cloning is supported by the CosyVoice backends (`cosyvoice2` /
`cosyvoice3`): pass `reference_audio_b64` or `reference_audio_url` plus
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

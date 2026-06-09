"""RedQueen TTS API.

A thin, OpenAI-compatible text-to-speech front-door. It runs no model locally;
it routes each request to one of several TTS engines reached over reverse SSH
tunnels to a GPU host (halo-office, Strix Halo gfx1151).

Public surface (OpenAI-compatible):
  GET  /health                 -> liveness + per-backend probe
  GET  /v1/models              -> advertised TTS voices/models
  GET  /v1/audio/voices        -> per-model voice catalog (best effort)
  POST /v1/audio/speech        -> synthesize; routes by ``model`` field

Each public ``model`` id maps to one backend in TTS_BACKENDS (JSON env). A
backend declares a ``kind`` describing how to call it:

  * ``openai`` -- backend already exposes POST /v1/audio/speech (Kokoro-FastAPI,
    etc.); the request body is forwarded as-is (after model-name rewrite) and the
    audio bytes streamed back.
  * ``raw``    -- backend exposes a simple POST {text, voice, format} -> audio
    bytes endpoint at ``path`` (our custom rocm/vllm server.py wrappers).

Auth: optional bearer key (TTS_API_KEY) and/or RapidAPI proxy secret
(TTS_RAPIDAPI_PROXY_SECRET), mirroring redqueen-nsfw.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment / .env)
# ---------------------------------------------------------------------------
# TTS_BACKENDS is a JSON object: { "<model_id>": { ... }, ... }
# Each backend:
#   url              base URL of the backend (e.g. http://127.0.0.1:8321)
#   kind             "openai" | "raw"
#   upstream_model   model name to send to the backend (openai kind); optional
#   path             endpoint path for "raw" kind (default /tts)
#   voices           optional list of advertised voice ids
#   default_voice    optional default voice
#   languages        optional list of language codes (advertised)
#   note             optional human description
_DEFAULT_BACKENDS = {
    "kokoro": {
        "url": "http://127.0.0.1:8321",
        "kind": "openai",
        "upstream_model": "kokoro",
        "default_voice": "af_heart",
        "languages": ["en", "zh", "ja", "es", "fr"],
        "note": "Kokoro-82M (apache-2.0), fixed voices, fast.",
    },
}

try:
    TTS_BACKENDS: dict[str, dict[str, Any]] = json.loads(
        os.environ.get("TTS_BACKENDS", "") or "null"
    ) or _DEFAULT_BACKENDS
except json.JSONDecodeError as exc:  # pragma: no cover - config error
    raise RuntimeError(f"invalid TTS_BACKENDS json: {exc}") from exc

API_KEY = os.environ.get("TTS_API_KEY", "")
RAPIDAPI_PROXY_SECRET = os.environ.get("TTS_RAPIDAPI_PROXY_SECRET", "")
REQUEST_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "120"))
PUBLIC_NAME = os.environ.get("TTS_PUBLIC_NAME", "redqueen-tts")

# Map response_format -> content type for streaming back to the client.
_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",
}

# Formats we transcode to with ffmpeg when the backend can't emit them itself
# (our raw backends return WAV). ffmpeg args per target container/codec.
FFMPEG_BIN = os.environ.get("TTS_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
TRANSCODE_ENABLED = os.environ.get("TTS_TRANSCODE", "1") == "1"
_FFMPEG_ARGS: dict[str, list[str]] = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
    "opus": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac"],
}
_FFMPEG_CT = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}


async def _transcode(audio: bytes, fmt: str) -> tuple[bytes, str] | None:
    """Transcode raw audio bytes to ``fmt`` via ffmpeg (reads stdin -> stdout).

    Returns (bytes, content_type) on success, or None to fall back to passthrough.
    """
    args = _FFMPEG_ARGS.get(fmt)
    if not args or not TRANSCODE_ENABLED:
        return None
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(input=audio), timeout=60)
    except Exception:  # noqa: BLE001 - any ffmpeg failure -> passthrough
        return None
    if proc.returncode != 0 or not out:
        return None
    return out, _FFMPEG_CT.get(fmt, "application/octet-stream")


app = FastAPI(title="RedQueen TTS API", version="0.1.0")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _check_auth(authorization: str, proxy_secret: str) -> None:
    """Allow the request if either auth method matches (when configured)."""
    if RAPIDAPI_PROXY_SECRET and proxy_secret == RAPIDAPI_PROXY_SECRET:
        return
    if API_KEY:
        token = ""
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token == API_KEY:
            return
    # If neither secret is configured, the front-door is open (dev mode).
    if not API_KEY and not RAPIDAPI_PROXY_SECRET:
        return
    raise HTTPException(status_code=401, detail="unauthorized")


def _resolve_backend(model: str) -> tuple[str, dict[str, Any]]:
    backend = TTS_BACKENDS.get(model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model '{model}'; available: {sorted(TTS_BACKENDS)}",
        )
    return model, backend


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for model, be in TTS_BACKENDS.items():
            url = be["url"].rstrip("/") + "/health"
            try:
                r = await client.get(url)
                results[model] = {"ok": r.status_code < 500, "status": r.status_code}
            except Exception as exc:  # noqa: BLE001 - report any probe failure
                results[model] = {"ok": False, "error": str(exc)[:120]}
    healthy = all(v.get("ok") for v in results.values()) if results else False
    return JSONResponse({"status": "ok" if healthy else "degraded", "backends": results})


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    data = []
    for model, be in TTS_BACKENDS.items():
        data.append(
            {
                "id": model,
                "object": "model",
                "owned_by": PUBLIC_NAME,
                "languages": be.get("languages", []),
                "default_voice": be.get("default_voice"),
                "note": be.get("note", ""),
            }
        )
    return {"object": "list", "data": data}


@app.get("/v1/audio/voices")
async def list_voices() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, be in TTS_BACKENDS.items():
        out[model] = be.get("voices", [])
    return {"object": "list", "voices": out}


@app.post("/v1/audio/speech")
async def speech(
    request: Request,
    authorization: str = Header(default=""),
    x_rapidapi_proxy_secret: str = Header(default=""),
) -> Response:
    _check_auth(authorization, x_rapidapi_proxy_secret)

    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json body: {exc}")

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="missing 'model'")
    text = body.get("input")
    if not text:
        raise HTTPException(status_code=400, detail="missing 'input' text")

    _, be = _resolve_backend(model)
    fmt = (body.get("response_format") or "mp3").lower()
    content_type = _CONTENT_TYPES.get(fmt, "application/octet-stream")
    voice = body.get("voice") or be.get("default_voice")

    kind = be.get("kind", "openai")
    base = be["url"].rstrip("/")

    if kind == "openai":
        payload = dict(body)
        payload["model"] = be.get("upstream_model", model)
        if voice:
            payload["voice"] = voice
        url = base + "/v1/audio/speech"
    elif kind == "raw":
        payload = {
            "text": text,
            "voice": voice,
            "language": body.get("language"),
            "format": fmt,
            "speed": body.get("speed", 1.0),
            "reference_audio": body.get("reference_audio"),
            "reference_audio_b64": body.get("reference_audio_b64"),
            "reference_audio_url": body.get("reference_audio_url"),
            "reference_text": body.get("reference_text"),
            "instructions": body.get("instructions"),
        }
        url = base + (be.get("path") or "/tts")
    else:  # pragma: no cover - config error
        raise HTTPException(status_code=500, detail=f"bad backend kind '{kind}'")

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=504, detail=f"backend '{model}' unreachable: {exc}")

    if r.status_code >= 400:
        detail = r.text[:300]
        raise HTTPException(status_code=502, detail=f"backend '{model}' error {r.status_code}: {detail}")

    audio = r.content
    backend_ct = r.headers.get("content-type", "")
    out_ct = backend_ct or content_type

    # Transcode to the requested format when the backend didn't already emit it.
    # Our raw backends return WAV; honor response_format=mp3/opus/aac/flac.
    want = _FFMPEG_CT.get(fmt)
    if want and want not in backend_ct.lower():
        result = await _transcode(audio, fmt)
        if result is not None:
            audio, out_ct = result

    headers = {
        "X-TTS-Model": model,
        "X-TTS-Latency-ms": str(round((time.time() - t0) * 1000, 1)),
    }
    return Response(content=audio, media_type=out_ct, headers=headers)

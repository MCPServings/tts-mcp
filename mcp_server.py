"""RedQueen TTS — MCP server (streamable-http / stdio).

A thin Model Context Protocol front-end over the existing OpenAI-compatible TTS
gateway (``app.py`` on the same host, ``TTS_MCP_GATEWAY`` URL). It exposes the
TTS catalogue and synthesis (incl. CosyVoice voice cloning) as MCP tools.

Transports (``TTS_MCP_TRANSPORT``):
  * ``streamable-http`` (default) — remote MCP over HTTP at ``/mcp``; deploy
    behind nginx. Per-IP rate limiting is enforced by an ASGI middleware.
  * ``stdio`` — local spawn (Claude Desktop / Cursor / VS Code). No rate limit
    (single trusted caller).

Rate limiting (per client IP, streamable-http only; all token-bucket per minute):
  * synthesis tools (text_to_speech, clone_voice): TTS_MCP_RL_SYNTH (default 15)
  * read-only tools (list_models, list_voices):     TTS_MCP_RL_READ  (default 60)
  * a global concurrency semaphore caps simultaneous synthesis to protect the
    single-GPU backend: TTS_MCP_MAX_CONCURRENCY (default 4).
"""
from __future__ import annotations

import os
import time
import threading
from collections import defaultdict, deque

import anyio
import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import AudioContent, TextContent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY = os.environ.get("TTS_MCP_GATEWAY", "http://127.0.0.1:18821").rstrip("/")
GATEWAY_TIMEOUT = float(os.environ.get("TTS_MCP_TIMEOUT", "300"))
TRANSPORT = os.environ.get("TTS_MCP_TRANSPORT", "streamable-http")
HOST = os.environ.get("TTS_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_MCP_PORT", "18822"))

RL_SYNTH = int(os.environ.get("TTS_MCP_RL_SYNTH", "15"))   # synth calls/min/IP
RL_READ = int(os.environ.get("TTS_MCP_RL_READ", "60"))     # read calls/min/IP
RL_WINDOW = float(os.environ.get("TTS_MCP_RL_WINDOW", "60"))
MAX_CONCURRENCY = int(os.environ.get("TTS_MCP_MAX_CONCURRENCY", "4"))
# Trust X-Forwarded-For (we sit behind nginx/Cloudflare). Off => use peer addr.
TRUST_XFF = os.environ.get("TTS_MCP_TRUST_XFF", "1") == "1"

# Hosts/origins allowed by MCP's DNS-rebinding protection. When serving behind
# nginx the upstream Host header is the public domain, so it must be allowlisted
# (comma-separated env). "*" disables the host/origin check entirely.
_ALLOWED_HOSTS = [h.strip() for h in os.environ.get(
    "TTS_MCP_ALLOWED_HOSTS", "127.0.0.1,127.0.0.1:18822,localhost,localhost:18822"
).split(",") if h.strip()]
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "TTS_MCP_ALLOWED_ORIGINS", ""
).split(",") if o.strip()]

from mcp.server.transport_security import TransportSecuritySettings

if "*" in _ALLOWED_HOSTS:
    _TS = TransportSecuritySettings(enable_dns_rebinding_protection=False)
else:
    _TS = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_ALLOWED_HOSTS,
        allowed_origins=_ALLOWED_ORIGINS or _ALLOWED_HOSTS,
    )

mcp = FastMCP(
    "redqueen-tts",
    instructions=(
        "Text-to-speech with multiple engines and voice cloning. Use "
        "list_models to discover voices/languages, text_to_speech to synthesize "
        "with a built-in voice, and clone_voice (CosyVoice) to synthesize in a "
        "cloned voice from a reference audio sample."
    ),
    host=HOST,
    port=PORT,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=_TS,
)


# ---------------------------------------------------------------------------
# Per-IP rate limiter (sliding window, thread-safe). Used by ASGI middleware.
# ---------------------------------------------------------------------------
class _SlidingWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            cutoff = now - window
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def sweep(self, window: float) -> None:
        """Drop stale per-IP queues so memory doesn't grow unbounded."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, q in self._hits.items() if not q or q[-1] < now - window]
            for k in stale:
                del self._hits[k]


_RL = _SlidingWindow()
# Global synthesis concurrency guard (protects the single-GPU backend).
_SYNTH_SEM = anyio.Semaphore(MAX_CONCURRENCY)


def _client_ip(scope) -> str:
    if TRUST_XFF:
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                first = value.decode("latin1").split(",")[0].strip()
                if first:
                    return first
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    """ASGI middleware: per-IP cap on MCP tool-call requests.

    The MCP streamable-http endpoint multiplexes everything over POST /mcp, so
    we can't cheaply distinguish synth vs read by path. We apply the (looser)
    read limit at the HTTP layer as a coarse DoS guard, and enforce the tighter
    synth limit + concurrency inside the tools themselves.
    """

    def __init__(self, app) -> None:
        self.app = app
        self._last_sweep = time.monotonic()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        now = time.monotonic()
        if now - self._last_sweep > RL_WINDOW:
            _RL.sweep(RL_WINDOW * 2)
            self._last_sweep = now

        ip = _client_ip(scope)
        # Coarse per-IP HTTP cap: read-limit + synth-limit headroom.
        if not _RL.allow(f"http:{ip}", RL_READ + RL_SYNTH, RL_WINDOW):
            await self._reject(send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send) -> None:
        body = b'{"error":"rate_limited","detail":"too many requests, slow down"}'
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", b"30"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _tool_ip(ctx: Context | None) -> str:
    """Best-effort client IP from the tool's request context (streamable-http)."""
    try:
        req = ctx.request_context.request  # starlette Request for http transport
        if req is not None:
            if TRUST_XFF:
                xff = req.headers.get("x-forwarded-for")
                if xff:
                    return xff.split(",")[0].strip()
            if req.client:
                return req.client.host
    except Exception:
        pass
    return "stdio"  # stdio transport has no remote peer


def _check_synth_rl(ctx: Context | None) -> None:
    ip = _tool_ip(ctx)
    if ip == "stdio":
        return  # local trusted caller, no limit
    if not _RL.allow(f"synth:{ip}", RL_SYNTH, RL_WINDOW):
        raise ValueError(
            f"rate limited: max {RL_SYNTH} synthesis calls per "
            f"{int(RL_WINDOW)}s per IP; please slow down"
        )


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------
_MIME = {
    "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
    "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/L16",
}


async def _gateway_get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(GATEWAY + path)
        r.raise_for_status()
        return r.json()


async def _synthesize(payload: dict, fmt: str) -> AudioContent:
    import base64

    async with _SYNTH_SEM:
        async with httpx.AsyncClient(timeout=GATEWAY_TIMEOUT) as client:
            r = await client.post(GATEWAY + "/v1/audio/speech", json=payload)
    if r.status_code >= 400:
        raise ValueError(f"tts backend error {r.status_code}: {r.text[:300]}")
    mime = r.headers.get("content-type", _MIME.get(fmt, "audio/wav")).split(";")[0]
    return AudioContent(
        type="audio",
        data=base64.b64encode(r.content).decode("ascii"),
        mimeType=mime,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_models() -> str:
    """List available TTS models with their languages and default voice.

    Returns a human-readable catalogue. Use the model ``id`` values with
    ``text_to_speech`` / ``clone_voice``.
    """
    data = await _gateway_get("/v1/models")
    lines = ["Available TTS models:"]
    for m in data.get("data", []):
        langs = ",".join(m.get("languages", []) or [])
        clone = " [voice-clone]" if "clone" in (m.get("note", "").lower()) else ""
        dv = m.get("default_voice")
        dv_s = f" default_voice={dv}" if dv else ""
        lines.append(f"  - {m['id']}{clone}: langs=[{langs}]{dv_s} — {m.get('note','')}")
    return "\n".join(lines)


@mcp.tool()
async def list_voices(model: str | None = None) -> str:
    """List the available voices per model (best-effort).

    Args:
        model: optional model id to filter to a single model.
    """
    data = await _gateway_get("/v1/audio/voices")
    voices = data.get("voices", {})
    if model:
        voices = {model: voices.get(model, [])}
    lines = ["Voices:"]
    for mid, vs in voices.items():
        vs_s = ", ".join(vs) if vs else "(dynamic / single default)"
        lines.append(f"  - {mid}: {vs_s}")
    return "\n".join(lines)


@mcp.tool()
async def text_to_speech(
    text: str,
    model: str = "kokoro",
    voice: str | None = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    language: str | None = None,
    ctx: Context = None,
) -> AudioContent:
    """Synthesize speech from text using a built-in voice.

    Args:
        text: the text to speak.
        model: TTS model id (see list_models), e.g. kokoro, melotts,
            qwen3-tts-1.7b, cosyvoice3, indextts2, chatterbox.
        voice: voice id; omit to use the model's default.
        response_format: mp3 (default), wav, opus, aac, flac, or pcm.
        speed: speaking rate multiplier (1.0 = normal).
        language: optional language hint (e.g. en, zh, ja).

    Returns the synthesized audio.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    _check_synth_rl(ctx)
    fmt = (response_format or "mp3").lower()
    payload: dict = {"model": model, "input": text, "response_format": fmt, "speed": speed}
    if voice:
        payload["voice"] = voice
    if language:
        payload["language"] = language
    return await _synthesize(payload, fmt)


@mcp.tool()
async def clone_voice(
    text: str,
    reference_audio_b64: str | None = None,
    reference_audio_url: str | None = None,
    reference_text: str | None = None,
    model: str = "cosyvoice3",
    response_format: str = "wav",
    speed: float = 1.0,
    ctx: Context = None,
) -> AudioContent:
    """Synthesize speech in a cloned voice from a reference audio sample.

    Supported by the voice-clone models: cosyvoice3 / cosyvoice2 (zh/en,
    multilingual), indextts2 (zh/en, also supports emotion control), and
    chatterbox (English). Provide the reference voice either inline as base64
    (``reference_audio_b64``) or as an http(s) URL (``reference_audio_url``).
    Supplying ``reference_text`` (the transcript of the reference clip)
    improves clone quality.

    Args:
        text: the text to speak in the cloned voice.
        reference_audio_b64: base64-encoded reference audio (wav/mp3/flac/...).
        reference_audio_url: http(s) URL to the reference audio.
        reference_text: transcript of the reference audio (recommended).
        model: cosyvoice3 (default), cosyvoice2, indextts2, or chatterbox.
        response_format: wav (default), flac, or mp3.
        speed: speaking rate multiplier.

    Returns the synthesized audio in the cloned voice.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    if not reference_audio_b64 and not reference_audio_url:
        raise ValueError("provide reference_audio_b64 or reference_audio_url")
    _check_synth_rl(ctx)
    fmt = (response_format or "wav").lower()
    payload: dict = {"model": model, "input": text, "response_format": fmt, "speed": speed}
    if reference_audio_b64:
        payload["reference_audio_b64"] = reference_audio_b64
    if reference_audio_url:
        payload["reference_audio_url"] = reference_audio_url
    if reference_text:
        payload["reference_text"] = reference_text
    return await _synthesize(payload, fmt)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
        return
    # streamable-http: wrap the MCP ASGI app with the rate-limit middleware and
    # serve with uvicorn.
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(RateLimitMiddleware)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("TTS_MCP_LOG", "info"))


if __name__ == "__main__":
    main()

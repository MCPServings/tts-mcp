from __future__ import annotations

import io
import os
from typing import Any

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from kokoro_onnx import Kokoro

PORT = int(os.environ.get("PORT", "8221"))
MODEL_PATH = os.environ.get("KOKORO_ONNX_MODEL", "/work/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_ONNX_VOICES", "/work/models/voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.environ.get("KOKORO_DEFAULT_LANG", "en-us")

VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella",
    "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

app = FastAPI(title="Kokoro ONNX TTS", version="0.1.0")
_model: Kokoro | None = None


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str | None = None
    response_format: str | None = "wav"
    speed: float | None = 1.0
    language: str | None = None


class RawRequest(BaseModel):
    text: str
    voice: str | None = None
    format: str | None = "wav"
    speed: float | None = 1.0
    language: str | None = None


def _load() -> Kokoro:
    global _model
    if _model is None:
        _model = Kokoro(MODEL_PATH, VOICES_PATH)
    return _model


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _load()
        return {
            "status": "ok",
            "model": "onnx-community/Kokoro-82M-v1.0-ONNX",
            "runtime": "kokoro-onnx",
            "sample_rate": 24000,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


@app.get("/voices")
def voices() -> dict[str, Any]:
    return {"voices": VOICES, "default_voice": DEFAULT_VOICE}


def _wav_response(text: str, voice: str | None, speed: float | None, language: str | None) -> Response:
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty input")
    selected_voice = voice or DEFAULT_VOICE
    lang = language or DEFAULT_LANG
    if selected_voice not in VOICES:
        # The model has more voices than this small list; allow explicit custom ids.
        pass
    samples, sample_rate = _load().create(
        text,
        voice=selected_voice,
        speed=float(speed or 1.0),
        lang=lang,
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    return _wav_response(req.input, req.voice, req.speed, req.language)


@app.post("/tts")
def raw_tts(req: RawRequest) -> Response:
    return _wav_response(req.text, req.voice, req.speed, req.language)

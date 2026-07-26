from __future__ import annotations

import io
import os
from typing import Any

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from kokoro_onnx import Kokoro

# misaki g2p is installed separately for languages unsupported by espeak.
import sys as _sys

_MISAKI_PATH = os.environ.get("KOKORO_MISAKI_PATH", "/work/misaki")
if _MISAKI_PATH and _MISAKI_PATH not in _sys.path:
    _sys.path.append(_MISAKI_PATH)

_zh_g2p = None


def _get_zh_g2p():
    global _zh_g2p
    if _zh_g2p is None:
        from misaki import zh as _mzh
        _zh_g2p = _mzh.ZHG2P()
    return _zh_g2p


PORT = int(os.environ.get("PORT", "8221"))
MODEL_PATH = os.environ.get("KOKORO_ONNX_MODEL", "/work/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_ONNX_VOICES", "/work/models/voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.environ.get("KOKORO_DEFAULT_LANG", "en-us")

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
    return {"voices": sorted(_load().voices.files), "default_voice": DEFAULT_VOICE}


def _norm_lang(language: str | None) -> str:
    lang = (language or DEFAULT_LANG).lower()
    return {
        "en": "en-us",
        "en_us": "en-us",
        "en-gb": "en-gb",
        "en_uk": "en-gb",
        "fr": "fr-fr",
    }.get(lang, lang)


def _wav_response(text: str, voice: str | None, speed: float | None, language: str | None) -> Response:
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty input")
    selected_voice = voice or DEFAULT_VOICE
    lang = _norm_lang(language)
    model = _load()
    if selected_voice not in model.voices.files:
        raise HTTPException(status_code=400, detail=f"unknown voice '{selected_voice}'")
    if lang in ("zh", "cmn", "zh-cn"):
        try:
            phonemes, _ = _get_zh_g2p()(text)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"zh g2p failed: {exc}")
        samples, sample_rate = model.create(
            phonemes,
            voice=selected_voice,
            speed=float(speed or 1.0),
            is_phonemes=True,
        )
    else:
        samples, sample_rate = model.create(
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

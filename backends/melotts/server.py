import io, os, time, tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# MeloTTS: per-language model, loaded lazily.
_LANGS = {
    "en": "EN", "zh": "ZH", "es": "ES", "fr": "FR", "jp": "JP", "ja": "JP", "kr": "KR", "ko": "KR",
}
_models = {}
_DEV = os.environ.get("MELO_DEVICE", "cpu")

app = FastAPI(title="MeloTTS backend")


def _get_model(lang_code: str):
    key = _LANGS.get(lang_code.lower(), "EN")
    if key not in _models:
        from melo.api import TTS
        _models[key] = TTS(language=key, device=_DEV)
    return _models[key]


class TtsReq(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    reference_audio: Optional[str] = None
    instructions: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "device": _DEV, "loaded": list(_models.keys())}


@app.get("/voices")
def voices():
    out = {}
    for lc, key in _LANGS.items():
        try:
            m = _get_model(lc)
            out[key] = list(m.hps.data.spk2id.keys())
        except Exception as e:  # noqa
            out[key] = f"err: {e}"
    return out


@app.post("/tts")
def tts(req: TtsReq):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text required")
    t0 = time.time()
    m = _get_model(req.language or "en")
    spk2id = m.hps.data.spk2id
    spk = None
    if req.voice and req.voice in spk2id:
        spk = spk2id[req.voice]
    else:
        spk = list(spk2id.values())[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        m.tts_to_file(req.text, spk, out_path, speed=req.speed or 1.0)
        with open(out_path, "rb") as fh:
            data = fh.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    media = "audio/wav"
    fmt = (req.format or "wav").lower()
    if fmt in ("mp3", "flac", "ogg", "opus"):
        try:
            import soundfile as sf
            import numpy as np  # noqa
            audio, sr = sf.read(io.BytesIO(data))
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="FLAC" if fmt == "flac" else "OGG" if fmt in ("ogg", "opus") else "WAV")
            data = buf.getvalue()
            media = {"flac": "audio/flac", "ogg": "audio/ogg", "opus": "audio/ogg"}.get(fmt, "audio/wav")
        except Exception:
            media = "audio/wav"
    return Response(content=data, media_type=media,
                    headers={"X-Latency-ms": str(round((time.time() - t0) * 1000, 1))})

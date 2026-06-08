import os, io, time
from typing import Optional
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
PORT = int(os.environ.get("PORT", "8223"))
DEFAULT_SPK = os.environ.get("QWEN_TTS_SPEAKER", "Ryan")

_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "jp": "Japanese",
    "ko": "Korean", "kr": "Korean", "de": "German", "fr": "French",
    "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
    "auto": "Auto",
}

state = {}
app = FastAPI(title="Qwen3-TTS backend")


@app.on_event("startup")
def _load():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=dev, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    state["model"] = m
    try:
        state["speakers"] = list(m.get_supported_speakers())
    except Exception:
        state["speakers"] = []
    try:
        state["languages"] = list(m.get_supported_languages())
    except Exception:
        state["languages"] = []
    print("[startup] loaded", MODEL_ID, "spk=", state["speakers"], flush=True)


class TtsReq(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "auto"
    format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None
    reference_audio: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in state else "loading",
        "model": MODEL_ID,
        "speakers": state.get("speakers", []),
        "languages": state.get("languages", []),
    }


@app.get("/voices")
def voices():
    return {"speakers": state.get("speakers", []), "languages": state.get("languages", [])}


@app.post("/tts")
def tts(req: TtsReq):
    m = state.get("model")
    if m is None:
        raise HTTPException(503, "model loading")
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text required")
    t0 = time.time()
    lang = _LANG_MAP.get((req.language or "auto").lower(), "Auto")
    spk = req.voice or DEFAULT_SPK
    if state.get("speakers") and spk not in state["speakers"]:
        spk = DEFAULT_SPK if DEFAULT_SPK in state["speakers"] else state["speakers"][0]
    try:
        wavs, sr = m.generate_custom_voice(
            text=req.text, language=lang, speaker=spk, instruct=req.instructions or "",
        )
    except Exception as e:
        raise HTTPException(500, f"synthesis failed: {e}")
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Latency-ms": str(round((time.time() - t0) * 1000, 1)),
                             "X-Speaker": spk})

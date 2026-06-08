import io
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

REPO = os.environ.get("INDEXTTS_REPO", "/work/index-tts")
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")
DEFAULT_VOICE = os.environ.get("INDEXTTS_DEFAULT_VOICE", "examples/voice_01.wav")
PORT = int(os.environ.get("PORT", "8227"))
SR = 22050

# IndexTTS2 uses relative paths (./checkpoints/hf_cache) -> must run from repo root
os.chdir(REPO)

app = FastAPI()
_state = {"tts": None, "err": None}


@app.on_event("startup")
def _load():
    try:
        from indextts.infer_v2 import IndexTTS2

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        tts = IndexTTS2(
            cfg_path=CFG_PATH,
            model_dir=MODEL_DIR,
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
            device=dev,
        )
        _state["tts"] = tts
        print("[startup] loaded IndexTTS2 dev=", dev, flush=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        _state["err"] = str(e)


class TTSReq(BaseModel):
    text: str
    voice: str | None = None
    language: str | None = None
    format: str | None = "wav"
    speed: float | None = 1.0
    reference_audio: str | None = None
    instructions: str | None = None


@app.get("/health")
def health():
    if _state["err"]:
        return JSONResponse({"status": "error", "error": _state["err"]}, status_code=503)
    if _state["tts"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": "IndexTTS-2", "sample_rate": SR}


@app.get("/voices")
def voices():
    return {
        "default": DEFAULT_VOICE,
        "note": "voice cloning model; pass reference_audio (path inside container) to clone a voice",
    }


def _resolve_ref(req: TTSReq) -> str:
    if req.reference_audio and os.path.exists(req.reference_audio):
        return req.reference_audio
    if req.voice and os.path.exists(req.voice):
        return req.voice
    return DEFAULT_VOICE


def _encode(arr: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    fmt = (fmt or "wav").lower()
    if fmt == "mp3":
        fmt = "wav"
    subtype = "PCM_16" if fmt == "wav" else None
    sf.write(buf, arr, sr, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


@app.post("/tts")
def tts(req: TTSReq):
    t = _state["tts"]
    if t is None:
        return JSONResponse({"error": _state["err"] or "loading"}, status_code=503)
    try:
        ref = _resolve_ref(req)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            t.infer(spk_audio_prompt=ref, text=req.text, output_path=out_path, verbose=False)
            data_arr, sr = sf.read(out_path, dtype="float32")
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
        data = _encode(data_arr, sr, req.format or "wav")
        mt = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(
            (req.format or "wav").lower(), "audio/wav"
        )
        return Response(content=data, media_type=mt)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

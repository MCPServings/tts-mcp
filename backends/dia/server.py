import io
import os
import sys

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

MODEL_ID = os.environ.get("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
PORT = int(os.environ.get("PORT", "8229"))
DTYPE = os.environ.get("DIA_DTYPE", "float16")

app = FastAPI()
_state = {"model": None, "sr": 44100, "err": None}


@app.on_event("startup")
def _load():
    try:
        from dia.model import Dia

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = DTYPE if dev.type == "cuda" else "float32"
        m = Dia.from_pretrained(MODEL_ID, compute_dtype=dtype, device=dev)
        _state["model"] = m
        print("[startup] loaded", MODEL_ID, "dev=", dev, "dtype=", dtype, flush=True)
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
    if _state["model"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID, "sample_rate": _state["sr"]}


@app.get("/voices")
def voices():
    return {"default": "[S1]/[S2] dialogue tags; non-verbal (laughs) (sighs) supported"}


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
    m = _state["model"]
    if m is None:
        return JSONResponse({"error": _state["err"] or "loading"}, status_code=503)
    try:
        text = req.text
        # Dia is a dialogue model; bare text must carry at least one speaker tag.
        if "[S1]" not in text and "[S2]" not in text:
            text = "[S1] " + text
        audio_prompt = None
        if req.reference_audio and os.path.exists(req.reference_audio):
            audio_prompt = req.reference_audio
        with torch.inference_mode():
            out = m.generate(
                text,
                use_torch_compile=False,
                verbose=False,
                cfg_scale=3.0,
                temperature=1.3,
                top_p=0.95,
                cfg_filter_top_k=45,
                audio_prompt=audio_prompt,
            )
        if isinstance(out, list):
            out = out[0]
        arr = np.asarray(out, dtype=np.float32)
        data = _encode(arr, _state["sr"], req.format or "wav")
        mt = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(
            (req.format or "wav").lower(), "audio/wav"
        )
        return Response(content=data, media_type=mt)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

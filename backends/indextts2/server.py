import base64
import io
import os
import re
import tempfile
import threading
import urllib.request

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

REPO = os.environ.get("INDEXTTS_REPO", "/work/index-tts")
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")
DEFAULT_VOICE = os.environ.get("INDEXTTS_DEFAULT_VOICE", "/work/default_ref.wav")
# Upstream recommends bf16, but on ROCm (MI300X, torch 2.10) bf16 makes the GPT repeat/insert
# syllables: zh CER 12-37% vs 0-3% in fp32 at the same speed (2026-11-04 Whisper check).
USE_BF16 = os.environ.get("INDEXTTS_BF16", "0") == "1"
PORT = int(os.environ.get("PORT", "8227"))
SR = 22050
_MAX_REF_BYTES = int(os.environ.get("INDEXTTS_MAX_REF_BYTES", str(15 * 1024 * 1024)))

# IndexTTS uses relative paths (./checkpoints/hf_cache) -> must run from repo root
os.chdir(REPO)

app = FastAPI()
_state = {"tts": None, "err": None}
# infer() mutates per-speaker caches on the model object; serialize requests.
_lock = threading.Lock()

_LANGS = {"zh", "en", "ja", "es", "ar"}
_ALIASES = {
    "zh-cn": "zh", "zh_cn": "zh", "zh-hans": "zh", "cmn": "zh", "chinese": "zh", "mandarin": "zh",
    "en-us": "en", "en_us": "en", "en-gb": "en", "english": "en",
    "ja-jp": "ja", "japanese": "ja", "es-es": "es", "spanish": "es", "arabic": "ar",
}
_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")


@app.on_event("startup")
def _load():
    try:
        import torchaudio as _ta

        def _ta_save_sf(filepath, src, sample_rate, *args, **kwargs):
            arr = src.detach().cpu().numpy() if hasattr(src, "detach") else np.asarray(src)
            if arr.ndim == 2 and arr.shape[0] <= arr.shape[1]:
                arr = arr.T
            sf.write(str(filepath), arr, int(sample_rate), format="WAV", subtype="PCM_16")

        _ta.save = _ta_save_sf
        from indextts.infer_v2_5 import IndexTTS2

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        tts = IndexTTS2(
            cfg_path=CFG_PATH,
            model_dir=MODEL_DIR,
            use_bf16=USE_BF16,
            device=dev,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=False,
        )
        _state["tts"] = tts
        print("[startup] loaded IndexTTS-2.5 dev=", dev, "bf16=", USE_BF16, flush=True)
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
    reference_audio_b64: str | None = None
    reference_audio_url: str | None = None
    instructions: str | None = None


@app.get("/health")
def health():
    if _state["err"]:
        return JSONResponse({"status": "error", "error": _state["err"]}, status_code=503)
    if _state["tts"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": "IndexTTS-2.5", "sample_rate": SR}


@app.get("/voices")
def voices():
    return {
        "default": DEFAULT_VOICE,
        "note": "voice cloning model; pass reference_audio_b64 / reference_audio_url (or a path inside the container) to clone a voice",
    }


def _materialize_reference(req: TTSReq) -> str | None:
    """Inline reference (base64 or http(s) URL) -> temp 16-bit wav path; caller unlinks it."""
    data = None
    if req.reference_audio_b64:
        raw = req.reference_audio_b64.strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
    elif req.reference_audio_url:
        url = req.reference_audio_url.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("reference_audio_url must be http(s)")
        rq = urllib.request.Request(url, headers={"User-Agent": "redqueen-tts/1.0"})
        with urllib.request.urlopen(rq, timeout=20) as resp:
            data = resp.read(_MAX_REF_BYTES + 1)
    if not data:
        return None
    if len(data) > _MAX_REF_BYTES:
        raise ValueError("reference audio too large (> %d bytes)" % _MAX_REF_BYTES)
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        arr = arr.mean(axis=1)
    except Exception:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tf.write(data)
            tmp_in = tf.name
        try:
            arr, sr = librosa.load(tmp_in, sr=None, mono=True)
        finally:
            os.unlink(tmp_in)
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    sf.write(out.name, arr, int(sr), format="WAV", subtype="PCM_16")
    return out.name


def _resolve_ref(req: TTSReq) -> str:
    if req.reference_audio and os.path.exists(req.reference_audio):
        return req.reference_audio
    if req.voice and os.path.exists(req.voice):
        return req.voice
    return DEFAULT_VOICE


def _resolve_lang(req: TTSReq) -> str:
    lang = (req.language or "").strip().lower()
    lang = _ALIASES.get(lang, lang)
    if lang in _LANGS:
        return lang
    if _KANA.search(req.text):
        return "ja"
    if _HAN.search(req.text):
        return "zh"
    if _ARABIC.search(req.text):
        return "ar"
    return "en"


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
        ref_tmp = _materialize_reference(req)
    except Exception as e:
        return JSONResponse({"error": f"bad reference_audio: {e}"}, status_code=400)
    try:
        ref = ref_tmp or _resolve_ref(req)
        lang = _resolve_lang(req)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            with _lock:
                t.infer(spk_audio_prompt=ref, text=req.text, output_path=out_path, lang=lang, verbose=False)
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
    finally:
        if ref_tmp and os.path.exists(ref_tmp):
            os.remove(ref_tmp)

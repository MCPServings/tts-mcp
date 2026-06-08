"""IndexTTS-2 backend server (FastAPI).

Voice-cloning + emotion-control TTS. Optimized for throughput on a single GPU:

  * use_fp16=True            — half precision (~2x faster, less VRAM)
  * use_accel=True           — IndexTTS2's vLLM-style accel engine (paged KV
                               cache + CUDA/HIP graphs + internal segment
                               batching). Falls back to the plain path if it
                               fails to initialize on this GPU.
  * use_cuda_kernel=False    — BigVGAN fused CUDA kernel is NVIDIA-only.
  * a bounded thread pool + semaphore lets concurrent HTTP requests run without
    serializing on the event loop while capping simultaneous GPU work.

Inline reference audio (base64 / http(s) URL) is materialized into a per-port
scratch dir and removed after inference (orphans purged on startup).

Env:
  INDEXTTS_REPO, INDEXTTS_MODEL_DIR, INDEXTTS_CFG, INDEXTTS_DEFAULT_VOICE
  INDEXTTS_FP16=1, INDEXTTS_ACCEL=1, INDEXTTS_MAX_CONCURRENCY=2
  INDEXTTS_MAX_TEXT_TOKENS=120, INDEXTTS_MAX_REF_BYTES, PORT=8227
"""
import asyncio
import base64
import io
import os
import tempfile
import threading
import urllib.request

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# torchaudio 2.9 routes torchaudio.save through torchcodec (not available on this
# ROCm image). IndexTTS calls torchaudio.save(path, wav_int16, sr) internally;
# replace it with a soundfile-based writer covering that call shape.
import torchaudio as _ta


def _ta_save_sf(filepath, src, sample_rate, *args, **kwargs):
    arr = src.detach().cpu().numpy() if hasattr(src, "detach") else np.asarray(src)
    # torchaudio uses (channels, frames); soundfile wants (frames, channels)
    if arr.ndim == 2:
        arr = arr.T
    subtype = "PCM_16" if np.issubdtype(arr.dtype, np.integer) else None
    sf.write(str(filepath), arr, int(sample_rate), subtype=subtype)


_ta.save = _ta_save_sf

REPO = os.environ.get("INDEXTTS_REPO", "/opt/tts/indextts2/index-tts")
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")
DEFAULT_VOICE = os.environ.get("INDEXTTS_DEFAULT_VOICE", "examples/voice_01.wav")
PORT = int(os.environ.get("PORT", "8227"))
USE_FP16 = os.environ.get("INDEXTTS_FP16", "1") == "1"
# accel = IndexTTS2's vLLM-style engine. On gfx1100 (ROCm) its sampler does NOT
# honor the mel stop-token and generates a full ~30s clip for every request,
# making it both wrong and slower; keep it OFF by default here. Flip
# INDEXTTS_ACCEL=1 only on a GPU where it's been verified to stop correctly.
USE_ACCEL = os.environ.get("INDEXTTS_ACCEL", "0") == "1"
# torch.compile the s2mel flow-matching estimator (the diffusion decoder).
USE_TORCH_COMPILE = os.environ.get("INDEXTTS_TORCH_COMPILE", "0") == "1"
MAX_CONCURRENCY = int(os.environ.get("INDEXTTS_MAX_CONCURRENCY", "2"))
MAX_TEXT_TOKENS = int(os.environ.get("INDEXTTS_MAX_TEXT_TOKENS", "120"))
MAX_REF_BYTES = int(os.environ.get("INDEXTTS_MAX_REF_BYTES", str(15 * 1024 * 1024)))
SR = 22050

# IndexTTS2 uses relative paths (./checkpoints/...) -> run from repo root.
os.chdir(REPO)

# Per-port scratch dir for inline reference audio; purge orphans on startup.
REF_TMP_DIR = os.environ.get("INDEXTTS_REF_TMP_DIR", "/tmp/idx_ref/%d" % PORT)
os.makedirs(REF_TMP_DIR, exist_ok=True)
for _orphan in os.listdir(REF_TMP_DIR):
    try:
        os.unlink(os.path.join(REF_TMP_DIR, _orphan))
    except OSError:
        pass

app = FastAPI()
_state = {"tts": None, "err": None, "accel": False}
# Cap simultaneous GPU inference; run blocking infer() off the event loop.
_SEM = asyncio.Semaphore(MAX_CONCURRENCY)
_GPU_LOCK = threading.Lock()  # serialize the non-reentrant model when needed


@app.on_event("startup")
def _load():
    try:
        from indextts.infer_v2 import IndexTTS2

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        accel = USE_ACCEL and torch.cuda.is_available()
        try:
            tts = IndexTTS2(
                cfg_path=CFG_PATH,
                model_dir=MODEL_DIR,
                use_fp16=USE_FP16,
                use_cuda_kernel=False,
                use_deepspeed=False,
                use_accel=accel,
                use_torch_compile=USE_TORCH_COMPILE,
                device=dev,
            )
            _state["accel"] = accel
        except Exception as e:
            # accel engine failed to init (e.g. HIP graph capture) -> retry plain
            import traceback
            traceback.print_exc()
            print(f"[startup] accel init failed ({e}); retrying without accel", flush=True)
            tts = IndexTTS2(
                cfg_path=CFG_PATH,
                model_dir=MODEL_DIR,
                use_fp16=USE_FP16,
                use_cuda_kernel=False,
                use_deepspeed=False,
                use_accel=False,
                use_torch_compile=USE_TORCH_COMPILE,
                device=dev,
            )
            _state["accel"] = False
        _state["tts"] = tts
        print(f"[startup] loaded IndexTTS2 dev={dev} fp16={USE_FP16} "
              f"accel={_state['accel']}", flush=True)
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
    reference_text: str | None = None
    # emotion control (IndexTTS-2)
    emo_audio_b64: str | None = None
    emo_audio_url: str | None = None
    emo_alpha: float | None = 1.0
    emo_text: str | None = None
    use_emo_text: bool | None = None
    instructions: str | None = None


@app.get("/health")
def health():
    if _state["err"]:
        return JSONResponse({"status": "error", "error": _state["err"]}, status_code=503)
    if _state["tts"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": "IndexTTS-2", "sample_rate": SR,
            "accel": _state["accel"], "fp16": USE_FP16}


@app.get("/voices")
def voices():
    return {
        "default": DEFAULT_VOICE,
        "note": "voice cloning + emotion control; pass reference_audio_b64/url to "
                "clone a voice, emo_audio_* or emo_text/use_emo_text for emotion",
    }


def _fetch_inline(b64: str | None, url: str | None) -> bytes | None:
    if b64:
        raw = b64.strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        return base64.b64decode(raw)
    if url:
        u = url.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            raise ValueError("reference url must be http(s)")
        rq = urllib.request.Request(u, headers={"User-Agent": "redqueen-tts/1.0"})
        with urllib.request.urlopen(rq, timeout=20) as resp:
            return resp.read(MAX_REF_BYTES + 1)
    return None


def _materialize(b64: str | None, url: str | None) -> str | None:
    """Decode inline audio to a temp wav in REF_TMP_DIR; caller unlinks it."""
    data = _fetch_inline(b64, url)
    if not data:
        return None
    if len(data) > MAX_REF_BYTES:
        raise ValueError("reference audio too large (> %d bytes)" % MAX_REF_BYTES)
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        arr = arr.mean(axis=1)
    except Exception:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir=REF_TMP_DIR) as tf:
            tf.write(data)
            tmp_in = tf.name
        try:
            arr, sr = librosa.load(tmp_in, sr=None, mono=True)
        finally:
            os.unlink(tmp_in)
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=REF_TMP_DIR)
    out.close()
    sf.write(out.name, arr, int(sr), format="WAV", subtype="PCM_16")
    return out.name


def _resolve_spk(req: TTSReq) -> tuple[str, str | None]:
    """Return (speaker_prompt_path, temp_to_unlink_or_None)."""
    tmp = _materialize(req.reference_audio_b64, req.reference_audio_url)
    if tmp:
        return tmp, tmp
    if req.reference_audio and os.path.exists(req.reference_audio):
        return req.reference_audio, None
    if req.voice and os.path.exists(req.voice):
        return req.voice, None
    return DEFAULT_VOICE, None


def _encode(arr: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    fmt = (fmt or "wav").lower()
    if fmt == "mp3":
        fmt = "wav"
    subtype = "PCM_16" if fmt == "wav" else None
    sf.write(buf, arr, sr, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


def _run_infer(req: TTSReq) -> bytes:
    t = _state["tts"]
    spk, spk_tmp = _resolve_spk(req)
    emo_tmp = None
    try:
        emo_tmp = _materialize(req.emo_audio_b64, req.emo_audio_url)
        kwargs = {}
        if emo_tmp:
            kwargs["emo_audio_prompt"] = emo_tmp
            kwargs["emo_alpha"] = req.emo_alpha if req.emo_alpha is not None else 1.0
        if req.use_emo_text or (req.emo_text and req.use_emo_text is None):
            kwargs["use_emo_text"] = True
            if req.emo_text:
                kwargs["emo_text"] = req.emo_text
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=REF_TMP_DIR) as tmp:
            out_path = tmp.name
        try:
            with _GPU_LOCK:
                t.infer(
                    spk_audio_prompt=spk, text=req.text, output_path=out_path,
                    max_text_tokens_per_segment=MAX_TEXT_TOKENS, verbose=False,
                    **kwargs,
                )
            arr, sr = sf.read(out_path, dtype="float32")
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
        return _encode(arr, sr, req.format or "wav")
    finally:
        for p in (spk_tmp, emo_tmp):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


@app.post("/tts")
async def tts(req: TTSReq):
    if _state["tts"] is None:
        return JSONResponse({"error": _state["err"] or "loading"}, status_code=503)
    if not req.text or not req.text.strip():
        return JSONResponse({"error": "text required"}, status_code=400)
    try:
        async with _SEM:
            data = await asyncio.get_event_loop().run_in_executor(None, _run_infer, req)
    except ValueError as e:
        return JSONResponse({"error": f"bad reference: {e}"}, status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    mt = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(
        (req.format or "wav").lower(), "audio/wav"
    )
    return Response(content=data, media_type=mt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)

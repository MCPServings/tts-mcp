import base64
import io
import os
import sys
import tempfile
import urllib.request

import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# Compat shim: diffusers 0.27.2 (pinned by CosyVoice) imports `cached_download`,
# which was removed from huggingface_hub >= 0.26. transformers 4.51.3 needs the
# newer hub, so we can't downgrade — alias the removed symbol instead.
import huggingface_hub as _hf
if not hasattr(_hf, "cached_download"):
    _hf.cached_download = _hf.hf_hub_download

# torchaudio 2.10 routes torchaudio.load through torchcodec (ignoring the
# deprecated backend= arg). torchcodec isn't available on this ROCm image, so
# replace torchaudio.load with a soundfile-based reader covering all internal
# CosyVoice uses (load_wav, spk2info prompts, etc.).
import torchaudio as _ta


def _ta_load_sf(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    # soundfile returns (frames, channels); torchaudio expects (channels, frames)
    wav = torch.from_numpy(data).T.contiguous()
    return wav, sr


_ta.load = _ta_load_sf

# repo + Matcha-TTS must be importable
REPO = os.environ.get("COSYVOICE_REPO", "/opt/CosyVoice")
for p in (REPO, os.path.join(REPO, "third_party", "Matcha-TTS")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_DIR = os.environ["COSYVOICE_MODEL_DIR"]
IS_V3 = os.environ.get("COSYVOICE_V3", "0") == "1"
PORT = int(os.environ.get("PORT", "8225"))
# default bundled reference for zero-shot cloning
PROMPT_WAV = os.environ.get(
    "COSYVOICE_PROMPT_WAV", os.path.join(REPO, "asset", "zero_shot_prompt.wav")
)
PROMPT_TEXT = os.environ.get(
    "COSYVOICE_PROMPT_TEXT",
    "希望你以后能够做的比我还好呦。",
)

# Scratch dir for inline reference audio (base64/URL), per-port so cosy2/cosy3
# never purge each other's files. Orphans left by a crash (where the request
# handler's finally-block never ran) are cleared on startup below.
REF_TMP_DIR = os.environ.get("COSYVOICE_REF_TMP_DIR", "/tmp/cosy_ref/%d" % PORT)
os.makedirs(REF_TMP_DIR, exist_ok=True)
for _orphan in os.listdir(REF_TMP_DIR):
    try:
        os.unlink(os.path.join(REF_TMP_DIR, _orphan))
    except OSError:
        pass

app = FastAPI()
_state = {"model": None, "prompt16k": None, "sr": 24000, "err": None}


@app.on_event("startup")
def _load():
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3
        from cosyvoice.utils.file_utils import load_wav

        if IS_V3:
            # CosyVoice3.__init__ has no load_jit kwarg
            m = CosyVoice3(MODEL_DIR, load_trt=False, fp16=False)
        else:
            m = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
        # This CosyVoice API revision takes the prompt as a wav *path*, not a tensor.
        _state["model"] = m
        _state["prompt16k"] = PROMPT_WAV
        _state["sr"] = int(getattr(m, "sample_rate", 24000))
        print(
            "[startup] loaded", MODEL_DIR, "sr=", _state["sr"], "v3=", IS_V3, flush=True
        )
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
    instructions: str | None = None


@app.get("/health")
def health():
    if _state["err"]:
        return JSONResponse({"status": "error", "error": _state["err"]}, status_code=503)
    if _state["model"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_DIR, "sample_rate": _state["sr"]}


@app.get("/voices")
def voices():
    return {"default": "zero-shot (bundled reference voice)"}


def _encode(wav: torch.Tensor, sr: int, fmt: str) -> bytes:
    arr = wav.squeeze().detach().cpu().numpy()
    buf = io.BytesIO()
    fmt = (fmt or "wav").lower()
    if fmt == "mp3":
        # soundfile has no mp3 encode universally; fall back to wav
        fmt = "wav"
    subtype = "PCM_16" if fmt == "wav" else None
    sf.write(buf, arr, sr, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


_MAX_REF_BYTES = int(os.environ.get("COSYVOICE_MAX_REF_BYTES", str(15 * 1024 * 1024)))


def _materialize_reference(req: "TTSReq") -> str | None:
    """Decode inline reference audio (base64 or http(s) URL) into a temp 16-bit
    wav and return its path; None when no inline audio is supplied. Raises
    ValueError on oversized / malformed input. Caller must unlink the path."""
    data = None
    if req.reference_audio_b64:
        raw = req.reference_audio_b64.strip()
        # tolerate data URLs: "data:audio/wav;base64,...."
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
    elif req.reference_audio_url:
        url = req.reference_audio_url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("reference_audio_url must be http(s)")
        rq = urllib.request.Request(url, headers={"User-Agent": "redqueen-tts/1.0"})
        with urllib.request.urlopen(rq, timeout=20) as resp:
            data = resp.read(_MAX_REF_BYTES + 1)
    if not data:
        return None
    if len(data) > _MAX_REF_BYTES:
        raise ValueError("reference audio too large (> %d bytes)" % _MAX_REF_BYTES)
    # soundfile handles wav/flac/ogg directly; fall back to librosa for mp3/m4a/etc.
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


@app.post("/tts")
def tts(req: TTSReq):
    m = _state["model"]
    if m is None:
        return JSONResponse({"error": _state["err"] or "loading"}, status_code=503)
    ref_tmp = None
    try:
        # prompt is a wav *path* in this CosyVoice API revision
        prompt16k = _state["prompt16k"]
        prompt_text = PROMPT_TEXT
        # custom reference voice: inline base64/URL takes priority, then a
        # server-local path (back-compat). reference_text should carry the
        # transcript of the prompt audio for best zero-shot cloning.
        try:
            ref_tmp = _materialize_reference(req)
        except Exception as e:
            return JSONResponse(
                {"error": f"bad reference_audio: {e}"}, status_code=400
            )
        if ref_tmp:
            prompt16k = ref_tmp
            if req.reference_text:
                prompt_text = req.reference_text
        elif req.reference_audio and os.path.exists(req.reference_audio):
            prompt16k = req.reference_audio
            if req.reference_text:
                prompt_text = req.reference_text

        text = req.text
        if IS_V3 and "<|endofprompt|>" not in text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

        chunks = []
        if req.instructions:
            gen = m.inference_instruct2(
                text, req.instructions, prompt16k, stream=False,
                speed=req.speed or 1.0, text_frontend=False,
            )
        else:
            gen = m.inference_zero_shot(
                text, prompt_text, prompt16k, stream=False,
                speed=req.speed or 1.0, text_frontend=False,
            )
        for out in gen:
            chunks.append(out["tts_speech"])
        wav = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        data = _encode(wav, _state["sr"], req.format or "wav")
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
            try:
                os.unlink(ref_tmp)
            except OSError:
                pass

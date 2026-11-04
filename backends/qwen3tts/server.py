import base64, io, os, tempfile, time, urllib.request
from typing import Optional
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
# CustomVoice only has preset speakers; cloning needs the Base checkpoint ("" disables cloning).
BASE_MODEL_ID = os.environ.get("QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
PORT = int(os.environ.get("PORT", "8223"))
DEFAULT_SPK = os.environ.get("QWEN_TTS_SPEAKER", "Ryan")
_MAX_REF_BYTES = int(os.environ.get("QWEN_TTS_MAX_REF_BYTES", str(15 * 1024 * 1024)))

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
    kwargs = {}
    if dev != "cpu":
        kwargs.update(device_map=dev, dtype=torch.bfloat16, attn_implementation="sdpa")
    m = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
    state["device"] = dev
    state["model"] = m
    try:
        state["speakers"] = list(m.get_supported_speakers())
    except Exception:
        state["speakers"] = []
    try:
        state["languages"] = list(m.get_supported_languages())
    except Exception:
        state["languages"] = []
    if BASE_MODEL_ID:
        state["base"] = Qwen3TTSModel.from_pretrained(BASE_MODEL_ID, **kwargs)
    print("[startup] loaded", MODEL_ID, "base=", BASE_MODEL_ID or None, "spk=", state["speakers"], flush=True)


class TtsReq(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "auto"
    format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    instructions: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_audio_b64: Optional[str] = None
    reference_audio_url: Optional[str] = None
    reference_text: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok" if "model" in state else "loading",
        "model": MODEL_ID,
        "clone": "base" in state,
        "speakers": state.get("speakers", []),
        "languages": state.get("languages", []),
    }


@app.get("/voices")
def voices():
    return {"speakers": state.get("speakers", []), "languages": state.get("languages", [])}


def _materialize_reference(req: TtsReq) -> Optional[str]:
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
        ref_tmp = _materialize_reference(req)
    except Exception as e:
        raise HTTPException(400, f"bad reference_audio: {e}")
    ref = ref_tmp or (req.reference_audio or "").strip()
    try:
        if ref:
            if not os.path.exists(ref):
                raise HTTPException(400, f"reference_audio not found: {ref}")
            base = state.get("base")
            if base is None:
                raise HTTPException(501, "voice cloning unavailable: no Base model loaded")
            ref_text = (req.reference_text or "").strip() or None
            # Without a transcript only the speaker embedding is usable (no ICL).
            wavs, sr = base.generate_voice_clone(
                text=req.text, language=lang, ref_audio=ref, ref_text=ref_text,
                x_vector_only_mode=ref_text is None, non_streaming_mode=True,
            )
        else:
            wavs, sr = m.generate_custom_voice(
                text=req.text, language=lang, speaker=spk, instruct=req.instructions or "",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"synthesis failed: {e}")
    finally:
        if ref_tmp and os.path.exists(ref_tmp):
            os.remove(ref_tmp)
    # speed 此前被静默忽略；用变速不变调的时间伸缩实现
    try:
        s = float(req.speed or 1.0)
    except Exception:
        s = 1.0
    if abs(s - 1.0) > 0.01:
        import librosa
        import numpy as np
        wavs = [librosa.effects.time_stretch(np.asarray(w, dtype="float32"), rate=s)
                for w in wavs]
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Latency-ms": str(round((time.time() - t0) * 1000, 1)),
                             "X-Speaker": spk,
                             "X-Mode": ("clone" if ref else "custom_voice")})

import io
import os

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

MODEL_ID = os.environ.get("ORPHEUS_MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
SNAC_ID = os.environ.get("ORPHEUS_SNAC_ID", "hubertsiuzdak/snac_24khz")
PORT = int(os.environ.get("PORT", "8228"))
DEFAULT_VOICE = os.environ.get("ORPHEUS_DEFAULT_VOICE", "tara")
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
SR = 24000

# Orpheus special tokens (canonical, from canopylabs model card)
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
CODE_OFFSET = 128266

app = FastAPI()
_state = {"model": None, "tok": None, "snac": None, "dev": None, "err": None}


@app.on_event("startup")
def _load():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
        model.to(dev)
        model.eval()
        snac = SNAC.from_pretrained(SNAC_ID).eval().to(dev)
        _state.update(model=model, tok=tok, snac=snac, dev=dev)
        print("[startup] loaded", MODEL_ID, "dev=", dev, flush=True)
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
    return {"status": "ok", "model": MODEL_ID, "sample_rate": SR}


@app.get("/voices")
def voices():
    return {"voices": VOICES, "default": DEFAULT_VOICE}


def _redistribute_codes(code_list, dev):
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(len(code_list) // 7):
        b = code_list[7 * i : 7 * i + 7]
        layer_1.append(b[0])
        layer_2.append(b[1] - 4096)
        layer_3.append(b[2] - 2 * 4096)
        layer_3.append(b[3] - 3 * 4096)
        layer_2.append(b[4] - 4 * 4096)
        layer_3.append(b[5] - 5 * 4096)
        layer_3.append(b[6] - 6 * 4096)
    codes = [
        torch.tensor(layer_1, device=dev).unsqueeze(0),
        torch.tensor(layer_2, device=dev).unsqueeze(0),
        torch.tensor(layer_3, device=dev).unsqueeze(0),
    ]
    return codes


def _encode(arr: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    fmt = (fmt or "wav").lower()
    if fmt == "mp3":
        fmt = "wav"
    subtype = "PCM_16" if fmt == "wav" else None
    sf.write(buf, arr, SR, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


@app.post("/tts")
def tts(req: TTSReq):
    model = _state["model"]
    if model is None:
        return JSONResponse({"error": _state["err"] or "loading"}, status_code=503)
    try:
        tok = _state["tok"]
        snac = _state["snac"]
        dev = _state["dev"]
        voice = (req.voice or DEFAULT_VOICE).lower()
        if voice not in VOICES:
            voice = DEFAULT_VOICE

        prompt = f"{voice}: {req.text}"
        input_ids = tok(prompt, return_tensors="pt").input_ids
        start = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        ids = torch.cat([start, input_ids, end], dim=1).to(dev)
        attn = torch.ones_like(ids)

        with torch.inference_mode():
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=1800,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH,
                pad_token_id=END_OF_SPEECH,
            )

        # crop to tokens after the last START_OF_SPEECH marker
        row = gen[0]
        sos_idx = (row == START_OF_SPEECH).nonzero(as_tuple=True)[0]
        if len(sos_idx) > 0:
            row = row[sos_idx[-1].item() + 1 :]
        # drop END_OF_SPEECH markers
        row = row[row != END_OF_SPEECH]
        n = (row.shape[0] // 7) * 7
        if n == 0:
            return JSONResponse({"error": "no audio codes generated"}, status_code=500)
        code_list = (row[:n] - CODE_OFFSET).tolist()
        codes = _redistribute_codes(code_list, dev)

        with torch.inference_mode():
            audio = snac.decode(codes)
        arr = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        data = _encode(arr, req.format or "wav")
        mt = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(
            (req.format or "wav").lower(), "audio/wav"
        )
        return Response(content=data, media_type=mt)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

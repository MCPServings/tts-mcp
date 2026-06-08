#!/bin/bash
# Qwen3-TTS on the ROCm image. Reuse system ROCm torch/torchaudio (do NOT create
# a venv: /opt/python is a uv python whose venv shadows torch). Install qwen-tts +
# pinned transformers into a persistent --target dir prepended via PYTHONPATH so
# it wins over the system transformers 5.x.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PORT=${PORT:-8223}
PYD=/work/pydeps
export PYTHONPATH="$PYD"

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] installing qwen-tts deps into $PYD ..."
  mkdir -p "$PYD"
  echo "numpy==2.1.3" > /work/constraints.txt
  python -m pip install --target="$PYD" --no-deps qwen-tts >/tmp/pip_qwen.log 2>&1
  python -m pip install --target="$PYD" -c /work/constraints.txt \
      transformers==4.57.3 librosa soundfile sox onnxruntime >/tmp/pip_deps.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[bootstrap] pip failed rc=$rc"; tail -8 /tmp/pip_deps.log
    sleep 3600; exit 1
  fi
  echo "[bootstrap] downloading weights ..."
  python - <<PY >/tmp/dl.log 2>&1
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz")
PY
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[bootstrap] weight download failed rc=$rc"; tail -8 /tmp/dl.log
    sleep 3600; exit 1
  fi
  touch /work/.ready
  echo "[bootstrap] done"
fi

exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

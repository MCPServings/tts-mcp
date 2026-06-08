#!/bin/bash
# Idempotent bootstrap + serve for canopylabs Orpheus-3B TTS. NO global 'set -e'.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export PORT=${PORT:-8228}
export PYTHONPATH=/work/pydeps
# The rocm image sets PIP_EXTRA_INDEX_URL to the AMD prerelease index, which serves
# corrupt/mismatched artifacts. Also the international link to pypi.org (this host is on a
# China network) corrupts large downloads -> use the Tsinghua domestic mirror.
unset PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
PYD=/work/pydeps

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] orpheus first run..."
  mkdir -p "$PYD"
  # snac touches torch -> install --no-deps; supply only the pure leaves it needs.
  # Rely on system transformers/torch/numpy/safetensors from the rocm/vllm image.
  python -m pip install --no-cache-dir --target="$PYD" --no-deps snac >/tmp/pip_snac.log 2>&1
  rc_snac=$?
  python -m pip install --no-cache-dir --target="$PYD" \
    hf_transfer einops soundfile fastapi "uvicorn[standard]" pydantic >/tmp/pip_deps.log 2>&1
  rc_deps=$?
  if [ $(( rc_snac + rc_deps )) -ne 0 ]; then
    echo "[bootstrap] pip failed (snac=$rc_snac deps=$rc_deps); tails:"
    tail -8 /tmp/pip_snac.log; tail -8 /tmp/pip_deps.log
    echo "[bootstrap] sleep 3600"; sleep 3600; exit 1
  fi
  touch /work/.ready
  echo "[bootstrap] done"
fi

# Prefetch weights with a resumable retry loop (the flaky egress stalls inline
# from_pretrained downloads). hf_transfer does chunked, resumable, parallel fetch.
python -c "import hf_transfer" 2>/dev/null || \
  python -m pip install --no-cache-dir --target="$PYD" hf_transfer >/tmp/pip_hft.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_ID=${ORPHEUS_MODEL_ID:-unsloth/orpheus-3b-0.1-ft}
if [ ! -f /work/.model_ready ]; then
  echo "[prefetch] downloading $MODEL_ID + SNAC..."
  n=0
  while [ $n -lt 20 ]; do
    python - "$MODEL_ID" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
mid = sys.argv[1]
snapshot_download(mid, allow_patterns=["*.safetensors","*.bin","*.json","*.model","tokenizer*","*.txt"])
snapshot_download("hubertsiuzdak/snac_24khz")
print("PREFETCH_OK")
PYEOF
    [ $? -eq 0 ] && { touch /work/.model_ready; break; }
    n=$((n+1)); echo "[prefetch] attempt $n failed, resuming in 5s..."; sleep 5
  done
  if [ ! -f /work/.model_ready ]; then
    echo "[prefetch] giving up after $n attempts"; sleep 3600; exit 1
  fi
  echo "[prefetch] done"
fi

exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

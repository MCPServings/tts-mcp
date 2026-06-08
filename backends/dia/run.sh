#!/bin/bash
# Idempotent bootstrap + serve for nari-labs Dia-1.6B. NO global 'set -e'.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export PORT=${PORT:-8229}
export PYTHONPATH=/work/pydeps
# The rocm image sets PIP_EXTRA_INDEX_URL to the AMD prerelease index, which serves
# corrupt/mismatched artifacts. Also the international link to pypi.org (this host is on a
# China network) corrupts large downloads -> use the Tsinghua domestic mirror.
unset PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
PYD=/work/pydeps

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] dia first run..."
  apt-get update >/tmp/apt.log 2>&1
  apt-get install -y --no-install-recommends git build-essential >>/tmp/apt.log 2>&1

  mkdir -p "$PYD"
  # retry wrapper: the halo-office egress occasionally corrupts large pip downloads
  # (hash mismatch). --no-cache-dir avoids reusing a corrupt cached file; retry a few times.
  pipretry() {
    local log=$1; shift
    local n=0
    while [ $n -lt 5 ]; do
      python -m pip install --no-cache-dir --retries 10 "$@" >"$log" 2>&1 && return 0
      n=$((n+1)); echo "[bootstrap] pip attempt $n failed for $log, retrying..."; sleep 5
    done
    return 1
  }
  # torch-touching packages installed --no-deps (else pip pulls CUDA torch + 2GB nvidia wheels).
  # Rely on system torch/torchaudio/numpy/safetensors from the rocm/vllm image.
  pipretry /tmp/pip_dia.log --target="$PYD" --no-deps \
    "git+https://github.com/nari-labs/dia.git" \
    descript-audio-codec descript-audiotools julius
  rc_dia=$?
  # remaining pure-python deps for dac + audiotools + dia leaves.
  # Pin numpy/numba to versions WITH cp313 wheels (match system numpy 2.1.3) so the
  # librosa->numba chain never tries to source-build numpy (no cp313 sdist on py3.13).
  pipretry /tmp/pip_deps.log --target="$PYD" \
    "numpy==2.1.3" "numba==0.61.2" \
    soundfile pydantic huggingface_hub hf_transfer einops \
    argbind ffmpy flatten-dict importlib-resources pyloudnorm \
    scipy librosa rich markdown2 randomname protobuf tqdm tensorboard \
    matplotlib fastapi "uvicorn[standard]"
  rc_deps=$?
  if [ $(( rc_dia + rc_deps )) -ne 0 ]; then
    echo "[bootstrap] pip failed (dia=$rc_dia deps=$rc_deps); tails:"
    tail -8 /tmp/pip_dia.log; tail -8 /tmp/pip_deps.log
    echo "[bootstrap] sleep 3600 to avoid crash-loop"; sleep 3600; exit 1
  fi
  touch /work/.ready
  echo "[bootstrap] done"
fi

# Prefetch the Dia weights with a resumable retry loop (inline from_pretrained over the
# flaky international link can hang indefinitely). hf_transfer = chunked/resumable.
python -c "import hf_transfer" 2>/dev/null || \
  python -m pip install --no-cache-dir --target="$PYD" hf_transfer >/tmp/pip_hft.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_ID=${DIA_MODEL_ID:-nari-labs/Dia-1.6B-0626}
if [ ! -f /work/.model_ready ]; then
  echo "[prefetch] downloading $MODEL_ID..."
  n=0
  while [ $n -lt 20 ]; do
    python - "$MODEL_ID" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
print("PREFETCH_OK")
PYEOF
    [ $? -eq 0 ] && { touch /work/.model_ready; break; }
    n=$((n+1)); echo "[prefetch] attempt $n failed, resuming in 5s..."; sleep 5
  done
  [ -f /work/.model_ready ] || { echo "[prefetch] gave up"; sleep 3600; exit 1; }
  echo "[prefetch] done"
fi

exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

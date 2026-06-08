#!/bin/bash
# Lightweight Kokoro ONNX backend. NO global set -e.
cd /work
export PORT=${PORT:-8221}
export HF_HOME=/work/hf
export PYTHONPATH=/work/pydeps
unset PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
PYD=/work/pydeps
MODELD=/work/models
MODEL=$MODELD/kokoro-v1.0.onnx
VOICES=$MODELD/voices-v1.0.bin

pipretry() {
  log=$1; shift
  n=0
  while [ $n -lt 5 ]; do
    python -m pip install --no-cache-dir --retries 10 "$@" >"$log" 2>&1 && return 0
    n=$((n+1)); echo "[bootstrap] pip attempt $n failed for $log, retrying..."; sleep 5
  done
  return 1
}

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] kokoro-onnx first run..."
  mkdir -p "$PYD" "$MODELD"
  apt-get update >/tmp/apt.log 2>&1
  apt-get install -y --no-install-recommends ca-certificates curl libsndfile1 >>/tmp/apt.log 2>&1
  pipretry /tmp/pip_deps.log --target="$PYD" \
    "kokoro-onnx==0.5.0" soundfile fastapi "uvicorn[standard]" pydantic
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[bootstrap] pip failed:"; tail -40 /tmp/pip_deps.log
    echo "[bootstrap] sleep 3600"; sleep 3600; exit 1
  fi
  touch /work/.ready
  echo "[bootstrap] done"
fi

if [ ! -f "$MODEL" ]; then
  echo "[download] kokoro-v1.0.onnx..."
  n=0
  while [ $n -lt 8 ]; do
    curl -L --retry 5 --retry-delay 5 -o "$MODEL.tmp" \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
      && mv "$MODEL.tmp" "$MODEL" && break
    n=$((n+1)); echo "[download] model attempt $n failed"; sleep 5
  done
fi
if [ ! -f "$VOICES" ]; then
  echo "[download] voices-v1.0.bin..."
  n=0
  while [ $n -lt 8 ]; do
    curl -L --retry 5 --retry-delay 5 -o "$VOICES.tmp" \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
      && mv "$VOICES.tmp" "$VOICES" && break
    n=$((n+1)); echo "[download] voices attempt $n failed"; sleep 5
  done
fi

if [ ! -f "$MODEL" ] || [ ! -f "$VOICES" ]; then
  echo "[download] missing model files"; ls -lh "$MODELD"
  echo "[download] sleep 3600"; sleep 3600; exit 1
fi

exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

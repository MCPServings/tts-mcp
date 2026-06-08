#!/bin/bash
# Idempotent bootstrap + serve for IndexTeam/IndexTTS-2. NO global 'set -e'.
# Relies on system torch/torchaudio/transformers/numpy/safetensors from the
# rocm/vllm image; only the missing leaves go into /work/pydeps.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export PORT=${PORT:-8227}
export PYTHONPATH=/work/pydeps:/work/index-tts
# numba's LLVM JIT segfaults during codegen on this gfx1151 box's CPU while librosa
# imports (@guvectorize compile). Force a generic CPU target so numba doesn't emit
# host-specific instructions that crash llvmlite. (DISABLE_JIT breaks librosa.normalize.)
export NUMBA_CPU_NAME=generic
export NUMBA_CPU_FEATURES=
export NUMBA_OPT=0
export NUMBA_LOOP_VECTORIZE=0
export NUMBA_SLP_VECTORIZE=0
# The rocm image sets PIP_EXTRA_INDEX_URL to the AMD prerelease index, which serves
# corrupt/mismatched artifacts. Also the international link to pypi.org (this host is on a
# China network) corrupts large downloads -> use the Tsinghua domestic mirror.
unset PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
PYD=/work/pydeps
REPO=/work/index-tts

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] indextts2 first run..."
  apt-get update >/tmp/apt.log 2>&1
  apt-get install -y --no-install-recommends git build-essential ffmpeg >>/tmp/apt.log 2>&1

  if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 https://github.com/index-tts/index-tts.git "$REPO" >/tmp/git.log 2>&1
  fi

  mkdir -p "$PYD"
  # retry wrapper: halo-office egress occasionally corrupts large pip downloads (hash mismatch).
  pipretry() {
    local log=$1; shift
    local n=0
    while [ $n -lt 5 ]; do
      python -m pip install --no-cache-dir --retries 10 "$@" >"$log" 2>&1 && return 0
      n=$((n+1)); echo "[bootstrap] pip attempt $n failed for $log, retrying..."; sleep 5
    done
    return 1
  }
  # torch-touching packages installed --no-deps to avoid CUDA torch pollution.
  # accelerate depends on torch -> also --no-deps (its other deps are system-provided).
  pipretry /tmp/pip_nodep.log --target="$PYD" --no-deps \
    vector-quantize-pytorch descript-audiotools descript-audio-codec accelerate julius torch-stoi
  rc_nodep=$?
  # pure-python deps. Pin numpy/numba to cp313-wheel versions (match system numpy 2.1.3)
  # so the librosa->numba chain never source-builds numpy. Rely on system torch/transformers.
  pipretry /tmp/pip_deps.log --target="$PYD" \
    "numpy==2.1.3" "numba==0.61.2" \
    omegaconf json5 munch einops einx argbind \
    librosa soundfile scipy \
    modelscope sentencepiece jieba cn2an pypinyin \
    wetext matplotlib tqdm \
    flatten-dict ffmpy pyloudnorm importlib-resources \
    randomname markdown2 rich protobuf docstring-parser \
    tensorboard fastapi "uvicorn[standard]" pydantic
  rc_deps=$?
  if [ $(( rc_nodep + rc_deps )) -ne 0 ]; then
    echo "[bootstrap] pip failed (nodep=$rc_nodep deps=$rc_deps); tails:"
    tail -10 /tmp/pip_nodep.log; tail -10 /tmp/pip_deps.log
    echo "[bootstrap] sleep 3600"; sleep 3600; exit 1
  fi

  # Never let target-installed PyPI torch/torchaudio shadow the ROCm build baked
  # into the container; a stale copy here breaks torchaudio with libtorch_hip.so.
  rm -rf "$PYD"/torch "$PYD"/torch-*.dist-info "$PYD"/torchgen \
      "$PYD"/torchaudio "$PYD"/torchaudio-*.dist-info \
      "$PYD"/triton "$PYD"/triton-*.dist-info \
      "$PYD"/nvidia* "$PYD"/cuda* \
      "$PYD"/numba "$PYD"/numba-*.dist-info \
      "$PYD"/llvmlite "$PYD"/llvmlite-*.dist-info

  # download IndexTTS-2 weights into repo checkpoints/
  python - <<'PYEOF' >/tmp/dl.log 2>&1
import os
from huggingface_hub import snapshot_download
snapshot_download("IndexTeam/IndexTTS-2", local_dir="/work/index-tts/checkpoints",
                  allow_patterns=None)
print("download ok")
PYEOF
  rc_dl=$?
  if [ $rc_dl -ne 0 ]; then
    echo "[bootstrap] weight download failed:"; tail -15 /tmp/dl.log
    echo "[bootstrap] sleep 3600"; sleep 3600; exit 1
  fi

  touch /work/.ready
  echo "[bootstrap] done"
fi

# IndexTTS2's vendored generation code targets transformers 4.52.1 (system image ships
# transformers 5.x which removed OffloadedCache etc). transformers has NO torch dep, so
# pinning it into pydeps (which precedes system on PYTHONPATH) is safe. Install everything
# --no-deps so pip never pulls numpy 2.4 (which breaks numba); rely on system numpy 2.1.3.
if [ ! -f /work/.tfpin_done ]; then
  echo "[tfpin] installing transformers==4.52.1 tokenizers==0.21.0 (no-deps) into pydeps..."
  # nuke any half-overwritten numpy in pydeps (dual dist-info -> import segfault); the
  # system image already ships a clean numpy 2.1.3 which will be used instead.
  rm -rf "$PYD"/numpy "$PYD"/numpy-*.dist-info "$PYD"/numpy.libs
  tfok=1
  for spec in "transformers==4.52.1 tokenizers==0.21.0" "accelerate==1.8.1"; do
    n=0; got=0
    while [ $n -lt 5 ]; do
      python -m pip install --no-cache-dir --retries 10 --target="$PYD" --upgrade --no-deps $spec >>/tmp/pip_tf.log 2>&1 && { got=1; break; }
      n=$((n+1)); echo "[tfpin] attempt $n failed for [$spec], retrying..."; sleep 5
    done
    [ $got -eq 1 ] || tfok=0
  done
  if [ $tfok -eq 1 ] && python -c "import numpy,transformers,tokenizers,accelerate" 2>/dev/null; then
    echo "[tfpin] ok (numpy=$(python -c 'import numpy;print(numpy.__version__)') tf=$(python -c 'import transformers;print(transformers.__version__)'))"
    touch /work/.tfpin_done
  else
    echo "[tfpin] install or import failed; sleep 60 then let container restart to retry"
    sleep 60; exit 1
  fi
fi

# IndexTTS2 imports Linux text normalizers from tn.{chinese,english}. The real
# WeTextProcessing package depends on pynini/OpenFST, which has no cp313 wheel
# here and fails to build. Provide a minimal pass-through normalizer so the
# service can run; upstream text normalization can be revisited separately.
if [ ! -f "$PYD/tn/chinese/normalizer.py" ]; then
  mkdir -p "$PYD/tn/chinese" "$PYD/tn/english"
  touch "$PYD/tn/__init__.py" "$PYD/tn/chinese/__init__.py" "$PYD/tn/english/__init__.py"
  cat > "$PYD/tn/chinese/normalizer.py" <<'PYEOF'
class Normalizer:
    def __init__(self, *args, **kwargs):
        pass

    def normalize(self, text):
        return text
PYEOF
  cat > "$PYD/tn/english/normalizer.py" <<'PYEOF'
class Normalizer:
    def __init__(self, *args, **kwargs):
        pass

    def normalize(self, text):
        return text
PYEOF
fi

# Also clean on every start, not only during first bootstrap: /work/pydeps is a
# persistent volume and older failed installs may have left PyPI torch behind.
rm -rf "$PYD"/torch "$PYD"/torch-*.dist-info "$PYD"/torchgen \
  "$PYD"/torchaudio "$PYD"/torchaudio-*.dist-info \
  "$PYD"/triton "$PYD"/triton-*.dist-info \
  "$PYD"/nvidia* "$PYD"/cuda* \
  "$PYD"/numba "$PYD"/numba-*.dist-info \
  "$PYD"/llvmlite "$PYD"/llvmlite-*.dist-info

# server.py lives in /work and chdir()s into the repo itself at import time.
exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

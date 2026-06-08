#!/bin/bash
# Idempotent bootstrap + serve for CosyVoice2/3. NO global 'set -e' so a transient
# failure does not crash-loop the container before /work/.ready is written.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export PORT=${PORT:-8225}
export COSYVOICE_REPO=/work/CosyVoice
export PYTHONPATH=/work/pydeps:$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS
PYD=/work/pydeps

if [ ! -f /work/.ready ]; then
  echo "[bootstrap] cosyvoice first run..."
  apt-get update >/tmp/apt.log 2>&1
  apt-get install -y --no-install-recommends git git-lfs build-essential pkg-config sox libsox-dev unzip >>/tmp/apt.log 2>&1

  if [ ! -d "$COSYVOICE_REPO/.git" ]; then
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_REPO" >/tmp/clone.log 2>&1
    (cd "$COSYVOICE_REPO" && git submodule update --init --recursive >>/tmp/clone.log 2>&1)
  fi

  mkdir -p "$PYD"
  # reuse system ROCm torch/torchaudio: install cosyvoice deps WITHOUT torch.
  python -m pip install --target="$PYD" --upgrade pip >/tmp/pip0.log 2>&1
  # CRITICAL: many deps (lightning, torchmetrics, diffusers...) declare a torch
  # requirement; if pip is allowed to resolve it, it pulls a fresh CUDA torch +
  # 2GB of nvidia-* wheels, shadowing the system ROCm torch. So we install the
  # heavy/torch-touching packages with --no-deps and supply the *non-torch*
  # transitive deps explicitly. A constraints file pins torch to the installed
  # ROCm build so any stray resolution is satisfied by the system package.
  TVER=$(python -c "import torch;print(torch.__version__)")
  TAVER=$(python -c "import torchaudio;print(torchaudio.__version__)")
  cat > /work/constraints.txt <<EOF
torch==$TVER
torchaudio==$TAVER
EOF
  # NOTE: openai-whisper (transcription only) and WeTextProcessing (needs pynini,
  # compile-heavy) are intentionally omitted; text_frontend is skipped at inference.
  # Packages that pull torch -> installed --no-deps:
  python -m pip install --target="$PYD" --no-deps \
    lightning==2.2.4 pytorch-lightning==2.2.4 lightning-utilities torchmetrics==1.4.0 \
    diffusers==0.27.2 conformer==0.3.2 wetext >/tmp/pip1a.log 2>&1
  rc1a=$?
  # CosyVoice frontend.py does an unconditional `import whisper` (only uses
  # whisper.log_mel_spectrogram) and tokenizer.py does `from whisper.tokenizer
  # import Tokenizer`. The upstream openai-whisper sdist fails to build on this
  # image and drags CUDA deps, so we drop in a faithful light package shim.
  rm -f "$PYD/whisper.py"
  mkdir -p "$PYD/whisper"
  cp /work/whisper_stub.py "$PYD/whisper/__init__.py"
  cp /work/whisper_tokenizer_stub.py "$PYD/whisper/tokenizer.py"
  # Remaining pure-python / non-torch deps resolved normally (constrained):
  python -m pip install --target="$PYD" -c /work/constraints.txt \
    "numpy<2" "transformers==4.51.3" \
    hydra-core==1.3.2 omegaconf==2.3.0 \
    inflect==7.3.1 librosa==0.10.2 \
    onnxruntime modelscope HyperPyYAML==1.2.2 \
    soundfile fastapi "uvicorn[standard]" pydantic \
    einops antlr4-python3-runtime==4.9.3 gdown >/tmp/pip1.log 2>&1
  rc1=$(( $? + rc1a ))
  # extra non-torch CosyVoice deps imported at module load (flow_matching etc.)
  # Leave matplotlib/onnx unpinned so py3.13 gets prebuilt wheels (the pinned
  # 3.7.5 / 1.16.0 have no cp313 wheels and fail to build from source).
  python -m pip install --target="$PYD" -c /work/constraints.txt \
    matplotlib networkx==3.1 onnx pyworld==0.3.4 \
    rich wget grpcio grpcio-tools pyarrow tiktoken einx >/tmp/pip1b.log 2>&1
  python -m pip install --target="$PYD" --no-deps x-transformers==2.11.24 >/tmp/pip_xt.log 2>&1
  # HyperPyYAML 1.2.2 uses ruamel.yaml's Loader.max_depth which was removed in
  # ruamel.yaml >= 0.18; pin to a compatible older release.
  python -m pip install --target="$PYD" "ruamel.yaml==0.17.40" >/tmp/pip_ruamel.log 2>&1
  # pynini is optional (text normalization); attempt but ignore failure
  python -m pip install --target="$PYD" pynini==2.1.6 >/tmp/pip_pynini.log 2>&1
  if [ $rc1 -ne 0 ]; then
    echo "[bootstrap] core pip failed rc=$rc1; tail:"; tail -8 /tmp/pip1.log
    echo "[bootstrap] sleep 3600 to avoid crash-loop; inspect /tmp/*.log"
    sleep 3600; exit 1
  fi

  echo "[bootstrap] downloading weights..."
  HF_HUB_ETAG_TIMEOUT=30 PYTHONPATH="$PYD" python - <<PY >/tmp/dl.log 2>&1
import os
from huggingface_hub import snapshot_download
mid = os.environ["COSYVOICE_MODEL_ID"]
p = snapshot_download(mid)
print("OK", mid, p)
# symlink to stable model dir
import pathlib
dst = os.environ["COSYVOICE_MODEL_DIR"]
pathlib.Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
if os.path.islink(dst) or os.path.exists(dst):
    pass
else:
    os.symlink(p, dst)
print("DIR", dst)
PY
  if [ $? -ne 0 ]; then
    echo "[bootstrap] weight dl failed; tail:"; tail -10 /tmp/dl.log
    sleep 3600; exit 1
  fi
  touch /work/.ready
  echo "[bootstrap] done"
fi

exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

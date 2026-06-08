#!/bin/bash
# Idempotent bootstrap + serve. NO global 'set -e' so a transient failure does
# not crash-loop the container before /work/venv/.ready is written.
cd /work
export HF_HOME=/work/hf
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PORT=${PORT:-8222}

if [ ! -f /work/venv/.ready ]; then
  echo "[bootstrap] installing (first run)..."
  apt-get update >/tmp/apt.log 2>&1
  apt-get install -y --no-install-recommends build-essential git curl mecab libmecab-dev mecab-ipadic-utf8 pkg-config >>/tmp/apt.log 2>&1
  python -m venv /work/venv
  /work/venv/bin/pip install -U pip wheel >/tmp/pip0.log 2>&1
  # pin matched torch+torchaudio CPU pair (MeloTTS-compatible); mismatched versions break torchaudio _torchaudio.abi3.so load
  /work/venv/bin/pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu >/tmp/pip_torch.log 2>&1 \
    && /work/venv/bin/pip install "numpy<2" "git+https://github.com/myshell-ai/MeloTTS.git" fastapi 'uvicorn[standard]' soundfile >/tmp/pip1.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[bootstrap] pip failed rc=$rc; tails:"; tail -5 /tmp/pip_torch.log /tmp/pip1.log
    echo "[bootstrap] sleeping 3600s to avoid crash-loop; inspect /tmp/*.log"
    sleep 3600
    exit 1
  fi
  /work/venv/bin/python -m unidic download >/tmp/unidic.log 2>&1
  /work/venv/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')" >/tmp/nltk.log 2>&1
  touch /work/venv/.ready
  echo "[bootstrap] done"
fi

exec /work/venv/bin/uvicorn server:app --host 0.0.0.0 --port "$PORT"

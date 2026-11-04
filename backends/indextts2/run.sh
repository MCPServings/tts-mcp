#!/bin/bash
# Serve IndexTeam/IndexTTS-2.5 (HF rev c39ce5ba) with index-tts @ d9e41aac on ROCm.
# All Python deps are prebuilt into ./pydeps by build-pydeps.sh (torch/torchaudio come from the image).
# Aux models (w2v-bert-2.0, semantic codec, campplus, bigvgan) live in index-tts/checkpoints/hf_cache.
W=/work
cd "$W"
export HF_HOME=$W/hf
export PORT=${PORT:-8227}
export PYTHONPATH=$W/pydeps:$W/index-tts
# Same numba settings as the IndexTTS-2 service (generic CPU target for librosa's JIT).
export NUMBA_CPU_NAME=generic
export NUMBA_CPU_FEATURES=
export NUMBA_OPT=0
export NUMBA_LOOP_VECTORIZE=0
export NUMBA_SLP_VECTORIZE=0
exec python -m uvicorn server:app --host 0.0.0.0 --port "$PORT"

#!/bin/bash
# Build /work/pydeps for IndexTTS-2.5 on a rocm/pytorch image (py3.12, ROCm torch 2.10 from the image).
# Run once in a throwaway container with internet; the service container only reads pydeps.
set -uo pipefail
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL
PYD=/work/pydeps
mkdir -p "$PYD"
pipt() { python -m pip install --no-cache-dir --disable-pip-version-check --retries 10 --target="$PYD" "$@"; }
# torch-touching packages: never let pip pull a CUDA torch/triton
pipt --no-deps accelerate==1.8.1 descript-audiotools==0.7.2 descript-audio-codec vector-quantize-pytorch julius torch-stoi openai-whisper==20250625 || exit 1
pipt "numpy==2.1.3" "numba==0.61.2" "transformers==4.52.1" "tokenizers==0.21.0" "huggingface_hub==0.36.2" safetensors \
  omegaconf json5 munch einops einx argbind "librosa==0.10.2.post1" soundfile "modelscope==1.27.0" sentencepiece \
  jieba cn2an pypinyin matplotlib tqdm flatten-dict ffmpy pyloudnorm importlib-resources randomname markdown2 rich \
  protobuf docstring-parser tensorboard fastapi "uvicorn[standard]" pydantic regex filelock packaging PyYAML requests psutil \
  tiktoken more-itertools fugashi mecab-python3 unidic-lite textstat WeTextProcessing nemo-text-processing || exit 1
rm -rf "$PYD"/torch "$PYD"/torch-*.dist-info "$PYD"/torchgen "$PYD"/torchaudio "$PYD"/torchaudio-*.dist-info \
       "$PYD"/triton "$PYD"/triton-*.dist-info "$PYD"/nvidia* "$PYD"/cuda* "$PYD"/bin/torchrun
echo "== versions"
PYTHONPATH=$PYD python - <<'PY'
import importlib.metadata as md
for p in ("numpy", "numba", "torch", "torchaudio", "transformers", "tokenizers", "librosa", "pynini", "nemo_text_processing", "WeTextProcessing", "tiktoken", "fugashi", "mecab-python3", "modelscope", "accelerate"):
    try: print(f"   {p} {md.version(p)}")
    except Exception as e: print(f"   {p} MISSING")
PY
echo BUILD_DONE

#!/bin/bash
set -euo pipefail
W=${W:-/work}
COMMIT=d9e41aac89fd00b3d71497fddb287b7f24613712
HFREV=c39ce5ba981572cb187443877ff559dfb246ce63
mkdir -p "$W"
cd "$W"
if [ ! -d index-tts/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/index-tts/index-tts.git index-tts
fi
git -C index-tts -c advice.detachedHead=false checkout -q "$COMMIT"
echo "code: $(git -C index-tts log -1 --format='%H %cd %s')"
mkdir -p index-tts/checkpoints/qwen0.6bemo4-merge
cd index-tts/checkpoints
B="https://huggingface.co/IndexTeam/IndexTTS-2.5/resolve/$HFREV"
for f in LICENSE README.md config.yaml feat1.pt feat2.pt wav2vec2bert_stats.pt multilingual_zh_ja_yue_char_del.tiktoken s2mel.pth codec.pth gpt.pth \
         qwen0.6bemo4-merge/Modelfile qwen0.6bemo4-merge/added_tokens.json qwen0.6bemo4-merge/chat_template.jinja qwen0.6bemo4-merge/config.json \
         qwen0.6bemo4-merge/generation_config.json qwen0.6bemo4-merge/merges.txt qwen0.6bemo4-merge/model.safetensors qwen0.6bemo4-merge/special_tokens_map.json \
         qwen0.6bemo4-merge/tokenizer.json qwen0.6bemo4-merge/tokenizer_config.json qwen0.6bemo4-merge/vocab.json; do
  [ -s "$f" ] || curl -fsSL --retry 5 -o "$f.part" "$B/$f" && { [ -f "$f.part" ] && mv "$f.part" "$f" || true; }
done
echo "$HFREV" > HF_REVISION
python3 - <<'PY'
import json, ssl, urllib.request, hashlib, os
t = json.load(urllib.request.urlopen("https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5/tree/c39ce5ba981572cb187443877ff559dfb246ce63?recursive=true", timeout=60))
bad = 0
for x in t:
    if x.get("type") != "file" or x["path"] == ".gitattributes":
        continue
    p = x["path"]; size = os.path.getsize(p) if os.path.exists(p) else -1
    if size != x["size"]:
        print("SIZE MISMATCH", p, size, x["size"]); bad += 1; continue
    oid = (x.get("lfs") or {}).get("oid")
    if oid:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != oid:
            print("SHA MISMATCH", p); bad += 1
print("weights verified" if bad == 0 else f"{bad} problems")
PY
du -sh "$W/index-tts/checkpoints"
# Provide a default speaker reference at "$W/default_ref.wav" (server.py falls back to it).
echo FETCH_DONE

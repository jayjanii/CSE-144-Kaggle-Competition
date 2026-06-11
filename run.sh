#!/usr/bin/env bash
# run the whole thing: bash run.sh (from the repo root)
set -e

PY=$(command -v python || command -v python3 || true)
if [ -z "$PY" ]; then
  echo "no python found" >&2
  exit 1
fi

# on colab torch etc. are already there (don't reinstall, it breaks the gpu);
# anywhere else grab the full requirements.
if "$PY" -c "import google.colab" 2>/dev/null; then
  "$PY" -m pip install -q open_clip_torch
else
  "$PY" -m pip install -q -r requirements.txt
fi

"$PY" -m pip freeze > env.txt  # note what we actually ran with

# cache + output go local by default; point these at drive to reuse embeddings
GOPT=${GOPT:-cache/gopt}
SO=${SO:-cache/so400m512}
OUT=${OUT:-submission.csv}
PSEUDO=${PSEUDO:-0.85}  # self-training on; set PSEUDO=0 to turn off
GPU=${GPU:-auto}        # auto-detects a100/l4/t4; set GPU=a100 to force

"$PY" src/ensemble_probes.py \
  --backbone ViT-gopt-16-SigLIP2-384:webli:"$GOPT" \
  --backbone ViT-SO400M-16-SigLIP2-512:webli:"$SO" \
  --gpu "$GPU" \
  --pseudo "$PSEUDO" \
  --write "$OUT"

# figure for the report
"$PY" src/probe_curve.py --emb "$GOPT/train.npy" --out probe_curve.png

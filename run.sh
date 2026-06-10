#!/usr/bin/env bash
# End-to-end reproduction. Run from the repo root: `bash run.sh`
# Works on a Colab GPU runtime and on a local machine.
set -e

# Colab already ships torch, torchvision, scikit-learn, numpy, pillow and
# matplotlib with the correct CUDA build. Reinstalling torch from PyPI can
# break the GPU, so only install the two packages Colab is missing.
pip install -q open_clip_torch kagglehub

# Record the exact environment that produced this run.
pip freeze > env.txt

# The competition data downloads automatically via kagglehub on first run.
# This needs a Kaggle token (~/.kaggle/kaggle.json) and you must have joined
# the competition. Alternatively drop the data under ./data/{train,test}.

python ensemble_probes.py \
  --backbone ViT-gopt-16-SigLIP2-384:webli:cache/gopt \
  --backbone ViT-SO400M-16-SigLIP2-512:webli:cache/so400m512 \
  --names class_names.csv --C 10 \
  --write submission.csv

# Loss / accuracy figure for the report (reuses the cached gopt embeddings).
python probe_curve.py --emb cache/gopt/train.npy --out probe_curve.png

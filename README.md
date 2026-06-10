# CSE 144 Final — SigLIP-2 frozen-probe ensemble

Two SigLIP-2 backbones (gopt-384 and SO400M-512) are kept completely frozen and
used as feature extractors. Each backbone gives:

- a **logistic probe** trained on its frozen image features, and
- a **zero-shot text head** built from the class names.

The four members are averaged with weights tuned on a leak-free 5-fold OOF.

## Run

One command, end to end (Colab GPU runtime or local):

```bash
bash run.sh
```

On Colab, clone the repo, `cd` into it, then `!bash run.sh`. The script installs
only the packages Colab is missing (`open_clip_torch`, `kagglehub`) so it does
not disturb the preinstalled CUDA build of torch.

Or run the steps directly:

```bash
pip install -r requirements.txt

python ensemble_probes.py \
  --backbone ViT-gopt-16-SigLIP2-384:webli:cache/gopt \
  --backbone ViT-SO400M-16-SigLIP2-512:webli:cache/so400m512 \
  --names class_names.csv --C 10 \
  --write submission.csv
```

The data is pulled from Kaggle automatically on first run (needs a Kaggle token
and competition access). Image embeddings are cached per backbone, so reruns are
fast.

`class_names.csv` maps each class id to a readable label (`class_id,name`). It
was built by hand from the four source datasets and is required for the text
head.

## Loss / accuracy curve

```bash
python probe_curve.py --emb cache/gopt/train.npy --out probe_curve.png
```

## Reproducibility

- One seed (`42`, override with `--seed`) is applied to `random`, `numpy`,
  `torch`, and the sklearn solver; cuDNN runs in deterministic mode.
- The 5-fold split is a seeded `StratifiedKFold`, so the OOF scores and the
  searched member weights are identical on every run.
- Train/test images are loaded in a fixed order (sorted by class id and numeric
  filename), independent of the filesystem.
- The GPU encoding step is cached to `cache/<backbone>/{train,test}.npy`. Keep
  those files and the probe + ensemble are bit-for-bit reproducible on CPU;
  only the one-time encode depends on the GPU/driver.
- Every run prints the torch / sklearn / numpy versions. Capture the full
  environment with `pip freeze > env.txt`.

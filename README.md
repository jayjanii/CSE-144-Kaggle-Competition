# CSE 144 Final — SigLIP-2 frozen-probe ensemble

Two SigLIP-2 backbones (gopt-384 and SO400M-512) are kept completely frozen and
used as feature extractors. Each backbone gives:

- a **logistic probe** trained on its frozen image features, and
- a **zero-shot text head** built from the class names.

The four members are averaged with weights tuned on a leak-free 5-fold OOF.

## Layout

```
src/
  data.py            data download / dir resolution
  ensemble_probes.py builds the submission (probe + text ensemble)
  probe_curve.py     loss / accuracy figure for the report
class_names.csv      class id -> readable name (hand-built)
run.sh               one-command reproduction
requirements.txt
```

Run scripts from the repo root (e.g. `python src/ensemble_probes.py ...`) so the
relative `class_names.csv`, `cache/`, and `data/` paths resolve.

## Run

One command, end to end (Colab GPU runtime or local):

```bash
bash run.sh
```

`run.sh` installs what it needs for the environment it finds itself in: on Colab
it adds only the missing packages (`open_clip_torch`, `kagglehub`) so it does not
disturb the preinstalled CUDA build of torch; on a bare machine it installs the
full `requirements.txt`. On Colab, clone the repo, `cd` into it, then
`!bash run.sh`.

Or run the steps directly:

```bash
pip install -r requirements.txt

python src/ensemble_probes.py \
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
python src/probe_curve.py --emb cache/gopt/train.npy --out probe_curve.png
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

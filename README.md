# CSE 144 Final: SigLIP-2 frozen-probe ensemble

Two frozen SigLIP-2 backbones (gopt-384, SO400M-512). Each gives a logistic
probe and a zero-shot text head; these candidates are averaged with weights
tuned on a 5-fold OOF (weak members can drop to zero). See the report for
details.

## Run

```bash
bash run.sh
```

Or directly:

```bash
pip install -r requirements.txt
python src/ensemble_probes.py \
  --backbone ViT-gopt-16-SigLIP2-384:webli:cache/gopt \
  --backbone ViT-SO400M-16-SigLIP2-512:webli:cache/so400m512 \
  --names class_names.csv --C 10 --write submission.csv
```

Run from the repo root. The dataset ships in `data/` (`train/<class>/*.jpg`,
`test/*.jpg`), so no Kaggle download is needed; if `data/` is ever missing it
falls back to kagglehub (`~/.kaggle/kaggle.json`). `class_names.csv`
(`class_id,name`) was built by hand and feeds the text head.

Loss/accuracy figure:

```bash
python src/probe_curve.py --emb cache/gopt/train.npy --out probe_curve.png
```

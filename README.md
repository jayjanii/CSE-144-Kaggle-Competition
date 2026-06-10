# CSE 144 Final — SigLIP-2 frozen-probe ensemble

Two SigLIP-2 backbones (gopt-384 and SO400M-512) are kept completely frozen and
used as feature extractors. Each backbone gives:

- a **logistic probe** trained on its frozen image features, and
- a **zero-shot text head** built from the class names.

The four members are averaged with weights tuned on a leak-free 5-fold OOF.

## Run

```bash
pip install -r requirements.txt

python ensemble_probes.py \
  --backbone ViT-gopt-16-SigLIP2-384:webli:cache/gopt \
  --backbone ViT-SO400M-16-SigLIP2-512:webli:cache/so400m512 \
  --names class_names.csv --C 10 \
  --write submission.csv
```

The data is pulled from Kaggle automatically on first run. Image embeddings are
cached per backbone, so reruns are fast.

`class_names.csv` maps each class id to a readable label (`class_id,name`). It
was built by hand from the four source datasets and is required for the text
head.

## Loss / accuracy curve

```bash
python probe_curve.py --emb cache/gopt/train.npy --out probe_curve.png
```

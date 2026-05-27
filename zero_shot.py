"""Zero-shot CLIP/SigLIP classifier — no training required.

Reads class_names.csv (you fill in via inspect_classes.py), encodes prompts
for every class, encodes every test image, and writes submission_probs.csv
in the same format as inference.py — so it slots straight into ensemble.py.

For classes you couldn't identify, leave name blank — the row gets a uniform
prior (1/num_classes) so it doesn't bias the ensemble. Partial coverage is
still hugely valuable: a confident SigLIP zero-shot vote on the 70 classes
you DID identify will likely outvote your fine-tuned models on those.

Usage:
    pip install open_clip_torch
    python zero_shot.py \
        --names class_names.csv \
        --test-dir data/test \
        --out zero_shot/submission_probs.csv \
        --model ViT-SO400M-14-SigLIP-384 --pretrained webli \
        --templates "a photo of a {}.,a close-up photo of a {}.,a {} in the wild."
"""

import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image


def load_names(path, num_classes=100):
    """Return list[str] of length num_classes; blank means 'unknown'."""
    names = [""] * num_classes
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["class_id"])
            if i < num_classes:
                names[i] = (row.get("name") or "").strip()
    return names


def build_text_features(model, tokenizer, names, templates, device):
    """For each class, encode every template, L2-normalize, mean-pool, re-normalize.

    Standard 'prompt ensembling' from CLIP paper — robust to wording.
    """
    feats = []
    with torch.inference_mode():
        for name in names:
            if not name:
                feats.append(None)
                continue
            prompts = [t.format(name) for t in templates]
            toks = tokenizer(prompts).to(device)
            te = model.encode_text(toks)
            te = te / te.norm(dim=-1, keepdim=True)
            te = te.mean(0)
            te = te / te.norm()
            feats.append(te)
    # replace unknowns with the mean of all known feats (so they get near-uniform similarity)
    known = [f for f in feats if f is not None]
    fallback = torch.stack(known).mean(0)
    fallback = fallback / fallback.norm()
    feats = [f if f is not None else fallback for f in feats]
    return torch.stack(feats)  # [C, D]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", required=True, help="class_names.csv")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--out", default="zero_shot/submission_probs.csv")
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--model", default="ViT-SO400M-14-SigLIP-384")
    p.add_argument("--pretrained", default="webli")
    p.add_argument(
        "--templates",
        default="a photo of a {}.,a close-up photo of a {}.,a {} on a plain background.",
        help="Comma-separated prompt templates with {} for the class name"
    )
    args = p.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} ({args.pretrained}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    names = load_names(args.names, args.num_classes)
    n_known = sum(1 for n in names if n)
    print(f"Loaded {n_known}/{args.num_classes} class names from {args.names}")

    templates = [t.strip() for t in args.templates.split(",")]
    print(f"Using {len(templates)} prompt templates")

    text_feats = build_text_features(model, tokenizer, names, templates, device)

    fnames = sorted(
        [f for f in os.listdir(args.test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda f: int(os.path.splitext(f)[0]),
    )

    all_probs = []
    with torch.inference_mode():
        for i, fname in enumerate(fnames):
            img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            ie = model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            logits = (ie @ text_feats.T) * 100.0  # CLIP temperature scale
            probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()
            all_probs.append((fname, probs))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(fnames)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [str(c) for c in range(args.num_classes)])
        for fname, probs in all_probs:
            w.writerow([fname] + [f"{v:.6f}" for v in probs])
    print(f"\nWrote {out} ({len(all_probs)} rows) — slot into ensemble.py like any other member.")


if __name__ == "__main__":
    main()

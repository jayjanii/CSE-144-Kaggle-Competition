"""Cluster the 100 training classes into visual themes via CLIP embeddings.

Encodes one or more representative images per class with SigLIP, then either:
  (a) k-means clusters them into K groups (you pick K based on how many source
      datasets you suspect are in play), OR
  (b) ranks each class by similarity to a set of "anchor" prompts you provide
      (e.g. "food, car, airplane, bird, dog, flower"), labeling each class
      with its closest anchor.

(b) is usually what you want: gives you a per-class guess like
    class 0  ->  food   (sim 0.42)
    class 1  ->  food   (sim 0.39)
    class 5  ->  car    (sim 0.51)
    class 12 ->  airplane (sim 0.44)
    ...
which tells you the source dataset cluster for each class in one pass.

Usage:
    pip install open_clip_torch
    python cluster_classes.py \
        --anchors "food,car,airplane,bird,dog,flower,texture,furniture"
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=None,
                   help="Path to train/; falls back to kagglehub")
    p.add_argument("--out", default="class_clusters.csv")
    p.add_argument(
        "--anchors",
        default="food dish,car,airplane,bird,dog,flower,furniture,texture,building,musical instrument",
        help="Comma-separated themes to compare each class against. Pick categories "
             "that match the source datasets you suspect are in play."
    )
    p.add_argument("--images-per-class", type=int, default=3,
                   help="How many of each class's images to average. More=more robust.")
    p.add_argument("--model", default="ViT-SO400M-14-SigLIP-384")
    p.add_argument("--pretrained", default="webli")
    args = p.parse_args()

    # download data if needed
    train_dir = args.train_dir
    if not train_dir or not os.path.isdir(train_dir) or not os.listdir(train_dir):
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG)
        if train_dir:
            cfg["train_dir"] = train_dir
        maybe_download_data(cfg)
        train_dir = cfg["train_dir"]
    print(f"Using train_dir = {train_dir}")

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    anchors = [a.strip() for a in args.anchors.split(",")]
    print(f"Anchors: {anchors}")
    with torch.inference_mode():
        prompts = [f"a photo of a {a}." for a in anchors]
        toks = tokenizer(prompts).to(device)
        anchor_feats = model.encode_text(toks)
        anchor_feats = anchor_feats / anchor_feats.norm(dim=-1, keepdim=True)

    classes = sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s)
    rows = []
    counter = Counter()
    print(f"\nClassifying {len(classes)} classes against {len(anchors)} anchors...\n")
    for cls in classes:
        cls_dir = Path(train_dir) / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.iterdir())[: args.images_per_class]
        with torch.inference_mode():
            xs = torch.stack([
                preprocess(Image.open(f).convert("RGB")) for f in files
            ]).to(device)
            f = model.encode_image(xs)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0, keepdim=True)
            f = f / f.norm()
            sims = (f @ anchor_feats.T).squeeze(0)
        top2 = sims.topk(2)
        a1, a2 = anchors[top2.indices[0]], anchors[top2.indices[1]]
        s1, s2 = top2.values[0].item(), top2.values[1].item()
        margin = s1 - s2
        confidence = "high" if margin > 0.05 else ("medium" if margin > 0.02 else "low")
        counter[a1] += 1
        rows.append((int(cls), a1, s1, a2, s2, confidence))
        print(f"  class {int(cls):3d}  ->  {a1:20s}  sim={s1:.3f}  "
              f"(2nd: {a2}, {s2:.3f}, margin {margin:+.3f}) [{confidence}]")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "best_anchor", "best_sim", "second_anchor", "second_sim", "confidence"])
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {args.out}\n")
    print("Cluster sizes (these should hint at how the dataset was assembled):")
    for anchor, count in counter.most_common():
        print(f"  {anchor:20s}  {count} classes")
    print(
        "\nNext: for each cluster, pick the candidate source dataset (food→Food-101, "
        "car→Stanford Cars, airplane→FGVC-Aircraft, etc.), download it, and run "
        "verify_dataset.py on one class from that cluster to confirm exact-source match."
    )


if __name__ == "__main__":
    main()

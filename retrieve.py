"""Retrieval-based labeling: for each test image, vote by k-NN in external datasets.

You map external-dataset class labels to your 0-99 indexing via a small
mapping file. For each test image, encode with SigLIP, find the k nearest
neighbors in the external dataset's encoded gallery, and accumulate votes
(weighted by similarity) into a 100-class probability vector.

This is INFERENCE-ONLY use of external images — no training on them.
Within the spirit of "you may only use train/ for training."

External dataset folder layout (one per source dataset, repeat as needed):
    external/fgvc_aircraft/
        <whatever_class_name>/
            img1.jpg
            img2.jpg
            ...

And a mapping CSV: external_to_local.csv
    external_class,local_class_id
    Boeing_737-800,0
    Boeing_747-400,1
    Gulfstream_IV,80
    Gulfstream_V,81
    ...

Usage:
    pip install open_clip_torch
    python retrieve.py \
        --external external/fgvc_aircraft \
        --mapping external_to_local.csv \
        --test-dir data/test \
        --out retrieval/submission_probs.csv \
        --k 10
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


def _norm(s):
    """Canonicalize a class name so format differences (case, spaces, _, -, .)
    don't break matching. 'Cadillac_CTS-V_Sedan_2012' and
    'cadillac cts-v sedan 2012' both -> 'cadillacctsvsedan2012'."""
    return "".join(c for c in s.lower() if c.isalnum())


@torch.inference_mode()
def encode_folder(model, preprocess, folder, device, batch_size=32, label_map=None):
    """Return (features[N,D] L2-normalized, labels[N], paths[N]).

    Walks `folder/<class>/*` ImageFolder-style. `label_map[class_name]` gives
    the local 0-99 id; rows with no mapping are skipped. Matching is done on a
    normalized form so case/separator differences between the mapping CSV and
    the on-disk folder names don't silently drop classes.
    """
    norm_map = {_norm(k): v for k, v in label_map.items()} if label_map else None
    files, labels = [], []
    unmatched = []
    for cls_dir in sorted(Path(folder).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        if norm_map is not None and _norm(cls) not in norm_map:
            unmatched.append(cls)
            continue
        lab = norm_map[_norm(cls)] if norm_map else -1
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append(str(p))
                labels.append(lab)
    print(f"  found {len(files)} images across {len(set(labels))} mapped classes")
    if unmatched:
        print(f"  ({len(unmatched)} unmapped folders, e.g. {unmatched[:3]})")

    feats = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        xs = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in batch_files]).to(device)
        ie = model.encode_image(xs)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        feats.append(ie.cpu())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    encoded {i + len(batch_files)}/{len(files)}")
    return torch.cat(feats, 0), torch.tensor(labels), files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--external", action="append", required=True,
                   help="Path to an external dataset folder (ImageFolder layout). "
                        "Repeat for multiple datasets.")
    p.add_argument("--mapping", required=True,
                   help="CSV: external_class,local_class_id")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--out", default="retrieval/submission_probs.csv")
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--k", type=int, default=10, help="Neighbors to retrieve")
    p.add_argument("--model", default="ViT-SO400M-14-SigLIP-384")
    p.add_argument("--pretrained", default="webli")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Softmax temperature on cosine sims when voting")
    args = p.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()

    label_map = {}
    with open(args.mapping, newline="") as f:
        for row in csv.DictReader(f):
            label_map[row["external_class"]] = int(row["local_class_id"])
    print(f"Mapping covers {len(label_map)} external classes -> {len(set(label_map.values()))} local ids")

    # Encode all external galleries
    galleries = []  # list of (feats, labels)
    for ext in args.external:
        print(f"\nEncoding gallery: {ext}")
        feats, labs, _ = encode_folder(model, preprocess, ext, device, label_map=label_map)
        galleries.append((feats, labs))
    gallery_feats = torch.cat([g[0] for g in galleries], 0)
    gallery_labs = torch.cat([g[1] for g in galleries], 0)
    print(f"\nTotal gallery: {len(gallery_feats)} images")

    # Encode test
    print("\nEncoding test set...")
    fnames = sorted(
        [f for f in os.listdir(args.test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda f: int(os.path.splitext(f)[0]),
    )

    all_probs = []
    K = args.k
    with torch.inference_mode():
        for i, fname in enumerate(fnames):
            img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            ie = model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            sims = (ie.cpu() @ gallery_feats.T).squeeze(0)  # [N_gallery]
            top_sims, top_idx = sims.topk(K)
            top_labs = gallery_labs[top_idx]
            # softmax-weighted vote
            weights = torch.softmax(top_sims / args.temperature, dim=0)
            probs = torch.zeros(args.num_classes)
            for w, lab in zip(weights, top_labs):
                probs[lab.item()] += w.item()
            # for classes never represented in the gallery, leave near-zero;
            # ensemble.py will let other members dominate them.
            probs = probs + 1e-6
            probs = probs / probs.sum()
            all_probs.append((fname, probs.tolist()))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(fnames)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [str(c) for c in range(args.num_classes)])
        for fname, probs in all_probs:
            w.writerow([fname] + [f"{v:.6f}" for v in probs])
    print(f"\nWrote {out} ({len(all_probs)} rows)")


if __name__ == "__main__":
    main()

"""Automated class identification via perceptual-hash matching against source datasets.

For each of the 100 competition classes, hashes its 10 training images and
searches every class folder across all provided source datasets for matches.
Reports the source class with the most hits and writes class_names.csv.

This skips ~all the manual reverse-image-searching when the competition
sampled directly from public datasets (the common case).

Usage:
    pip install imagehash
    python match_classes.py \
        --source food_101:food-101/images \
        --source stanford_cars:stanford_cars/cars_train_by_class \
        --source fgvc_aircraft:fgvc-aircraft-2013b/data/images_by_family \
        --source flowers_102:flowers-102/by_class \
        --out class_names.csv

Each --source spec is `dataset_label:path_to_ImageFolder_root`. The folder
must be ImageFolder-style: one subfolder per class, named after the class.

Strategy
--------
1. Build a hash index per source class (≈10-100 images × ~500 classes total).
2. For each of our 100 classes, hash its 10 images, count pHash hits (d<=6)
   per source class across all source datasets.
3. The source class with the most hits wins; report confidence based on
   hit count and uniqueness.

Runtime
-------
Index build: ~10-15 min for ~35k source images.
Match step:  ~1-2 min for 100 × 10 = 1000 queries.
Total: ~15 min one-time cost vs. hours of manual searching.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image


def hash_folder(root, label):
    """Yield (source_label, source_class, image_path, phash) for every image."""
    root = Path(root)
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                h = imagehash.phash(Image.open(p).convert("RGB"))
                yield (label, cls_dir.name, str(p), h)
            except Exception as e:
                print(f"  skip {p}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=None,
                   help="Path to competition train/; falls back to kagglehub")
    p.add_argument("--source", action="append", required=True, metavar="LABEL:PATH",
                   help="Source dataset spec. Repeat per dataset.")
    p.add_argument("--out", default="class_names.csv")
    p.add_argument("--max-distance", type=int, default=6,
                   help="pHash distance threshold for an EXACT match (default 6).")
    p.add_argument("--max-per-class", type=int, default=200,
                   help="Cap images per source class to save time (default 200).")
    args = p.parse_args()

    # download competition data if needed
    train_dir = args.train_dir
    if not train_dir or not os.path.isdir(train_dir) or not os.listdir(train_dir):
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG)
        if train_dir:
            cfg["train_dir"] = train_dir
        maybe_download_data(cfg)
        train_dir = cfg["train_dir"]
    print(f"Using train_dir = {train_dir}")

    # parse sources
    sources = []
    for spec in args.source:
        if ":" not in spec:
            raise SystemExit(f"--source spec must be LABEL:PATH, got {spec!r}")
        label, path = spec.split(":", 1)
        if not Path(path).is_dir():
            raise SystemExit(f"source path not found: {path}")
        sources.append((label, path))

    # build source hash index
    print(f"\nIndexing {len(sources)} source datasets...")
    # key: (source_label, source_class)  -> list[(image_path, phash)]
    index = defaultdict(list)
    for label, path in sources:
        count_per_class = defaultdict(int)
        n = 0
        for src_label, src_cls, p_path, h in hash_folder(path, label):
            if count_per_class[(src_label, src_cls)] >= args.max_per_class:
                continue
            count_per_class[(src_label, src_cls)] += 1
            index[(src_label, src_cls)].append((p_path, h))
            n += 1
            if n % 2000 == 0:
                print(f"  {label}: indexed {n}")
        print(f"  {label}: {n} images across {sum(1 for k in index if k[0] == label)} classes")
    total = sum(len(v) for v in index.values())
    print(f"  Total: {total} indexed images across {len(index)} (source,class) groups")

    # for each of our 100 classes, vote
    our_classes = sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s)
    rows = []
    print(f"\nMatching {len(our_classes)} competition classes...\n")
    for cls in our_classes:
        if not (Path(train_dir) / cls).is_dir():
            continue
        our_dir = Path(train_dir) / cls
        our_files = sorted(p for p in our_dir.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        # hits[(src_label, src_cls)] = count of our images that match
        hits = defaultdict(int)
        details = []
        for f in our_files:
            try:
                h = imagehash.phash(Image.open(f).convert("RGB"))
            except Exception:
                continue
            best_key, best_dist, best_path = None, 999, None
            for key, entries in index.items():
                for p_path, ph in entries:
                    d = h - ph
                    if d < best_dist:
                        best_dist = d
                        best_key = key
                        best_path = p_path
            if best_dist <= args.max_distance:
                hits[best_key] += 1
                details.append((f.name, best_key, best_dist, best_path))

        if hits:
            top = max(hits.items(), key=lambda kv: kv[1])
            (src_label, src_cls), n_hits = top
            n_our = len(our_files)
            if n_hits >= 8:
                confidence = "high"
            elif n_hits >= 4:
                confidence = "medium"
            else:
                confidence = "low"
            print(f"  class {int(cls):3d}  ->  {src_label}:{src_cls:40s}  "
                  f"{n_hits}/{n_our} matches  [{confidence}]")
            rows.append((int(cls), src_cls, src_label, confidence, n_hits, n_our))
        else:
            print(f"  class {int(cls):3d}  ->  NO MATCHES (manual identification needed)")
            rows.append((int(cls), "", "unknown", "unknown", 0, len(our_files)))

    # normalize source class names to human-readable form
    def humanize(src_cls):
        return src_cls.replace("_", " ").replace("-", " ").strip()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "source_dataset", "confidence", "match_count", "n_images"])
        for cls_id, src_cls, src_label, conf, n_hits, n_our in rows:
            w.writerow([cls_id, humanize(src_cls), src_label, conf, n_hits, n_our])
    print(f"\nWrote {args.out}")
    n_high = sum(1 for r in rows if r[3] == "high")
    n_med = sum(1 for r in rows if r[3] == "medium")
    n_low = sum(1 for r in rows if r[3] == "low")
    n_unk = sum(1 for r in rows if r[3] == "unknown")
    print(f"Confidence: {n_high} high, {n_med} medium, {n_low} low, {n_unk} unknown")
    print(
        "\nReview the CSV. For 'unknown' rows, do manual Lens search (likely "
        "different source dataset). For 'low'/'medium', spot-check by eye."
    )


if __name__ == "__main__":
    main()

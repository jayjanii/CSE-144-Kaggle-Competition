"""Export one 10-image grid per class for fast manual identification.

Goal: identify which public dataset each of the 100 classes came from, and
what each class's real-world name is. With names you can do zero-shot CLIP +
retrieval against the source dataset (see retrieve.py / zero_shot.py).

Workflow:
    1. python inspect_classes.py --out class_grids/
    2. Open class_grids/class_00.jpg, class_01.jpg, ... in Drive viewer
    3. For each, reverse-image-search 1-2 of the 10 tiles via Google Lens
       (right-click the image → Search with Google → switch to "Lens").
       Usually you'll see the source dataset's webpage or a Wikipedia hit
       that names the breed/model/species.
    4. Fill in class_names.csv (template printed at end).

Estimated time: 2-3 hours for all 100 classes. This single step is probably
worth more than any further model training at this point.
"""

import argparse
import csv
import os
from pathlib import Path

from PIL import Image


def make_grid(images, cols=5, tile_size=224, pad=4, bg=(20, 20, 20)):
    """Return a single PIL image: cols-wide grid of tiles resized to tile_size."""
    n = len(images)
    rows = (n + cols - 1) // cols
    W = cols * tile_size + (cols + 1) * pad
    H = rows * tile_size + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        # short-side resize then center crop to tile_size
        w, h = img.size
        scale = tile_size / min(w, h)
        nw, nh = max(tile_size, round(w * scale)), max(tile_size, round(h * scale))
        rs = img.resize((nw, nh), Image.BICUBIC)
        l = (nw - tile_size) // 2
        t = (nh - tile_size) // 2
        tile = rs.crop((l, t, l + tile_size, t + tile_size))
        x = pad + c * (tile_size + pad)
        y = pad + r * (tile_size + pad)
        canvas.paste(tile, (x, y))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--out", default="class_grids")
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--cols", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = sorted(os.listdir(args.train_dir), key=lambda s: int(s) if s.isdigit() else s)
    for cls in classes:
        cls_dir = Path(args.train_dir) / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.iterdir())[:10]
        imgs = [Image.open(f).convert("RGB") for f in files]
        grid = make_grid(imgs, cols=args.cols, tile_size=args.tile_size)
        grid.save(out / f"class_{int(cls):02d}.jpg", quality=85)

    print(f"Wrote {len(classes)} grids to {out}/")
    print("\nNow fill in this template (class_names.csv):\n")
    print("class_id,name,source_dataset,confidence")
    for cls in classes:
        print(f"{int(cls)},,,low")


if __name__ == "__main__":
    main()

"""Build class_names.csv from the 4 source datasets, assuming the discovered
block-alphabetical layout:

    classes  0-24  →  Food-101         (alphabetical)
    classes 25-49  →  Oxford Flowers 102 (alphabetical)
    classes 50-74  →  Stanford Cars    (alphabetical)
    classes 75-99  →  FGVC-Aircraft variants (alphabetical)

After running, verify by spot-checking against the training images. If any
block is in a different order, run match_classes.py to disambiguate.
"""

import argparse
import csv
from pathlib import Path


def list_alpha_classes(root, take=25):
    """Return sorted subdirectory names of `root` (treated as ImageFolder)."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(classes) < take:
        raise SystemExit(f"{root} has only {len(classes)} classes, need {take}")
    return classes[:take]


def humanize(name):
    """Convert dataset-internal class names to natural language for CLIP prompts."""
    return name.replace("_", " ").replace("-", " ").strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--food", required=True, help="Path to food-101/images")
    p.add_argument("--flowers", required=True, help="Path to flowers-102 ImageFolder root")
    p.add_argument("--cars", required=True, help="Path to Stanford Cars ImageFolder root")
    p.add_argument("--aircraft", required=True, help="Path to FGVC-Aircraft variant ImageFolder root")
    p.add_argument("--out", default="class_names.csv")
    args = p.parse_args()

    food = list_alpha_classes(args.food)
    flowers = list_alpha_classes(args.flowers)
    cars = list_alpha_classes(args.cars)
    aircraft = list_alpha_classes(args.aircraft)

    rows = []
    for i, name in enumerate(food):
        rows.append((i, humanize(name), "food_101", "high"))
    for i, name in enumerate(flowers):
        rows.append((25 + i, humanize(name), "flowers_102", "high"))
    for i, name in enumerate(cars):
        rows.append((50 + i, humanize(name), "stanford_cars", "high"))
    for i, name in enumerate(aircraft):
        rows.append((75 + i, humanize(name), "fgvc_aircraft", "high"))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "source_dataset", "confidence"])
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.out} ({len(rows)} rows)\n")
    print("Sanity check — first and last entry of each block:")
    print(f"  food     0  = {food[0]}")
    print(f"  food    24  = {food[24]}")
    print(f"  flowers 25  = {flowers[0]}")
    print(f"  flowers 49  = {flowers[24]}")
    print(f"  cars    50  = {cars[0]}")
    print(f"  cars    74  = {cars[24]}")
    print(f"  aircraft 75 = {aircraft[0]}")
    print(f"  aircraft 99 = {aircraft[24]}")
    print(
        "\nVerify these match the corresponding class_grids/class_NN.jpg. "
        "If any block is wrong, run match_classes.py to discover the true mapping."
    )


if __name__ == "__main__":
    main()

"""Stitch and analyze k-fold out-of-fold (OOF) predictions from train.py --kfold.

train.py --kfold N --fold-idx i writes oof_fold<i>.csv (index,path,label,0..C-1) of
leak-free TTA predictions for the held-out fold. Run all N folds for a model, then:

    # combine one model's folds into a single leak-free prediction per image:
    python oof.py stitch m1_dinov2/fold0/oof_fold0.csv ... \
        m1_dinov2/fold4/oof_fold4.csv --out m1_dinov2/oof.csv

    # cross-model CV, leak-free ensemble-weight search, and label audit:
    python oof.py analyze m1_dinov2/oof.csv m2_eva02/oof.csv m3_siglip/oof.csv

Because OOF covers every labeled image with no leakage, its accuracy is a far more
trustworthy CV signal than the 216-image val split or the ~11% public LB, and the
ensemble weights it finds are tuned on ~1000+ images instead of a noisy slice.
"""

import argparse
import csv
import itertools

import numpy as np


def read_oof(path):
    """Return (indices, paths, labels, probs[N,C]) sorted by index."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header: index,path,label,0..C-1
        idx, paths, labels, probs = [], [], [], []
        for row in r:
            idx.append(int(row[0]))
            paths.append(row[1])
            labels.append(int(row[2]))
            probs.append([float(x) for x in row[3:]])
    order = np.argsort(idx)
    return (
        np.array(idx)[order],
        [paths[i] for i in order],
        np.array(labels)[order],
        np.array(probs)[order],
    )


def cmd_stitch(args):
    all_idx, all_paths, all_lab, all_pr = [], [], [], []
    seen = set()
    for path in args.files:
        idx, paths, lab, pr = read_oof(path)
        dup = seen.intersection(idx.tolist())
        if dup:
            raise SystemExit(
                f"{path}: {len(dup)} indices overlap a previous fold "
                f"(e.g. {sorted(dup)[:5]}) — folds must be disjoint"
            )
        seen.update(idx.tolist())
        all_idx.append(idx)
        all_paths += paths
        all_lab.append(lab)
        all_pr.append(pr)

    idx = np.concatenate(all_idx)
    lab = np.concatenate(all_lab)
    pr = np.concatenate(all_pr)
    order = np.argsort(idx)
    idx, lab, pr = idx[order], lab[order], pr[order]
    paths = [all_paths[i] for i in order]

    acc = (pr.argmax(1) == lab).mean()
    print(
        f"Stitched {len(args.files)} folds → {len(idx)} images, "
        f"OOF acc {acc:.4f} ({int(acc * len(idx))}/{len(idx)})"
    )
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "path", "label"] + [str(c) for c in range(pr.shape[1])])
        for i in range(len(idx)):
            w.writerow([idx[i], paths[i], lab[i]] + [f"{x:.6f}" for x in pr[i]])
    print(f"Wrote {args.out}")


def cmd_analyze(args):
    models = [read_oof(p) for p in args.files]
    ref_idx = models[0][0]
    for path, (idx, *_) in zip(args.files[1:], models[1:]):
        if not np.array_equal(idx, ref_idx):
            raise SystemExit(
                f"{path} covers a different/reordered image set than "
                f"{args.files[0]} — stitch each model over the same folds+seed first"
            )
    labels = models[0][2]
    paths = models[0][1]
    probs = [m[3] for m in models]
    n = len(labels)

    print(f"Aligned {len(models)} models over {n} OOF images (1 image = {100/n:.2f}%)\n")
    print("--- single-model OOF accuracy ---")
    for path, pr in zip(args.files, probs):
        print(f"  {path:42s} {(pr.argmax(1) == labels).mean():.4f}")

    eq = sum(probs) / len(probs)
    eq_acc = (eq.argmax(1) == labels).mean()
    print(f"\nequal-weight ensemble OOF acc: {eq_acc:.4f} ({int(eq_acc * n)}/{n})")

    # Leak-free weight search over all OOF images.
    grid = [round(x, 2) for x in np.arange(0.5, 2.01, 0.25)]
    best, best_w = -1.0, None
    for ws in itertools.product(grid, repeat=len(probs)):
        mix = sum(w * p for w, p in zip(ws, probs))
        a = (mix.argmax(1) == labels).mean()
        if a > best:
            best, best_w = a, ws
    gain = (best - eq_acc) * n
    print(
        f"best grid weights {best_w} -> {best:.4f}  "
        f"(+{gain:.1f} images over equal-weight)"
    )
    if gain < 2:
        print("  → gain is within noise (<2 images); prefer equal weights.")

    # Label audit: images the ensemble confidently calls differently than the label.
    pred = eq.argmax(1)
    conf = eq.max(1)
    wrong = np.where(pred != labels)[0]
    wrong = wrong[np.argsort(-conf[wrong])]
    print(
        f"\n--- label audit: {len(wrong)} OOF errors, top {min(args.top, len(wrong))} "
        f"by ensemble confidence ---"
    )
    print(f"{'conf':>5}  {'given':>5} {'pred':>5}  path")
    for i in wrong[: args.top]:
        print(f"{conf[i]:5.2f}  {labels[i]:5d} {pred[i]:5d}  {paths[i]}")
    print(
        "\nHigh-confidence rows where the ensemble disagrees with the given label are "
        "label-error candidates: verify the image, then encode confirmed fixes in "
        "label_overrides.csv (deterministic + reproducible)."
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stitch", help="combine one model's per-fold OOF files")
    s.add_argument("files", nargs="+", help="oof_fold*.csv for a single model")
    s.add_argument("--out", required=True, help="combined OOF output path")
    s.set_defaults(func=cmd_stitch)

    a = sub.add_parser("analyze", help="cross-model CV + weight search + label audit")
    a.add_argument("files", nargs="+", help="one stitched oof.csv per model")
    a.add_argument("--top", type=int, default=40, help="label-audit rows to print")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Tune multi-backbone probe ensemble weights on the aligned OOF probabilities.

multi_probe.py saves oof_<tag>.npy (leak-free per-train-image probs) for each
backbone, all over the same seeded split, plus labels.npy. This finds the
weight combination that maximizes OOF accuracy — a 1079-image signal, far more
trustworthy than the ~114-image public LB.

Usage:
    python tune_probe_oof.py --cache multi_probe/cache
    # then apply the printed weights to the test probe CSVs via ensemble.py
"""

import argparse
import itertools
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="multi_probe/cache")
    p.add_argument("--grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0",
                   help="weight grid per backbone")
    args = p.parse_args()

    cache = Path(args.cache)
    oof_files = sorted(cache.glob("oof_*.npy"))
    if not oof_files:
        raise SystemExit(f"no oof_*.npy in {cache} — run multi_probe.py first")
    labels = np.load(cache / "labels.npy")

    tags = [f.stem[4:] for f in oof_files]          # strip 'oof_'
    oofs = [np.load(f) for f in oof_files]
    N = len(labels)
    print(f"{len(tags)} backbones over {N} OOF images:\n")
    for tag, o in zip(tags, oofs):
        print(f"  {tag:55s} OOF acc {(o.argmax(1)==labels).mean():.4f}")

    # normalize each to row-stochastic (probes already are, but be safe)
    oofs = [o / (o.sum(1, keepdims=True) + 1e-12) for o in oofs]

    eq = sum(oofs) / len(oofs)
    eq_acc = (eq.argmax(1) == labels).mean()
    print(f"\nequal-weight ensemble OOF acc: {eq_acc:.4f} ({int(eq_acc*N)}/{N})")

    grid = [float(x) for x in args.grid.split(",")]
    best, best_w = -1, None
    # fix first weight to 1.0 (scale-invariant) to shrink the search
    for combo in itertools.product(grid, repeat=len(oofs) - 1):
        w = (1.0,) + combo
        mix = sum(wi * o for wi, o in zip(w, oofs))
        a = (mix.argmax(1) == labels).mean()
        if a > best:
            best, best_w = a, w
    gain = (best - eq_acc) * N
    print(f"best weights {[f'{x:g}' for x in best_w]} -> {best:.4f} "
          f"({gain:+.1f} imgs vs equal)")
    print("\nbackbone : weight")
    for tag, wi in zip(tags, best_w):
        print(f"  {tag:55s} {wi:g}")
    if gain < 2:
        print("\n(gain < 2 images — within noise; equal weights are fine.)")
    print("\nApply these weights to the matching test probe CSVs with ensemble.py "
          "(--mode rank or arith), in the SAME backbone order.")


if __name__ == "__main__":
    main()

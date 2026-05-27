"""Tune ensemble fusion settings on stitched OOF data.

You usually have aligned OOF probs for only a SUBSET of your ensemble members
(e.g. SigLIP + EVA-02 here — DINOv2 was trained with a different split). That's
fine: we fit the *relative* fusion settings (temperatures + weight ratios + the
fusion mode) on the OOF members and reuse them at submission time. For members
without OOF you pass a single scalar weight (e.g. --extra-weight 1.0) and sweep
that at submission time if you want.

Inputs are stitched per-model OOF files written by `oof.py stitch`, format:
    index,path,label,0,1,...,C-1

What it does
------------
1. Reports single-member OOF acc/NLL (raw and after per-member temperature).
2. Fits one scalar temperature per member by golden-section search on OOF NLL.
3. Grid-searches weights for the OOF members under three fusion modes
   (arithmetic, geometric, rank) and prints the best of each.
4. Prints a ready-to-paste `python ensemble.py ...` command using the winning
   settings, slotting in the no-OOF member with the user-supplied weight.

Usage
-----
    python tune_ensemble.py \
        --oof m3_siglip/oof.csv:weights/m3_siglip/full/submission_probs.csv \
        --oof m2_eva02/oof.csv:weights/m2_eva02/full/submission_probs.csv \
        --extra BEST_0945/submission_probs.csv:dinov2:1.2 \
        --out-cmd suggested_ensemble.sh

The `--oof` spec is `<oof_csv>:<test_probs_csv>` for an OOF member.
The `--extra` spec is `<test_probs_csv>:<label>:<weight>` for a no-OOF member.
The script writes a shell-runnable line that produces submission_ensemble.csv.
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


# ── IO ────────────────────────────────────────────────────────────────────────


def read_oof(path):
    """Return (labels[N], probs[N,C], paths[N]) sorted by index."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        idx, paths, labels, probs = [], [], [], []
        for row in r:
            idx.append(int(row[0]))
            paths.append(row[1])
            labels.append(int(row[2]))
            probs.append([float(x) for x in row[3:]])
    order = np.argsort(idx)
    return (
        np.array(labels)[order],
        np.array(probs)[order],
        [paths[i] for i in order],
    )


# ── Core math ─────────────────────────────────────────────────────────────────


def temper(probs, T, eps=1e-12):
    """Re-softmax probs with scalar temperature T (axis=-1). Vectorized."""
    if T == 1.0:
        return probs
    log_p = np.log(np.clip(probs, eps, 1.0)) / T
    log_p -= log_p.max(axis=-1, keepdims=True)
    exp = np.exp(log_p)
    return exp / exp.sum(axis=-1, keepdims=True)


def nll(probs, labels, eps=1e-12):
    return -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean()


def acc(probs, labels):
    return float((probs.argmax(1) == labels).mean())


def fit_temperature(probs, labels, lo=0.25, hi=4.0, iters=60):
    """Golden-section search for T minimizing OOF NLL."""
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    a, b = lo, hi
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = nll(temper(probs, c), labels)
    fd = nll(temper(probs, d), labels)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = nll(temper(probs, c), labels)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = nll(temper(probs, d), labels)
    T = (a + b) / 2
    return T, nll(temper(probs, T), labels)


def fuse(member_probs, weights, mode, eps=1e-12):
    """member_probs: list of [N,C] arrays (already tempered). weights: list of floats."""
    w = np.asarray(weights, dtype=float)
    ws = w.sum()
    if mode == "arith":
        out = np.zeros_like(member_probs[0])
        for p, wi in zip(member_probs, w):
            out += wi * p
        return out / ws
    if mode == "geom":
        log_out = np.zeros_like(member_probs[0])
        for p, wi in zip(member_probs, w):
            log_out += wi * np.log(np.clip(p, eps, 1.0))
        log_out /= ws
        log_out -= log_out.max(axis=-1, keepdims=True)
        exp = np.exp(log_out)
        return exp / exp.sum(axis=-1, keepdims=True)
    if mode == "rank":
        # average per-image rank, then min-max normalize → pseudo-prob
        out = np.zeros_like(member_probs[0])
        for p, wi in zip(member_probs, w):
            r = p.argsort(axis=1).argsort(axis=1).astype(float) + 1  # ranks 1..C
            out += wi * r
        out /= ws
        mn = out.min(axis=1, keepdims=True)
        mx = out.max(axis=1, keepdims=True)
        rng = np.where(mx > mn, mx - mn, 1.0)
        out = (out - mn) / rng
        return out / out.sum(axis=1, keepdims=True)
    raise ValueError(mode)


def grid_search_weights(member_probs, labels, mode, grid):
    """Brute-force search over weight tuples; one weight fixed to 1.0 (scale-invariant)."""
    n = len(member_probs)
    best = (-1.0, None)
    # fix first weight = 1.0 to break scale invariance
    for combo in itertools.product(grid, repeat=n - 1):
        w = (1.0,) + combo
        a = acc(fuse(member_probs, w, mode), labels)
        if a > best[0]:
            best = (a, w)
    return best


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--oof", action="append", required=True, metavar="OOF_CSV:TEST_PROBS_CSV",
        help="For each OOF member: stitched oof.csv path AND the matching test "
             "submission_probs.csv (same model, full-train). Repeat per member."
    )
    p.add_argument(
        "--extra", action="append", default=[], metavar="TEST_PROBS_CSV:LABEL:WEIGHT",
        help="For each no-OOF member: its test probs, a label for printing, and "
             "the weight to use in the final ensemble (you choose this — sweep at "
             "submission time if uncertain). Repeat per member."
    )
    p.add_argument(
        "--out-cmd", default="suggested_ensemble.sh",
        help="Where to write the suggested ensemble.py invocation"
    )
    p.add_argument(
        "--ensemble-out", default="submission_ensemble_tuned.csv",
        help="Path the suggested command will write the submission to"
    )
    p.add_argument(
        "--grid", default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Weight grid for the OOF members (comma-separated)"
    )
    args = p.parse_args()

    grid = [float(x) for x in args.grid.split(",")]

    # ── load OOF members ──
    oof_specs = []  # list of (oof_path, test_probs_path, label, labels_arr, probs_arr)
    for spec in args.oof:
        if ":" not in spec:
            raise SystemExit(f"--oof spec must be OOF:TEST, got {spec!r}")
        oof_path, test_path = spec.split(":", 1)
        labels, probs, _ = read_oof(oof_path)
        label = Path(oof_path).parent.name or Path(oof_path).stem
        oof_specs.append((oof_path, test_path, label, labels, probs))

    # sanity: all OOF members must cover the same labeled set in the same order
    ref_labels = oof_specs[0][3]
    for op, _, lab, ll, _ in oof_specs[1:]:
        if not np.array_equal(ll, ref_labels):
            raise SystemExit(
                f"OOF labels for {lab} ({op}) don't match {oof_specs[0][2]} — "
                "stitch both with the same kfold+seed first"
            )
    labels = ref_labels
    N, C = oof_specs[0][4].shape
    print(f"Aligned {len(oof_specs)} OOF members over {N} labeled images (1 image = {100/N:.2f}%)\n")

    # ── per-member single accuracy + fit temperature ──
    print("--- single-member OOF ---")
    temps = []
    tempered = []
    for _, _, label, _, probs in oof_specs:
        raw_acc = acc(probs, labels)
        raw_nll = nll(probs, labels)
        T, T_nll = fit_temperature(probs, labels)
        tp = temper(probs, T)
        t_acc = acc(tp, labels)
        print(f"  {label:20s} acc={raw_acc:.4f}  nll={raw_nll:.4f}  "
              f"-> T*={T:.3f}  nll'={T_nll:.4f}  acc'={t_acc:.4f}")
        temps.append(T)
        tempered.append(tp)

    # ── ensemble search across fusion modes ──
    print("\n--- OOF ensemble (across fusion modes, tempered members) ---")
    results = {}
    for mode in ("arith", "geom", "rank"):
        eq_w = [1.0] * len(tempered)
        eq_acc = acc(fuse(tempered, eq_w, mode), labels)
        best_acc, best_w = grid_search_weights(tempered, labels, mode, grid)
        gain = (best_acc - eq_acc) * N
        results[mode] = (eq_acc, best_acc, best_w)
        print(f"  {mode:5s}  equal-weight {eq_acc:.4f}   "
              f"best{[f'{w:.2f}' for w in best_w]}={best_acc:.4f}  "
              f"({gain:+.1f} images vs equal)")

    best_mode = max(results, key=lambda m: results[m][1])
    best_acc, best_eq_acc, best_w = results[best_mode][1], results[best_mode][0], results[best_mode][2]
    print(f"\n=> chose mode={best_mode}, OOF weights={list(best_w)}, OOF acc={best_acc:.4f}")
    if (best_acc - best_eq_acc) * N < 2:
        print("   (gain over equal-weight is <2 images — within noise. Equal weights may be safer.)")

    # ── assemble final ensemble.py command ──
    extras = []  # (test_path, label, weight)
    for spec in args.extra:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--extra spec must be TEST:LABEL:WEIGHT, got {spec!r}")
        extras.append((parts[0], parts[1], float(parts[2])))

    all_test_paths = [oof[1] for oof in oof_specs] + [e[0] for e in extras]
    all_weights = list(best_w) + [e[2] for e in extras]
    # extras get T=1.0 (no OOF to fit on)
    all_temps = list(temps) + [1.0] * len(extras)
    all_labels = [oof[2] for oof in oof_specs] + [e[1] for e in extras]

    cmd = (
        f"python ensemble.py \\\n  "
        + " \\\n  ".join(all_test_paths)
        + f" \\\n  --mode {best_mode} \\\n"
        + f"  --weights {','.join(f'{w:g}' for w in all_weights)} \\\n"
        + f"  --temperatures {','.join(f'{t:.3f}' for t in all_temps)} \\\n"
        + f"  --out {args.ensemble_out}"
    )
    print("\n=== suggested submission command ===")
    print("# members:", ", ".join(all_labels))
    print(cmd)

    Path(args.out_cmd).write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n")
    print(f"\nWrote {args.out_cmd} (chmod +x and run, or paste the line above).")
    print(
        "\nNote: temperatures/weights for OOF members were tuned on leak-free OOF\n"
        "  (~1000 images). Extras (no OOF) use T=1.0 and the weight you passed; if\n"
        "  unsure, sweep {0.8, 1.0, 1.2, 1.5} and submit the median."
    )


if __name__ == "__main__":
    main()

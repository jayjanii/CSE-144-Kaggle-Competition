"""Average per-member probabilities into one ensemble submission.

Each input is a submission_probs.csv produced by inference.py:
    image_id,0,1,...,99
Rows are matched by image_id (order-independent), fused (optionally weighted and
temperature-calibrated), and the argmax is written as the ensemble submission.

Fusion modes:
    --mode arith   arithmetic mean of probs (default; classic soft-voting)
    --mode geom    geometric mean of probs == arithmetic mean of log-probs.
                   Robust when members disagree about calibration sharpness
                   (CLIP/SigLIP softmax tends to be sharper than DINOv2's).
    --mode rank    average of per-image class ranks. Calibration-free; useful
                   when one member is consistently over/under-confident.

Per-member temperatures (--temperatures T1,T2,...) divide each member's logits
by T before re-softmaxing. T>1 softens an over-confident member, T<1 sharpens.
Fit on OOF data with tune_ensemble.py.

Usage:
    python ensemble.py runA/submission_probs.csv runB/submission_probs.csv \
        --out submission_ensemble.csv
    python ensemble.py a.csv b.csv c.csv --weights 1,1,0.5 \
        --mode geom --temperatures 1.0,1.3,1.1 \
        --out sub.csv --probs-out ensemble_probs.csv
"""

import argparse
import csv
import math


def read_probs(path):
    """Return (class_columns, {image_id: [float probs]})."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)  # ["image_id", "0", "1", ...]
        classes = header[1:]
        probs = {row[0]: [float(x) for x in row[1:]] for row in r}
    return classes, probs


def _apply_temperature(probs_row, T, eps=1e-12):
    """Re-softmax probs with temperature T (T>1 softens, T<1 sharpens)."""
    if T == 1.0:
        return list(probs_row)
    # logits = log(p); rescale by 1/T; renormalize via softmax in log-space
    logits = [math.log(max(p, eps)) / T for p in probs_row]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _fuse(rows, weights, mode, eps=1e-12):
    """Fuse a list of per-member prob vectors (already temperature-applied)."""
    C = len(rows[0])
    wsum = sum(weights)
    if mode == "arith":
        out = [0.0] * C
        for r, w in zip(rows, weights):
            for k in range(C):
                out[k] += w * r[k]
        return [v / wsum for v in out]
    if mode == "geom":
        # weighted geometric mean = exp(sum(w_i * log p_i) / sum(w))
        log_out = [0.0] * C
        for r, w in zip(rows, weights):
            for k in range(C):
                log_out[k] += w * math.log(max(r[k], eps))
        m = max(log_out)
        exps = [math.exp(l / wsum - m / wsum) for l in log_out]
        s = sum(exps)
        return [e / s for e in exps]
    if mode == "rank":
        # per-member ranks (1..C, higher=more likely), weighted-averaged, then
        # normalize to a pseudo-probability so downstream code treats it the same.
        out = [0.0] * C
        for r, w in zip(rows, weights):
            order = sorted(range(C), key=lambda k: r[k])  # ascending
            ranks = [0] * C
            for rk, k in enumerate(order):
                ranks[k] = rk + 1
            for k in range(C):
                out[k] += w * ranks[k]
        # rescale to [0,1] and normalize so it sums to 1 (pseudo-prob)
        mn, mx = min(out), max(out)
        if mx > mn:
            out = [(v - mn) / (mx - mn) for v in out]
        s = sum(out) or 1.0
        return [v / s for v in out]
    raise ValueError(f"unknown fusion mode: {mode}")


def main():
    p = argparse.ArgumentParser(description="Fuse per-member probs into an ensemble submission")
    p.add_argument("probs", nargs="+", help="submission_probs.csv files, one per member")
    p.add_argument("--weights", default=None,
                   help="Comma-separated per-member weights (default: equal)")
    p.add_argument("--temperatures", default=None,
                   help="Comma-separated per-member temperatures (default: all 1.0). "
                        "T>1 softens, T<1 sharpens. Fit on OOF with tune_ensemble.py.")
    p.add_argument("--mode", choices=("arith", "geom", "rank"), default="arith",
                   help="Fusion mode (default: arith). 'geom' is robust to "
                        "miscalibrated members; 'rank' is calibration-free.")
    p.add_argument("--out", default="submission_ensemble.csv", help="Output submission path")
    p.add_argument("--probs-out", default=None,
                   help="Optional: write averaged probs CSV (e.g. for round-2 pseudo-labels)")
    p.add_argument("--pseudo-out", default=None, metavar="CSV",
                   help="Optional: write a pseudo_labels.csv (filename,label,confidence) from the "
                        "ENSEMBLE-averaged probs — cleaner than any single model's. Consumed by "
                        "train.py --pseudo-labels for round-2 self-training.")
    p.add_argument("--pseudo-threshold", type=float, default=0.95,
                   help="Min ensemble confidence to keep a pseudo-label (default: 0.95)")
    p.add_argument("--pseudo-topk", type=int, default=None, metavar="N",
                   help="Keep the N most-confident predictions instead of a fixed threshold "
                        "(overrides --pseudo-threshold)")
    args = p.parse_args()

    n_members = len(args.probs)
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != n_members:
            raise SystemExit(f"--weights has {len(weights)} values but {n_members} files given")
    else:
        weights = [1.0] * n_members
    if args.temperatures:
        temps = [float(t) for t in args.temperatures.split(",")]
        if len(temps) != n_members:
            raise SystemExit(f"--temperatures has {len(temps)} values but {n_members} files given")
    else:
        temps = [1.0] * n_members

    classes = ids = None
    member_probs = []  # list of {image_id: [tempered probs]}
    for path, T in zip(args.probs, temps):
        c, probs = read_probs(path)
        if classes is None:
            classes, ids = c, list(probs.keys())
        else:
            if c != classes:
                raise SystemExit(f"class-column mismatch in {path}")
            if set(probs.keys()) != set(ids):
                raise SystemExit(f"image_id set mismatch in {path}")
        if T != 1.0:
            probs = {i: _apply_temperature(v, T) for i, v in probs.items()}
        member_probs.append(probs)

    acc = {}
    for i in ids:
        rows = [mp[i] for mp in member_probs]
        acc[i] = _fuse(rows, weights, args.mode)

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["image_id", "predicted_class"])
        for i in ids:
            pi = acc[i]
            pred = max(range(len(classes)), key=lambda k: pi[k])
            wtr.writerow([i, int(classes[pred])])
    print(f"Ensemble of {n_members} members "
          f"(mode={args.mode}, weights={weights}, temps={temps}) "
          f"→ {args.out} ({len(ids)} rows)")

    if args.probs_out:
        with open(args.probs_out, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["image_id"] + classes)
            for i in ids:
                wtr.writerow([i] + [f"{v:.6f}" for v in acc[i]])
        print(f"Averaged probs → {args.probs_out}")

    if args.pseudo_out:
        # Score every test image by its ensemble-averaged top-1 confidence, then keep
        # the most confident (mirrors inference.py so train.py consumes it identically).
        scored = []
        for i in ids:
            pi = acc[i]
            lab = max(range(len(classes)), key=lambda k: pi[k])
            scored.append((pi[lab], i, int(classes[lab])))  # (conf, filename, label)
        scored.sort(reverse=True)  # most confident first

        if args.pseudo_topk:
            kept = scored[: args.pseudo_topk]
            criterion = f"top-{args.pseudo_topk}"
        else:
            kept = [s for s in scored if s[0] >= args.pseudo_threshold]
            criterion = f"thr={args.pseudo_threshold}"

        with open(args.pseudo_out, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["filename", "label", "confidence"])
            for conf, fname, lab in kept:
                wtr.writerow([fname, lab, f"{conf:.4f}"])
        min_conf = kept[-1][0] if kept else 0.0
        print(f"Pseudo-labels: kept {len(kept)}/{len(ids)} "
              f"({criterion}, min conf {min_conf:.3f}) → {args.pseudo_out}")


if __name__ == "__main__":
    main()

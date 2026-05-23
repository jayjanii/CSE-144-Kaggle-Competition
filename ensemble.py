"""Average softmax probabilities from multiple runs into one ensemble submission.

Each input is a submission_probs.csv produced by inference.py:
    image_id,0,1,...,99
Rows are matched by image_id (order-independent), averaged (optionally weighted),
and the argmax is written as the ensemble submission.

Usage:
    python ensemble.py runA/submission_probs.csv runB/submission_probs.csv \
        --out submission_ensemble.csv
    # weighted, and also emit averaged probs for iterative pseudo-labeling:
    python ensemble.py a.csv b.csv c.csv --weights 1,1,0.5 \
        --out sub.csv --probs-out ensemble_probs.csv
"""

import argparse
import csv


def read_probs(path):
    """Return (class_columns, {image_id: [float probs]})."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)  # ["image_id", "0", "1", ...]
        classes = header[1:]
        probs = {row[0]: [float(x) for x in row[1:]] for row in r}
    return classes, probs


def main():
    p = argparse.ArgumentParser(description="Average run probs into an ensemble submission")
    p.add_argument("probs", nargs="+", help="submission_probs.csv files, one per member")
    p.add_argument("--weights", default=None,
                   help="Comma-separated per-member weights (default: equal)")
    p.add_argument("--out", default="submission_ensemble.csv", help="Output submission path")
    p.add_argument("--probs-out", default=None,
                   help="Optional: write averaged probs CSV (e.g. for round-2 pseudo-labels)")
    args = p.parse_args()

    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(args.probs):
            raise SystemExit(f"--weights has {len(weights)} values but {len(args.probs)} files given")
    else:
        weights = [1.0] * len(args.probs)
    wsum = sum(weights)

    classes = ids = acc = None
    for path, w in zip(args.probs, weights):
        c, probs = read_probs(path)
        if classes is None:
            classes, ids = c, list(probs.keys())
            acc = {i: [0.0] * len(classes) for i in ids}
        else:
            if c != classes:
                raise SystemExit(f"class-column mismatch in {path}")
            if set(probs.keys()) != set(ids):
                raise SystemExit(f"image_id set mismatch in {path}")
        for i in ids:
            row, ai = probs[i], acc[i]
            for k in range(len(classes)):
                ai[k] += w * row[k]

    for i in ids:
        acc[i] = [v / wsum for v in acc[i]]

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["image_id", "predicted_class"])
        for i in ids:
            pi = acc[i]
            pred = max(range(len(classes)), key=lambda k: pi[k])
            wtr.writerow([i, int(classes[pred])])
    print(f"Ensemble of {len(args.probs)} members (weights={weights}) → {args.out} ({len(ids)} rows)")

    if args.probs_out:
        with open(args.probs_out, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["image_id"] + classes)
            for i in ids:
                wtr.writerow([i] + [f"{v:.6f}" for v in acc[i]])
        print(f"Averaged probs → {args.probs_out}")


if __name__ == "__main__":
    main()

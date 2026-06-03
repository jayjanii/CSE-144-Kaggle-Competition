"""Sinkhorn class-balancing for the final submission.

The assignment spec states there are exactly 10 evaluation images per class
(100 classes x 10 = 1000 test images, perfectly balanced). That is a strong,
LEGITIMATE prior — it comes straight from the problem statement, uses no
external data.

Sinkhorn-Knopp rebalances a prediction matrix P[N, C] so that:
  - each row (image) still distributes over classes by confidence, and
  - each column (class) is predicted ~N/C times (the known balance).

This corrects systematic bias where the model over-predicts some classes and
starves others. Argmax of the balanced matrix is the final prediction.

Usage:
    python sinkhorn.py submission_probs.csv --out submission_balanced.csv
    # if the per-class count is not uniform, pass --per-class differently,
    # but per the spec it is exactly 10.
"""

import argparse
import csv

import numpy as np


def read_probs(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        classes = header[1:]
        ids, rows = [], []
        for row in r:
            ids.append(row[0])
            rows.append([float(x) for x in row[1:]])
    return classes, ids, np.array(rows)


def sinkhorn_balance(P, col_target, n_iters=100, tau=1.0, eps=1e-12):
    """Balance P[N,C] toward uniform column marginals via Sinkhorn iterations.

    P:          [N, C] non-negative scores (softmax probs work well)
    col_target: desired sum per class (e.g. N/C = 10 here)
    tau:        temperature applied to log P before balancing. tau<1 sharpens
                (trust the model more), tau>1 softens (let the balance prior
                dominate). 1.0 is a sensible default.
    """
    # work from sharpened/softened probabilities
    logP = np.log(np.clip(P, eps, 1.0)) / tau
    logP -= logP.max(axis=1, keepdims=True)
    M = np.exp(logP)  # [N, C]

    N, C = M.shape
    r = np.ones((N, 1))            # each image gets total mass 1 (one label)
    c = np.full((1, C), col_target)  # each class gets ~N/C predictions

    for _ in range(n_iters):
        # scale columns to hit the class marginal
        M *= (c / (M.sum(axis=0, keepdims=True) + eps))
        # scale rows back to unit mass per image
        M *= (r / (M.sum(axis=1, keepdims=True) + eps))
    return M


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probs", help="submission_probs.csv (image_id,0,1,...,C-1)")
    p.add_argument("--out", default="submission_balanced.csv")
    p.add_argument("--probs-out", default=None,
                   help="Optional: also write the balanced prob matrix")
    p.add_argument("--per-class", type=float, default=None,
                   help="Target predictions per class (default: N/C, i.e. balanced)")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Temperature before balancing. <1 trusts the model more, "
                        ">1 trusts the balance prior more (default 1.0).")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    classes, ids, P = read_probs(args.probs)
    N, C = P.shape
    col_target = args.per_class if args.per_class is not None else N / C
    print(f"{N} images, {C} classes -> target {col_target:.2f} predictions/class")

    # before
    pred0 = P.argmax(1)
    counts0 = np.bincount(pred0, minlength=C)
    print(f"Before: class-count range [{counts0.min()}, {counts0.max()}], "
          f"std {counts0.std():.2f}")

    B = sinkhorn_balance(P, col_target, n_iters=args.iters, tau=args.tau)

    pred1 = B.argmax(1)
    counts1 = np.bincount(pred1, minlength=C)
    print(f"After:  class-count range [{counts1.min()}, {counts1.max()}], "
          f"std {counts1.std():.2f}")
    changed = int((pred0 != pred1).sum())
    print(f"Changed {changed}/{N} predictions ({changed/N:.1%})")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_class"])
        for i, pr in zip(ids, pred1):
            w.writerow([i, int(classes[pr])])
    print(f"Wrote {args.out}")

    if args.probs_out:
        # renormalize rows to sum to 1 for a clean prob file
        Bn = B / (B.sum(1, keepdims=True) + 1e-12)
        with open(args.probs_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id"] + classes)
            for i, row in zip(ids, Bn):
                w.writerow([i] + [f"{v:.6f}" for v in row])
        print(f"Wrote {args.probs_out}")


if __name__ == "__main__":
    main()

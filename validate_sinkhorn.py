"""Validate whether Sinkhorn balancing helps, on leak-free OOF predictions.

The competition classes are uneven (train has 1079 images, some classes ~4,
others ~13), so a *uniform* 10-per-class prior may be wrong. This script uses
the cached frozen embeddings to build OOF probabilities via stratified CV,
then compares OOF accuracy:
    (a) raw probe argmax
    (b) + Sinkhorn with a UNIFORM class prior
    (c) + Sinkhorn with a prior PROPORTIONAL to the train class counts
        (the best guess for the test distribution if test mirrors train)

Whichever wins on OOF is the prior to use for the real submission. If (a)
wins, skip Sinkhorn entirely.

Usage:
    python validate_sinkhorn.py \
        --emb multi_probe/cache/openclip_ViT-SO400M-16-SigLIP2-384_webli_train.npy \
        --train-dir <auto> --tau 1.0
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def load_labels(train_dir):
    labels = []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    labels.append(int(cls))
    return np.array(labels)


def sinkhorn(P, col_target, n_iters=100, tau=1.0, eps=1e-12):
    logP = np.log(np.clip(P, eps, 1.0)) / tau
    logP -= logP.max(1, keepdims=True)
    M = np.exp(logP)
    N, C = M.shape
    r = np.ones((N, 1))
    c = col_target.reshape(1, C)
    for _ in range(n_iters):
        M *= (c / (M.sum(0, keepdims=True) + eps))
        M *= (r / (M.sum(1, keepdims=True) + eps))
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True, help="cached *_train.npy embeddings")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    train_dir = args.train_dir
    if not train_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg); train_dir = cfg["train_dir"]

    X = np.load(args.emb)
    y = load_labels(train_dir)
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"
    N, C = len(y), args.num_classes

    # OOF probabilities via stratified CV
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
    oof = np.zeros((N, C))
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pr = clf.predict_proba(X[va])
        for j, cls in enumerate(clf.classes_):
            oof[va, cls] = pr[:, j]

    raw_acc = (oof.argmax(1) == y).mean()

    # priors
    counts = np.bincount(y, minlength=C).astype(float)
    uniform = np.full(C, N / C)
    proportional = counts / counts.sum() * N

    acc_uniform = (sinkhorn(oof, uniform, tau=args.tau).argmax(1) == y).mean()
    acc_prop = (sinkhorn(oof, proportional, tau=args.tau).argmax(1) == y).mean()

    print(f"OOF over {N} images, {C} classes (tau={args.tau})\n")
    print(f"  raw probe argmax            {raw_acc:.4f}")
    print(f"  + Sinkhorn (uniform prior)  {acc_uniform:.4f}  ({(acc_uniform-raw_acc)*N:+.1f} imgs)")
    print(f"  + Sinkhorn (train-prop)     {acc_prop:.4f}  ({(acc_prop-raw_acc)*N:+.1f} imgs)")

    best = max([("raw", raw_acc), ("uniform", acc_uniform), ("proportional", acc_prop)],
               key=lambda t: t[1])
    print(f"\n=> best on OOF: {best[0]} ({best[1]:.4f})")
    if best[0] == "raw":
        print("   Sinkhorn does NOT help on this data — skip it for the submission.")
    else:
        print(f"   Use the {best[0]} prior for the real submission.")

    # sweep tau for the winning prior style (quick)
    print("\ntau sweep (proportional prior):")
    for t in (0.5, 1.0, 1.5, 2.0, 3.0):
        a = (sinkhorn(oof, proportional, tau=t).argmax(1) == y).mean()
        print(f"  tau={t:>3}  {a:.4f}")


if __name__ == "__main__":
    main()

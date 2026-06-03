"""Plot the logistic-probe training curve (train + held-out cross-entropy).

The submission probe uses sklearn LogisticRegression, which fits to convergence
silently. To VISUALIZE convergence (a presentation artifact — does not affect
predictions), this trains an equivalent linear softmax head with explicit
full-batch gradient steps on the cached frozen embeddings and logs the
cross-entropy on the train split and a held-out 20% split each iteration.

Reads the cached embeddings written by siglip2_pipeline.py / ensemble_probes.py,
so it needs no GPU and runs in seconds.

Usage:
    python probe_curve.py \
        --emb siglip2/cache_gopt/train.npy \
        --iters 60 --out probe_curve.png
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

SEED = 42


def load_labels(train_dir):
    labels = []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    labels.append(int(cls))
    return np.array(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb", required=True, help="cached train embeddings .npy")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="L2 (torch built-in, per-parameter — mirrors the probe's regularization)")
    p.add_argument("--out", default="probe_curve.png")
    p.add_argument("--title", default="Logistic-probe training curve\n"
                   "(cross-entropy on frozen SigLIP-2 gopt features)")
    args = p.parse_args()

    train_dir = args.train_dir
    if not train_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg); train_dir = cfg["train_dir"]

    X = np.load(args.emb).astype(np.float32)
    y = load_labels(train_dir)
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    torch.manual_seed(SEED)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)

    # balanced class weights (mirrors class_weight="balanced")
    counts = np.bincount(ytr, minlength=args.num_classes).astype(np.float32)
    w = len(ytr) / (args.num_classes * np.maximum(counts, 1))
    cw = torch.tensor(w, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    crit_plain = nn.CrossEntropyLoss()

    clf = nn.Linear(X.shape[1], args.num_classes)
    # Adam full-batch with torch's built-in (properly-scaled) weight decay.
    # Gives a smooth monotonic descent for a small linear probe.
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tr_losses, va_losses = [], []
    for it in range(args.iters):
        clf.train(); opt.zero_grad()
        loss = crit(clf(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            tr = crit_plain(clf(Xtr_t), ytr_t).item()
            va = crit_plain(clf(Xva_t), yva_t).item()
            va_acc = (clf(Xva_t).argmax(1) == yva_t).float().mean().item()
        tr_losses.append(tr); va_losses.append(va)
        if (it + 1) % 20 == 0:
            print(f"iter {it+1:3d}  train CE {tr:.3f}  val CE {va:.3f}  val acc {va_acc:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(1, args.iters + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(xs, tr_losses, color="#1E2761", lw=2.6, label="training loss")
    ax.plot(xs, va_losses, color="#E0A23B", lw=2.6, label="validation loss (held-out 20%)")
    ax.set_xlabel("iteration", fontsize=12)
    ax.set_ylabel("cross-entropy loss", fontsize=12)
    ax.set_title(args.title, fontsize=14, fontweight="bold", color="#1E2761")
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_xlim(1, args.iters)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}  (final: train {tr_losses[-1]:.3f}, val {va_losses[-1]:.3f})")


if __name__ == "__main__":
    main()

# Loss / accuracy curve for the linear probe (a figure for the report).
# sklearn's LogisticRegression fits silently, so to actually see convergence we
# train an equivalent linear softmax head with full-batch Adam on the cached
# frozen embeddings and log train/val cross-entropy and val accuracy each step.
# Runs on CPU in a few seconds; does not touch the submission predictions.

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data import get_data_dirs

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
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True, help="cached train embeddings .npy")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--out", default="probe_curve.png")
    p.add_argument("--title", default="Linear-probe training curve")
    args = p.parse_args()

    train_dir = args.train_dir or get_data_dirs()[0]

    X = np.load(args.emb).astype(np.float32)
    y = load_labels(train_dir)
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)

    # balanced class weights, mirroring the probe's class_weight="balanced"
    counts = np.bincount(ytr, minlength=args.num_classes).astype(np.float32)
    cw = torch.tensor(len(ytr) / (args.num_classes * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    crit_plain = nn.CrossEntropyLoss()

    clf = nn.Linear(X.shape[1], args.num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tr_losses, va_losses, va_accs = [], [], []
    for it in range(args.iters):
        clf.train()
        opt.zero_grad()
        crit(clf(Xtr_t), ytr_t).backward()
        opt.step()
        clf.eval()
        with torch.no_grad():
            tr = crit_plain(clf(Xtr_t), ytr_t).item()
            va = crit_plain(clf(Xva_t), yva_t).item()
            acc = (clf(Xva_t).argmax(1) == yva_t).float().mean().item()
        tr_losses.append(tr)
        va_losses.append(va)
        va_accs.append(acc)
        if (it + 1) % 20 == 0:
            print(f"iter {it+1:3d}  train CE {tr:.3f}  val CE {va:.3f}  val acc {acc:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(1, args.iters + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(xs, tr_losses, color="#1E2761", lw=2.4, label="train")
    ax1.plot(xs, va_losses, color="#E0A23B", lw=2.4, label="val (held-out 20%)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)
    ax2.plot(xs, va_accs, color="#1E2761", lw=2.4)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy")
    ax2.grid(alpha=0.25)
    for ax in (ax1, ax2):
        ax.set_xlim(1, args.iters)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(args.title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}  (final train {tr_losses[-1]:.3f}, val {va_losses[-1]:.3f}, acc {va_accs[-1]:.4f})")


if __name__ == "__main__":
    main()

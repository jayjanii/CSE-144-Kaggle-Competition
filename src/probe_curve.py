# loss/acc curves across all folds for the report. trains a linear head on each
# fold of the cached embeddings to show convergence (faint per-fold + bold mean).

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

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


def train_fold(Xtr, ytr, Xva, yva, num_classes, iters, lr, weight_decay):
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    cw = torch.tensor(len(ytr) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    crit_plain = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    clf = nn.Linear(Xtr.shape[1], num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    tr_loss, va_loss, va_acc = [], [], []
    for _ in range(iters):
        clf.train()
        opt.zero_grad()
        crit(clf(Xtr_t), ytr_t).backward()
        opt.step()
        clf.eval()
        with torch.no_grad():
            tr_loss.append(crit_plain(clf(Xtr_t), ytr_t).item())
            va_loss.append(crit_plain(clf(Xva_t), yva_t).item())
            va_acc.append((clf(Xva_t).argmax(1) == yva_t).float().mean().item())
    return tr_loss, va_loss, va_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True, help="cached train embeddings .npy")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--out", default="probe_curve.png")
    p.add_argument("--title", default="Linear-probe convergence (per fold)")
    args = p.parse_args()

    train_dir = args.train_dir or get_data_dirs()[0]
    X = np.load(args.emb).astype(np.float32)
    y = load_labels(train_dir)
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    tr, va, acc = [], [], []
    for fi, (i_tr, i_va) in enumerate(skf.split(X, y)):
        a, b, c = train_fold(X[i_tr], y[i_tr], X[i_va], y[i_va],
                             args.num_classes, args.iters, args.lr, args.weight_decay)
        tr.append(a)
        va.append(b)
        acc.append(c)
        print(f"fold {fi}: final val acc {c[-1]:.4f}")
    tr, va, acc = np.array(tr), np.array(va), np.array(acc)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(1, args.iters + 1)
    navy, gold = "#1E2761", "#E0A23B"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # loss: faint per-fold val + bold mean train/val
    for v in va:
        ax1.plot(xs, v, color=gold, lw=0.8, alpha=0.35)
    ax1.plot(xs, va.mean(0), color=gold, lw=2.4, label="val (mean)")
    ax1.plot(xs, tr.mean(0), color=navy, lw=2.4, label="train (mean)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    # accuracy: faint per-fold + bold mean
    for a in acc:
        ax2.plot(xs, a, color=navy, lw=0.8, alpha=0.35)
    ax2.plot(xs, acc.mean(0), color=navy, lw=2.4, label="mean")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"Validation accuracy ({args.folds}-fold)")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    for ax in (ax1, ax2):
        ax.set_xlim(1, args.iters)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(args.title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    fa = acc[:, -1]
    print(f"wrote {args.out}  (final val acc {fa.mean():.4f} +/- {fa.std():.4f})")


if __name__ == "__main__":
    main()

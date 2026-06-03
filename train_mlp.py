"""Train an MLP head on frozen SigLIP-2 embeddings — a real trained model.

Unlike the sklearn logistic probe (which fits silently), this trains a small
multi-layer head with an explicit loop, so it produces honest train/validation
LOSS and ACCURACY curves. It is also a genuine ensemble member: its softmax
outputs can be fused with the logistic probe + text branches.

Outputs:
  - <out>_curve.png      train/val loss + accuracy (two panels)
  - <out>_test_probs.csv softmax over the test set (for ensembling)
  - prints 5-fold OOF accuracy (leak-free, for honest comparison)

Reads cached embeddings from siglip2_pipeline.py / ensemble_probes.py — no GPU
re-encoding needed (though a GPU speeds up the MLP a little).

Usage:
    python train_mlp.py \
        --emb-train siglip2/cache_gopt/train.npy \
        --emb-test  siglip2/cache_gopt/test.npy \
        --out mlp_gopt
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def load_test_fnames(test_dir):
    return sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                  key=lambda f: int(os.path.splitext(f)[0]))


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def make_model(d_in, n_cls, hidden, dropout):
    torch.manual_seed(SEED)
    return MLP(d_in, n_cls, hidden, dropout).to(DEVICE)


def train_loop(Xtr, ytr, Xva, yva, n_cls, args, log=False):
    """Train an MLP, return (model, history). history has loss/acc per epoch."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.long, device=DEVICE)

    counts = np.bincount(ytr, minlength=n_cls).astype(np.float32)
    cw = torch.tensor(len(ytr) / (n_cls * np.maximum(counts, 1)),
                      dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    crit_eval = nn.CrossEntropyLoss()

    model = make_model(Xtr.shape[1], n_cls, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = len(Xtr_t)
    bs = args.batch_size
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    g = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lo_tr = model(Xtr_t); lo_va = model(Xva_t)
            hist["tr_loss"].append(crit_eval(lo_tr, ytr_t).item())
            hist["va_loss"].append(crit_eval(lo_va, yva_t).item())
            hist["tr_acc"].append((lo_tr.argmax(1) == ytr_t).float().mean().item())
            hist["va_acc"].append((lo_va.argmax(1) == yva_t).float().mean().item())
        if log and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d}  tr_loss {hist['tr_loss'][-1]:.3f} "
                  f"va_loss {hist['va_loss'][-1]:.3f}  va_acc {hist['va_acc'][-1]:.4f}")
    return model, hist


@torch.no_grad()
def predict(model, X):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return model(xb).softmax(1).cpu().numpy()


def make_paper_figure(hist, epochs, out):
    """Two-panel publication-style figure (loss | accuracy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # ---- publication rcParams ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "figure.dpi": 150,
    })
    TRAIN = "#2C3E70"   # deep slate blue
    VAL = "#C0504D"     # muted brick red
    GRID = "#B8B8B8"

    ep = np.arange(1, epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # best epoch marker (min val loss)
    best = int(np.argmin(hist["va_loss"]))

    # ---- loss panel ----
    a1.plot(ep, hist["tr_loss"], color=TRAIN, lw=1.8, label="Train")
    a1.plot(ep, hist["va_loss"], color=VAL, lw=1.8, label="Validation")
    a1.axvline(best + 1, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=0)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-entropy loss")
    a1.set_title("(a)  Loss", loc="left", fontweight="bold")
    a1.legend(frameon=False, handlelength=1.6)

    # ---- accuracy panel ----
    a2.plot(ep, np.array(hist["tr_acc"]) * 100, color=TRAIN, lw=1.8, label="Train")
    a2.plot(ep, np.array(hist["va_acc"]) * 100, color=VAL, lw=1.8, label="Validation")
    a2.axvline(best + 1, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=0)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy (%)")
    a2.set_title("(b)  Accuracy", loc="left", fontweight="bold")
    a2.legend(frameon=False, loc="lower right", handlelength=1.6)
    # annotate best val accuracy
    bva = hist["va_acc"][best] * 100
    a2.annotate(f"{bva:.1f}%", xy=(best + 1, bva),
                xytext=(best + 1 + epochs * 0.04, bva - 6),
                fontsize=10, color=VAL,
                arrowprops=dict(arrowstyle="-", color=VAL, lw=0.8))

    for ax in (a1, a2):
        ax.set_xlim(1, epochs)
        ax.grid(True, which="major", color=GRID, lw=0.5, alpha=0.5)
        ax.xaxis.set_minor_locator(MultipleLocator(max(1, epochs // 20)))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(which="both", top=False, right=False)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    # also a vector PDF — what papers actually embed
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-train", required=True)
    p.add_argument("--emb-test", required=True)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out", default="mlp_gopt")
    args = p.parse_args()

    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg)
        train_dir = train_dir or cfg["train_dir"]; test_dir = test_dir or cfg["test_dir"]

    X = np.load(args.emb_train).astype(np.float32)
    Xte = np.load(args.emb_test).astype(np.float32)
    y = load_labels(train_dir)
    te_fnames = load_test_fnames(test_dir)
    C = args.num_classes
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"

    # ---- 1) showcase curve: single 80/20 split ----
    print("Training MLP for the train/val curve (80/20 split)...")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    _, hist = train_loop(Xtr, ytr, Xva, yva, C, args, log=True)

    curve = f"{args.out}_curve.png"
    make_paper_figure(hist, args.epochs, curve)
    print(f"Wrote {curve}  (final val_acc {hist['va_acc'][-1]:.4f})")

    # ---- 2) leak-free OOF accuracy (5-fold) ----
    print("\n5-fold OOF for the MLP...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), C))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m, _ = train_loop(X[tr], y[tr], X[va], y[va], C, args)
        oof[va] = predict(m, X[va])
    oof_acc = (oof.argmax(1) == y).mean()
    print(f"MLP OOF acc: {oof_acc:.4f}  (compare to logistic probe 0.9453)")

    # ---- 3) test probs for ensembling (train on all data) ----
    print("\nTraining final MLP on all data for test predictions...")
    m, _ = train_loop(X, y, X, y, C, args)  # val=train just for the call signature
    Pte = predict(m, Xte)
    probs_path = f"{args.out}_test_probs.csv"
    with open(probs_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id"] + [str(c) for c in range(C)])
        for fn, row in zip(te_fnames, Pte):
            w.writerow([fn] + [f"{v:.6f}" for v in row])
    print(f"Wrote {probs_path}  — slot into ensemble_probes / ensemble.py as a member")


if __name__ == "__main__":
    main()

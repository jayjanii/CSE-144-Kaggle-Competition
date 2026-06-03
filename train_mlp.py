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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, args.epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(ep, hist["tr_loss"], color="#1E2761", lw=2.4, label="training loss")
    a1.plot(ep, hist["va_loss"], color="#E0A23B", lw=2.4, label="validation loss (held-out 20%)")
    a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy loss")
    a1.set_title("MLP head — loss", fontweight="bold", color="#1E2761")
    a1.legend(frameon=False); a1.grid(alpha=0.25); a1.set_xlim(1, args.epochs)
    a2.plot(ep, hist["tr_acc"], color="#1E2761", lw=2.4, label="training accuracy")
    a2.plot(ep, hist["va_acc"], color="#E0A23B", lw=2.4, label="validation accuracy")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy")
    a2.set_title("MLP head — accuracy", fontweight="bold", color="#1E2761")
    a2.legend(frameon=False, loc="lower right"); a2.grid(alpha=0.25); a2.set_xlim(1, args.epochs)
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    curve = f"{args.out}_curve.png"
    fig.savefig(curve, dpi=150)
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

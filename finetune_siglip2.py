"""Partial-unfreeze fine-tune of a SigLIP backbone — a genuine trained model.

Unlike siglip2_pipeline.py (frozen backbone + logistic probe), this UPDATES the
backbone weights: it freezes the early transformer blocks and trains the last N
blocks + final norm + head with layer-wise LR decay. That makes it
  (a) a real training run with honest train/val LOSS and ACCURACY curves, and
  (b) a decorrelated ensemble member — a fine-tuned model makes different
      mistakes than a frozen probe, which is exactly what helps a fusion.

It reuses the proven pieces from train.py (data loading, albumentations aug,
mixup/cutmix, model builder, normalization) so it stays consistent with the
0.917 SO400M run, and reuses train_mlp.make_paper_figure for the figure.

A single fine-tuned SO400M (~0.92) is individually WEAKER than the frozen probe
(~0.97). Its value is purely as an ensemble member — fuse its test probs with
the probe's. See the bottom of this file for the fuse snippet.

Outputs (under --out/):
  curve.png / curve.pdf     paper-style train/val loss + accuracy
  submission_probs.csv      image_id + 100 softmax columns (for fusion)
  submission.csv            id,label  (standalone Kaggle submission)
  test_probs.npy            same probs as an array aligned to sorted test files
  best.pth                  best-val checkpoint

Usage:
    # confirm the exact backbone tag first:
    python -c "import timm; print(timm.list_models('*so400m*siglip*'))"

    python finetune_siglip2.py \
        --model vit_so400m_patch16_siglip_384.v2_webli --input-size 384 \
        --unfreeze-blocks 6 --epochs 30 --out ft_so400m
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from train import (
    AlbumentationsDataset,
    LabelSmoothingCE,
    _numeric_image_folder,
    apply_mix,
    build_model,
    enable_grad_checkpointing,
    get_classifier,
    get_train_transform,
    get_val_transform,
    maybe_download_data,
    mixed_loss,
    resolve_normalization,
    set_seed,
)
from train_mlp import make_paper_figure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Partial unfreeze + layer-wise LR ──────────────────────────────────────────


def get_blocks(model):
    """Return the transformer-block ModuleList for a timm ViT-style model."""
    for attr in ("blocks", "stages"):
        if hasattr(model, attr):
            return list(getattr(model, attr))
    raise AttributeError("could not find transformer blocks on the model")


def partial_unfreeze(model, model_name, n_blocks):
    """Freeze everything, then unfreeze head + final norm + the last n_blocks.

    Returns the list of unfrozen blocks (deepest last) so the optimizer can give
    them layer-wise-decayed LRs.
    """
    for p in model.parameters():
        p.requires_grad = False

    head = get_classifier(model, model_name)
    for p in head.parameters():
        p.requires_grad = True

    for norm_attr in ("norm", "fc_norm", "norm_pre"):
        if hasattr(model, norm_attr) and isinstance(getattr(model, norm_attr), nn.Module):
            for p in getattr(model, norm_attr).parameters():
                p.requires_grad = True

    blocks = get_blocks(model)
    unfrozen = blocks[-n_blocks:] if n_blocks > 0 else []
    for blk in unfrozen:
        for p in blk.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Unfroze head + final norm + last {n_blocks}/{len(blocks)} blocks "
        f"-> training {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params "
        f"({100*n_train/n_total:.1f}%)"
    )
    return unfrozen


def build_optimizer(model, model_name, unfrozen_blocks, head_lr, backbone_lr,
                    llrd_decay, weight_decay):
    """AdamW with layer-wise LR decay across the unfrozen blocks.

    head/final-norm -> head_lr; deepest unfrozen block -> backbone_lr; each
    shallower unfrozen block -> *= llrd_decay. No weight decay on norms/biases.
    """
    seen = set()
    groups = []

    def add(params, lr):
        decay, no_decay = [], []
        for name, p in params:
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (no_decay if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name)
             else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    # head first (highest lr)
    add(get_classifier(model, model_name).named_parameters(), head_lr)

    # unfrozen blocks, deepest -> shallowest with LLRD
    n = len(unfrozen_blocks)
    for i, blk in enumerate(reversed(unfrozen_blocks)):  # i=0 is deepest
        add(blk.named_parameters(), backbone_lr * (llrd_decay ** i))

    # any remaining trainable params (final norm, etc.)
    add(model.named_parameters(), backbone_lr * (llrd_decay ** max(n - 1, 0)))

    lrs = [g["lr"] for g in groups]
    print(f"  optimizer: {len(groups)} groups, lr {min(lrs):.2e}–{max(lrs):.2e}")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


# ── Train / eval ──────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, cfg):
    model.train()
    optimizer.zero_grad()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, ya, yb, lam = apply_mix(x, y, cfg, use_mix=True)
        with torch.amp.autocast(DEVICE):
            logits = model(x)
            loss = mixed_loss(criterion, logits, y, ya, yb, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()


@torch.inference_mode()
def evaluate(model, loader):
    """Clean (no-aug, no-smoothing) loss + top-1 — comparable across train/val."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast(DEVICE):
            logits = model(x)
        tot_loss += ce(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return tot_loss / total, correct / total


# ── Test-time inference (TTA -> probs) ────────────────────────────────────────


def _tta_transforms(size, mean, std):
    """Center-crop + horizontal-flip + a second (larger) scale = 4 views."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    def build(resize, flip):
        ops = [A.SmallestMaxSize(max_size=resize),
               A.CenterCrop(height=size, width=size)]
        if flip:
            ops.append(A.HorizontalFlip(p=1.0))
        ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
        return A.Compose(ops)

    r1 = int(round(size * 480 / 448))
    r2 = int(round(size * 520 / 448))
    return [build(r1, False), build(r1, True), build(r2, False), build(r2, True)]


@torch.inference_mode()
def predict_pils(model, imgs, size, mean, std, num_classes=100, batch_size=32):
    """4-view TTA softmax over a list of RGB numpy images -> (n, C) probs."""
    model.eval()
    views = _tta_transforms(size, mean, std)
    n = len(imgs)
    probs = np.zeros((n, num_classes), dtype=np.float64)
    for tf in views:
        for i in range(0, n, batch_size):
            chunk = imgs[i:i + batch_size]
            xb = torch.stack([tf(image=im)["image"] for im in chunk]).to(DEVICE)
            with torch.amp.autocast(DEVICE):
                p = model(xb).softmax(1).float().cpu().numpy()
            probs[i:i + len(chunk)] += p
    probs /= len(views)
    return probs


def list_test_fnames(test_dir):
    return sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda f: int(os.path.splitext(f)[0]),
    )


def predict_test(model, test_dir, size, mean, std, num_classes=100, batch_size=32):
    fnames = list_test_fnames(test_dir)
    imgs = [np.array(Image.open(os.path.join(test_dir, f)).convert("RGB")) for f in fnames]
    probs = predict_pils(model, imgs, size, mean, std, num_classes, batch_size)
    return fnames, probs


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_so400m_patch14_siglip_378.webli_ft_in1k",
                   help="timm backbone. Try vit_so400m_patch16_siglip_384.v2_webli "
                        "(SigLIP 2). Confirm with timm.list_models('*so400m*siglip*').")
    p.add_argument("--input-size", type=int, default=378)
    p.add_argument("--unfreeze-blocks", type=int, default=6,
                   help="How many of the last transformer blocks to train (+head+norm)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--llrd-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    # aggressive-but-SigLIP-appropriate regularization (see module docstring)
    p.add_argument("--aug-style", default="convnext", choices=("default", "convnext"))
    p.add_argument("--mixup-alpha", type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--folds", type=int, default=None,
                   help="K-fold bagging: train K models on K stratified folds, "
                        "average their test probs, and emit a leak-free OOF matrix "
                        "(every image predicted by a model that never saw it). "
                        "Uses 100%% of the data. Omit for a single 85/15 split + curve.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--out", default="ft_so400m")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model_and_norm(args):
    """Build a fresh backbone, partially unfreeze it, return (model, mean, std, unfrozen)."""
    model = build_model(args.num_classes, args.model, args.drop_path_rate)
    mean, std = resolve_normalization(model, args.model)
    unfrozen = partial_unfreeze(model, args.model, args.unfreeze_blocks)
    if args.grad_checkpointing:
        enable_grad_checkpointing(model, args.model)
    return model, mean, std, unfrozen


def train_fold(args, cfg, base, tr_idx, va_idx, tag="", select_best=False):
    """Train one model on tr_idx, tracking metrics on va_idx.

    select_best=True keeps the best-val checkpoint (single-split curve mode).
    select_best=False keeps the FINAL-epoch weights — required for honest OOF in
    k-fold mode (selecting on the held-out fold you then score would leak).
    Returns (model, mean, std, hist).
    """
    model, mean, std, unfrozen = make_model_and_norm(args)
    train_tf = get_train_transform(args.input_size, mean, std, args.aug_style)
    eval_tf = get_val_transform(args.input_size, mean, std)
    train_ds = AlbumentationsDataset(Subset(base, tr_idx), train_tf)
    train_eval_ds = AlbumentationsDataset(Subset(base, tr_idx), eval_tf)
    val_ds = AlbumentationsDataset(Subset(base, va_idx), eval_tf)

    dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kw)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **dl_kw)

    criterion = LabelSmoothingCE(args.label_smoothing)
    optimizer = build_optimizer(model, args.model, unfrozen, args.head_lr,
                                args.backbone_lr, args.llrd_decay, args.weight_decay)
    steps = args.epochs * len(train_loader)
    warmup = int(args.warmup_frac * steps)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)
    scaler = torch.amp.GradScaler(DEVICE)

    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best_acc, best_sd = 0.0, None
    print(f"\nTraining {args.epochs} epochs {tag}...")
    for ep in range(args.epochs):
        t0 = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, cfg)
        tr_loss, tr_acc = evaluate(model, train_eval_loader)
        va_loss, va_acc = evaluate(model, val_loader)
        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
        hist["tr_acc"].append(tr_acc); hist["va_acc"].append(va_acc)
        star = " "
        if va_acc > best_acc:
            best_acc = va_acc
            if select_best:
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "*"
        print(f"[E{ep:02d}] tr_loss={tr_loss:.3f} va_loss={va_loss:.3f} "
              f"tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} {star} {time.time()-t0:.0f}s")

    if select_best and best_sd is not None:
        model.load_state_dict(best_sd)
        print(f"Loaded best-val weights ({best_acc:.4f}).")
    else:
        print(f"Keeping final-epoch weights (best seen {best_acc:.4f}).")
    return model, mean, std, hist


def _write_test_outputs(out, fnames, probs, num_classes):
    np.save(str(out / "test_probs.npy"), probs)
    with open(out / "submission_probs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id"] + [str(c) for c in range(num_classes)])
        for fn, row in zip(fnames, probs):
            w.writerow([fn] + [f"{v:.6f}" for v in row])
    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for fn, row in zip(fnames, probs):
            w.writerow([os.path.splitext(fn)[0], int(row.argmax())])
    print(f"Wrote {out/'submission_probs.csv'}, {out/'submission.csv'}, {out/'test_probs.npy'}")


def run_single(args, cfg, base, test_dir, out):
    """85/15 stratified split: train one model, draw the curve, predict test."""
    targets = np.array(base.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(targets)), targets))
    print(f"Single split: {len(tr_idx)} train / {len(va_idx)} val (stratified)")

    model, mean, std, hist = train_fold(args, cfg, base, tr_idx, va_idx, select_best=True)
    print(f"\nBest val acc: {max(hist['va_acc']):.4f}  (vs frozen probe ~0.945 OOF / 0.972 LB)")

    curve = str(out / "curve.png")
    make_paper_figure(hist, args.epochs, curve)
    print(f"Wrote {curve} (+ .pdf)")

    print("Predicting test set with 4-view TTA ...")
    fnames, probs = predict_test(model, test_dir, args.input_size, mean, std,
                                 args.num_classes, args.batch_size)
    _write_test_outputs(out, fnames, probs, args.num_classes)


def run_kfold(args, cfg, base, test_dir, out):
    """K-fold bagging: K models on K stratified folds -> leak-free OOF + bagged test."""
    from train import _fold_assignment

    targets = np.array(base.targets)
    N, C, K = len(targets), args.num_classes, args.folds
    fold_of = _fold_assignment(base.targets, K, args.seed)

    fnames = list_test_fnames(test_dir)
    test_imgs = [np.array(Image.open(os.path.join(test_dir, f)).convert("RGB")) for f in fnames]
    test_accum = np.zeros((len(fnames), C), dtype=np.float64)
    oof = np.zeros((N, C), dtype=np.float64)
    hist0 = None

    for f in range(K):
        va_idx = np.where(fold_of == f)[0]
        tr_idx = np.where(fold_of != f)[0]
        print(f"\n=== Fold {f}/{K}: {len(tr_idx)} train / {len(va_idx)} held-out ===")
        model, mean, std, hist = train_fold(
            args, cfg, base, tr_idx, va_idx, tag=f"[fold {f}] ", select_best=False)
        if f == 0:
            hist0 = hist  # representative curve

        # leak-free OOF: predict this fold's held-out images with its own model
        va_imgs = [np.array(base[i][0].convert("RGB")) for i in va_idx]
        oof[va_idx] = predict_pils(model, va_imgs, args.input_size, mean, std, C, args.batch_size)
        fold_oof_acc = (oof[va_idx].argmax(1) == targets[va_idx]).mean()
        print(f"  fold {f} held-out TTA acc: {fold_oof_acc:.4f}")

        # bagged test prediction
        test_accum += predict_pils(model, test_imgs, args.input_size, mean, std, C, args.batch_size)

        del model
        torch.cuda.empty_cache()

    oof_acc = (oof.argmax(1) == targets).mean()
    print(f"\n=== Bagged OOF acc ({K} folds, 4-view TTA): {oof_acc:.4f} "
          f"over {N} images (vs frozen probe ~0.945 OOF) ===")

    # paper curve from fold 0
    if hist0 is not None:
        curve = str(out / "curve.png")
        make_paper_figure(hist0, args.epochs, curve)
        print(f"Wrote {curve} (+ .pdf)  [fold-0 curve]")

    # ── OOF outputs in canonical (class, filename) order so they align with the
    #    probe pipeline's OOF for fusion ──
    order = sorted(range(N), key=lambda i: (int(targets[i]), base.samples[i][0]))
    oof_canon = oof[order]
    np.save(str(out / "oof_probs.npy"), oof_canon)
    with open(out / "oof_probs.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv); w.writerow(["path", "label"] + [str(c) for c in range(C)])
        for i in order:
            w.writerow([base.samples[i][0], int(targets[i])]
                       + [f"{v:.6f}" for v in oof[i]])
    print(f"Wrote {out/'oof_probs.npy'} (+ .csv) — canonical order for fusion")

    # ── bagged test outputs ──
    test_probs = test_accum / K
    _write_test_outputs(out, fnames, test_probs, C)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # resolve data dirs (reuse train.py's downloader/config if not given)
    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        from train import CONFIG
        cfg0 = dict(CONFIG)
        maybe_download_data(cfg0)
        train_dir = train_dir or cfg0["train_dir"]
        test_dir = test_dir or cfg0["test_dir"]

    print(f"Backbone {args.model} @ {args.input_size}px, unfreeze last "
          f"{args.unfreeze_blocks} blocks, aug={args.aug_style}")
    cfg = {
        "grad_clip": args.grad_clip,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "mixup_cutmix_prob": args.mix_prob,
    }
    base = _numeric_image_folder(train_dir)

    if args.folds and args.folds > 1:
        run_kfold(args, cfg, base, test_dir, out)
    else:
        run_single(args, cfg, base, test_dir, out)


# ── Fusing with the frozen-probe pipeline ─────────────────────────────────────
# K-fold mode emits two aligned matrices you need for an honest fusion:
#   <out>/oof_probs.npy   leak-free OOF over all 1079 train images (canonical
#                         (class, filename) order)
#   <out>/test_probs.npy  bagged test probs (sorted test filenames)
#
# Tune the fuse weight on OOF (never the public LB), then apply the SAME weight
# to the test probs:
#
#   import numpy as np
#   Q_oof = np.load("ft_so400m/oof_probs.npy")     # this fine-tune, OOF
#   P_oof = np.load("probe_oof_probs.npy")         # frozen probe+text, OOF (canonical order)
#   y     = np.load("oof_labels.npy")              # canonical-order labels
#   best_w, best_acc = 0.0, (P_oof.argmax(1) == y).mean()
#   for w in np.linspace(0, 0.6, 13):
#       acc = ((1 - w) * P_oof + w * Q_oof).argmax(1) == y
#       if acc.mean() > best_acc: best_w, best_acc = w, acc.mean()
#   print("best w", best_w, "OOF acc", best_acc)   # apply best_w to the test probs
#
# The fine-tune is weaker alone, so the winning w is usually small (0.15–0.30):
# its job is to break ties the probe gets wrong. If best_w == 0, it adds nothing
# and you ship the probe alone — that's a valid, honest outcome.

if __name__ == "__main__":
    main()

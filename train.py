"""DINOv2-Giant three-phase fine-tuning — single self-contained script."""

import csv
import math
import os
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from transformers import get_cosine_schedule_with_warmup

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "num_classes": 100,
    "input_size": 448,
    "batch_size": 128,
    "phase1_epochs": 5,
    "phase2_epochs": 30,
    "phase3_epochs": 15,
    "phase1_lr": 3e-3,
    "phase2_base_lr": 6e-5,
    "phase3_base_lr": 1e-5,
    "llrd_decay": 0.85,
    "head_lr_scale": 10,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.4,
    "cutmix_alpha": 1.0,
    "mixup_cutmix_prob": 0.5,
    "grad_clip": 1.0,
    "warmup_epochs": 1,
    "eta_min": 1e-7,
    "tta_sizes": [392, 448, 518],
    "train_dir": "data/train",
    "test_dir": "data/test",
    "output_dir": "outputs/",
}
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
NUM_BLOCKS = 40  # ViT-g/14 has 40 transformer blocks


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 100) -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    model.head = nn.Linear(1536, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model.to(DEVICE)


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_train_transform(size: int = 448) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
            border_mode=0, p=0.4,
        ),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.CoarseDropout(
            max_holes=8, max_height=56, max_width=56, min_holes=1,
            fill_value=0, p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
        ToTensorV2(),
    ])


def get_val_transform(size: int = 448) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(max_size=480),
        A.CenterCrop(size, size),
        A.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class AlbumentationsDataset(Dataset):
    """Wraps an ImageFolder (no torchvision transform) with an albumentations pipeline."""

    def __init__(self, dataset, transform: A.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # PIL Image, int
        img_np = np.array(img)
        return self.transform(image=img_np)["image"], label


def _numeric_image_folder(root: str) -> datasets.ImageFolder:
    """ImageFolder with class folder names mapped to their integer values."""
    ds = datasets.ImageFolder(root)  # no transform → PIL images
    ds.class_to_idx = {cls: int(cls) for cls in ds.classes}
    ds.targets = [ds.class_to_idx[ds.classes[t]] for t in ds.targets]
    ds.samples = [
        (path, ds.class_to_idx[ds.classes[old_idx]])
        for path, old_idx in ds.samples
    ]
    return ds


def get_dataloaders(cfg: dict):
    size = cfg["input_size"]
    bs = cfg["batch_size"]

    base_ds = _numeric_image_folder(cfg["train_dir"])
    n = len(base_ds)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[: int(0.8 * n)]
    val_idx = indices[int(0.8 * n) :]

    train_ds = AlbumentationsDataset(Subset(base_ds, train_idx), get_train_transform(size))
    val_ds = AlbumentationsDataset(Subset(base_ds, val_idx), get_val_transform(size))

    loader_kwargs = dict(num_workers=8, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ── MixUp / CutMix ───────────────────────────────────────────────────────────

def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_h = int(H * math.sqrt(1 - lam))
    cut_w = int(W * math.sqrt(1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], lam


def apply_mix(x, y, cfg, use_mix: bool):
    """Returns (input, ya, yb, lam). ya==yb==None means no mixing was applied."""
    if not use_mix or random.random() > cfg["mixup_cutmix_prob"]:
        return x, y, None, None
    if random.random() < 0.5:
        mixed, ya, yb, lam = _mixup(x, y, cfg["mixup_alpha"])
    else:
        mixed, ya, yb, lam = _cutmix(x, y, cfg["cutmix_alpha"])
    return mixed, ya, yb, lam


def mixed_loss(criterion, logits, y, ya, yb, lam):
    if ya is None:
        return criterion(logits, y)
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ── Loss ──────────────────────────────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


# ── Optimizer / LLRD ─────────────────────────────────────────────────────────

def _no_decay(name: str) -> bool:
    return "bias" in name or "norm" in name.lower()


def get_param_groups(named_params, lr: float, weight_decay: float) -> list:
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if _no_decay(name) else decay).append(p)
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def build_llrd_optimizer(
    model: nn.Module,
    base_lr: float,
    decay: float,
    head_lr_scale: float = 10,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """
    Layer-wise LR decay across NUM_BLOCKS transformer blocks.

    Group order: head, block[0..39] (low→high lr), patch_embed/rest.
    Each block is split into decay / no-decay sub-groups.
    """
    head_lr = base_lr * head_lr_scale
    patch_lr = base_lr * (decay ** NUM_BLOCKS)

    param_groups = []

    # ── head ──
    head_d, head_nd = [], []
    for name, p in model.head.named_parameters():
        if p.requires_grad:
            (head_nd if _no_decay(name) else head_d).append(p)
    param_groups += [
        {"params": head_d,  "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_nd, "lr": head_lr, "weight_decay": 0.0},
    ]
    print(f"  head                lr={head_lr:.2e}")

    # ── transformer blocks ──
    # block[i=39] → distance_from_top=0 → lr=base_lr (highest)
    # block[i=0]  → distance_from_top=39 → lr=base_lr*decay^39 (lowest)
    for i in range(NUM_BLOCKS):
        dist = NUM_BLOCKS - 1 - i
        block_lr = base_lr * (decay ** dist)
        blk_d, blk_nd = [], []
        for name, p in model.blocks[i].named_parameters():
            if p.requires_grad:
                (blk_nd if _no_decay(name) else blk_d).append(p)
        param_groups += [
            {"params": blk_d,  "lr": block_lr, "weight_decay": weight_decay},
            {"params": blk_nd, "lr": block_lr, "weight_decay": 0.0},
        ]
        if i in (0, NUM_BLOCKS - 1):
            print(f"  blocks[{i:2d}]          lr={block_lr:.2e}")

    # ── patch_embed, norm, cls_token, pos_embed, etc. ──
    accounted = (
        {id(p) for p in model.head.parameters()}
        | {id(p) for blk in model.blocks for p in blk.parameters()}
    )
    rest_d, rest_nd = [], []
    for name, p in model.named_parameters():
        if id(p) in accounted or not p.requires_grad:
            continue
        (rest_nd if _no_decay(name) else rest_d).append(p)
    param_groups += [
        {"params": rest_d,  "lr": patch_lr, "weight_decay": weight_decay},
        {"params": rest_nd, "lr": patch_lr, "weight_decay": 0.0},
    ]
    print(f"  patch_embed/rest    lr={patch_lr:.2e}")

    # Filter empty groups to avoid optimizer warnings
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def _get_lrs(optimizer) -> tuple[float, float]:
    """Return (lr_head, lr_last_block). Groups: 0=head_d, 1=head_nd, 2=block[0]_d ..."""
    lr_head = optimizer.param_groups[0]["lr"]
    # block[39] decay group: index 2 + 39*2 = 80 (if all groups are present)
    last_block_idx = 2 + (NUM_BLOCKS - 1) * 2
    lr_last = optimizer.param_groups[last_block_idx]["lr"]
    return lr_head, lr_last


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = total_top1 = total_top5 = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        top5 = logits.topk(5, dim=1).indices
        total_top1 += (top5[:, 0] == y).sum().item()
        total_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += x.size(0)
    return total_loss / total, total_top1 / total, total_top5 / total


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_ckpt(path: str, model, optimizer, scheduler, epoch: int, best_val_acc: float, phase: int):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "phase": phase,
        },
        path,
    )


def load_ckpt(path: str, model, optimizer=None, scheduler=None) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)


# ── CSV Logging ───────────────────────────────────────────────────────────────

def init_csv(path: str):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["phase", "epoch", "train_loss", "val_loss",
             "val_top1", "val_top5", "lr_head", "lr_last_block"]
        )


def append_csv(path: str, phase, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_lb):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [phase, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
             f"{val_top1:.4f}", f"{val_top5:.4f}", f"{lr_h:.2e}", f"{lr_lb:.2e}"]
        )


# ── Training loops ────────────────────────────────────────────────────────────

def train_phase1(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def train_phase2(model, loader, criterion, optimizer, scaler, scheduler, cfg) -> tuple[float, float]:
    """AMP + MixUp/CutMix + per-step scheduler (warmup+cosine from transformers)."""
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, ya, yb, lam = apply_mix(x, y, cfg, use_mix=True)

        optimizer.zero_grad()
        with torch.amp.autocast(DEVICE):
            logits = model(x)
            loss = mixed_loss(criterion, logits, y, ya, yb, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per-batch step for warmup scheduler

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def train_phase3(model, loader, criterion, optimizer, scaler, cfg) -> tuple[float, float]:
    """AMP + MixUp/CutMix; caller steps scheduler per epoch."""
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, ya, yb, lam = apply_mix(x, y, cfg, use_mix=True)

        optimizer.zero_grad()
        with torch.amp.autocast(DEVICE):
            logits = model(x)
            loss = mixed_loss(criterion, logits, y, ya, yb, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    best_ckpt = str(out / "best.pth")
    latest_ckpt = str(out / "latest.pth")
    log_path = str(out / "train_log.csv")
    init_csv(log_path)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print("Building DINOv2-Giant model...")
    model = build_model(cfg["num_classes"])

    print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(cfg)

    best_val_acc = 0.0

    # ── Phase 1: Head Warmup ──────────────────────────────────────────────────
    print("\n=== Phase 1: Head Warmup (frozen backbone) ===")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    criterion1 = LabelSmoothingCE(cfg["label_smoothing"])
    optimizer1 = torch.optim.AdamW(
        get_param_groups(model.head.named_parameters(), cfg["phase1_lr"], cfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=cfg["phase1_epochs"], eta_min=cfg["eta_min"]
    )

    for epoch in range(cfg["phase1_epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_phase1(model, train_loader, criterion1, optimizer1)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion1)
        scheduler1.step()

        lr_h = optimizer1.param_groups[0]["lr"]
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_ckpt(best_ckpt, model, optimizer1, scheduler1, epoch, best_val_acc, 1)
        save_ckpt(latest_ckpt, model, optimizer1, scheduler1, epoch, best_val_acc, 1)

        append_csv(log_path, 1, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_h)
        print(
            f"[P1 E{epoch:02d}] loss={train_loss:.4f} | acc={train_acc:.4f} | "
            f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
            f"lr={lr_h:.2e} | {time.time()-t0:.0f}s"
        )

    # ── Phase 2: Full Fine-tune with LLRD ─────────────────────────────────────
    print("\n=== Phase 2: Full LLRD Fine-tune ===")
    load_ckpt(best_ckpt, model)
    for p in model.parameters():
        p.requires_grad = True

    criterion2 = LabelSmoothingCE(cfg["label_smoothing"])
    print("LLRD param groups:")
    optimizer2 = build_llrd_optimizer(
        model,
        base_lr=cfg["phase2_base_lr"],
        decay=cfg["llrd_decay"],
        head_lr_scale=cfg["head_lr_scale"],
        weight_decay=cfg["weight_decay"],
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg["phase2_epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    scheduler2 = get_cosine_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler2 = torch.amp.GradScaler(DEVICE)

    for epoch in range(cfg["phase2_epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_phase2(
            model, train_loader, criterion2, optimizer2, scaler2, scheduler2, cfg
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion2)

        lr_h, lr_lb = _get_lrs(optimizer2)
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_ckpt(best_ckpt, model, optimizer2, scheduler2, epoch, best_val_acc, 2)
        save_ckpt(latest_ckpt, model, optimizer2, scheduler2, epoch, best_val_acc, 2)

        append_csv(log_path, 2, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_lb)
        print(
            f"[P2 E{epoch:02d}] loss={train_loss:.4f} | acc={train_acc:.4f} | "
            f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
            f"lr_h={lr_h:.2e} lr_lb={lr_lb:.2e} | {time.time()-t0:.0f}s"
        )

    # ── Phase 3: Cosine Restart ───────────────────────────────────────────────
    print("\n=== Phase 3: Cosine Restart ===")
    load_ckpt(best_ckpt, model)
    for p in model.parameters():
        p.requires_grad = True

    criterion3 = nn.CrossEntropyLoss()
    print("LLRD param groups (phase 3):")
    optimizer3 = build_llrd_optimizer(
        model,
        base_lr=cfg["phase3_base_lr"],
        decay=cfg["llrd_decay"],
        head_lr_scale=cfg["head_lr_scale"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer3, T_0=cfg["phase3_epochs"], eta_min=cfg["eta_min"]
    )
    scaler3 = torch.amp.GradScaler(DEVICE)

    for epoch in range(cfg["phase3_epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_phase3(
            model, train_loader, criterion3, optimizer3, scaler3, cfg
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion3)
        scheduler3.step()

        lr_h, lr_lb = _get_lrs(optimizer3)
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_ckpt(best_ckpt, model, optimizer3, scheduler3, epoch, best_val_acc, 3)
        save_ckpt(latest_ckpt, model, optimizer3, scheduler3, epoch, best_val_acc, 3)

        append_csv(log_path, 3, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_lb)
        print(
            f"[P3 E{epoch:02d}] loss={train_loss:.4f} | acc={train_acc:.4f} | "
            f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
            f"lr_h={lr_h:.2e} lr_lb={lr_lb:.2e} | {time.time()-t0:.0f}s"
        )

    print(f"\nDone. Best val_top1={best_val_acc:.4f}. Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

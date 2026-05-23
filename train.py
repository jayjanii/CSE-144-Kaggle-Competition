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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from transformers import get_cosine_schedule_with_warmup

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "num_classes": 100,
    "input_size": 518,
    "batch_size": 128,  # phase 1 (frozen backbone, cheap)
    "phase23_batch_size": 32,  # phases 2/3 (full backbone gradients)
    "grad_accum_steps": 4,  # effective batch = 32 * 4 = 128
    "phase1_epochs": 5,
    "phase2_epochs": 10,
    "phase3_epochs": 10,
    "phase1_lr": 3e-3,
    "phase2_base_lr": 6e-5,
    "phase3_base_lr": 1e-5,
    "llrd_decay": 0.9,
    "head_lr_scale": 10,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.4,
    "cutmix_alpha": 0.2,
    "mixup_cutmix_prob": 0.5,
    "grad_clip": 1.0,
    "warmup_epochs": 1,
    "eta_min": 1e-7,
    "tta_sizes": [448, 518, 588],  # keep centered on input_size (train res)
    "train_dir": "data/train",
    "test_dir": "data/test",
    "output_dir": os.environ.get("OUTPUT_DIR", "outputs/"),
    "kaggle_competition": "ucsc-cse-144-spring-2026-final-project",
    "seed": 42,
    "wandb_project": os.environ.get("WANDB_PROJECT", "cse144-final"),
    "wandb_entity": os.environ.get("WANDB_ENTITY", "jay_jani-university-of-california"),
    "wandb_mode": os.environ.get("WANDB_MODE", "online"),
}
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
NUM_BLOCKS = 40  # ViT-g/14 has 40 transformer blocks


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ── Data download ─────────────────────────────────────────────────────────────


def maybe_download_data(cfg: dict) -> None:
    """Download competition data via kagglehub if train_dir is missing."""
    if os.path.isdir(cfg["train_dir"]) and os.listdir(cfg["train_dir"]):
        print(f"Data found at {cfg['train_dir']}, skipping download.")
        return
    import kagglehub

    print(f"Downloading {cfg['kaggle_competition']} via kagglehub...")
    path = kagglehub.competition_download(cfg["kaggle_competition"])
    print(f"Downloaded to: {path}")
    cfg["train_dir"] = os.path.join(path, "train")
    cfg["test_dir"] = os.path.join(path, "test")


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 100) -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
    model.head = nn.Linear(1536, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model.to(DEVICE)


def enable_grad_checkpointing(model: nn.Module) -> None:
    """Wrap each transformer block's forward with torch.utils.checkpoint (idempotent)."""
    import torch.utils.checkpoint as cp

    if getattr(model, "_grad_ckpt_enabled", False):
        print("Gradient checkpointing already enabled — skipping.")
        return
    for blk in model.blocks:
        orig = blk.forward

        def _cp(x, _f=orig):
            return cp.checkpoint(_f, x, use_reentrant=False)

        blk.forward = _cp
    model._grad_ckpt_enabled = True
    print("Gradient checkpointing enabled.")


# ── Augmentation ──────────────────────────────────────────────────────────────


def get_train_transform(size: int = 448) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(size, size), scale=(0.4, 1.0), ratio=(0.75, 1.33)
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.4,
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, 56),
                hole_width_range=(1, 56),
                fill=0,
                p=0.4,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ToTensorV2(),
        ]
    )


def get_val_transform(size: int = 448) -> A.Compose:
    # Resize the short side slightly larger than the crop, scaled to `size`
    # (448 -> 480, same convention inference.py uses), so CenterCrop always fits.
    resize = int(round(size * 480 / 448))
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=resize),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ToTensorV2(),
        ]
    )


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


class PseudoLabelDataset(Dataset):
    """High-confidence test images with model-generated pseudo-labels (PIL output)."""

    def __init__(self, csv_path: str, test_dir: str):
        self.test_dir = test_dir
        with open(csv_path, newline="") as f:
            self.samples = [(r["filename"], int(r["label"])) for r in csv.DictReader(f)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert("RGB")
        return img, label


def _numeric_image_folder(root: str) -> datasets.ImageFolder:
    """ImageFolder with class folder names mapped to their integer values."""
    ds = datasets.ImageFolder(root)  # no transform → PIL images
    ds.class_to_idx = {cls: int(cls) for cls in ds.classes}
    ds.targets = [ds.class_to_idx[ds.classes[t]] for t in ds.targets]
    ds.samples = [
        (path, ds.class_to_idx[ds.classes[old_idx]]) for path, old_idx in ds.samples
    ]
    return ds


def get_datasets(cfg: dict, full_train: bool = False, pseudo_csv: str | None = None):
    """Build (train_ds, val_ds).

    full_train: use 100%% of labeled data, no val split (val_ds is None).
    pseudo_csv: append high-confidence pseudo-labeled test images to the train set.
    """
    size = cfg["input_size"]
    seed = cfg["seed"]
    train_tf = get_train_transform(size)
    base_ds = _numeric_image_folder(cfg["train_dir"])
    n = len(base_ds)

    if full_train:
        train_core = AlbumentationsDataset(base_ds, train_tf)
        val_ds = None
        print(f"Full-train mode: using all {n} labeled images, no val split.")
    else:
        indices = torch.randperm(
            n, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        train_idx = indices[: int(0.8 * n)]
        val_idx = indices[int(0.8 * n) :]
        train_core = AlbumentationsDataset(Subset(base_ds, train_idx), train_tf)
        val_ds = AlbumentationsDataset(
            Subset(base_ds, val_idx), get_val_transform(size)
        )

    if pseudo_csv:
        pseudo_base = PseudoLabelDataset(pseudo_csv, cfg["test_dir"])
        pseudo_ds = AlbumentationsDataset(pseudo_base, train_tf)
        train_ds = ConcatDataset([train_core, pseudo_ds])
        print(
            f"Added {len(pseudo_base)} pseudo-labeled samples → train size {len(train_ds)}."
        )
    else:
        train_ds = train_core

    return train_ds, val_ds


def get_val_raw(cfg: dict):
    """Return [(PIL image, label)] for the val split — raw images for TTA eval.

    Recreates the exact same seeded 80/20 split used by get_datasets().
    """
    base_ds = _numeric_image_folder(cfg["train_dir"])
    n = len(base_ds)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(cfg["seed"])
    ).tolist()
    val_idx = indices[int(0.8 * n) :]
    return [base_ds[i] for i in val_idx]  # (PIL image, int label)


def make_loaders(train_ds, val_ds, batch_size: int, seed: int = 42):
    kw = dict(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
        if val_ds is not None
        else None
    )
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
    """Returns (input, ya, yb, lam). ya is None means no mixing was applied."""
    if not use_mix or random.random() > cfg["mixup_cutmix_prob"]:
        return x, None, None, None
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
    patch_lr = base_lr * (decay**NUM_BLOCKS)

    param_groups = []

    # ── head ──
    head_d, head_nd = [], []
    for name, p in model.head.named_parameters():
        if p.requires_grad:
            (head_nd if _no_decay(name) else head_d).append(p)
    param_groups += [
        {"params": head_d, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_nd, "lr": head_lr, "weight_decay": 0.0},
    ]
    print(f"  head                lr={head_lr:.2e}")

    # ── transformer blocks ──
    # block[i=39] → distance_from_top=0 → lr=base_lr (highest)
    # block[i=0]  → distance_from_top=39 → lr=base_lr*decay^39 (lowest)
    for i in range(NUM_BLOCKS):
        dist = NUM_BLOCKS - 1 - i
        block_lr = base_lr * (decay**dist)
        blk_d, blk_nd = [], []
        for name, p in model.blocks[i].named_parameters():
            if p.requires_grad:
                (blk_nd if _no_decay(name) else blk_d).append(p)
        param_groups += [
            {"params": blk_d, "lr": block_lr, "weight_decay": weight_decay},
            {"params": blk_nd, "lr": block_lr, "weight_decay": 0.0},
        ]
        if i in (0, NUM_BLOCKS - 1):
            print(f"  blocks[{i:2d}]          lr={block_lr:.2e}")

    # ── patch_embed, norm, cls_token, pos_embed, etc. ──
    accounted = {id(p) for p in model.head.parameters()} | {
        id(p) for blk in model.blocks for p in blk.parameters()
    }
    rest_d, rest_nd = [], []
    for name, p in model.named_parameters():
        if id(p) in accounted or not p.requires_grad:
            continue
        (rest_nd if _no_decay(name) else rest_d).append(p)
    param_groups += [
        {"params": rest_d, "lr": patch_lr, "weight_decay": weight_decay},
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


def eval_and_track(model, val_loader, criterion, best_val_acc, full_train):
    """Evaluate if a val loader exists; decide whether to checkpoint.

    Returns (val_loss, val_top1, val_top5, do_save, best_val_acc).
    In full-train mode there is no val set, so every epoch is saved and the
    last epoch becomes the final model.
    """
    if val_loader is not None:
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion)
        do_save = val_top1 > best_val_acc
        best_val_acc = max(best_val_acc, val_top1)
        return val_loss, val_top1, val_top5, do_save, best_val_acc
    nan = float("nan")
    return nan, nan, nan, True, best_val_acc


def tta_val_eval(model, cfg, best_ckpt, run):
    """Evaluate the final (saved) model on the val split with test-time TTA.

    This is the best cheap predictor of the leaderboard score for a given recipe,
    since it measures exactly what inference.py measures (multi-scale + 5-crop +
    flip), just on held-out labeled data. Reloads best_ckpt so it scores the
    final saved model regardless of what's currently in memory.
    """
    from inference import run_tta  # local import avoids a circular import

    load_ckpt(best_ckpt, model)
    model.eval()
    val_raw = get_val_raw(cfg)
    correct = 0
    for img, label in val_raw:
        probs = run_tta(model, img, cfg)
        if int(probs.argmax().item()) == label:
            correct += 1
    acc = correct / len(val_raw)
    print(
        f"[TTA-VAL] TTA val_top1 = {acc:.4f} over {len(val_raw)} images "
        f"(tta_sizes={cfg['tta_sizes']}) — best single-crop val was logged above"
    )
    if run is not None:
        run.summary["tta_val_top1"] = acc
    return acc


def swa_accumulate(swa_sd, model, n):
    """Running average of model weights (SWA). Returns (swa_sd, new_n).

    ViT uses LayerNorm (no BatchNorm running stats), so a plain weight average
    needs no recalibration. Float tensors are averaged; the rest take the latest.
    """
    msd = model.state_dict()
    if swa_sd is None:
        return {k: v.detach().clone().float() for k, v in msd.items()}, 1
    for k, v in msd.items():
        if torch.is_floating_point(v):
            swa_sd[k].mul_(n / (n + 1)).add_(v.detach().float(), alpha=1 / (n + 1))
        else:
            swa_sd[k] = v.detach().clone()
    return swa_sd, n + 1


# ── Checkpointing ─────────────────────────────────────────────────────────────


def save_ckpt(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    phase: int,
    save_optim: bool = False,
):
    """Save a checkpoint. Model-only by default (inference + phase-boundary loads
    never read the optimizer state); pass save_optim=True for true resume support."""
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "phase": phase,
    }
    if save_optim:
        ckpt["optimizer"] = optimizer.state_dict()
        ckpt["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    torch.save(ckpt, path)


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
            [
                "phase",
                "epoch",
                "train_loss",
                "val_loss",
                "val_top1",
                "val_top5",
                "lr_head",
                "lr_last_block",
            ]
        )


def append_csv(
    path: str, phase, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_lb
):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                phase,
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_top1:.4f}",
                f"{val_top5:.4f}",
                f"{lr_h:.2e}",
                f"{lr_lb:.2e}",
            ]
        )


# ── Weights & Biases ──────────────────────────────────────────────────────────


def init_wandb(cfg: dict, args):
    """Return a wandb run, or None if disabled / unavailable."""
    if getattr(args, "no_wandb", False):
        print("wandb disabled (--no-wandb).")
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping experiment logging.")
        return None
    run = wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        mode=cfg["wandb_mode"],
        config={k: v for k, v in cfg.items() if not k.startswith("wandb_")},
    )
    return run


def wandb_log(
    run,
    phase,
    global_epoch,
    train_loss,
    train_acc,
    val_loss,
    val_top1,
    val_top5,
    lr_h,
    lr_lb,
):
    if run is None:
        return
    run.log(
        {
            "phase": phase,
            "global_epoch": global_epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/top1": val_top1,
            "val/top5": val_top5,
            "lr/head": lr_h,
            "lr/last_block": lr_lb,
        }
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


def _train_amp(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    cfg,
    scheduler=None,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """
    Shared AMP training loop for phases 2 and 3.
    Gradient accumulation over `accum_steps` micro-batches.
    If scheduler is provided it is stepped per optimizer step (transformers-style).
    """
    model.train()
    total_loss = total_correct = total = clean_total = 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, ya, yb, lam = apply_mix(x, y, cfg, use_mix=True)

        with torch.amp.autocast(DEVICE):
            logits = model(x)
            loss = mixed_loss(criterion, logits, y, ya, yb, lam) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * x.size(0)
        # train_acc is only meaningful on un-mixed batches
        if ya is None:
            total_correct += (logits.argmax(1) == y).sum().item()
            clean_total += x.size(0)
        total += x.size(0)

    # flush any leftover partial accumulation window
    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / total, total_correct / max(clean_total, 1)


def train_phase2(
    model, loader, criterion, optimizer, scaler, scheduler, cfg
) -> tuple[float, float]:
    return _train_amp(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        cfg,
        scheduler=scheduler,
        accum_steps=cfg["grad_accum_steps"],
    )


def train_phase3(
    model, loader, criterion, optimizer, scaler, cfg
) -> tuple[float, float]:
    return _train_amp(
        model,
        loader,
        criterion,
        optimizer,
        scaler,
        cfg,
        scheduler=None,
        accum_steps=cfg["grad_accum_steps"],
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="DINOv2-Giant three-phase fine-tuning")
    # phase control
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase 1 and start from phase 2 using existing best.pth",
    )
    p.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip phases 1+2 and start from phase 3 using existing best.pth",
    )
    p.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip phase 3 entirely; final model is the phase-2 result",
    )
    p.add_argument(
        "--no-swa",
        action="store_true",
        help="Disable SWA weight averaging over phase 3 (use per-epoch best instead)",
    )
    p.add_argument(
        "--tta-val",
        action="store_true",
        help="After training, TTA-evaluate the final model on the val split "
        "(LB-predictive number). Ignored in --full-train (no val split).",
    )
    # data / training mode
    p.add_argument(
        "--full-train",
        action="store_true",
        help="Train on 100%% of labeled data (no val split); keep the last epoch",
    )
    p.add_argument(
        "--pseudo-labels",
        default=None,
        metavar="CSV",
        help="Path to pseudo_labels.csv (from inference.py) to append to training",
    )
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--train-dir", default=CONFIG["train_dir"])
    p.add_argument("--test-dir", default=CONFIG["test_dir"])
    p.add_argument("--output-dir", default=CONFIG["output_dir"])
    # hyperparameter overrides (handy for Colab sweeps)
    p.add_argument("--phase1-epochs", type=int, default=CONFIG["phase1_epochs"])
    p.add_argument("--phase2-epochs", type=int, default=CONFIG["phase2_epochs"])
    p.add_argument("--phase3-epochs", type=int, default=CONFIG["phase3_epochs"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument(
        "--phase23-batch-size", type=int, default=CONFIG["phase23_batch_size"]
    )
    p.add_argument("--grad-accum-steps", type=int, default=CONFIG["grad_accum_steps"])
    # checkpointing
    p.add_argument(
        "--save-optimizer",
        action="store_true",
        help="Also store optimizer/scheduler state (larger ckpt; for future resume)",
    )
    # logging
    p.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg.update(
        {
            "seed": args.seed,
            "train_dir": args.train_dir,
            "test_dir": args.test_dir,
            "output_dir": args.output_dir,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "phase3_epochs": args.phase3_epochs,
            "batch_size": args.batch_size,
            "phase23_batch_size": args.phase23_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "full_train": args.full_train,
            "pseudo_labels": args.pseudo_labels,
        }
    )
    full_train = args.full_train
    save_optimizer = args.save_optimizer

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    best_ckpt = str(out / "best.pth")
    log_path = str(out / "train_log.csv")
    init_csv(log_path)
    print(f"Checkpoints → {best_ckpt}")

    set_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")

    run = init_wandb(cfg, args)
    global_epoch = 0

    maybe_download_data(cfg)

    print("Building DINOv2-Giant model...")
    model = build_model(cfg["num_classes"])

    print("Preparing datasets...")
    train_ds, val_ds = get_datasets(
        cfg, full_train=full_train, pseudo_csv=args.pseudo_labels
    )

    best_val_acc = 0.0

    # ── Phase 1: Head Warmup ──────────────────────────────────────────────────
    if args.skip_phase1 or args.skip_phase2:
        print(f"Skipping phase 1 — loading {best_ckpt}")
        _, best_val_acc = load_ckpt(best_ckpt, model)
    else:
        print("\n=== Phase 1: Head Warmup (frozen backbone) ===")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

        train_loader, val_loader = make_loaders(
            train_ds, val_ds, cfg["batch_size"], cfg["seed"]
        )

        criterion1 = LabelSmoothingCE(cfg["label_smoothing"])
        optimizer1 = torch.optim.AdamW(
            get_param_groups(
                model.head.named_parameters(), cfg["phase1_lr"], cfg["weight_decay"]
            ),
            betas=(0.9, 0.999),
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=cfg["phase1_epochs"], eta_min=cfg["eta_min"]
        )

        for epoch in range(cfg["phase1_epochs"]):
            t0 = time.time()
            train_loss, train_acc = train_phase1(
                model, train_loader, criterion1, optimizer1
            )
            val_loss, val_top1, val_top5, do_save, best_val_acc = eval_and_track(
                model, val_loader, criterion1, best_val_acc, full_train
            )
            scheduler1.step()

            lr_h = optimizer1.param_groups[0]["lr"]
            if do_save:
                save_ckpt(
                    best_ckpt,
                    model,
                    optimizer1,
                    scheduler1,
                    epoch,
                    best_val_acc,
                    1,
                    save_optim=save_optimizer,
                )

            append_csv(
                log_path, 1, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_h
            )
            wandb_log(
                run,
                1,
                global_epoch,
                train_loss,
                train_acc,
                val_loss,
                val_top1,
                val_top5,
                lr_h,
                lr_h,
            )
            global_epoch += 1
            print(
                f"[P1 E{epoch:02d}] loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
                f"lr={lr_h:.2e} | {time.time() - t0:.0f}s"
            )

        del train_loader, val_loader

    # ── Phase 2: Full Fine-tune with LLRD ─────────────────────────────────────
    torch.cuda.empty_cache()

    if args.skip_phase2:
        print(f"Skipping phase 2 — loading {best_ckpt}")
        _, best_val_acc = load_ckpt(best_ckpt, model)
    else:
        print("\n=== Phase 2: Full LLRD Fine-tune ===")
        load_ckpt(best_ckpt, model)
        for p in model.parameters():
            p.requires_grad = True
        enable_grad_checkpointing(model)

        train_loader2, val_loader2 = make_loaders(
            train_ds, val_ds, cfg["phase23_batch_size"], cfg["seed"]
        )
        steps_per_epoch = math.ceil(len(train_loader2) / cfg["grad_accum_steps"])

        criterion2 = LabelSmoothingCE(cfg["label_smoothing"])
        print("LLRD param groups:")
        optimizer2 = build_llrd_optimizer(
            model,
            base_lr=cfg["phase2_base_lr"],
            decay=cfg["llrd_decay"],
            head_lr_scale=cfg["head_lr_scale"],
            weight_decay=cfg["weight_decay"],
        )

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
                model, train_loader2, criterion2, optimizer2, scaler2, scheduler2, cfg
            )
            val_loss, val_top1, val_top5, do_save, best_val_acc = eval_and_track(
                model, val_loader2, criterion2, best_val_acc, full_train
            )

            lr_h, lr_lb = _get_lrs(optimizer2)
            if do_save:
                save_ckpt(
                    best_ckpt,
                    model,
                    optimizer2,
                    scheduler2,
                    epoch,
                    best_val_acc,
                    2,
                    save_optim=save_optimizer,
                )

            append_csv(
                log_path,
                2,
                epoch,
                train_loss,
                val_loss,
                val_top1,
                val_top5,
                lr_h,
                lr_lb,
            )
            wandb_log(
                run,
                2,
                global_epoch,
                train_loss,
                train_acc,
                val_loss,
                val_top1,
                val_top5,
                lr_h,
                lr_lb,
            )
            global_epoch += 1
            print(
                f"[P2 E{epoch:02d}] loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
                f"lr_h={lr_h:.2e} lr_lb={lr_lb:.2e} | {time.time() - t0:.0f}s"
            )

        del train_loader2, val_loader2

    # ── Phase 3: Cosine Restart + SWA ─────────────────────────────────────────
    if args.skip_phase3:
        print("\nSkipping phase 3 — final model is the phase-2 result.")
    else:
        print("\n=== Phase 3: Cosine Restart + SWA ===")
        torch.cuda.empty_cache()

        load_ckpt(best_ckpt, model)
        for p in model.parameters():
            p.requires_grad = True
        enable_grad_checkpointing(model)  # idempotent — no-op if phase 2 already enabled it

        train_loader3, val_loader3 = make_loaders(
            train_ds, val_ds, cfg["phase23_batch_size"], cfg["seed"]
        )

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

        use_swa = not args.no_swa
        swa_sd, swa_n = None, 0

        for epoch in range(cfg["phase3_epochs"]):
            t0 = time.time()
            train_loss, train_acc = train_phase3(
                model, train_loader3, criterion3, optimizer3, scaler3, cfg
            )
            val_loss, val_top1, val_top5, do_save, best_val_acc = eval_and_track(
                model, val_loader3, criterion3, best_val_acc, full_train
            )
            scheduler3.step()

            if use_swa:
                swa_sd, swa_n = swa_accumulate(swa_sd, model, swa_n)

            lr_h, lr_lb = _get_lrs(optimizer3)
            # Keep per-epoch saves so best.pth always holds the best single epoch
            # (val mode) / latest epoch (full-train); SWA may overwrite it below.
            if do_save:
                save_ckpt(
                    best_ckpt,
                    model,
                    optimizer3,
                    scheduler3,
                    epoch,
                    best_val_acc,
                    3,
                    save_optim=save_optimizer,
                )

            append_csv(
                log_path, 3, epoch, train_loss, val_loss, val_top1, val_top5, lr_h, lr_lb
            )
            wandb_log(
                run,
                3,
                global_epoch,
                train_loss,
                train_acc,
                val_loss,
                val_top1,
                val_top5,
                lr_h,
                lr_lb,
            )
            global_epoch += 1
            print(
                f"[P3 E{epoch:02d}] loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
                f"lr_h={lr_h:.2e} lr_lb={lr_lb:.2e} | {time.time() - t0:.0f}s"
            )

        # ── Finalize SWA: load the averaged weights and decide whether to keep ──
        if use_swa and swa_sd is not None:
            model.load_state_dict(swa_sd)
            if val_loader3 is not None:
                _, swa_top1, _ = evaluate(model, val_loader3, criterion3)
                print(
                    f"[SWA] averaged {swa_n} epochs → val_top1={swa_top1:.4f} "
                    f"(best single epoch was {best_val_acc:.4f})"
                )
                if run is not None:
                    run.summary["swa_val_top1"] = swa_top1
                if swa_top1 >= best_val_acc:
                    best_val_acc = swa_top1
                    save_ckpt(
                        best_ckpt, model, optimizer3, scheduler3,
                        cfg["phase3_epochs"], best_val_acc, 3,
                        save_optim=save_optimizer,
                    )
                    print(f"[SWA] saved averaged model as best ({best_val_acc:.4f}).")
                else:
                    print("[SWA] averaged model worse than best epoch — keeping best epoch.")
                    load_ckpt(best_ckpt, model)  # restore best epoch into memory
            else:
                # full-train: no val, so the SWA average is the robust endpoint
                save_ckpt(
                    best_ckpt, model, optimizer3, scheduler3,
                    cfg["phase3_epochs"], best_val_acc, 3,
                    save_optim=save_optimizer,
                )
                print(f"[SWA] saved averaged model ({swa_n} epochs) as final (full-train).")

        del train_loader3, val_loader3

    # Optional: TTA evaluation on the val split — best cheap LB predictor
    if args.tta_val and not full_train:
        print("\nRunning TTA evaluation on the val split...")
        tta_val_eval(model, cfg, best_ckpt, run)
    elif args.tta_val and full_train:
        print("\n--tta-val ignored: no val split in --full-train mode.")

    if run is not None:
        if not full_train:
            run.summary["best_val_top1"] = best_val_acc
        run.finish()

    if full_train:
        print(f"\nDone (full-train, no val). Final checkpoint: {best_ckpt}")
    else:
        print(f"\nDone. Best val_top1={best_val_acc:.4f}. Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

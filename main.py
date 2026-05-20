import argparse
import os

import torch
import torch.nn as nn
import wandb
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from config import (
    ACCUM_STEPS,
    ACCUM_STEPS_PHASE3,
    BATCH_SIZE_PHASE1,
    BATCH_SIZE_PHASE2,
    BATCH_SIZE_PHASE3,
    CKPT,
    CUTMIX_ALPHA,
    DATA_DIR,
    DEVICE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    EPOCHS_PHASE3,
    LABEL_SMOOTHING,
    LR_BACKBONE_PHASE2,
    LR_BLOCKS_REST_PHASE3,
    LR_BLOCKS_TOP_PHASE3,
    LR_HEAD_PHASE1,
    LR_HEAD_PHASE2,
    LR_HEAD_PHASE3,
    MIXUP_ALPHA,
    MIXUP_PROB,
    MIXUP_SWITCH_PROB,
    MODEL_NAME,
    NUM_CLASSES,
    PATIENCE_PHASE1,
    PATIENCE_PHASE2,
    PATIENCE_PHASE3,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WEIGHT_DECAY,
)
from data import download_data, get_dataloaders
from evaluate import evaluate
from model import create_model, freeze_backbone, unfreeze_all, unfreeze_top_blocks
from train import EarlyStopper, train_one_epoch, train_one_epoch_phase2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02")
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip head-only training (phase 1) and go straight to full fine-tuning.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="Path to a checkpoint (.pth) to load before training. "
        "Use with --skip-phase1 to resume from a saved phase-1 checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing the competition data (default: DATA_DIR env / 'data').",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=CKPT,
        help="Where to save the best checkpoint (default: CKPT_PATH env / 'best.pth').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("KAGGLE_API_TOKEN"):
        print(
            "Warning: KAGGLE_API_TOKEN not set — data download will fail if data is missing."
        )

    download_data(args.data_dir)

    model = create_model(pretrained=True)

    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {args.resume}")

    make_loaders = get_dataloaders(model, args.data_dir)
    model = torch.compile(model)

    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        prob=MIXUP_PROB,
        switch_prob=MIXUP_SWITCH_PROB,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=NUM_CLASSES,
    )
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "epochs_phase1": EPOCHS_PHASE1,
            "epochs_phase2": EPOCHS_PHASE2,
            "epochs_phase3": EPOCHS_PHASE3,
            "batch_size_phase1": BATCH_SIZE_PHASE1,
            "batch_size_phase2": BATCH_SIZE_PHASE2,
            "batch_size_phase3": BATCH_SIZE_PHASE3,
            "effective_batch_size_p2": BATCH_SIZE_PHASE2 * ACCUM_STEPS,
            "effective_batch_size_p3": BATCH_SIZE_PHASE3 * ACCUM_STEPS_PHASE3,
            "accum_steps": ACCUM_STEPS,
            "accum_steps_phase3": ACCUM_STEPS_PHASE3,
            "lr_head_phase1": LR_HEAD_PHASE1,
            "lr_head_phase2": LR_HEAD_PHASE2,
            "lr_backbone_phase2": LR_BACKBONE_PHASE2,
            "lr_head_phase3": LR_HEAD_PHASE3,
            "lr_blocks_top_phase3": LR_BLOCKS_TOP_PHASE3,
            "lr_blocks_rest_phase3": LR_BLOCKS_REST_PHASE3,
            "patience_phase1": PATIENCE_PHASE1,
            "patience_phase2": PATIENCE_PHASE2,
            "patience_phase3": PATIENCE_PHASE3,
            "model": MODEL_NAME,
            "skip_phase1": args.skip_phase1,
            "resume": args.resume,
        },
    )

    best_val_acc = 0.0

    # ------------------------------------------------------------------
    # p1
    # ------------------------------------------------------------------
    train_loader, val_loader = make_loaders(1)

    if not args.skip_phase1:
        print("Phase 1: Training head only...")
        freeze_backbone(model)

        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=LR_HEAD_PHASE1, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler(DEVICE)
        stopper = EarlyStopper(patience=PATIENCE_PHASE1)

        for epoch in range(EPOCHS_PHASE1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                train_criterion,
                optimizer,
                DEVICE,
                scaler,
                mixup_fn,
            )
            val_loss, val_acc = evaluate(model, val_loader, val_criterion, DEVICE)

            run.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch,
                    "phase": 1,
                }
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.ckpt)
                run.save(args.ckpt)

            print(
                f"[P1] Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if stopper(val_acc):
                print(
                    f"[P1] Early stop — val_acc flat for {PATIENCE_PHASE1} epochs. Advancing to phase 2."
                )
                break
    else:
        print("Skipping phase 1.")
        # Track whatever the loaded checkpoint achieves so phase 2 still saves improvements.
        _, best_val_acc = evaluate(model, val_loader, val_criterion, DEVICE)
        print(f"Loaded checkpoint val_acc={best_val_acc:.4f}")

    del train_loader, val_loader

    # ------------------------------------------------------------------
    # p2
    # ------------------------------------------------------------------
    print("Phase 2: top blocks + head...")

    param_groups = unfreeze_top_blocks(model)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PHASE2
    )
    scaler = torch.amp.GradScaler(DEVICE)
    stopper = EarlyStopper(patience=PATIENCE_PHASE2)

    train_loader, val_loader = make_loaders(2)

    for epoch in range(EPOCHS_PHASE2):
        train_loss, train_acc = train_one_epoch_phase2(
            model,
            train_loader,
            train_criterion,
            optimizer,
            DEVICE,
            ACCUM_STEPS,
            scaler,
            mixup_fn,
        )
        val_loss, val_acc = evaluate(model, val_loader, val_criterion, DEVICE)
        scheduler.step()

        phase1_offset = 0 if args.skip_phase1 else EPOCHS_PHASE1
        run.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch": phase1_offset + epoch,
                "phase": 2,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.ckpt)
            run.save(args.ckpt)

        print(
            f"[P2] Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if stopper(val_acc):
            print(
                f"[P2] Early stop — val_acc flat for {PATIENCE_PHASE2} epochs. Advancing to phase 3."
            )
            break

    del train_loader, val_loader

    # ------------------------------------------------------------------
    # p3
    # ------------------------------------------------------------------
    print("Phase 3: full model fine-tuning...")
    param_groups = unfreeze_all(model)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PHASE3
    )
    scaler = torch.amp.GradScaler(DEVICE)
    stopper = EarlyStopper(patience=PATIENCE_PHASE3)

    train_loader, val_loader = make_loaders(3)

    for epoch in range(EPOCHS_PHASE3):
        train_loss, train_acc = train_one_epoch_phase2(
            model,
            train_loader,
            train_criterion,
            optimizer,
            DEVICE,
            ACCUM_STEPS_PHASE3,
            scaler,
            mixup_fn,
        )
        val_loss, val_acc = evaluate(model, val_loader, val_criterion, DEVICE)
        scheduler.step()

        phase_offset = (0 if args.skip_phase1 else EPOCHS_PHASE1) + EPOCHS_PHASE2
        run.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch": phase_offset + epoch,
                "phase": 3,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.ckpt)
            run.save(args.ckpt)

        print(
            f"[P3] Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if stopper(val_acc):
            print(
                f"[P3] Early stop — val_acc flat for {PATIENCE_PHASE3} epochs. Training complete."
            )
            break

    del train_loader, val_loader

    run.finish()
    print(f"Done. Best val_acc={best_val_acc:.4f}. Checkpoint saved to {args.ckpt}")


if __name__ == "__main__":
    main()

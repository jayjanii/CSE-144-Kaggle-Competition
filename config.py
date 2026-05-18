import os
import torch

# Model
MODEL_NAME = "eva02_large_patch14_448"
NUM_CLASSES = 100

# Phase 1 — head-only training
EPOCHS_PHASE1 = 10
BATCH_SIZE_PHASE1 = 128
NUM_WORKERS_PHASE1 = 8
LR_HEAD_PHASE1 = 1e-3

# Phase 2 — full fine-tune with AMP + grad accumulation
EPOCHS_PHASE2 = 20
BATCH_SIZE_PHASE2 = 16
NUM_WORKERS_PHASE2 = 4
ACCUM_STEPS = 8
LR_HEAD_PHASE2 = 1e-3
LR_BACKBONE_PHASE2 = 1e-5

WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths — override via environment variables for Colab / different setups
DATA_DIR = os.environ.get("DATA_DIR", "data")
CKPT = os.environ.get("CKPT_PATH", "best.pth")

# WandB
WANDB_PROJECT = "cse144-final"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jay_jani-university-of-california")

# Kaggle competition slug
KAGGLE_COMPETITION = "ucsc-cse-144-spring-2026-final-project"

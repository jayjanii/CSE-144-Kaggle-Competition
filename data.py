import os
import kagglehub
import timm
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE_PHASE1, BATCH_SIZE_PHASE2, BATCH_SIZE_PHASE3,
    NUM_WORKERS_PHASE1, NUM_WORKERS_PHASE2, NUM_WORKERS_PHASE3,
    KAGGLE_COMPETITION,
)


def download_data(data_dir: str) -> None:
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print("Data already downloaded, skipping.")
        return
    path = kagglehub.competition_download(KAGGLE_COMPETITION, output_dir=data_dir)
    print("Downloaded to:", path)


def get_transforms(model):
    data_config = timm.data.resolve_data_config({}, model=model)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    return train_transform, val_transform


def _numeric_folder_dataset(root, transform):
    """ImageFolder with class folders sorted numerically instead of alphabetically."""
    ds = datasets.ImageFolder(root, transform=transform)
    # remap: class name (folder name as string) → int, sorted numerically
    ds.class_to_idx = {cls: int(cls) for cls in ds.classes}
    ds.targets = [ds.class_to_idx[ds.classes[t]] for t in ds.targets]
    ds.samples = [(path, ds.class_to_idx[ds.classes[old_idx]]) for path, old_idx in ds.samples]
    return ds


def get_dataloaders(model, data_dir: str):
    train_transform, val_transform = get_transforms(model)

    train_base = _numeric_folder_dataset(os.path.join(data_dir, "train"), transform=train_transform)
    val_base   = _numeric_folder_dataset(os.path.join(data_dir, "train"), transform=val_transform)

    n = len(train_base)
    train_size = int(0.8 * n)
    indices = torch.randperm(n).tolist()

    train_dataset = Subset(train_base, indices[:train_size])
    val_dataset   = Subset(val_base,   indices[train_size:])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PHASE1, shuffle=True,
        num_workers=NUM_WORKERS_PHASE1, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_PHASE1, shuffle=False,
        num_workers=NUM_WORKERS_PHASE1, pin_memory=True,
    )
    train_loader_phase2 = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PHASE2, shuffle=True,
        num_workers=NUM_WORKERS_PHASE2, pin_memory=True,
    )
    val_loader_phase2 = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_PHASE2, shuffle=False,
        num_workers=NUM_WORKERS_PHASE2, pin_memory=True,
    )
    train_loader_phase3 = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PHASE3, shuffle=True,
        num_workers=NUM_WORKERS_PHASE3, pin_memory=True,
    )
    val_loader_phase3 = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_PHASE3, shuffle=False,
        num_workers=NUM_WORKERS_PHASE3, pin_memory=True,
    )

    return train_loader, val_loader, train_loader_phase2, val_loader_phase2, train_loader_phase3, val_loader_phase3

import csv
import os
import kagglehub
import timm
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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
    train_transform = timm.data.create_transform(
        **data_config,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        color_jitter=0.4,
    )
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    return train_transform, val_transform


def _numeric_folder_dataset(root, transform):
    """ImageFolder with class folders sorted numerically instead of alphabetically."""
    ds = datasets.ImageFolder(root, transform=transform)
    ds.class_to_idx = {cls: int(cls) for cls in ds.classes}
    ds.targets = [ds.class_to_idx[ds.classes[t]] for t in ds.targets]
    ds.samples = [(path, ds.class_to_idx[ds.classes[old_idx]]) for path, old_idx in ds.samples]
    return ds


class PseudoLabelDataset(Dataset):
    """High-confidence test images with model-generated pseudo-labels."""

    def __init__(self, csv_path: str, test_dir: str, transform):
        self.test_dir = test_dir
        self.transform = transform
        with open(csv_path, newline="") as f:
            self.samples = [(row["filename"], int(row["label"])) for row in csv.DictReader(f)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert("RGB")
        return self.transform(img), label


_PHASE_CONFIG = {
    1: (BATCH_SIZE_PHASE1, NUM_WORKERS_PHASE1),
    2: (BATCH_SIZE_PHASE2, NUM_WORKERS_PHASE2),
    3: (BATCH_SIZE_PHASE3, NUM_WORKERS_PHASE3),
}


def get_dataloaders(model, data_dir, *, full_train=False, pseudo_labels_csv=None):
    """Return a make_loaders(phase) callable.

    full_train: use all labeled data for training (no val split, no early stopping).
    pseudo_labels_csv: path to CSV produced by pseudo_label.py; those test images
        are appended to the training set.
    """
    train_transform, val_transform = get_transforms(model)

    train_base = _numeric_folder_dataset(os.path.join(data_dir, "train"), transform=train_transform)

    if full_train:
        train_dataset = train_base
        val_dataset = None
    else:
        val_base = _numeric_folder_dataset(os.path.join(data_dir, "train"), transform=val_transform)
        n = len(train_base)
        train_size = int(0.8 * n)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        train_dataset = Subset(train_base, indices[:train_size])
        val_dataset = Subset(val_base, indices[train_size:])

    if pseudo_labels_csv:
        pseudo_ds = PseudoLabelDataset(
            pseudo_labels_csv, os.path.join(data_dir, "test"), train_transform
        )
        train_dataset = ConcatDataset([train_dataset, pseudo_ds])
        print(f"Added {len(pseudo_ds)} pseudo-labeled test samples to training set.")

    def make_loaders(phase: int):
        batch_size, num_workers = _PHASE_CONFIG[phase]
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        )
        val_loader = (
            DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
            )
            if val_dataset is not None else None
        )
        return train_loader, val_loader

    return make_loaders

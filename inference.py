"""TTA inference — writes submission.csv and submission_probs.csv."""

import csv
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Re-use CONFIG and model builder from train.py
from train import CONFIG, DEVICE, DINOV2_MEAN, DINOV2_STD, build_model

# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZE_MEAN = torch.tensor(DINOV2_MEAN).view(3, 1, 1)
_NORMALIZE_STD = torch.tensor(DINOV2_STD).view(3, 1, 1)


def _to_tensor_normalized(pil_img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(pil_img)  # [3, H, W], float32 in [0,1]
    return (t - _NORMALIZE_MEAN) / _NORMALIZE_STD


def _five_crop(img: Image.Image, size: int) -> list[Image.Image]:
    """Return TL, TR, BL, BR, center crops of `size` from `img`."""
    W, H = img.size
    crops = [
        img.crop((0, 0, size, size)),               # TL
        img.crop((W - size, 0, W, size)),            # TR
        img.crop((0, H - size, size, H)),            # BL
        img.crop((W - size, H - size, W, H)),        # BR
        img.crop(
            ((W - size) // 2, (H - size) // 2,
             (W + size) // 2, (H + size) // 2)
        ),                                           # center
    ]
    return crops


def run_tta(model: nn.Module, image: Image.Image, cfg: dict) -> torch.Tensor:
    """
    3 resolutions × 5 crops × 2 (orig + hflip) = 30 forward passes.
    Returns averaged softmax probabilities of shape [num_classes].
    """
    model.eval()
    all_probs: list[torch.Tensor] = []

    with torch.inference_mode():
        for size in cfg["tta_sizes"]:
            # Resize so the shortest side is slightly larger than the crop size,
            # matching the val-transform convention (480 → crop 448).
            resize_to = int(round(size * (480 / 448)))
            W, H = image.size
            scale = resize_to / min(W, H)
            new_W, new_H = int(round(W * scale)), int(round(H * scale))
            # Ensure both dimensions are at least `size`
            new_W, new_H = max(new_W, size), max(new_H, size)
            resized = image.resize((new_W, new_H), Image.BICUBIC)

            crops = _five_crop(resized, size)
            for crop in crops:
                for flipped in (False, True):
                    aug = TF.hflip(crop) if flipped else crop
                    tensor = _to_tensor_normalized(aug).unsqueeze(0).to(DEVICE)
                    with torch.amp.autocast(DEVICE):
                        probs = model(tensor).softmax(dim=1).squeeze(0).cpu()
                    all_probs.append(probs)

    return torch.stack(all_probs).mean(dim=0)


# ── Test dataset ──────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, test_dir: str):
        self.test_dir = test_dir
        self.filenames = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            key=lambda f: int(os.path.splitext(f)[0]),
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert("RGB")
        return img, fname


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    cfg = CONFIG
    parser = argparse.ArgumentParser(description="TTA inference for DINOv2-Giant")
    parser.add_argument("--ckpt", default=os.path.join(cfg["output_dir"], "best.pth"),
                        help="Path to best checkpoint")
    parser.add_argument("--test-dir", default=cfg["test_dir"])
    parser.add_argument("--output-dir", default=cfg["output_dir"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sub_path = os.path.join(args.output_dir, "submission.csv")
    probs_path = os.path.join(args.output_dir, "submission_probs.csv")

    print("Loading model...")
    model = build_model(cfg["num_classes"])
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt} "
          f"(phase={ckpt.get('phase','?')}, best_val_acc={ckpt.get('best_val_acc',0):.4f})")

    dataset = TestDataset(args.test_dir)
    print(f"Running TTA ({cfg['tta_sizes']} × 5 crops × 2 flips = 30 passes) "
          f"on {len(dataset)} test images...")

    all_ids: list[str] = []
    all_preds: list[int] = []
    all_probs: list[torch.Tensor] = []

    for i, (img, fname) in enumerate(dataset):
        probs = run_tta(model, img, cfg)
        image_id = os.path.splitext(fname)[0]
        all_ids.append(image_id)
        all_preds.append(int(probs.argmax().item()))
        all_probs.append(probs)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset)}")

    # submission.csv
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_class"])
        w.writerows(zip(all_ids, all_preds))
    print(f"Saved predictions → {sub_path}")

    # submission_probs.csv
    prob_matrix = torch.stack(all_probs).numpy()
    with open(probs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [str(c) for c in range(cfg["num_classes"])])
        for image_id, row in zip(all_ids, prob_matrix):
            w.writerow([image_id] + [f"{v:.6f}" for v in row.tolist()])
    print(f"Saved softmax probs  → {probs_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm

from config import DEVICE, DATA_DIR, CKPT, MODEL_NAME, NUM_CLASSES
from model import create_model


class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.transform = transform
        # sort numerically so output order matches sample_submission
        self.filenames = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            key=lambda f: int(os.path.splitext(f)[0])
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert("RGB")
        return self.transform(img), fname


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on the test set")
    parser.add_argument("--ckpt", type=str, default=CKPT, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Competition data directory")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    model = create_model(pretrained=False)
    state = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    test_dir = os.path.join(args.data_dir, "test")
    dataset = TestDataset(test_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    with torch.inference_mode():
        for images, filenames in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for fname, pred in zip(filenames, preds):
                rows.append((fname, pred))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

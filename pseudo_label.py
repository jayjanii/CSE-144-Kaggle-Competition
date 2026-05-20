import argparse
import csv
import os
import torch
import timm

from config import DEVICE, DATA_DIR, CKPT
from data import download_data
from infer import TestDataset
from model import create_model
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for unlabeled test images")
    parser.add_argument("--ckpt", type=str, default=CKPT)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output", type=str, default="pseudo_labels.csv")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Minimum softmax confidence to include a sample (default: 0.9)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    download_data(args.data_dir)

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    test_dir = os.path.join(args.data_dir, "test")
    dataset = TestDataset(test_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    results = []
    with torch.inference_mode():
        for images, filenames in loader:
            images = images.to(DEVICE)
            probs = model(images).softmax(dim=1)
            confidences, preds = probs.max(dim=1)
            for fname, pred, conf in zip(filenames, preds.cpu().tolist(), confidences.cpu().tolist()):
                results.append((fname, pred, conf))

    above = [(f, l, c) for f, l, c in results if c >= args.threshold]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for fname, label, conf in above:
            writer.writerow([fname, label, f"{conf:.4f}"])

    total = len(results)
    kept = len(above)
    print(f"Total test images : {total}")
    print(f"Above threshold   : {kept} ({100 * kept / total:.1f}%)")
    print(f"Saved to          : {args.output}")
    print(f"\nNext step: python main.py --pseudo-labels {args.output}")


if __name__ == "__main__":
    main()

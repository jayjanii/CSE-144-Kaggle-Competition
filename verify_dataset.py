"""Check whether a class's training images came from a candidate external dataset.

Two complementary checks:
  1. Perceptual hashing (pHash) — decisive for exact-source matches even after
     resizing / JPEG re-encoding. If pHash distance <= 6, the image is almost
     certainly the same photo. If <=10, very likely a near-duplicate.
  2. CLIP embedding cosine similarity — catches "same category, different
     photo" matches that pHash misses. Sim >= 0.85 in the same category is
     normal; sim >= 0.95 with a specific image suggests near-duplicate.

Workflow:
    pip install imagehash open_clip_torch
    # Download the candidate dataset first (Food-101 example below).
    python verify_dataset.py \
        --our-class 0 \
        --train-dir data/train \
        --candidate-dir food-101/images/apple_pie \
        --report

If you see lots of pHash distances <= 6, your dataset was literally sampled
from the candidate. If pHash is all >12 but CLIP sims are >0.85, it's the
same category but a different photo source.
"""

import argparse
import os
from pathlib import Path

from PIL import Image


def phash_check(our_paths, cand_paths, max_dist=10):
    """Print closest candidate match for each of our images, by perceptual hash."""
    import imagehash

    print(f"\n=== pHash check: {len(our_paths)} ours vs {len(cand_paths)} candidates ===")
    cand_hashes = []
    for p in cand_paths:
        try:
            cand_hashes.append((p, imagehash.phash(Image.open(p).convert("RGB"))))
        except Exception as e:
            print(f"  skip {p}: {e}")
    if not cand_hashes:
        print("  no candidates loaded — bad path?")
        return

    exact_hits = 0
    near_hits = 0
    for our in our_paths:
        h = imagehash.phash(Image.open(our).convert("RGB"))
        best = min(cand_hashes, key=lambda x: h - x[1])
        dist = h - best[1]
        if dist <= 6:
            tag = "EXACT"
            exact_hits += 1
        elif dist <= max_dist:
            tag = "near"
            near_hits += 1
        else:
            tag = "miss"
        print(f"  {Path(our).name:30s}  d={dist:3d}  {tag}  →  {Path(best[0]).name}")
    n = len(our_paths)
    print(f"\nSummary: {exact_hits}/{n} EXACT (same image), {near_hits}/{n} near (likely dup), "
          f"{n - exact_hits - near_hits}/{n} miss.")
    if exact_hits >= n // 2:
        print("VERDICT: dataset was sampled from this source (or a near-mirror).")
    elif exact_hits + near_hits >= n // 2:
        print("VERDICT: likely same source, possibly with mild re-encoding.")
    else:
        print("VERDICT: NOT this exact source. Try CLIP similarity to check the category.")


def clip_check(our_paths, cand_paths, model_name="ViT-SO400M-14-SigLIP-384", pretrained="webli"):
    """Encode both sides, print average and max cosine similarity from ours -> candidates."""
    import torch
    import open_clip

    print(f"\n=== CLIP similarity check ({model_name}) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    def encode(paths, batch_size=16):
        feats = []
        with torch.inference_mode():
            for i in range(0, len(paths), batch_size):
                xs = torch.stack([
                    preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]
                ]).to(device)
                f = model.encode_image(xs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        return torch.cat(feats, 0)

    our_f = encode(our_paths)
    cand_f = encode(cand_paths)
    sims = our_f @ cand_f.T  # [n_ours, n_cands]
    top_sim, top_idx = sims.max(dim=1)
    mean_sim = sims.mean()

    for i, our in enumerate(our_paths):
        print(f"  {Path(our).name:30s}  top sim={top_sim[i]:.3f}  →  {Path(cand_paths[top_idx[i]]).name}")
    print(f"\nmean(top-1 sim) = {top_sim.mean():.3f}   mean(all pairs) = {mean_sim:.3f}")
    if top_sim.mean() >= 0.95:
        print("VERDICT: same images (near-duplicates) — almost certainly the source dataset.")
    elif top_sim.mean() >= 0.85:
        print("VERDICT: same category, different photos — likely the same conceptual class.")
    else:
        print("VERDICT: probably a different category or a much harder/easier subset.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--our-class", required=True,
                   help="Class id to test (e.g. 0). Reads train_dir/<class>/*")
    p.add_argument("--train-dir", default="data/train",
                   help="Path to competition train/. Falls back to kagglehub.")
    p.add_argument("--candidate-dir", required=True,
                   help="Folder of candidate-dataset images for the SAME class "
                        "(e.g. food-101/images/apple_pie/)")
    p.add_argument("--max-candidates", type=int, default=200,
                   help="Cap candidate images to speed things up.")
    p.add_argument("--skip-clip", action="store_true",
                   help="Skip the CLIP similarity check (saves model download)")
    args = p.parse_args()

    # Reuse train.py's downloader if needed
    train_dir = args.train_dir
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG)
        cfg["train_dir"] = train_dir
        maybe_download_data(cfg)
        train_dir = cfg["train_dir"]

    our_dir = Path(train_dir) / args.our_class
    if not our_dir.is_dir():
        raise SystemExit(f"No such class folder: {our_dir}")
    our_paths = sorted(str(p) for p in our_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cand_paths = sorted(str(p) for p in Path(args.candidate_dir).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cand_paths = cand_paths[: args.max_candidates]
    print(f"Comparing {len(our_paths)} ours (class {args.our_class}) "
          f"vs {len(cand_paths)} candidates from {args.candidate_dir}")

    phash_check(our_paths, cand_paths)
    if not args.skip_clip:
        clip_check(our_paths, cand_paths)


if __name__ == "__main__":
    main()

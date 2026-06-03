"""Frozen-backbone logistic-probe pipeline (no fine-tuning).

Replicates the clean, fully-legitimate recipe:
  1. ENCODE   frozen SigLIP-2 vision tower, 4 deterministic TTA views per image,
              pooled into one embedding.
  2. PROBE    a single multinomial logistic regression on the frozen embeddings.
  3. ZERO-SHOT the same model's text tower scores class-name prompts -> Z.
  4. FUSE     alpha * P + (1-alpha) * Z.
  (5. BALANCE  run sinkhorn.py on the fused probs — separate step.)

Nothing here trains the backbone; only the logistic head is fit on train/.
The text tower + class names are standard zero-shot CLIP usage. No external
image datasets are used.

Usage:
    pip install open_clip_torch scikit-learn
    python probe.py \
        --model ViT-SO400M-16-SigLIP2-384 --pretrained webli \
        --names class_names.csv \
        --out-probe probe/probe_probs.csv \
        --out-zeroshot probe/zeroshot_probs.csv \
        --out-fused probe/fused_probs.csv \
        --alpha 0.65

Then:
    python sinkhorn.py probe/fused_probs.csv --out submission_final.csv

Verify the SigLIP-2 identifier first:
    python -c "import open_clip; print([m for m in open_clip.list_pretrained() if 'SigLIP2' in m[0] or 'siglip2' in m[0].lower()])"
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# ── TTA views ─────────────────────────────────────────────────────────────────


def center_crop_view(img, size):
    """Aspect-preserving short-side resize, then center crop to size."""
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    r = img.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - size) // 2, (nh - size) // 2
    return r.crop((l, t, l + size, t + size))


def squash_view(img, size):
    """Resize the whole image to size x size, ignoring aspect ratio."""
    return img.resize((size, size), Image.BICUBIC)


def four_views(img, size):
    """center, center+flip, squash, squash+flip — the 4 deterministic views."""
    c = center_crop_view(img, size)
    s = squash_view(img, size)
    return [c, TF.hflip(c), s, TF.hflip(s)]


# ── Feature extraction ────────────────────────────────────────────────────────


@torch.inference_mode()
def encode_images(model, preprocess_norm, paths, size, device, batch_size=32):
    """Return [N, D] L2-normalized embeddings: 4 views per image, mean-pooled.

    preprocess_norm: (mean, std) tensors for normalization. We build views
    ourselves (not open_clip's preprocess) so the 4 deterministic views match
    the recipe exactly; we still apply the model's normalization stats.
    """
    mean, std = preprocess_norm
    feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        view_batch = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            for v in four_views(img, size):
                t = (TF.to_tensor(v) - mean) / std
                view_batch.append(t)
        x = torch.stack(view_batch).to(device)          # [B*4, 3, S, S]
        e = model.encode_image(x)
        e = F.normalize(e, dim=-1)
        e = e.reshape(len(batch_paths), 4, -1).mean(1)   # pool 4 views
        e = F.normalize(e, dim=-1)
        feats.append(e.cpu().float())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    encoded {i + len(batch_paths)}/{len(paths)}")
    return torch.cat(feats, 0).numpy()


# ── Data ──────────────────────────────────────────────────────────────────────


def load_train(train_dir):
    """Return (paths, labels) over train/<class>/*.jpg with integer class names."""
    paths, labels = [], []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(d, f))
                labels.append(int(cls))
    return paths, np.array(labels)


def load_test(test_dir):
    fnames = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    return [os.path.join(test_dir, f) for f in fnames], fnames


def load_names(path, num_classes):
    names = [""] * num_classes
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["class_id"])
            if i < num_classes:
                names[i] = (row.get("name") or "").strip()
    return names


def write_probs(path, ids, probs, num_classes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [str(c) for c in range(num_classes)])
        for i, row in zip(ids, probs):
            w.writerow([i] + [f"{v:.6f}" for v in row])


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ViT-SO400M-16-SigLIP2-384",
                   help="open_clip model. Verify with open_clip.list_pretrained().")
    p.add_argument("--pretrained", default="webli")
    p.add_argument("--size", type=int, default=384, help="View resolution")
    p.add_argument("--names", default="class_names.csv",
                   help="For the zero-shot text branch")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 reg for logreg")
    p.add_argument("--alpha", type=float, default=0.65,
                   help="Fusion weight on the probe (image) probs; text gets 1-alpha")
    p.add_argument("--templates",
                   default="a photo of a {}.,a close-up photo of a {}.,a {} on a plain background.")
    p.add_argument("--out-probe", default="probe/probe_probs.csv")
    p.add_argument("--out-zeroshot", default="probe/zeroshot_probs.csv")
    p.add_argument("--out-fused", default="probe/fused_probs.csv")
    p.add_argument("--cache", default="probe/cache",
                   help="Where to cache extracted embeddings (.npy)")
    args = p.parse_args()

    # resolve data dirs via train.py's downloader if not given
    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg)
        train_dir = train_dir or cfg["train_dir"]
        test_dir = test_dir or cfg["test_dir"]
    print(f"train_dir={train_dir}\ntest_dir={test_dir}")

    import open_clip
    from sklearn.linear_model import LogisticRegression

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # normalization stats from open_clip's config
    cfg_vis = model.visual.preprocess_cfg if hasattr(model.visual, "preprocess_cfg") else {}
    mean = torch.tensor(cfg_vis.get("mean", (0.5, 0.5, 0.5))).view(3, 1, 1)
    std = torch.tensor(cfg_vis.get("std", (0.5, 0.5, 0.5))).view(3, 1, 1)
    print(f"norm mean={mean.flatten().tolist()} std={std.flatten().tolist()}")

    # ── 1. ENCODE ──
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    tr_paths, tr_labels = load_train(train_dir)
    te_paths, te_fnames = load_test(test_dir)

    tr_cache = cache / "train_emb.npy"
    te_cache = cache / "test_emb.npy"
    if tr_cache.exists():
        Xtr = np.load(tr_cache); print(f"loaded cached train emb {Xtr.shape}")
    else:
        print(f"Encoding {len(tr_paths)} train images...")
        Xtr = encode_images(model, (mean, std), tr_paths, args.size, device)
        np.save(tr_cache, Xtr)
    if te_cache.exists():
        Xte = np.load(te_cache); print(f"loaded cached test emb {Xte.shape}")
    else:
        print(f"Encoding {len(te_paths)} test images...")
        Xte = encode_images(model, (mean, std), te_paths, args.size, device)
        np.save(te_cache, Xte)

    # ── 2. PROBE ──
    print(f"Fitting logistic probe (C={args.C}, balanced)...")
    clf = LogisticRegression(
        C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1,
    )
    clf.fit(Xtr, tr_labels)
    train_acc = clf.score(Xtr, tr_labels)
    print(f"  train acc {train_acc:.4f}")
    P = clf.predict_proba(Xte)                      # [N_test, C]
    # align columns to 0..C-1 (LogisticRegression sorts classes)
    full_P = np.zeros((len(te_paths), args.num_classes))
    for j, cls in enumerate(clf.classes_):
        full_P[:, cls] = P[:, j]
    P = full_P
    write_probs(args.out_probe, te_fnames, P, args.num_classes)
    print(f"  wrote {args.out_probe}")

    # ── 3. ZERO-SHOT (text tower) ──
    names = load_names(args.names, args.num_classes)
    templates = [t.strip() for t in args.templates.split(",")]
    with torch.inference_mode():
        text_feats = []
        for nm in names:
            if not nm:
                text_feats.append(None); continue
            toks = tokenizer([t.format(nm) for t in templates]).to(device)
            tf_ = F.normalize(model.encode_text(toks), dim=-1).mean(0)
            text_feats.append(F.normalize(tf_, dim=0))
        known = torch.stack([t for t in text_feats if t is not None]).mean(0)
        known = F.normalize(known, dim=0)
        text_feats = torch.stack([t if t is not None else known for t in text_feats])  # [C, D]
        logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
        Z = (torch.tensor(Xte).to(device) @ text_feats.T) * logit_scale
        Z = Z.softmax(1).cpu().numpy()
    write_probs(args.out_zeroshot, te_fnames, Z, args.num_classes)
    print(f"  wrote {args.out_zeroshot}")

    # ── 4. FUSE ──
    fused = args.alpha * P + (1 - args.alpha) * Z
    fused = fused / fused.sum(1, keepdims=True)
    write_probs(args.out_fused, te_fnames, fused, args.num_classes)
    print(f"  wrote {args.out_fused} (alpha={args.alpha})")
    print("\nNext: python sinkhorn.py "
          f"{args.out_fused} --out submission_final.csv")


if __name__ == "__main__":
    main()

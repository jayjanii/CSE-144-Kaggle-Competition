"""Single-backbone SigLIP-2 probe pipeline (competitor-style, end to end).

Faithful reimplementation of the 0.96 recipe:

    1. ENCODE   Each image -> 4 deterministic TTA views
                (center crop, center+flip, squash resize, squash+flip),
                passed through the FROZEN SigLIP-2 SoViT-400m vision tower,
                mean-pooled into one L2-normalized 1152-dim embedding.
    2. PROBE    A single multinomial logistic regression (balanced classes,
                L2-regularized) on the frozen train embeddings -> probs P.
    3. ZERO-SHOT The SigLIP-2 text tower scores class-name prompts against the
                image embeddings -> probs Z.
    4. FUSE     0.65 * P + 0.35 * Z.
    5. BALANCE  Sinkhorn over the test (and validation) probabilities to match
                a uniform class prior.

Only the logistic head is trained; the backbone is frozen. The text branch and
class names are standard zero-shot CLIP. No external image datasets.

Reproducibility: fixed seed, deterministic views, embeddings cached to .npy.

Usage:
    pip install open_clip_torch scikit-learn
    # verify the model id first:
    python -c "import open_clip; print([m for m in open_clip.list_pretrained() if 'iglip2' in m[0].lower()])"

    python siglip2_pipeline.py \
        --model ViT-SO400M-16-SigLIP2-384 --pretrained webli \
        --names class_names.csv \
        --out submission.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# ── 1. Four deterministic TTA views ───────────────────────────────────────────


def center_crop_view(img, size):
    """Aspect-preserving short-side resize, then center crop to size x size."""
    w, h = img.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    r = img.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - size) // 2, (nh - size) // 2
    return r.crop((l, t, l + size, t + size))


def four_views(img, size):
    """center crop, center+flip, squash resize, squash+flip."""
    c = center_crop_view(img, size)
    sq = img.resize((size, size), Image.BICUBIC)
    return [c, TF.hflip(c), sq, TF.hflip(sq)]


@torch.inference_mode()
def encode(model, mean, std, paths, size, batch_size=24):
    """4-view pooled, L2-normalized embeddings for a list of image paths."""
    feats = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        views = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            for v in four_views(img, size):
                views.append((TF.to_tensor(v) - mean) / std)
        x = torch.stack(views).to(DEVICE)
        e = model.encode_image(x)
        e = F.normalize(e, dim=-1).reshape(len(chunk), 4, -1).mean(1)
        feats.append(F.normalize(e, dim=-1).cpu().float())
        if (i // batch_size + 1) % 15 == 0:
            print(f"    {i + len(chunk)}/{len(paths)}")
    return torch.cat(feats, 0).numpy()


# ── Data ──────────────────────────────────────────────────────────────────────


def load_train(train_dir):
    paths, labels = [], []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(d, f)); labels.append(int(cls))
    return paths, np.array(labels)


def load_test(test_dir):
    fn = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                key=lambda f: int(os.path.splitext(f)[0]))
    return [os.path.join(test_dir, f) for f in fn], fn


def load_names(path, C):
    names = [""] * C
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["class_id"])
            if i < C:
                names[i] = (row.get("name") or "").strip()
    return names


# ── 5. Sinkhorn class balancing ───────────────────────────────────────────────


def sinkhorn(P, col_target, n_iters=100, tau=1.0, eps=1e-12):
    """Balance P[N,C] toward the given per-class marginal via Sinkhorn iters."""
    logP = np.log(np.clip(P, eps, 1.0)) / tau
    logP -= logP.max(1, keepdims=True)
    M = np.exp(logP)
    N, C = M.shape
    r = np.ones((N, 1))
    c = np.asarray(col_target).reshape(1, C)
    for _ in range(n_iters):
        M *= (c / (M.sum(0, keepdims=True) + eps))
        M *= (r / (M.sum(1, keepdims=True) + eps))
    return M


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ViT-SO400M-16-SigLIP2-384")
    p.add_argument("--pretrained", default="webli")
    p.add_argument("--names", default="class_names.csv")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 reg for logreg")
    p.add_argument("--alpha", type=float, default=0.65, help="weight on probe P (text gets 1-alpha)")
    p.add_argument("--templates",
                   default="a photo of a {}.,a close-up photo of a {}.,a {} on a plain background.")
    p.add_argument("--sinkhorn", dest="sinkhorn", action="store_true", default=True,
                   help="apply Sinkhorn balancing (default ON, competitor-faithful)")
    p.add_argument("--no-sinkhorn", dest="sinkhorn", action="store_false")
    p.add_argument("--sinkhorn-tau", type=float, default=1.0)
    p.add_argument("--cache", default="siglip2/cache")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--out-probs", default="siglip2/fused_probs.csv")
    args = p.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    # data dirs
    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg)
        train_dir = train_dir or cfg["train_dir"]
        test_dir = test_dir or cfg["test_dir"]

    tr_paths, ytr = load_train(train_dir)
    te_paths, te_fnames = load_test(test_dir)
    C = args.num_classes
    print(f"{len(tr_paths)} train, {len(te_paths)} test, {C} classes")

    # model
    import open_clip
    print(f"Loading FROZEN {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(DEVICE).eval()
    vis = getattr(model.visual, "preprocess_cfg", {})
    mean = torch.tensor(vis.get("mean", (0.5, 0.5, 0.5))).view(3, 1, 1)
    std = torch.tensor(vis.get("std", (0.5, 0.5, 0.5))).view(3, 1, 1)
    size = vis.get("size", (384, 384))
    size = size[0] if isinstance(size, (tuple, list)) else size
    scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    print(f"  view size {size}, norm mean {mean.flatten().tolist()}")

    # ── 1. ENCODE (cached) ──
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    tr_cache, te_cache = cache / "train.npy", cache / "test.npy"
    if tr_cache.exists():
        Xtr = np.load(tr_cache); print(f"cached train emb {Xtr.shape}")
    else:
        print(f"Encoding {len(tr_paths)} train images (4 views each)...")
        Xtr = encode(model, mean, std, tr_paths, size); np.save(tr_cache, Xtr)
    if te_cache.exists():
        Xte = np.load(te_cache); print(f"cached test emb {Xte.shape}")
    else:
        print(f"Encoding {len(te_paths)} test images...")
        Xte = encode(model, mean, std, te_paths, size); np.save(te_cache, Xte)

    # ── held-out validation report (for the presentation) ──
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=SEED, stratify=ytr)
    val_clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1)
    val_clf.fit(Xa, ya)
    print(f"\nHeld-out 20% val: probe acc {val_clf.score(Xb, yb):.4f}")

    # leak-free OOF accuracy over all train (CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr))
    for tr, va in skf.split(Xtr, ytr):
        c = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1)
        c.fit(Xtr[tr], ytr[tr]); oof[va] = c.predict(Xtr[va])
    print(f"5-fold OOF: probe acc {(oof == ytr).mean():.4f}")

    # ── 2. PROBE (fit on all train, predict test) ──
    print("\nFitting final probe on all train...")
    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    P = np.zeros((len(te_paths), C))
    pr = clf.predict_proba(Xte)
    for j, cls in enumerate(clf.classes_):
        P[:, cls] = pr[:, j]

    # ── 3. ZERO-SHOT text ──
    names = load_names(args.names, C)
    templates = [t.strip() for t in args.templates.split(",")]
    with torch.inference_mode():
        feats = []
        for nm in names:
            if not nm:
                feats.append(None); continue
            toks = tokenizer([t.format(nm) for t in templates]).to(DEVICE)
            tf_ = F.normalize(model.encode_text(toks), dim=-1).mean(0)
            feats.append(F.normalize(tf_, dim=0))
        known = F.normalize(torch.stack([t for t in feats if t is not None]).mean(0), dim=0)
        feats = torch.stack([t if t is not None else known for t in feats])
        Z = ((torch.tensor(Xte).to(DEVICE) @ feats.T) * scale).softmax(1).cpu().numpy()

    # ── 4. FUSE ──
    fused = args.alpha * P + (1 - args.alpha) * Z
    fused = fused / fused.sum(1, keepdims=True)
    print(f"\nFused 0.{int(args.alpha*100)}*P + 0.{int((1-args.alpha)*100)}*Z")

    # ── 5. BALANCE ──
    if args.sinkhorn:
        before = np.bincount(fused.argmax(1), minlength=C)
        fused = sinkhorn(fused, np.full(C, len(te_paths) / C), tau=args.sinkhorn_tau)
        after = np.bincount(fused.argmax(1), minlength=C)
        print(f"Sinkhorn: class-count std {before.std():.2f} -> {after.std():.2f}")

    # ── write ──
    Path(args.out_probs).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_probs, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id"] + [str(c) for c in range(C)])
        for fn, row in zip(te_fnames, fused):
            w.writerow([fn] + [f"{v:.6f}" for v in row])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "predicted_class"])
        for fn, row in zip(te_fnames, fused):
            w.writerow([fn, int(row.argmax())])
    print(f"\nWrote {args.out} and {args.out_probs}")


if __name__ == "__main__":
    main()

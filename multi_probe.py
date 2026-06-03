"""Multi-backbone frozen logistic-probe ensemble (no fine-tuning).

Improves on the single-backbone probe recipe by exploiting architecture
diversity: each frozen backbone gets its own logistic probe, and the probes
are ensembled. CLIP-family backbones (SigLIP-2) additionally contribute a
zero-shot text branch. A final Sinkhorn step (run separately) enforces the
known 10-per-class test balance.

    backbone_i  -->  4-view pooled embedding  -->  logistic probe  -->  P_i
    SigLIP-2    -->  text tower on class names ------------------->  Z
    fused = sum_i w_i * P_i  +  w_text * Z   (renormalized)
    submission = sinkhorn(fused)

Nothing trains a backbone; only the per-backbone logistic heads are fit on
train/. Text branch + class names are standard zero-shot CLIP. No external
image datasets.

Backbone spec: "type:name[:pretrained]"  where type in {openclip, timm, dinov2}.
  openclip  -> open_clip.create_model_and_transforms (has text tower)
  timm      -> timm.create_model(..., num_classes=0)  (features only)
  dinov2    -> torch.hub facebookresearch/dinov2      (CLS embedding)

Usage:
    pip install open_clip_torch timm scikit-learn
    python multi_probe.py \
        --backbone openclip:ViT-SO400M-16-SigLIP2-384:webli \
        --backbone dinov2:dinov2_vitg14_reg \
        --backbone timm:eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \
        --names class_names.csv \
        --text-backbone openclip:ViT-SO400M-16-SigLIP2-384:webli \
        --alpha-text 0.35 \
        --out-fused multi_probe/fused_probs.csv

    python sinkhorn.py multi_probe/fused_probs.csv --out submission_final.csv
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
from sklearn.model_selection import StratifiedKFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── TTA views (center, center+flip, squash, squash+flip) ──────────────────────


def center_crop_view(img, size):
    w, h = img.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    r = img.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - size) // 2, (nh - size) // 2
    return r.crop((l, t, l + size, t + size))


def four_views(img, size):
    c = center_crop_view(img, size)
    sq = img.resize((size, size), Image.BICUBIC)
    return [c, TF.hflip(c), sq, TF.hflip(sq)]


# ── Backbone loaders → unified (image_fn, text_fn, mean, std, size) ────────────


def load_backbone(spec):
    """spec = 'type:name[:pretrained]'. Returns a dict with encode fns + norm."""
    parts = spec.split(":")
    btype, name = parts[0], parts[1]
    pretrained = parts[2] if len(parts) > 2 else None

    if btype == "openclip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained or "webli")
        tokenizer = open_clip.get_tokenizer(name)
        model = model.to(DEVICE).eval()
        vis = getattr(model.visual, "preprocess_cfg", {})
        mean = torch.tensor(vis.get("mean", (0.5, 0.5, 0.5))).view(3, 1, 1)
        std = torch.tensor(vis.get("std", (0.5, 0.5, 0.5))).view(3, 1, 1)
        size = vis.get("size", (384, 384))
        size = size[0] if isinstance(size, (tuple, list)) else size

        def image_fn(x):
            return F.normalize(model.encode_image(x), dim=-1)

        def text_fn(prompts):
            toks = tokenizer(prompts).to(DEVICE)
            return F.normalize(model.encode_text(toks), dim=-1)

        scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
        return dict(image_fn=image_fn, text_fn=text_fn, mean=mean, std=std,
                    size=size, logit_scale=scale, model=model)

    if btype == "timm":
        import timm
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
        except TypeError:
            model = timm.create_model(name, pretrained=True, num_classes=0)
        model = model.to(DEVICE).eval()
        dc = timm.data.resolve_model_data_config(model)
        mean = torch.tensor(dc["mean"]).view(3, 1, 1)
        std = torch.tensor(dc["std"]).view(3, 1, 1)
        size = dc["input_size"][-1]

        def image_fn(x):
            return F.normalize(model(x), dim=-1)

        return dict(image_fn=image_fn, text_fn=None, mean=mean, std=std,
                    size=size, logit_scale=None, model=model)

    if btype == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", name).to(DEVICE).eval()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        size = 518

        def image_fn(x):
            return F.normalize(model(x), dim=-1)   # CLS embedding (1536-d)

        return dict(image_fn=image_fn, text_fn=None, mean=mean, std=std,
                    size=size, logit_scale=None, model=model)

    raise SystemExit(f"unknown backbone type: {btype}")


@torch.inference_mode()
def encode(image_fn, mean, std, paths, size, batch_size=24):
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
        e = image_fn(x).reshape(len(chunk), 4, -1).mean(1)
        feats.append(F.normalize(e, dim=-1).cpu().float())
        if (i // batch_size + 1) % 15 == 0:
            print(f"    {i + len(chunk)}/{len(paths)}")
    return torch.cat(feats, 0).numpy()


# ── Data helpers ──────────────────────────────────────────────────────────────


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


def write_probs(path, ids, probs, C):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id"] + [str(c) for c in range(C)])
        for i, row in zip(ids, probs):
            w.writerow([i] + [f"{v:.6f}" for v in row])


def fit_probe_with_cv(Xtr, ytr, Xte, C_reg, num_classes, n_splits=5):
    """Fit a probe; report OOF accuracy via stratified CV; predict test probs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(ytr))
    for tr, va in skf.split(Xtr, ytr):
        clf = LogisticRegression(C=C_reg, max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr[tr], ytr[tr])
        oof[va] = clf.predict(Xtr[va])
    oof_acc = (oof == ytr).mean()

    clf = LogisticRegression(C=C_reg, max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    full = np.zeros((Xte.shape[0], num_classes))
    for j, cls in enumerate(clf.classes_):
        full[:, cls] = P[:, j]
    return full, oof_acc


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", action="append", required=True,
                   help="'type:name[:pretrained]'. Repeat per backbone.")
    p.add_argument("--weights", default=None,
                   help="Comma-separated per-backbone probe weights (default: equal)")
    p.add_argument("--text-backbone", default=None,
                   help="openclip spec to use for the zero-shot text branch")
    p.add_argument("--alpha-text", type=float, default=0.35,
                   help="Weight on the zero-shot text probs in the final fuse")
    p.add_argument("--names", default="class_names.csv")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 reg for logreg")
    p.add_argument("--templates",
                   default="a photo of a {}.,a close-up photo of a {}.,a {} on a plain background.")
    p.add_argument("--cache", default="multi_probe/cache")
    p.add_argument("--out-fused", default="multi_probe/fused_probs.csv")
    p.add_argument("--out-dir", default="multi_probe",
                   help="Also writes per-backbone probe_<name>.csv here")
    args = p.parse_args()

    # resolve data dirs
    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        from train import CONFIG, maybe_download_data
        cfg = dict(CONFIG); maybe_download_data(cfg)
        train_dir = train_dir or cfg["train_dir"]
        test_dir = test_dir or cfg["test_dir"]

    tr_paths, ytr = load_train(train_dir)
    te_paths, te_fnames = load_test(test_dir)
    C = args.num_classes
    print(f"{len(tr_paths)} train, {len(te_paths)} test, {C} classes\n")

    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    probe_probs = []
    oof_accs = []
    for spec in args.backbone:
        tag = spec.replace(":", "_").replace("/", "_").replace(".", "_")
        print(f"=== backbone {spec} ===")
        tr_cache = cache / f"{tag}_train.npy"
        te_cache = cache / f"{tag}_test.npy"
        if tr_cache.exists() and te_cache.exists():
            Xtr, Xte = np.load(tr_cache), np.load(te_cache)
            print(f"  cached: train {Xtr.shape}, test {Xte.shape}")
        else:
            bb = load_backbone(spec)
            print(f"  encoding train ({len(tr_paths)}) @ size {bb['size']}...")
            Xtr = encode(bb["image_fn"], bb["mean"], bb["std"], tr_paths, bb["size"])
            print(f"  encoding test ({len(te_paths)})...")
            Xte = encode(bb["image_fn"], bb["mean"], bb["std"], te_paths, bb["size"])
            np.save(tr_cache, Xtr); np.save(te_cache, Xte)
            del bb; torch.cuda.empty_cache()

        P, oof = fit_probe_with_cv(Xtr, ytr, Xte, args.C, C)
        print(f"  probe OOF acc = {oof:.4f}\n")
        probe_probs.append(P); oof_accs.append(oof)
        write_probs(Path(args.out_dir) / f"probe_{tag}.csv", te_fnames, P, C)

    # weights
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        assert len(w) == len(probe_probs), "weights count != backbone count"
    else:
        w = [1.0] * len(probe_probs)
    print("Per-backbone OOF:", [f"{a:.4f}" for a in oof_accs], "weights:", w)

    fused = sum(wi * Pi for wi, Pi in zip(w, probe_probs))
    fused = fused / fused.sum()  # rough normalize; renormalized below

    # zero-shot text branch
    if args.text_backbone:
        print(f"\n=== text branch: {args.text_backbone} ===")
        bb = load_backbone(args.text_backbone)
        assert bb["text_fn"] is not None, "text-backbone must be a CLIP-family (openclip) model"
        tag = args.text_backbone.replace(":", "_").replace("/", "_").replace(".", "_")
        te_cache = cache / f"{tag}_test.npy"
        Xte_t = np.load(te_cache) if te_cache.exists() else \
            encode(bb["image_fn"], bb["mean"], bb["std"], te_paths, bb["size"])
        names = load_names(args.names, C)
        templates = [t.strip() for t in args.templates.split(",")]
        with torch.inference_mode():
            feats = []
            for nm in names:
                if not nm:
                    feats.append(None); continue
                tf_ = bb["text_fn"]([t.format(nm) for t in templates]).mean(0)
                feats.append(F.normalize(tf_, dim=0))
            known = F.normalize(torch.stack([t for t in feats if t is not None]).mean(0), dim=0)
            feats = torch.stack([t if t is not None else known for t in feats])
            Z = (torch.tensor(Xte_t).to(DEVICE) @ feats.T) * bb["logit_scale"]
            Z = Z.softmax(1).cpu().numpy()
        write_probs(Path(args.out_dir) / "zeroshot.csv", te_fnames, Z, C)

        # renormalize the probe mixture to a proper distribution, then fuse with text
        Pmix = sum(wi * Pi for wi, Pi in zip(w, probe_probs))
        Pmix = Pmix / Pmix.sum(1, keepdims=True)
        fused = (1 - args.alpha_text) * Pmix + args.alpha_text * Z
    fused = fused / fused.sum(1, keepdims=True)

    write_probs(args.out_fused, te_fnames, fused, C)
    print(f"\nWrote {args.out_fused}")
    print(f"Next: python sinkhorn.py {args.out_fused} --out submission_final.csv")


if __name__ == "__main__":
    main()

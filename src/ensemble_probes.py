# frozen siglip-2 ensemble: each backbone gives a logistic probe + a zero-shot
# text head, averaged with weights picked on the oof.

import argparse
import csv
import itertools
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data import get_data_dirs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# a few classes have <5 images, so stratified k-fold cant fill every fold; fine
warnings.filterwarnings("ignore", message="The least populated class")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


@torch.inference_mode()
def encode(image_fn, mean, std, paths, size, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        views = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            for v in four_views(img, size):
                views.append((TF.to_tensor(v) - mean) / std)
        x = torch.stack(views).to(DEVICE)
        # frozen backbone, so half precision on gpu is free speed
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            e = image_fn(x).reshape(len(chunk), 4, -1).mean(1)
        feats.append(F.normalize(e.float(), dim=-1).cpu())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    {i + len(chunk)}/{len(paths)}")
    return torch.cat(feats, 0).numpy()


def load_backbone(model_name, pretrained):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model = model.to(DEVICE).eval()
    vis = getattr(model.visual, "preprocess_cfg", {})
    mean = torch.tensor(vis.get("mean", (0.5, 0.5, 0.5))).view(3, 1, 1)
    std = torch.tensor(vis.get("std", (0.5, 0.5, 0.5))).view(3, 1, 1)
    size = vis.get("size", (384, 384))
    size = size[0] if isinstance(size, (tuple, list)) else size
    scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    return dict(
        image_fn=lambda x: F.normalize(model.encode_image(x), dim=-1),
        text_fn=lambda pr: F.normalize(model.encode_text(tok(pr).to(DEVICE)), dim=-1),
        mean=mean, std=std, size=size, scale=scale, model=model,
    )


def load_train(train_dir):
    paths, labels = [], []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(d, f))
                    labels.append(int(cls))
    return paths, np.array(labels)


def load_test(test_dir):
    fn = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    return [os.path.join(test_dir, f) for f in fn], fn


def load_names(path, C):
    names = [""] * C
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["class_id"])
            if i < C:
                names[i] = (row.get("name") or "").strip()
    return names


def _logreg(C_reg):
    return LogisticRegression(C=C_reg, max_iter=2000, class_weight="balanced",
                              n_jobs=-1, random_state=SEED)


def oof_probe(X, y, C_reg, K, Xex=None, yex=None):
    # Xex/yex: optional pseudo-labeled rows added to every fold's train (never val)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), K))
    for tr, va in skf.split(X, y):
        Xf, yf = X[tr], y[tr]
        if Xex is not None and len(Xex):
            Xf, yf = np.concatenate([Xf, Xex]), np.concatenate([yf, yex])
        clf = _logreg(C_reg).fit(Xf, yf)
        pr = clf.predict_proba(X[va])
        for j, cls in enumerate(clf.classes_):
            oof[va, cls] = pr[:, j]
    return oof


def full_probe(X, y, Xte, C_reg, K, Xex=None, yex=None):
    Xf, yf = X, y
    if Xex is not None and len(Xex):
        Xf, yf = np.concatenate([X, Xex]), np.concatenate([y, yex])
    clf = _logreg(C_reg).fit(Xf, yf)
    P = np.zeros((len(Xte), K))
    pr = clf.predict_proba(Xte)
    for j, cls in enumerate(clf.classes_):
        P[:, cls] = pr[:, j]
    return P


def text_probs(bb, names, templates, X):
    with torch.inference_mode():
        feats = []
        for nm in names:
            if not nm:
                feats.append(None)
                continue
            t = bb["text_fn"]([tpl.format(nm) for tpl in templates]).mean(0)
            feats.append(F.normalize(t, dim=0))
        # unnamed class -> just use the average text vector
        known = F.normalize(torch.stack([t for t in feats if t is not None]).mean(0), dim=0)
        feats = torch.stack([t if t is not None else known for t in feats])
        return ((torch.tensor(X).to(DEVICE) @ feats.T) * bb["scale"]).softmax(1).cpu().numpy()


def probe_members(backbones, y, C, K, pseudo=None):
    # assemble probe (re-fit, optionally with pseudo) + cached text member per backbone
    oof, test, names = [], [], []
    for b in backbones:
        Xex = yex = None
        if pseudo is not None:
            mask, labels = pseudo
            Xex, yex = b["Xte"][mask], labels
        oof.append(oof_probe(b["Xtr"], y, C, K, Xex, yex))
        test.append(full_probe(b["Xtr"], y, b["Xte"], C, K, Xex, yex))
        names.append(f"{b['name']}[probe]")
        oof.append(b["Zo"])
        test.append(b["Zt"])
        names.append(f"{b['name']}[text]")
    return oof, test, names


def fit_weights(members_oof, y, grid):
    # first member pinned at 1, grid-search the rest on the oof
    best = (-1.0, None)
    for combo in itertools.product(grid, repeat=len(members_oof) - 1):
        w = (1.0,) + combo
        if sum(w) == 0:
            continue
        mix = sum(wi * mo for wi, mo in zip(w, members_oof))
        acc = (mix.argmax(1) == y).mean()
        if acc > best[0]:
            best = (float(acc), w)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", action="append", required=True,
                   help="model:pretrained:cache_dir (repeatable)")
    p.add_argument("--names", default="class_names.csv")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--templates",
                   default="a photo of a {}.,a close-up photo of a {}.,a {} on a plain background.")
    p.add_argument("--baseline", type=float, default=0.9592)
    p.add_argument("--grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--pseudo", type=float, default=0.0,
                   help="self-train: add test preds with confidence >= this (0 = off)")
    p.add_argument("--write", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    set_seed(args.seed)
    import sklearn
    print(f"seed {args.seed} | torch {torch.__version__} | "
          f"sklearn {sklearn.__version__} | numpy {np.__version__} | device {DEVICE}")

    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        d_train, d_test = get_data_dirs()
        train_dir = train_dir or d_train
        test_dir = test_dir or d_test

    tr_paths, y = load_train(train_dir)
    te_paths, te_fnames = load_test(test_dir)
    K = args.num_classes
    names = load_names(args.names, K)
    templates = [t.strip() for t in args.templates.split(",")]

    # encode each backbone once and keep its embeddings + zero-shot text members
    backbones = []
    for spec in args.backbone:
        model_name, pretrained, cache = spec.split(":", 2)
        cdir = Path(cache)
        cdir.mkdir(parents=True, exist_ok=True)
        trc, tec = cdir / "train.npy", cdir / "test.npy"

        print(f"\n{model_name}: loading backbone...")
        bb = load_backbone(model_name, pretrained)
        if trc.exists() and tec.exists():
            Xtr, Xte = np.load(trc), np.load(tec)
            print(f"  cached embeddings {Xtr.shape}")
        else:
            print(f"  encoding train ({len(tr_paths)}) @ {bb['size']}...")
            Xtr = encode(bb["image_fn"], bb["mean"], bb["std"], tr_paths, bb["size"], args.batch_size)
            print(f"  encoding test ({len(te_paths)})...")
            Xte = encode(bb["image_fn"], bb["mean"], bb["std"], te_paths, bb["size"], args.batch_size)
            np.save(trc, Xtr)
            np.save(tec, Xte)

        Zo = text_probs(bb, names, templates, Xtr)
        Zt = text_probs(bb, names, templates, Xte)
        print(f"  text  OOF acc {(Zo.argmax(1) == y).mean():.4f}")
        backbones.append(dict(name=model_name, Xtr=Xtr, Xte=Xte, Zo=Zo, Zt=Zt))
        del bb
        torch.cuda.empty_cache()

    grid = [float(x) for x in args.grid.split(",")]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def evaluate(pseudo, tag):
        m_oof, m_test, m_names = probe_members(backbones, y, args.C, K, pseudo)
        a, w = fit_weights(m_oof, y, grid)
        print(f"\n[{tag}] best weights:")
        for nm, wi in zip(m_names, w):
            print(f"  {nm:40s} {wi:g}")
        print(f"[{tag}] ensemble OOF: {a:.4f}   baseline: {args.baseline:.4f}")
        oof_mix = sum(wi * mo for wi, mo in zip(w, m_oof))
        fa = [float((oof_mix[va].argmax(1) == y[va]).mean()) for _, va in skf.split(oof_mix, y)]
        print("  per-fold: " + "  ".join(f"{x:.4f}" for x in fa) +
              f"  (mean {np.mean(fa):.4f}, std {np.std(fa):.4f})")
        test_mix = sum(wi * mt for wi, mt in zip(w, m_test))
        return a, test_mix / test_mix.sum(1, keepdims=True)

    acc, test_mix = evaluate(None, "frozen")

    if args.pseudo > 0:
        conf, pred = test_mix.max(1), test_mix.argmax(1)
        mask = conf >= args.pseudo
        print(f"\npseudo-labeling: {int(mask.sum())}/{len(mask)} test imgs >= {args.pseudo}")
        if mask.sum() > 0:
            acc, test_mix = evaluate((mask, pred[mask].astype(int)), "pseudo")
            print("  (note: pseudo-OOF is mildly optimistic; trust the public LB)")

    if args.write:
        with open(args.write, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["image_id", "predicted_class"])
            for fn, row in zip(te_fnames, test_mix):
                wr.writerow([fn, int(row.argmax())])
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()

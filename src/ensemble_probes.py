# frozen siglip-2 ensemble: each backbone gives a logistic probe + a zero-shot
# text head, averaged with weights picked on the oof.

import argparse
import csv
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data import get_data_dirs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# quiet the hf-hub weight/tokenizer download bars on cold start
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# a few classes have <5 images, so stratified k-fold cant fill every fold; fine
warnings.filterwarnings("ignore", message="The least populated class")

# hardware presets: bigger batch + bf16/tf32 fast matmul on the strong cards.
# encoding is the only gpu-bound step, so this is where it pays off.
# note: encode stacks 4 tta views, so real forward batch is 4x these numbers.
# measured end-to-end, cold encode of both backbones:
#   a100 80gb: batch 256, ~67gb peak, ~9 min
#   l4   24gb: batch 64,  ~18gb peak, ~25 min
GPU_PROFILES = {
    "a100": dict(batch=128, fast=True, dtype=torch.bfloat16),  # 80gb bumps to 256; also h100
    "l4":   dict(batch=64,  fast=False, dtype=torch.float16),
    "t4":   dict(batch=32,  fast=False, dtype=torch.float16),
}


def pick_profile(name):
    if name != "auto":
        return name
    if DEVICE != "cuda":
        return "t4"
    dev = torch.cuda.get_device_name(0).lower()
    if "l4" in dev:
        return "l4"
    if "t4" in dev:
        return "t4"
    if "a100" in dev or "h100" in dev:
        return "a100"
    # unknown card (h200, blackwell, a40, etc.): fall back by vram so a strong
    # card isnt stuck on the conservative t4 preset. 40gb+ datacenter cards are
    # ampere or newer, so bf16/tf32 in the a100 preset is safe.
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 40:
        return "a100"
    if vram >= 20:
        return "l4"
    return "t4"


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
def encode(image_fn, mean, std, paths, size, batch_size=32, dtype=torch.float16, desc="encode"):
    feats = []
    pbar = tqdm(total=len(paths), desc=desc, unit="img", leave=True)
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        views = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            for v in four_views(img, size):
                views.append((TF.to_tensor(v) - mean) / std)
        x = torch.stack(views).to(DEVICE)
        # frozen backbone, so low precision on gpu is free speed
        with torch.autocast(DEVICE, dtype=dtype, enabled=DEVICE == "cuda"):
            e = image_fn(x).reshape(len(chunk), 4, -1).mean(1)
        feats.append(F.normalize(e.float(), dim=-1).cpu())
        pbar.update(len(chunk))
    pbar.close()
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


def text_probs(bb, names, X, temp=1.0):
    with torch.inference_mode():
        feats = []
        for i, nm in enumerate(names):
            if not nm:
                feats.append(None)
                continue
            # category-specific prompts to pin down ambiguous names
            if i < 25:        # food
                tpls = ["a photo of {}, a type of food.",
                        "a plate of delicious {}.",
                        "a close-up photo of the food {}."]
            elif i < 50:      # flowers
                tpls = ["a close-up photo of a {} flower.",
                        "a photo of {}, a type of flower.",
                        "the beautiful {} flower."]
            elif i < 75:      # cars
                tpls = ["a photo of the car model {}.",
                        "a photo of the vehicle {}.",
                        "a {} driving on the street."]
            else:             # aircraft
                tpls = ["a photo of the {} aircraft.",
                        "the airplane {} in flight.",
                        "a photo of the {} plane."]
            t = bb["text_fn"]([tpl.format(nm) for tpl in tpls]).mean(0)
            feats.append(F.normalize(t, dim=0))
        # unnamed class -> just use the average text vector
        known = F.normalize(torch.stack([t for t in feats if t is not None]).mean(0), dim=0)
        feats = torch.stack([t if t is not None else known for t in feats])
        # temperature on top of the model's own logit scale
        logits = (torch.tensor(X).to(DEVICE) @ feats.T) * (bb["scale"] / temp)
        return logits.softmax(1).cpu().numpy()


def probe_members(backbones, y, C, K, pseudo=None):
    # assemble probe (re-fit, optionally with pseudo) + cached text member per backbone
    oof, test, names = [], [], []
    print("  members (oof acc):")
    for b in backbones:
        Xex = yex = None
        if pseudo is not None:
            mask, labels = pseudo
            Xex, yex = b["Xte"][mask], labels
        po = oof_probe(b["Xtr"], y, C, K, Xex, yex)
        probe_acc = (po.argmax(1) == y).mean()
        text_acc = (b["Zo"].argmax(1) == y).mean()
        print(f"    {b['name']:28s} probe {probe_acc:.4f}  text {text_acc:.4f} (temp {b['temp']:.2f})")
        oof.append(po)
        test.append(full_probe(b["Xtr"], y, b["Xte"], C, K, Xex, yex))
        names.append(f"{b['name']}[probe]")
        oof.append(b["Zo"])
        test.append(b["Zt"])
        names.append(f"{b['name']}[text]")

    # early-fusion probe on the concatenated embeddings
    if len(backbones) > 1:
        Xtr_concat = np.concatenate([b["Xtr"] for b in backbones], axis=-1)
        Xte_concat = np.concatenate([b["Xte"] for b in backbones], axis=-1)
        # renormalize after concat
        Xtr_concat = Xtr_concat / np.linalg.norm(Xtr_concat, axis=-1, keepdims=True)
        Xte_concat = Xte_concat / np.linalg.norm(Xte_concat, axis=-1, keepdims=True)
        Xex_concat = yex_concat = None
        if pseudo is not None:
            mask, labels = pseudo
            Xex_concat, yex_concat = Xte_concat[mask], labels
        po_concat = oof_probe(Xtr_concat, y, C, K, Xex_concat, yex_concat)
        print(f"    {'concat':28s} probe {(po_concat.argmax(1) == y).mean():.4f}")
        oof.append(po_concat)
        test.append(full_probe(Xtr_concat, y, Xte_concat, C, K, Xex_concat, yex_concat))
        names.append("concat[probe]")

    return oof, test, names


def fit_weights(members_oof, y, grid, passes=4, restarts=60):
    # random-restart coordinate ascent on the oof. first member pinned at 1, the
    # rest tuned one at a time; restarts escape the local optima that plain greedy
    # falls into. scales linearly with members (exhaustive grid blows up past ~3).
    M = len(members_oof)
    rng = np.random.RandomState(SEED)

    def acc_of(ww):
        mix = sum(wi * mo for wi, mo in zip(ww, members_oof))
        return float((mix.argmax(1) == y).mean())

    def ascend(w):
        best = acc_of(w)
        for _ in range(passes):
            improved = False
            for i in range(1, M):
                best_g = w[i]
                for g in grid:
                    w[i] = g
                    a = acc_of(w)
                    if a > best:
                        best, best_g, improved = a, g, True
                w[i] = best_g
            if not improved:
                break
        return best, tuple(w)

    starts = [[1.0] + [0.0] * (M - 1)]  # canonical start
    for _ in range(restarts):
        starts.append([1.0] + [float(rng.choice(grid)) for _ in range(M - 1)])

    best = (-1.0, None)
    for s in starts:
        a, w = ascend(s)
        if a > best[0]:
            best = (a, w)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", action="append", required=True,
                   help="model:pretrained:cache_dir (repeatable)")
    p.add_argument("--names", default="class_names.csv")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--gpu", choices=["auto", "a100", "l4", "t4"], default="auto",
                   help="hardware preset: batch size + fast-matmul knobs")
    p.add_argument("--batch-size", type=int, default=None, help="override preset batch")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--baseline", type=float, default=0.9592)
    p.add_argument("--grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--pseudo", type=float, default=0.0,
                   help="self-train: add test preds with confidence >= this (0 = off)")
    p.add_argument("--write", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    set_seed(args.seed)

    prof_name = pick_profile(args.gpu)
    prof = GPU_PROFILES[prof_name]
    batch_size = args.batch_size or prof["batch"]
    # a100 ships in 40 and 80gb; only use the big batch when the vram is there
    if args.batch_size is None and prof_name == "a100" and DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 256 if vram >= 70 else 128
    if prof["fast"] and DEVICE == "cuda":
        # tf32 matmul is faster and still deterministic; leave cudnn deterministic.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dev_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    import sklearn
    print(f"seed {args.seed} | device {dev_name} | preset {prof_name} (batch {batch_size})")
    print(f"torch {torch.__version__} | sklearn {sklearn.__version__} | numpy {np.__version__}")

    train_dir, test_dir = args.train_dir, args.test_dir
    if not train_dir or not test_dir:
        d_train, d_test = get_data_dirs()
        train_dir = train_dir or d_train
        test_dir = test_dir or d_test

    tr_paths, y = load_train(train_dir)
    te_paths, te_fnames = load_test(test_dir)
    K = args.num_classes
    names = load_names(args.names, K)

    # encode each backbone once and keep its embeddings + zero-shot text members
    print("\n[encode]")
    backbones = []
    for spec in args.backbone:
        model_name, pretrained, cache = spec.split(":", 2)
        cdir = Path(cache)
        cdir.mkdir(parents=True, exist_ok=True)
        trc, tec = cdir / "train.npy", cdir / "test.npy"

        bb = load_backbone(model_name, pretrained)
        if trc.exists() and tec.exists():
            Xtr, Xte = np.load(trc), np.load(tec)
            print(f"  {model_name:28s} cached {Xtr.shape}")
        else:
            print(f"  {model_name} @ {bb['size']}")
            Xtr = encode(bb["image_fn"], bb["mean"], bb["std"], tr_paths, bb["size"],
                         batch_size, prof["dtype"], desc="    train")
            Xte = encode(bb["image_fn"], bb["mean"], bb["std"], te_paths, bb["size"],
                         batch_size, prof["dtype"], desc="    test ")
            np.save(trc, Xtr)
            np.save(tec, Xte)

        # tune the text-head temperature on oof log-loss
        best_temp, best_loss = 1.0, float("inf")
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            Zo_temp = text_probs(bb, names, Xtr, temp=temp)
            eps = 1e-15
            loss = -np.mean(np.log(np.clip(Zo_temp[np.arange(len(y)), y], eps, 1.0 - eps)))
            if loss < best_loss:
                best_loss, best_temp = loss, temp

        Zo = text_probs(bb, names, Xtr, temp=best_temp)
        Zt = text_probs(bb, names, Xte, temp=best_temp)
        backbones.append(dict(name=model_name, Xtr=Xtr, Xte=Xte, Zo=Zo, Zt=Zt, temp=best_temp))
        del bb
        torch.cuda.empty_cache()

    grid = [float(x) for x in args.grid.split(",")]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def evaluate(pseudo):
        m_oof, m_test, m_names = probe_members(backbones, y, args.C, K, pseudo)
        a, w = fit_weights(m_oof, y, grid)
        print("  weights:")
        for nm, wi in zip(m_names, w):
            print(f"    {nm:34s} {wi:g}")
        print(f"  ensemble oof {a:.4f}  (baseline {args.baseline:.4f})")
        oof_mix = sum(wi * mo for wi, mo in zip(w, m_oof))
        fa = [float((oof_mix[va].argmax(1) == y[va]).mean()) for _, va in skf.split(oof_mix, y)]
        print(f"  per-fold {' '.join(f'{x:.4f}' for x in fa)}"
              f"  (mean {np.mean(fa):.4f} std {np.std(fa):.4f})")
        test_mix = sum(wi * mt for wi, mt in zip(w, m_test))
        return a, test_mix / test_mix.sum(1, keepdims=True)

    print("\n[frozen]")
    acc, test_mix = evaluate(None)

    if args.pseudo > 0:
        conf, pred = test_mix.max(1), test_mix.argmax(1)
        mask = conf >= args.pseudo
        n = int(mask.sum())
        print(f"\n[pseudo >= {args.pseudo:g}]  {n}/{len(mask)} test imgs added")
        if n > 0:
            acc, test_mix = evaluate((mask, pred[mask].astype(int)))
            print("  (oof optimistic)")

    if args.write:
        with open(args.write, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["ID", "Label"])
            for fn, row in zip(te_fnames, test_mix):
                wr.writerow([fn, int(row.argmax())])
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()

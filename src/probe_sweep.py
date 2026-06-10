# extensive grid search for the linear probe, scored on leak-free 5-fold OOF.
# reuses the cached embeddings, so it runs on cpu in a few minutes.

import argparse
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

SEED = 42
warnings.filterwarnings("ignore")  # quiet sklearn convergence/fold chatter


def load_labels(train_dir):
    labels = []
    for cls in sorted(os.listdir(train_dir), key=lambda s: int(s) if s.isdigit() else s):
        d = os.path.join(train_dir, cls)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    labels.append(int(cls))
    return np.array(labels)


def oof_acc(X, y, K, **kw):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), K))
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, n_jobs=-1, random_state=SEED, **kw)
        clf.fit(X[tr], y[tr])
        pr = clf.predict_proba(X[va])
        for j, cls in enumerate(clf.classes_):
            oof[va, cls] = pr[:, j]
    return float((oof.argmax(1) == y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True, help="cached train embeddings .npy")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--C", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--saga", action="store_true", help="also sweep l1/elasticnet via saga (slow)")
    args = p.parse_args()

    train_dir = args.train_dir
    if train_dir is None:
        from data import get_data_dirs
        train_dir = get_data_dirs()[0]
    X = np.load(args.emb).astype(np.float32)
    y = load_labels(train_dir)
    assert len(X) == len(y), f"emb {len(X)} != labels {len(y)}"
    K = args.num_classes
    Cs = [float(c) for c in args.C.split(",")]

    # (penalty, solver) pairs; lbfgs is multinomial, liblinear is one-vs-rest
    combos = [("l2", "lbfgs"), ("l1", "liblinear")]
    results = []
    for penalty, solver in combos:
        for cw in (None, "balanced"):
            for C in Cs:
                acc = oof_acc(X, y, K, C=C, penalty=penalty, solver=solver, class_weight=cw)
                cfg = dict(C=C, penalty=penalty, solver=solver, class_weight=cw)
                results.append((acc, cfg))
                print(f"{acc:.4f}  C={C:<7g} {penalty:<3} {solver:<10} cw={cw}")

    if args.saga:
        for l1r in (0.1, 0.5, 0.9):
            for cw in (None, "balanced"):
                for C in Cs:
                    acc = oof_acc(X, y, K, C=C, penalty="elasticnet", solver="saga",
                                  l1_ratio=l1r, class_weight=cw)
                    cfg = dict(C=C, penalty="elasticnet", solver="saga", l1_ratio=l1r, class_weight=cw)
                    results.append((acc, cfg))
                    print(f"{acc:.4f}  C={C:<7g} elasticnet saga l1r={l1r} cw={cw}")

    results.sort(key=lambda r: -r[0])
    print(f"\ntop {args.top}:")
    for acc, cfg in results[:args.top]:
        print(f"  {acc:.4f}  {cfg}")
    best_acc, best_cfg = results[0]
    print(f"\nbest: {best_acc:.4f}  {best_cfg}")


if __name__ == "__main__":
    main()

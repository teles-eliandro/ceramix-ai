"""
CERAMIX-AI — Training & honest evaluation.

Trains the ensemble described in the spec (CatBoost + XGBoost + MLP) for:
  - RGB regression          (colour)
  - colour-family classification (9 classes)
  - surface/texture classification (9 classes)
  - transparency classification (4 classes)

Every metric reported here is measured on the fixed held-out test split.
Nothing is extrapolated or estimated — if the numbers are poor, they are
reported as poor.

Reference: GlazyBench, arXiv:2605.06641. MIT licence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import build_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ RGB ----
def train_rgb(Xtr, ytr, Xte, yte) -> dict:
    """Predict R,G,B. Also report DeltaE-like distance in RGB space."""
    targets = ["R", "G", "B"]
    mask_tr = ytr[targets].notna().all(axis=1)
    mask_te = yte[targets].notna().all(axis=1)
    Xtr_, Xte_ = Xtr[mask_tr], Xte[mask_te]
    Ytr, Yte = ytr.loc[mask_tr, targets].values, yte.loc[mask_te, targets].values
    log(f"RGB: train={Xtr_.shape} test={Xte_.shape}")

    pipes = {
        "catboost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", __import__("catboost").CatBoostRegressor(
                iterations=1200, depth=8, learning_rate=0.05,
                l2_leaf_reg=3.0, loss_function="MultiRMSE",
                random_seed=SEED, verbose=False, thread_count=4)),
        ]),
        "xgboost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", __import__("xgboost").XGBRegressor(
                n_estimators=900, max_depth=8, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                random_state=SEED, n_jobs=4, tree_method="hist")),
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                random_state=SEED, n_jobs=4)),
        ]),
        "mlp": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                               activation="relu", alpha=1e-3,
                               learning_rate_init=1e-3, max_iter=400,
                               early_stopping=True, n_iter_no_change=20,
                               random_state=SEED)),
        ]),
    }

    preds, scores = {}, {}
    for name, pipe in pipes.items():
        t0 = time.time()
        pipe.fit(Xtr_, Ytr)
        p = pipe.predict(Xte_)
        preds[name] = p
        scores[name] = {
            "r2": float(r2_score(Yte, p, multioutput="uniform_average")),
            "mae": float(mean_absolute_error(Yte, p)),
            "rmse": float(np.sqrt(mean_squared_error(Yte, p))),
            "secs": round(time.time() - t0, 1),
        }
        log(f"  {name:9} R2={scores[name]['r2']:.4f}  MAE={scores[name]['mae']:.2f}  "
            f"RMSE={scores[name]['rmse']:.2f}  ({scores[name]['secs']}s)")

    # weighted ensemble (weights from the spec: 0.4 / 0.3 / 0.3)
    W = {"catboost": 0.4, "xgboost": 0.3, "mlp": 0.3}
    ens = sum(preds[k] * w for k, w in W.items())
    scores["ensemble"] = {
        "r2": float(r2_score(Yte, ens, multioutput="uniform_average")),
        "mae": float(mean_absolute_error(Yte, ens)),
        "rmse": float(np.sqrt(mean_squared_error(Yte, ens))),
        "weights": W,
    }
    log(f"  ENSEMBLE  R2={scores['ensemble']['r2']:.4f}  "
        f"MAE={scores['ensemble']['mae']:.2f}  RMSE={scores['ensemble']['rmse']:.2f}")

    # per-channel R2 — brightness channels usually differ a lot
    per = {}
    for i, ch in enumerate(targets):
        per[ch] = float(r2_score(Yte[:, i], ens[:, i]))
    scores["per_channel_r2"] = per
    log(f"  per-channel R2: {per}")

    # Euclidean RGB error (proxy for perceptual distance)
    dist = np.linalg.norm(Yte - ens, axis=1)
    scores["rgb_dist_mean"] = float(dist.mean())
    scores["rgb_dist_p50"] = float(np.percentile(dist, 50))
    scores["rgb_dist_p90"] = float(np.percentile(dist, 90))
    log(f"  RGB distance: mean={dist.mean():.2f} p50={np.percentile(dist,50):.2f} "
        f"p90={np.percentile(dist,90):.2f}")

    return scores, {k: v for k, v in pipes.items()}


# ------------------------------------------------------- classification ----
def train_classifier(Xtr, ytr, Xte, yte, target: str, kind: str) -> dict:
    """Train the 3-model ensemble for a classification target."""
    mtr, mte = ytr[target].notna(), yte[target].notna()
    Xtr_, Xte_ = Xtr[mtr], Xte[mte]
    ytr_, yte_ = ytr.loc[mtr, target].astype(str), yte.loc[mte, target].astype(str)
    le = LabelEncoder().fit(ytr_)
    ytr_e, yte_e = le.transform(ytr_), le.transform(yte_)
    log(f"{target}: train={Xtr_.shape} test={Xte_.shape} classes={list(le.classes_)}")

    import catboost
    import xgboost

    models = {
        "catboost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", catboost.CatBoostClassifier(
                iterations=900, depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
                loss_function="MultiClass", random_seed=SEED, verbose=False,
                thread_count=4)),
        ]),
        "xgboost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", xgboost.XGBClassifier(
                n_estimators=700, max_depth=8, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                random_state=SEED, n_jobs=4, tree_method="hist")),
        ]),
        "mlp": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", MLPClassifier(hidden_layer_sizes=(256, 128),
                                alpha=1e-3, max_iter=500,
                                early_stopping=True, n_iter_no_change=20,
                                random_state=SEED)),
        ]),
    }

    probas, scores = {}, {}
    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(Xtr_, ytr_e)
        p = pipe.predict(Xte_)
        probas[name] = pipe.predict_proba(Xte_)
        scores[name] = {
            "accuracy": float(accuracy_score(yte_e, p)),
            "f1_macro": float(f1_score(yte_e, p, average="macro", zero_division=0)),
            "secs": round(time.time() - t0, 1),
        }
        log(f"  {name:9} acc={scores[name]['accuracy']:.4f} "
            f"f1={scores[name]['f1_macro']:.4f} ({scores[name]['secs']}s)")

    W = {"catboost": 0.4, "xgboost": 0.3, "mlp": 0.3}
    ep = sum(probas[k] * w for k, w in W.items())
    ens = ep.argmax(axis=1)
    scores["ensemble"] = {
        "accuracy": float(accuracy_score(yte_e, ens)),
        "f1_macro": float(f1_score(yte_e, ens, average="macro", zero_division=0)),
        "weights": W,
    }
    log(f"  ENSEMBLE  acc={scores['ensemble']['accuracy']:.4f} "
        f"f1={scores['ensemble']['f1_macro']:.4f}")

    # honest baseline: always predict the most frequent class
    from collections import Counter
    maj = Counter(ytr_e).most_common(1)[0][0]
    base_acc = float((yte_e == maj).mean())
    scores["majority_baseline_acc"] = base_acc
    log(f"  majority-class baseline acc={base_acc:.4f}  "
        f"(lift={scores['ensemble']['accuracy']-base_acc:+.4f})")

    scores["classes"] = list(le.classes_)
    return scores, le


# ---------------------------------------------------------------- main ----
def main() -> None:
    t_all = time.time()
    log("building features ...")
    from features import ingredient_vocab
    vocab = ingredient_vocab("train", 40)
    log(f"ingredient vocabulary (from train, n={len(vocab)}): {vocab[:6]} ...")
    Xtr, ytr = build_split("train", ing_vocab=vocab)
    Xte, yte = build_split("test", ing_vocab=vocab)

    # strictly align columns — test may lack a rare material, train may have extra
    Xtr, Xte = Xtr.align(Xte, join="inner", axis=1)
    Xtr = Xtr.fillna(0.0)
    Xte = Xte.fillna(0.0)
    log(f"train X={Xtr.shape}  test X={Xte.shape}  (aligned)")

    report = {"dataset": {
        "name": "GlazyBench", "source": "AlpachinoNLP/GlazyBench (HuggingFace)",
        "arxiv": "arXiv:2605.06641", "licence": "MIT",
        "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
        "n_features": int(Xtr.shape[1]),
    }, "tasks": {}}

    log("=== RGB regression ===")
    report["tasks"]["rgb"], _ = train_rgb(Xtr, ytr, Xte, yte)

    log("=== colour family ===")
    report["tasks"]["color_family"], _ = train_classifier(
        Xtr, ytr, Xte, yte, "color_family", "color")

    log("=== surface / texture ===")
    report["tasks"]["surface"], _ = train_classifier(
        Xtr, ytr, Xte, yte, "surface", "texture")

    log("=== transparency ===")
    report["tasks"]["transparency"], _ = train_classifier(
        Xtr, ytr, Xte, yte, "transparency", "transparency")

    report["total_secs"] = round(time.time() - t_all, 1)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    log(f"done in {report['total_secs']}s -> models/metrics.json")

    # concise summary
    print("\n" + "=" * 66)
    print("SUMMARY (held-out test set, n=%d)" % len(Xte))
    print("=" * 66)
    r = report["tasks"]["rgb"]["ensemble"]
    print(f"RGB regression   R2={r['r2']:.4f}  MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}")
    for t in ("color_family", "surface", "transparency"):
        e = report["tasks"][t]["ensemble"]
        b = report["tasks"][t]["majority_baseline_acc"]
        print(f"{t:16} acc={e['accuracy']:.4f}  f1={e['f1_macro']:.4f}  "
              f"(baseline {b:.4f})")


if __name__ == "__main__":
    main()

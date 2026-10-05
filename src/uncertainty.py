"""
CERAMIX-AI — Uncertainty-aware colour prediction.

The dataset's label noise is large (see models/variance_decomp.json), so a
single point estimate is misleading. This module trains quantile regressors
that output an *interval* for each colour channel, plus a K-nearest-neighbour
conformal wrapper that produces empirically calibrated intervals.

Output: for a given recipe, instead of "R = 138", the system reports
"R = 138 [115, 162] (90% coverage)".

Method
------
1. Gradient-boosted quantile regression (CatBoost Quantile:alpha).
   Trained at alpha = 0.05 / 0.5 / 0.95 for each channel.
2. Split-conformal calibration on a held-out slice of train: we measure the
   actual coverage on unseen data and widen the interval so the nominal
   coverage is honest.

References: GlazyBench (arXiv:2605.06641); conformal prediction following
Vovk et al.; quantile regression in gradient boosting.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "models"
SEED = 42
np.random.seed(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def qreg(alpha: float):
    import catboost
    return catboost.CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=700, depth=7, learning_rate=0.06,
        random_seed=SEED, verbose=False, thread_count=4)


def main():
    t0 = time.time()
    from features import build_split, ingredient_vocab

    log("loading data ...")
    vocab = ingredient_vocab("train", 40)
    Xtr, ytr = build_split("train", ing_vocab=vocab)
    Xte, yte = build_split("test", ing_vocab=vocab)
    Xtr, Xte = Xtr.align(Xte, join="inner", axis=1)
    Xtr, Xte = Xtr.fillna(0), Xte.fillna(0)

    cols = ["R", "G", "B"]
    mtr = ytr[cols].notna().all(axis=1).values
    mte = yte[cols].notna().all(axis=1).values
    Ytr = ytr.loc[mtr, cols].values.astype(float)
    Yte = yte.loc[mte, cols].values.astype(float)

    # split train into proper-train and calibration
    n = len(Ytr)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr_i, cal_i = idx[:cut], idx[cut:]
    Xa, Ya = Xtr[mtr].iloc[tr_i], Ytr[tr_i]
    Xc, Yc = Xtr[mtr].iloc[cal_i], Ytr[cal_i]
    log(f"proper-train={len(Ya)}  calibration={len(Xc)}  test={len(Yte)}")

    result = {"method": "gradient-boosted quantile regression + split conformal",
              "nominal_coverage": 0.90, "channels": {}}
    lo_all, hi_all = [], []

    for j, ch in enumerate(cols):
        log(f"quantile models for {ch} ...")
        pipes = {}
        for a in (0.05, 0.5, 0.95):
            p = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", qreg(a))])
            p.fit(Xa, Ya[:, j])
            pipes[a] = p

        # conformal calibration: measure |y - median| on calibration set
        qc = np.abs(Yc[:, j] - pipes[0.5].predict(Xc))
        # widen so that the empirical coverage matches 90%
        width = float(np.quantile(qc, 0.90))

        med = pipes[0.5].predict(Xte)
        lo = med - width
        hi = med + width
        cov = float(np.mean((Yte[:, j] >= lo) & (Yte[:, j] <= hi)))
        mae = float(np.mean(np.abs(Yte[:, j] - med)))

        # also record the model's own (uncalibrated) interval for comparison
        lo_m = pipes[0.05].predict(Xte); hi_m = pipes[0.95].predict(Xte)
        cov_m = float(np.mean((Yte[:, j] >= lo_m) & (Yte[:, j] <= hi_m)))

        result["channels"][ch] = {
            "conformal_width": width,
            "coverage_conformal": cov,
            "coverage_model_interval": cov_m,
            "median_mae": mae,
        }
        log(f"  {ch}: width=±{width:.1f} conformal_cov={cov:.3f} "
            f"model_cov={cov_m:.3f} MAE={mae:.2f}")
        lo_all.append(lo); hi_all.append(hi)

    LO = np.vstack(lo_all).T; HI = np.vstack(hi_all).T
    # joint (all three channels inside their intervals at once)
    joint = float(np.mean(np.all((Yte >= LO) & (Yte <= HI), axis=1)))
    result["joint_coverage"] = joint
    result["total_secs"] = round(time.time() - t0, 1)
    log(f"joint coverage (all 3 channels simultaneously) = {joint:.3f}")

    (OUT / "uncertainty.json").write_text(json.dumps(result, indent=2))
    log("saved models/uncertainty.json")

    print("\n" + "=" * 70)
    print("UNCERTAINTY MODEL — calibrated 90% prediction intervals")
    print("=" * 70)
    for ch, v in result["channels"].items():
        print(f"  {ch}: ±{v['conformal_width']:5.1f}  "
              f"coverage={v['coverage_conformal']:.3f}  MAE={v['median_mae']:.2f}")
    print(f"  joint coverage (all channels) = {joint:.3f}")


if __name__ == "__main__":
    main()

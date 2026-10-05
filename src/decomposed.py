"""
CERAMIX-AI — Decomposed / conditioned models.

Instead of one hard multi-output model (colour AND texture), we build a
family of constrained models where part of the problem is fixed and only
the remaining degrees of freedom are learned. This is the classic
"reduce the hypothesis space" strategy: conditioning shrinks the target
variance, so accuracy on the conditioned task rises.

Model families implemented
--------------------------
M1  Global RGB regression            (baseline, unconstrained)
M2  RGB per colour-family            (colour family given as input)
M3  RGB per atmosphere x cone band   (firing regime given as input)
M4  RGB for texture-conditioned subsets (glossy / matte / satin)
M5  RGB per colour family, with a
    restricted "delta" target         (predict deviation from family mean)
M6  Surface (texture) classification
    conditioned on transparency
M7  Recipe-local models: group by
    dominant flux system (alkali / alkaline-earth / mixed)

Every metric is measured on the untouched held-out test split.

Reference: GlazyBench, arXiv:2605.06641 (MIT).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cat_reg():
    import catboost
    return catboost.CatBoostRegressor(
        iterations=900, depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
        loss_function="MultiRMSE", random_seed=SEED, verbose=False, thread_count=4)


def xgb_reg():
    import xgboost
    return xgboost.XGBRegressor(
        n_estimators=700, max_depth=8, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=1.0, random_state=SEED,
        n_jobs=4, tree_method="hist")


def eval_rgb(Y_true, P):
    return {
        "n": int(len(Y_true)),
        "r2": float(r2_score(Y_true, P, multioutput="uniform_average")),
        "mae": float(mean_absolute_error(Y_true, P)),
        "rmse": float(np.sqrt(np.mean((Y_true - P) ** 2))),
        "dist_mean": float(np.linalg.norm(Y_true - P, axis=1).mean()),
    }


def load_conditioned():
    """Load features + a rich set of conditioning variables."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from features import build_split, ingredient_vocab

    vocab = ingredient_vocab("train", 40)
    Xtr, ytr = build_split("train", ing_vocab=vocab)
    Xte, yte = build_split("test", ing_vocab=vocab)
    Xtr, Xte = Xtr.align(Xte, join="inner", axis=1)

    # conditioning columns (present in both X and y after align)
    cond_tr = Xtr[["atmo_oxidation", "atmo_reduction", "cone_mean"]].copy()
    cond_te = Xte[["atmo_oxidation", "atmo_reduction", "cone_mean"]].copy()
    return Xtr.fillna(0), ytr, Xte.fillna(0), yte, cond_tr, cond_te


def cone_band(c):
    if not np.isfinite(c):
        return "unknown"
    if c < 5:
        return "low(<5)"
    if c <= 7:
        return "mid(5-7)"
    if c <= 9:
        return "high(8-9)"
    return "veryhigh(>9)"


def atmosphere(row):
    if row.get("atmo_reduction", 0) > 0.5:
        return "reduction"
    if row.get("atmo_oxidation", 0) > 0.5:
        return "oxidation"
    return "other"


def main():
    t0 = time.time()
    log("loading features + conditioning variables ...")
    Xtr, ytr, Xte, yte, ctr, cte = load_conditioned()
    log(f"train X={Xtr.shape} test X={Xte.shape}")

    rgb_cols = ["R", "G", "B"]
    mtr = ytr[rgb_cols].notna().all(axis=1).values
    mte = yte[rgb_cols].notna().all(axis=1).values
    Ytr = ytr.loc[mtr, rgb_cols].values.astype(float)
    Yte = yte.loc[mte, rgb_cols].values.astype(float)

    report = {"dataset": {
        "name": "GlazyBench", "arxiv": "arXiv:2605.06641",
        "licence": "MIT", "source": "AlpachinoNLP/GlazyBench",
        "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
        "n_features": int(Xtr.shape[1])}, "models": {}}

    # ---------------- M1: global baseline -----------------------------
    log("=== M1 global RGB (baseline) ===")
    P, _ = _fit(Xtr[mtr], Ytr, Xte[mte])
    report["models"]["M1_global"] = eval_rgb(Yte, P)
    log(f"  {report['models']['M1_global']}")

    # ---------------- conditioning keys -------------------------------
    fam_tr = ytr.loc[mtr, "color_family"].astype(str).values
    fam_te = yte.loc[mte, "color_family"].astype(str).values
    atm_tr = ctr[mtr].apply(atmosphere, axis=1).values
    atm_te = cte[mte].apply(atmosphere, axis=1).values
    cb_tr = ctr[mtr]["cone_mean"].apply(cone_band).values
    cb_te = cte[mte]["cone_mean"].apply(cone_band).values
    surf_tr = np.array([str(s) for s in ytr.loc[mtr, "surface"].values])
    surf_te = np.array([str(s) for s in yte.loc[mte, "surface"].values])

    # ---------------- M2: per colour family ---------------------------
    log("=== M2 RGB conditioned on colour family ===")
    report["models"]["M2_by_color_family"] = _grouped_model(
        Xtr[mtr], Ytr, Xte[mte], Yte, fam_tr, fam_te, min_n=80, label="color_family")

    # ---------------- M3: per firing regime ---------------------------
    log("=== M3 RGB conditioned on firing regime (atmosphere x cone band) ===")
    reg_tr = np.array([f"{a}|{b}" for a, b in zip(atm_tr, cb_tr)])
    reg_te = np.array([f"{a}|{b}" for a, b in zip(atm_te, cb_te)])
    report["models"]["M3_by_firing_regime"] = _grouped_model(
        Xtr[mtr], Ytr, Xte[mte], Yte, reg_tr, reg_te, min_n=80, label="firing_regime")

    # ---------------- M4: per texture ---------------------------------
    log("=== M4 RGB conditioned on texture ===")
    report["models"]["M4_by_texture"] = _grouped_model(
        Xtr[mtr], Ytr, Xte[mte], Yte, surf_tr, surf_te, min_n=80, label="texture")

    # ---------------- M5: delta target within family ------------------
    log("=== M5 per colour family, predicting deviation from family mean ===")
    report["models"]["M5_delta_within_family"] = _delta_model(
        Xtr[mtr], Ytr, Xte[mte], Yte, fam_tr, fam_te)

    # ---------------- M6: texture classification conditioned ----------
    log("=== M6 texture classification conditioned on transparency ===")
    report["models"]["M6_texture_given_transparency"] = _cond_classifier(
        Xtr, ytr, Xte, yte, cond="transparency", target="surface")

    # ---------------- M7: flux-system local models --------------------
    log("=== M7 RGB by dominant flux system ===")
    flux_tr = np.array([_flux_system(Xtr[mtr].iloc[i]) for i in range(mtr.sum())])
    flux_te = np.array([_flux_system(Xte[mte].iloc[i]) for i in range(mte.sum())])
    report["models"]["M7_by_flux_system"] = _grouped_model(
        Xtr[mtr], Ytr, Xte[mte], Yte, flux_tr, flux_te, min_n=150, label="flux_system")

    report["total_secs"] = round(time.time() - t0, 1)
    (OUT / "metrics_decomposed.json").write_text(json.dumps(report, indent=2))
    log(f"done in {report['total_secs']}s -> models/metrics_decomposed.json")

    print("\n" + "=" * 74)
    print("DECOMPOSED MODELS — held-out test performance (RGB)")
    print("=" * 74)
    print(f"{'model':34} {'n':>6} {'R2':>8} {'MAE':>8} {'dist':>8}")
    for k, v in report["models"].items():
        if "r2" in v:
            print(f"{k:34} {v['n']:>6} {v['r2']:>8.4f} {v['mae']:>8.2f} {v['dist_mean']:>8.2f}")
        elif "accuracy" in v:
            print(f"{k:34} {v.get('n',0):>6} acc={v['accuracy']:.4f} f1={v.get('f1_macro',0):.4f}")


def _fit(Xtr, Ytr, Xte):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", cat_reg())])
    pipe.fit(Xtr, Ytr)
    return pipe.predict(Xte), pipe


def _flux_system(row):
    """Classify a recipe's dominant flux chemistry."""
    r2o = float(row.get("R2O_wt", 0) or 0)
    ro = float(row.get("RO_wt", 0) or 0)
    if r2o + ro < 1e-6:
        return "unknown"
    f = r2o / (r2o + ro)
    if f > 0.6:
        return "alkali-rich"
    if f < 0.25:
        return "alkaline-earth-rich"
    return "mixed"


def _grouped_model(Xtr, Ytr, Xte, Yte, gtr, gte, min_n=80, label=""):
    """Train one model per group; report in-group R2 and overall R2."""
    # normalise to plain strings so sorting/np.where never sees mixed types
    gtr = np.array([str(g) for g in gtr])
    gte = np.array([str(g) for g in gte])
    out = {"groups": {}, "label": label}
    preds = np.full_like(Yte, np.nan)
    all_pred, all_true = [], []
    for g in sorted(set(gtr)):
        itr = np.where(gtr == g)[0]
        ite = np.where(gte == g)[0]
        if len(itr) < min_n or len(ite) < 20:
            out["groups"][g] = {"n_train": int(len(itr)), "n_test": int(len(ite)),
                                "status": "skipped (n too small)"}
            continue
        try:
            P = _fit(Xtr.iloc[itr], Ytr[itr], Xte.iloc[ite])[0]
            preds[ite] = P
            m = eval_rgb(Yte[ite], P)
            m["status"] = "ok"
            out["groups"][g] = m
            all_pred.append(P); all_true.append(Yte[ite])
            log(f"    {str(g):26} n_tr={len(itr):5} n_te={len(ite):5} R2={m['r2']:.4f}")
        except Exception as e:
            out["groups"][g] = {"status": f"error: {type(e).__name__}"}
            log(f"    {str(g):26} ERRO {type(e).__name__}")

    if all_pred:
        A = np.vstack(all_true); B = np.vstack(all_pred)
        out.update(eval_rgb(A, B))
        out["coverage"] = float(len(A) / len(Yte))
    return out


def _delta_model(Xtr, Ytr, Xte, Yte, gtr, gte):
    """Predict deviation from the group mean instead of the absolute value.
    Removing the group offset reduces target variance for the learner."""
    gtr = np.array([str(g) for g in gtr])
    gte = np.array([str(g) for g in gte])
    out = {"groups": {}, "label": "delta_within_family"}
    all_pred, all_true = [], []
    for g in sorted(set(gtr)):
        itr = np.where(gtr == g)[0]
        ite = np.where(gte == g)[0]
        if len(itr) < 80 or len(ite) < 20:
            continue
        mu = Ytr[itr].mean(axis=0)
        Dtr = Ytr[itr] - mu
        try:
            P = _fit(Xtr.iloc[itr], Dtr, Xte.iloc[ite])[0]
            pred = P + mu
            m = eval_rgb(Yte[ite], pred)
            m["n_train"] = int(len(itr)); m["n_test"] = int(len(ite))
            out["groups"][g] = m
            all_pred.append(pred); all_true.append(Yte[ite])
            log(f"    {str(g):26} R2={m['r2']:.4f} (delta)")
        except Exception as e:
            log(f"    {str(g):26} ERRO {type(e).__name__}")
    if all_pred:
        A = np.vstack(all_true); B = np.vstack(all_pred)
        out.update(eval_rgb(A, B))
        out["coverage"] = float(len(A) / len(Yte))
    return out


def _cond_classifier(Xtr, ytr, Xte, yte, cond, target):
    """P(target | given cond) — train one classifier per conditioning value."""
    out = {"groups": {}, "label": f"{target} given {cond}"}
    mtr = ytr[[cond, target]].notna().all(axis=1)
    mte = yte[[cond, target]].notna().all(axis=1)
    c_tr = ytr.loc[mtr, cond].astype(str).values
    c_te = yte.loc[mte, cond].astype(str).values
    t_tr = ytr.loc[mtr, target].astype(str).values
    t_te = yte.loc[mte, target].astype(str).values
    Xa, Xb = Xtr[mtr.values], Xte[mte.values]

    preds = np.array([""] * len(t_te), dtype=object)
    for g in sorted(set(c_tr)):
        itr = np.where(c_tr == g)[0]; ite = np.where(c_te == g)[0]
        if len(itr) < 100 or len(ite) < 30:
            continue
        le = LabelEncoder().fit(t_tr[itr])
        if len(le.classes_) < 2:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", __import__("catboost").CatBoostClassifier(
                             iterations=600, depth=7, learning_rate=0.06,
                             random_seed=SEED, verbose=False, thread_count=4))])
        pipe.fit(Xa.iloc[itr], le.transform(t_tr[itr]))
        preds[ite] = le.inverse_transform(pipe.predict(Xa.iloc[ite]))
        acc = accuracy_score(t_te[ite], preds[ite])
        out["groups"][g] = {"n_train": int(len(itr)), "n_test": int(len(ite)),
                            "accuracy": float(acc)}
        log(f"    {str(g):22} n_tr={len(itr):5} acc={acc:.4f}")

    m = preds != ""
    if m.sum() > 0:
        yt, yp = t_te[m], preds[m]
        out["accuracy"] = float(accuracy_score(yt, yp))
        out["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        out["n"] = int(m.sum())
        out["coverage"] = float(m.sum() / len(t_te))
        from collections import Counter
        maj = Counter(t_tr).most_common(1)[0][0]
        out["global_baseline"] = float((yt == maj).mean())
    return out


if __name__ == "__main__":
    main()

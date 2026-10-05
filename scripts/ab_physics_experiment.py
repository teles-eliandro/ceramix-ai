"""
A/B experiment: does adding physics-derived features improve the model?

Compares, on identical splits and identical model hyperparameters:
  BASE  = current feature set (oxides + UMF + derived + firing + ingredients)
  PHYS  = BASE + physics-derived features (COE, viscosity, structure)

Reports R2 / MAE for RGB, and macro-F1 for surface and transparency, plus a
paired bootstrap over test samples to say whether any improvement is real or
noise. Honest reporting: if the delta is inside the noise band, say so.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import build_split, ingredient_vocab, MOLAR_MASS  # noqa: E402
from physics import build_physics_features  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import f1_score, mean_absolute_error, r2_score  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models"


def r2_per_channel(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {ch: float(r2_score(y_true[:, i], y_pred[:, i]))
            for i, ch in enumerate(("R", "G", "B"))}


def bootstrap_delta_r2(y_true, pred_base, pred_phys, n_boot=1000, seed=0):
    """Paired bootstrap of (R2_phys - R2_base) over test samples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        # R2 over the flattened 3-channel vector so it matches the headline metric
        ss_res_b = ((yt - pred_base[idx]) ** 2).sum()
        ss_res_p = ((yt - pred_phys[idx]) ** 2).sum()
        ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum()
        deltas[b] = (1 - ss_res_p / ss_tot) - (1 - ss_res_b / ss_tot)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), float((deltas > 0).mean())


def main():
    t0 = time.time()
    print("[ab] building splits ...")
    vocab = ingredient_vocab("train", top_n=40)
    Xtr, ytr = build_split("train", ing_vocab=vocab)
    Xte, yte = build_split("test", ing_vocab=vocab)
    print(f"[ab] Xtr={Xtr.shape} Xte={Xte.shape}")

    ox_cols = [c for c in Xtr.columns if c in MOLAR_MASS]
    ox_tr = Xtr[ox_cols]
    ox_te = Xte[ox_cols]

    print("[ab] building physics features ...")
    P_tr = build_physics_features(ox_tr)
    P_te = build_physics_features(ox_te)
    # align columns (defensive)
    P_te = P_te.reindex(columns=P_tr.columns, fill_value=0.0)
    print(f"[ab] physics features: {P_tr.shape[1]}")

    Xtr_b = Xtr.fillna(0.0)
    Xte_b = Xte.fillna(0.0)
    Xtr_p = pd.concat([Xtr_b, P_tr], axis=1).fillna(0.0)
    Xte_p = pd.concat([Xte_b, P_te], axis=1).fillna(0.0)
    Xte_p = Xte_p.reindex(columns=Xtr_p.columns, fill_value=0.0)

    y_rgb = yte[["R", "G", "B"]].to_numpy(float)
    y_tr_rgb = ytr[["R", "G", "B"]].to_numpy(float)

    results = {"n_features_base": int(Xtr_b.shape[1]),
               "n_features_phys": int(Xtr_p.shape[1]),
               "physics_added": int(P_tr.shape[1])}

    # ---------------------------------------------------------------- RGB ---
    print("[ab] training RGB models ...")
    rf_b = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                 n_jobs=-1, random_state=0)
    rf_b.fit(Xtr_b, y_tr_rgb)
    pred_b = rf_b.predict(Xte_b)

    rf_p = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                 n_jobs=-1, random_state=0)
    rf_p.fit(Xtr_p, y_tr_rgb)
    pred_p = rf_p.predict(Xte_p)

    r2_b = float(r2_score(y_rgb, pred_b, multioutput="variance_weighted"))
    r2_p = float(r2_score(y_rgb, pred_p, multioutput="variance_weighted"))
    mae_b = float(mean_absolute_error(y_rgb, pred_b))
    mae_p = float(mean_absolute_error(y_rgb, pred_p))

    mean_d, lo, hi, pwin = bootstrap_delta_r2(y_rgb, pred_b, pred_p)
    results["rgb"] = {
        "base": {"r2": r2_b, "mae": mae_b, "per_channel_r2": r2_per_channel(y_rgb, pred_b)},
        "phys": {"r2": r2_p, "mae": mae_p, "per_channel_r2": r2_per_channel(y_rgb, pred_p)},
        "delta_r2": r2_p - r2_b,
        "delta_mae": mae_p - mae_b,
        "bootstrap": {"mean_delta": mean_d, "ci95_low": lo, "ci95_high": hi,
                      "prob_improvement": pwin},
    }
    print(f"[ab] RGB  base R2={r2_b:.4f} MAE={mae_b:.3f} | "
          f"phys R2={r2_p:.4f} MAE={mae_p:.3f} | "
          f"delta_R2={r2_p-r2_b:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")

    # ------------------------------------------- surface / transparency ---
    # ~44% of records have no surface/transparency label. Those rows must be
    # dropped FOR THESE TASKS ONLY (they remain valid for the RGB task), and
    # the same mask applied to train and test so the comparison is fair.
    from sklearn.ensemble import RandomForestClassifier
    for task in ("surface", "transparency"):
        print(f"[ab] training {task} classifiers ...")
        tr_mask = ytr[task].notna().to_numpy()
        te_mask = yte[task].notna().to_numpy()
        yl_tr = ytr.loc[tr_mask, task].astype(str)
        yl_te = yte.loc[te_mask, task].astype(str)
        Xtr_bt, Xte_bt = Xtr_b.loc[tr_mask], Xte_b.loc[te_mask]
        Xtr_pt, Xte_pt = Xtr_p.loc[tr_mask], Xte_p.loc[te_mask]
        print(f"[ab]   labelled: train={tr_mask.sum()} test={te_mask.sum()} "
              f"classes={yl_tr.nunique()}")

        clf_b = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=0, class_weight="balanced")
        clf_b.fit(Xtr_bt, yl_tr)
        pb = clf_b.predict(Xte_bt)
        clf_p = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=0, class_weight="balanced")
        clf_p.fit(Xtr_pt, yl_tr)
        pp = clf_p.predict(Xte_pt)

        f1_b = float(f1_score(yl_te, pb, average="macro", zero_division=0))
        f1_p = float(f1_score(yl_te, pp, average="macro", zero_division=0))
        wf1_b = float(f1_score(yl_te, pb, average="weighted", zero_division=0))
        wf1_p = float(f1_score(yl_te, pp, average="weighted", zero_division=0))
        acc_b = float((pb == yl_te).mean())
        acc_p = float((pp == yl_te).mean())
        # majority-class baseline for context
        maj = float(yl_te.value_counts(normalize=True).max())
        results[task] = {
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_classes": int(yl_tr.nunique()),
            "majority_baseline_acc": maj,
            "base": {"f1_macro": f1_b, "f1_weighted": wf1_b, "acc": acc_b},
            "phys": {"f1_macro": f1_p, "f1_weighted": wf1_p, "acc": acc_p},
            "delta_f1": f1_p - f1_b,
            "delta_f1_weighted": wf1_p - wf1_b,
            "delta_acc": acc_p - acc_b,
        }
        print(f"[ab] {task}: base F1={f1_b:.4f} acc={acc_b:.4f} | "
              f"phys F1={f1_p:.4f} acc={acc_p:.4f} | delta_F1={f1_p-f1_b:+.4f} "
              f"(majority acc={maj:.3f})")

    # ------------------------------------------ feature importance delta ---
    imp_base = pd.Series(rf_b.feature_importances_, index=Xtr_b.columns)
    imp_phys = pd.Series(rf_p.feature_importances_, index=Xtr_p.columns)
    phys_new = [c for c in P_tr.columns]
    top_phys = imp_phys[phys_new].sort_values(ascending=False).head(15)
    results["top_physics_importances"] = {k: float(v) for k, v in top_phys.items()}
    results["physics_importance_share"] = float(imp_phys[phys_new].sum())
    # which base features lost importance (i.e. physics captured them)
    lost = (imp_base - imp_phys.reindex(imp_base.index).fillna(0.0)).sort_values(ascending=False)
    results["base_features_displaced"] = {k: float(v) for k, v in lost.head(10).items()}

    results["runtime_secs"] = time.time() - t0
    (OUT / "ab_physics.json").write_text(json.dumps(results, indent=1))
    print(f"[ab] wrote {OUT/'ab_physics.json'}  ({results['runtime_secs']:.0f}s)")

    print("\n=== top physics features by importance ===")
    for k, v in top_phys.items():
        print(f"   {k:34s} {v:.5f}")
    print(f"\n   physics share of total importance: {results['physics_importance_share']:.4f}")


if __name__ == "__main__":
    main()

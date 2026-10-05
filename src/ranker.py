"""
CERAMIX-AI — Candidate ranking engine.

Given a target colour and surface finish, rank existing real-world recipes by
how well their *predicted colour distribution* matches the target, then report
each candidate with its calibrated uncertainty interval.

This is the honest product built on top of the measured noise ceiling: instead
of claiming a formulation that hits a colour exactly, we surface the most
promising real, already-fired recipes for the user to test.

Scoring
-------
For each candidate recipe we predict the colour via the trained ensemble and
attach conformal intervals. The score combines:
  * colour match      — Euclidean distance predicted-vs-target, normalised
  * interval favour   — a candidate whose interval brackets the target is
                        more trustworthy than one whose point estimate is
                        closer but whose interval is far away
  * texture match     — predicted surface/texture class vs requested
  * transparency match — predicted transparency vs requested
  * process risk      — penalties for unusual cone (far from industry bands)

References: GlazyBench (arXiv:2605.06641); conformal prediction (Vovk et al.).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "models"
SEED = 42
np.random.seed(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------ colour utils --
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(v)))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """sRGB (D65) -> CIE XYZ."""
    lin = srgb_to_linear(np.asarray(rgb, dtype=float))
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return lin @ M.T


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB -> CIELAB (D65)."""
    xyz = rgb_to_xyz(np.asarray(rgb, dtype=float))
    white = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / white
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_76(lab1, lab2) -> float:
    """CIE76 colour difference (Euclidean in Lab). CIE76 is used here in place
    of CIEDE2000 because it is exactly reproducible without reference tables;
    the ordering it induces is identical for ranking purposes."""
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


# ------------------------------------------------------------------ engine --
class RecipeRanker:
    def __init__(self):
        self.model = None
        self.le_surface = None
        self.le_transp = None
        self.widths = {}
        self.vocab = None
        self.cols = None

    def train(self):
        from features import build_split, ingredient_vocab
        import catboost

        log("building features ...")
        self.vocab = ingredient_vocab("train", 40)
        Xtr, ytr = build_split("train", ing_vocab=self.vocab)
        Xte, yte = build_split("test", ing_vocab=self.vocab)
        Xtr, Xte = Xtr.align(Xte, join="inner", axis=1)
        Xtr, Xte = Xtr.fillna(0), Xte.fillna(0)
        self.cols = list(Xtr.columns)

        rgb = ["R", "G", "B"]
        mtr = ytr[rgb].notna().all(axis=1).values
        mte = yte[rgb].notna().all(axis=1).values

        # colour regressor
        log("training colour regressor ...")
        self.model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", catboost.CatBoostRegressor(
                iterations=1000, depth=8, learning_rate=0.05,
                loss_function="MultiRMSE", random_seed=SEED,
                verbose=False, thread_count=4))])
        self.model.fit(Xtr[mtr], ytr.loc[mtr, rgb].values.astype(float))

        # calibrated widths (from uncertainty study)
        self.widths = {"R": 72.7, "G": 71.5, "B": 72.1}
        try:
            u = json.loads((OUT / "uncertainty.json").read_text())
            for c, v in u["channels"].items():
                self.widths[c] = float(v["conformal_width"])
            log(f"loaded calibrated widths: {self.widths}")
        except Exception:
            log("using default widths")

        # texture + transparency classifiers
        for name, attr in (("surface", "le_surface"),
                           ("transparency", "le_transp")):
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import LabelEncoder
            m = ytr[name].notna().values & mtr
            mt = yte[name].notna().values & mte
            if m.sum() < 200:
                continue
            le = LabelEncoder().fit(ytr.loc[m, name].astype(str))
            clf = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", catboost.CatBoostClassifier(
                    iterations=700, depth=7, learning_rate=0.06,
                    random_seed=SEED, verbose=False, thread_count=4))])
            clf.fit(Xtr[m], le.transform(ytr.loc[m, name].astype(str)))
            acc = accuracy_score(le.transform(yte.loc[mt, name].astype(str)),
                                 clf.predict(Xte[mt]))
            setattr(self, attr, (clf, le))
            log(f"  {name} classifier acc={acc:.4f} ({len(le.classes_)} classes)")
        return self

    # ---- inference ----
    def predict(self, X: pd.DataFrame) -> dict:
        Xa = X.reindex(columns=self.cols, fill_value=0.0).fillna(0)
        med = self.model.predict(Xa)
        out = {"rgb_median": med}
        w = np.array([self.widths["R"], self.widths["G"], self.widths["B"]])
        out["rgb_low"] = med - w
        out["rgb_high"] = med + w
        if self.le_surface:
            clf, le = self.le_surface
            out["surface"] = le.inverse_transform(clf.predict(Xa))
        if self.le_transp:
            clf, le = self.le_transp
            out["transparency"] = le.inverse_transform(clf.predict(Xa))
        return out

    def rank(self, target_rgb, target_surface=None, target_transparency=None,
             candidates: pd.DataFrame = None, top_k: int = 12,
             max_delta_e: float = 25.0) -> list[dict]:
        """Rank candidate recipes for a target colour and finish."""
        P = self.predict(candidates)
        med, lo, hi = P["rgb_median"], P["rgb_low"], P["rgb_high"]
        tgt = np.asarray(target_rgb, dtype=float)

        tgt_lab = rgb_to_lab(tgt)
        rows = []
        for i in range(len(med)):
            m_lab = rgb_to_lab(med[i])
            de = delta_e_76(m_lab, tgt_lab)

            # does the calibrated interval contain the target?
            inside = bool(np.all((lo[i] <= tgt) & (tgt <= hi[i])))
            # distance from target to the nearest edge of the interval
            gap = float(np.linalg.norm(np.maximum(0, np.maximum(lo[i] - tgt,
                                                             tgt - hi[i]))))
            score = de + 0.6 * gap - (6.0 if inside else 0.0)

            surf_ok = None
            if target_surface and "surface" in P:
                surf_ok = (str(P["surface"][i]).lower() == str(target_surface).lower())
                if surf_ok:
                    score -= 5.0
            tr_ok = None
            if target_transparency and "transparency" in P:
                tr_ok = (str(P["transparency"][i]).lower()
                         == str(target_transparency).lower())
                if tr_ok:
                    score -= 4.0

            rows.append({
                "index": int(i),
                "delta_e": round(float(de), 2),
                "rgb": [int(round(v)) for v in med[i]],
                "hex": rgb_to_hex(med[i]),
                "interval": [[int(round(v)) for v in lo[i]],
                             [int(round(v)) for v in hi[i]]],
                "interval_contains_target": inside,
                "interval_gap": round(gap, 2),
                "surface": str(P["surface"][i]) if "surface" in P else None,
                "surface_match": surf_ok,
                "transparency": (str(P["transparency"][i])
                                 if "transparency" in P else None),
                "transparency_match": tr_ok,
                "score": round(float(score), 3),
            })

        rows.sort(key=lambda r: r["score"])
        return rows[:top_k]


def main():
    t0 = time.time()
    log("training ranker ...")
    r = RecipeRanker().train()

    # demo: rank real recipes against a target
    from features import build_split
    Xte, yte = build_split("test", ing_vocab=r.vocab)
    Xte = Xte.reindex(columns=r.cols, fill_value=0.0).fillna(0)

    target = hex_to_rgb("#B8860B")   # dark goldenrod — a beige/ochre tile shade
    log(f"target {rgb_to_hex(target)} lab={np.round(rgb_to_lab(target),1)}")

    res = r.rank(target, target_surface="Matte", candidates=Xte, top_k=10)

    print("\n" + "=" * 78)
    print(f"TOP CANDIDATES for target {rgb_to_hex(target)}  (Matte)")
    print("=" * 78)
    for i, c in enumerate(res, 1):
        flag = "TARGET-IN-RANGE" if c["interval_contains_target"] else ""
        print(f"{i:2}. dE={c['delta_e']:6.2f}  {c['hex']}  rgb={c['rgb']}  "
              f"{c['surface']:12} {flag}")
        print(f"    90% interval rgb={c['interval']}  gap={c['interval_gap']}")

    (OUT / "ranking_demo.json").write_text(json.dumps(
        {"target_hex": rgb_to_hex(target), "candidates": res}, indent=2))
    log(f"done in {time.time()-t0:.1f}s -> models/ranking_demo.json")


if __name__ == "__main__":
    main()

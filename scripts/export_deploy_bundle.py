#!/usr/bin/env python3
"""Bake the trained ranker into a compact, deployable artifact.

The ranker normally retrains from the 48 MB GlazyBench dataset at startup,
which is much too slow/heavy for a free-tier web dyno. This script trains once
locally and persists everything inference needs:

  * the fitted CatBoost colour regressor pipeline
  * the surface / transparency classifier pipelines + label encoders
  * the calibrated interval widths
  * the candidate recipe pool (features already built) + display fields

Output: models/deploy_bundle.pkl  (single file, self-contained)

Run:  .venv/bin/python scripts/export_deploy_bundle.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "models"

# Display fields we want available to the UI alongside the feature matrix.
DISPLAY_KEYS = ["name", "surface", "transparency", "cone", "atmosphere",
                "R", "G", "B", "hex", "url"]


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    import pandas as pd
    from features import build_split
    from ranker import RecipeRanker

    log("training ranker (one-off) ...")
    r = RecipeRanker().train()

    log("rebuilding candidate pool ...")
    vocab = r.vocab
    Xte, yte = build_split("test", ing_vocab=vocab)
    Xtr, _ = build_split("train", ing_vocab=vocab)
    Xte = Xte.reindex(columns=r.cols, fill_value=0.0).fillna(0.0)

    # display columns that exist
    disp = {}
    for k in DISPLAY_KEYS:
        if k in yte.columns:
            disp[k] = yte[k].reset_index(drop=True)

    bundle = {
        "format": 1,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cols": r.cols,
        "widths": r.widths,
        "model": r.model,
        "le_surface": r.le_surface,
        "le_transp": r.le_transp,
        "candidate_X": Xte.reset_index(drop=True),
        "candidate_display": disp,
        "n_candidates": int(len(Xte)),
        "n_train_features": int(Xtr.shape[1]),
    }

    out = OUT / "deploy_bundle.pkl"
    with out.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=4)

    size_mb = out.stat().st_size / 1e6
    log(f"wrote {out} ({size_mb:.1f} MB) "
        f"candidates={len(Xte)} cols={len(r.cols)}")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

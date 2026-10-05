#!/usr/bin/env python3
"""Bake the trained ranker + full recipe records into a deployable artifact.

The ranker normally retrains from the 48 MB GlazyBench dataset at startup,
which is much too slow/heavy for a free-tier web dyno. This script trains once
locally and persists everything inference and reporting need:

  * the fitted CatBoost colour regressor pipeline
  * the surface / transparency classifier pipelines + label encoders
  * the calibrated interval widths
  * the candidate feature matrix
  * the FULL recipe record per candidate: oxide composition, ingredients with
    amounts, cone/atmosphere (firing), UMF, and target colour

Output: models/deploy_bundle.pkl  (single file, self-contained)

Run:  .venv/bin/python scripts/export_deploy_bundle.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "models"
DATA = ROOT / "data" / "glazybench"

# Fields we copy verbatim from the raw GlazyBench record onto each candidate.
RECIPE_FIELDS = [
    "id", "chemical_composition", "ingredients", "ingredient_count",
    "has_chemical_data", "cone", "cone_min", "cone_max", "atmosphere",
    "temperature", "temperature_value", "temperature_unit", "umf",
]

# Oxide symbols we surface in the report (order preserved for display).
# Anything else present in a recipe is appended after these.
OXIDE_ORDER = [
    "SiO2", "Al2O3", "B2O3", "Na2O", "K2O", "Li2O", "CaO", "MgO", "BaO",
    "SrO", "ZnO", "PbO", "P2O5", "TiO2", "ZrO2", "SnO2", "MnO", "MnO2",
    "CoO", "NiO", "CuO", "Cu2O", "Cr2O3", "V2O5", "CeO2", "Fe2O3", "FeO",
    "LOI", "F", "Sb2O3", "CdO", "SeO2", "Ag2O", "Au2O3", "Pr2O3", "Nd2O3",
]


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def load_records(split: str) -> list[dict]:
    """Load the raw GlazyBench recipes for a split, in order."""
    p = DATA / f"property_prediction_{split}_recipes.json"
    with p.open() as fh:
        return json.load(fh)


def clean_oxides(comp: dict | None) -> list[dict]:
    """Turn {'SiO2': 56.49, ...} into an ordered, non-zero list of oxides."""
    if not isinstance(comp, dict):
        return []
    vals = {}
    for k, v in comp.items():
        if k in ("Amt.", "Total", "total"):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            vals[k] = round(f, 3)
    ordered = [k for k in OXIDE_ORDER if k in vals]
    ordered += [k for k in vals if k not in OXIDE_ORDER]
    return [{"oxide": k, "wt_pct": vals[k]} for k in ordered]


def clean_ingredients(items) -> list[dict]:
    """Keep material/amount pairs, dropping Glazy's 'total' bookkeeping rows."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("material") or "").strip()
        if not name or name.lower().startswith(("total", "total base")):
            continue
        try:
            amt = round(float(it.get("amount")), 2)
        except (TypeError, ValueError):
            continue
        out.append({"material": name, "amount": amt})
    return out


def clean_umf(umf) -> list[dict]:
    """The unity molecular formula, ordered as a flux/amphoteric/glass former."""
    if not isinstance(umf, dict):
        return []
    vals = {}
    for k, v in umf.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            vals[k] = round(f, 4)
    ordered = [k for k in OXIDE_ORDER if k in vals]
    ordered += [k for k in vals if k not in OXIDE_ORDER]
    return [{"oxide": k, "value": vals[k]} for k in ordered]


def main() -> None:
    t0 = time.time()
    from features import build_split
    from ranker import RecipeRanker

    log("training ranker (one-off) ...")
    r = RecipeRanker().train()

    log("rebuilding candidate pool ...")
    vocab = r.vocab
    Xte, yte = build_split("test", ing_vocab=vocab)
    Xtr, _ = build_split("train", ing_vocab=vocab)
    Xte = Xte.reindex(columns=r.cols, fill_value=0.0).fillna(0.0)

    log("loading raw recipe records ...")
    records = load_records("test")
    if len(records) != len(Xte):
        log(f"WARNING: {len(records)} records vs {len(Xte)} feature rows")

    # Build one enriched record per candidate.
    recipes = []
    for i, rec in enumerate(records):
        if i >= len(Xte):
            break
        recipes.append({
            "id": str(rec.get("id", "")),
            "url": f"https://glazy.org/recipe/{rec.get('id')}" if rec.get("id") else None,
            "oxides": clean_oxides(rec.get("chemical_composition")),
            "ingredients": clean_ingredients(rec.get("ingredients")),
            "ingredient_count": rec.get("ingredient_count"),
            "umf": clean_umf(rec.get("umf")),
            "cone": rec.get("cone"),
            "cone_min": rec.get("cone_min"),
            "cone_max": rec.get("cone_max"),
            "atmosphere": rec.get("atmosphere"),
            "temperature": rec.get("temperature"),
            "temperature_value": rec.get("temperature_value"),
            "temperature_unit": rec.get("temperature_unit"),
            "has_chemical_data": rec.get("has_chemical_data"),
        })

    # Target colour (ground truth) per candidate, for the "measured vs predicted".
    disp = {}
    for k in ("surface", "transparency", "R", "G", "B"):
        if k in yte.columns:
            disp[k] = yte[k].reset_index(drop=True)

    bundle = {
        "format": 2,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cols": r.cols,
        "widths": r.widths,
        "model": r.model,
        "le_surface": r.le_surface,
        "le_transp": r.le_transp,
        "candidate_X": Xte.reset_index(drop=True),
        "candidate_display": disp,
        "candidate_recipes": recipes,
        "n_candidates": int(len(Xte)),
        "n_train_features": int(Xtr.shape[1]),
    }

    out = OUT / "deploy_bundle.pkl"
    with out.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=4)

    size_mb = out.stat().st_size / 1e6
    n_ing = sum(1 for x in recipes if x["ingredients"])
    n_ox = sum(1 for x in recipes if x["oxides"])
    n_umf = sum(1 for x in recipes if x["umf"])
    log(f"wrote {out} ({size_mb:.1f} MB)")
    log(f"candidates={len(Xte)} with_ingredients={n_ing} "
        f"with_oxides={n_ox} with_umf={n_umf}")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

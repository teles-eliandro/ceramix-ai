"""
CERAMIX-AI — Feature engineering from GlazyBench.

Builds the design matrix from raw glaze recipes:
  - 45 oxide wt% (chemical composition)
  - UMF / Seger unity mole ratios (fluxes normalised to 1.00)
  - Derived ceramic descriptors (silica/alumina ratio, flux balance,
    R2O:RO ratio, opacity index, colouring oxide totals)
  - Firing descriptors (cone -> temperature, atmosphere one-hot)

References
----------
GlazyBench: arXiv:2605.06641 (MIT licence).
Seger UMF formalism: standard ceramic glaze calculation (see
  Hansen, "Glaze calculation", and the classic Seger unity formula).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "glazybench"

# ---------------------------------------------------------------- oxides ---
# Molar masses (g/mol) for UMF conversion. LOI/Amt. are non-oxides.
MOLAR_MASS = {
    "SiO2": 60.084, "Al2O3": 101.961, "B2O3": 69.620, "Na2O": 61.979,
    "K2O": 94.196, "Li2O": 29.881, "CaO": 56.077, "MgO": 40.304,
    "BaO": 153.326, "SrO": 103.619, "ZnO": 81.379, "PbO": 223.199,
    "Fe2O3": 159.688, "FeO": 71.844, "TiO2": 79.866, "ZrO2": 123.218,
    "ZrO": 107.219, "SnO2": 150.708, "MnO": 70.937, "MnO2": 86.937,
    "CoO": 74.933, "NiO": 74.693, "CuO": 79.545, "Cu2O": 143.091,
    "Cr2O3": 151.990, "V2O5": 181.880, "P2O5": 141.944, "CeO2": 172.115,
    "Pr2O3": 329.813, "PrO2": 172.907, "Nd2O3": 336.478, "La2O3": 325.809,
    "Bi2O3": 465.959, "Sb2O3": 291.518, "As2O3": 197.841, "CdO": 128.410,
    "Ag2O": 231.735, "MoO3": 143.938, "WO3": 231.838, "Nb2O5": 265.810,
    "HfO2": 210.489, "Er2O3": 382.516, "U3O8": 842.082, "F": 18.998,
}

# Ceramic-chemistry classification
FLUX_R2O = ["Na2O", "K2O", "Li2O"]                       # alkalis
FLUX_RO = ["CaO", "MgO", "BaO", "SrO", "ZnO", "PbO", "FeO", "MnO", "CuO", "CoO", "NiO"]
GLASS_FORMERS = ["SiO2", "B2O3", "P2O5"]
STABILISERS = ["Al2O3"]
# Transition-metal chromophores -> colour
CHROMOPHORES = ["Fe2O3", "FeO", "CoO", "CuO", "Cu2O", "Cr2O3", "MnO", "MnO2",
                "NiO", "V2O5", "TiO2", "Pr2O3", "PrO2", "Nd2O3", "Er2O3",
                "CdO", "U3O8", "Ag2O"]
OPACIFIERS = ["ZrO2", "ZrO", "SnO2", "TiO2", "CeO2", "Sb2O3", "As2O3", "HfO2"]

# Orton cone -> approximate end-point temperature in °C
CONE_TEMP = {
    4: 1160, 5: 1196, 6: 1222, 7: 1240, 8: 1263, 9: 1280, 10: 1305,
    11: 1315, 12: 1326, 3: 1101, 2: 1080, 1: 1101, 0: 1080,
}


def _num(x) -> float:
    """Coerce anything to float, defaulting to 0.0."""
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def parse_cone(cone) -> tuple[float, float]:
    """'6-8' -> (6,8); '10' -> (10,10); None -> (nan,nan)."""
    if cone is None:
        return np.nan, np.nan
    if isinstance(cone, (int, float)):
        return float(cone), float(cone)
    nums = re.findall(r"\d+(?:\.\d+)?", str(cone))
    if not nums:
        return np.nan, np.nan
    vals = [float(n) for n in nums[:2]]
    return (vals[0], vals[-1]) if len(vals) > 1 else (vals[0], vals[0])


def oxide_frame(records: list[dict]) -> pd.DataFrame:
    """Extract the 45-oxide matrix (wt %)."""
    rows = []
    for r in records:
        comp = r.get("chemical_composition") or {}
        rows.append({ox: _num(comp.get(ox)) for ox in MOLAR_MASS})
    return pd.DataFrame(rows, columns=list(MOLAR_MASS))


def umf_features(ox: pd.DataFrame) -> pd.DataFrame:
    """Seger Unity Formula: normalise fluxes to 1.00 mole, express the rest
    as molar ratios — the industry-standard way to compare glazes."""
    out = {}
    moles = pd.DataFrame(index=ox.index)
    for c in ox.columns:
        mm = MOLAR_MASS[c]
        moles[c] = ox[c] / mm if mm else 0.0

    flux_cols = [c for c in FLUX_R2O + FLUX_RO if c in moles.columns]
    flux_total = moles[flux_cols].sum(axis=1).replace(0, np.nan)

    for c in moles.columns:
        out[f"umf_{c}"] = (moles[c] / flux_total).fillna(0.0)

    # canonical UMF descriptors
    out["umf_flux_total_mol"] = flux_total.fillna(0.0)
    out["umf_R2O"] = moles[FLUX_R2O].sum(axis=1) / flux_total
    out["umf_RO"] = moles[FLUX_RO].sum(axis=1) / flux_total
    out["umf_Al2O3"] = moles["Al2O3"] / flux_total
    out["umf_SiO2"] = moles["SiO2"] / flux_total
    return pd.DataFrame(out).fillna(0.0)


def derived_features(ox: pd.DataFrame) -> pd.DataFrame:
    """Ceramic-engineering descriptors with direct physical meaning."""
    d = {}
    sio2, al2o3 = ox["SiO2"], ox["Al2O3"].replace(0, np.nan)

    # Silica:alumina ratio — controls matteness / thermal expansion
    d["silica_alumina_ratio"] = (sio2 / al2o3).fillna(0.0)
    # Total flux content — viscosity at firing temperature
    fluxes = ox[[c for c in FLUX_R2O + FLUX_RO if c in ox.columns]].sum(axis=1)
    d["flux_total_wt"] = fluxes
    d["glass_former_wt"] = ox[[c for c in GLASS_FORMERS if c in ox.columns]].sum(axis=1)
    d["stabiliser_wt"] = ox["Al2O3"]

    # R2O:RO balance — alkali-rich = glossy/low viscosity, RO-rich = matte
    r2o = ox[[c for c in FLUX_R2O if c in ox.columns]].sum(axis=1)
    ro = ox[[c for c in FLUX_RO if c in ox.columns]].sum(axis=1)
    d["R2O_wt"] = r2o
    d["RO_wt"] = ro
    d["R2O_RO_ratio"] = (r2o / ro.replace(0, np.nan)).fillna(0.0)

    # Chromophore / opacifier totals
    d["chromophore_total"] = ox[[c for c in CHROMOPHORES if c in ox.columns]].sum(axis=1)
    d["opacifier_total"] = ox[[c for c in OPACIFIERS if c in ox.columns]].sum(axis=1)
    d["has_chromophore"] = (d["chromophore_total"] > 0.01).astype(int)

    # Iron redox proxy — oxidation state strongly modulates a*/b*
    fe3, fe2 = ox["Fe2O3"], ox["FeO"]
    d["iron_total"] = fe3 + fe2
    d["Fe_reduced_fraction"] = (fe2 / (fe3 + fe2).replace(0, np.nan)).fillna(0.0)

    # Seger-style stability indicators
    d["alumina_to_flux"] = (al2o3.fillna(0) / fluxes.replace(0, np.nan)).fillna(0.0)
    d["silica_to_flux"] = (sio2 / fluxes.replace(0, np.nan)).fillna(0.0)
    # LOI indicates carbonates/hydrates -> gas evolution during firing
    d["LOI"] = ox.get("LOI", pd.Series(0.0, index=ox.index))
    d["chem_sum"] = ox.sum(axis=1)
    return pd.DataFrame(d).fillna(0.0)


def firing_features(rec: pd.DataFrame) -> pd.DataFrame:
    """Firing schedule descriptors."""
    f = {}
    cones = rec["cone"].apply(parse_cone)
    f["cone_min"] = [c[0] for c in cones]
    f["cone_max"] = [c[1] for c in cones]
    f["cone_mean"] = np.nanmean(np.vstack([f["cone_min"], f["cone_max"]]),
                                axis=0, where=~np.isnan(np.vstack([f["cone_min"], f["cone_max"]])))

    # map mean cone to approximate peak temperature
    f["peak_temp_C"] = [_cone_to_temp(c) for c in f["cone_mean"]]

    atmo = rec["atmosphere"].fillna("Unknown").str.strip()
    f["atmo_oxidation"] = (atmo.str.lower() == "oxidation").astype(int)
    f["atmo_reduction"] = (atmo.str.lower() == "reduction").astype(int)
    f["atmo_neutral"] = (atmo.str.lower() == "neutral").astype(int)
    f["atmo_known"] = (f["atmo_oxidation"] + f["atmo_reduction"] + f["atmo_neutral"]).clip(0, 1)
    return pd.DataFrame(f)


def _cone_to_temp(c: float) -> float:
    if not np.isfinite(c):
        return np.nan
    keys = sorted(CONE_TEMP)
    return float(CONE_TEMP[min(keys, key=lambda k: abs(k - c))])


def ingredient_features(rec: pd.DataFrame, top_n: int = 40,
                        vocab: list[str] | None = None) -> pd.DataFrame:
    """Presence/amount of the most common raw materials.

    `vocab` must be derived from the TRAIN split and reused for test,
    otherwise the column sets diverge and the model cannot score.
    """
    from collections import Counter
    if vocab is None:
        cnt = Counter()
        for ing in rec["ingredients"]:
            if isinstance(ing, list):
                for i in ing:
                    m = (i or {}).get("material")
                    if m:
                        cnt[str(m).strip().lower()] += 1
        vocab = [m for m, _ in cnt.most_common(top_n)]

    data = {}
    for m in vocab:
        col = []
        for ing in rec["ingredients"]:
            amt = 0.0
            if isinstance(ing, list):
                for i in ing:
                    if str((i or {}).get("material", "")).strip().lower() == m:
                        amt = _num((i or {}).get("amount"))
                        break
            col.append(amt)
        data[f"ing_{m.replace(' ', '_')}"] = col
    return pd.DataFrame(data).fillna(0.0)


def build_split(split: str, ing_vocab: list[str] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y) for 'train' or 'test'.

    Pass the vocabulary returned by the train call into the test call so
    both splits produce identical feature columns.
    """
    rec = pd.read_json(DATA / f"property_prediction_{split}_recipes.json")
    tgt = pd.read_json(DATA / f"property_prediction_{split}_targets.json")
    rec = rec.reset_index(drop=True)
    tgt = tgt.set_index("id")
    rec = rec[rec["id"].isin(tgt.index)].reset_index(drop=True)
    tgt = tgt.loc[rec["id"]].reset_index(drop=True)

    ox = oxide_frame(rec.to_dict("records"))
    X = pd.concat(
        [ox, umf_features(ox), derived_features(ox),
         firing_features(rec), ingredient_features(rec, vocab=ing_vocab)],
        axis=1,
    )
    X = X.loc[:, ~X.columns.duplicated()]

    rgb = tgt["color_rgb"].apply(
        lambda c: pd.Series([_num((c or {}).get("r")),
                             _num((c or {}).get("g")),
                             _num((c or {}).get("b"))])
    )
    rgb.columns = ["R", "G", "B"]
    y = pd.concat(
        [rgb,
         tgt[["color_family", "surface", "transparency"]].reset_index(drop=True)],
        axis=1,
    )
    y["id"] = rec["id"].values
    return X, y


def ingredient_vocab(split: str = "train", top_n: int = 40) -> list[str]:
    """The raw-material vocabulary, derived from the train split only."""
    from collections import Counter
    rec = pd.read_json(DATA / f"property_prediction_{split}_recipes.json")
    cnt = Counter()
    for ing in rec["ingredients"]:
        if isinstance(ing, list):
            for i in ing:
                m = (i or {}).get("material")
                if m:
                    cnt[str(m).strip().lower()] += 1
    return [m for m, _ in cnt.most_common(top_n)]


if __name__ == "__main__":
    for s in ("train", "test"):
        X, y = build_split(s)
        print(f"[{s}] X={X.shape}  y={y.shape}")
        print(f"   features: {list(X.columns)[:8]} ... (+{len(X.columns)-8})")

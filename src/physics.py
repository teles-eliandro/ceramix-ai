"""
CERAMIX-AI — Physics-derived glass/glaze properties.

The premise: colour and surface are not functions of raw recipe but of the
*glass structure* the recipe produces on firing. Composition -> structure ->
appearance. The features here encode the structure side of that chain using
classical glass-science models that are decades old and well validated in the
ceramics literature.

Implemented
-----------
* Coefficient of thermal expansion (COE) — additive oxide models:
    Sankey, Appen, English & Turner, Hall, Winkelmann & Schott
  plus Appen's concentration-dependent variant terms and the empirical
  dilatometer correction.
* Viscosity / melting: Vogel-Fulcher-Tammann (VFT) readiness proxies,
  fusion-point temperature approximation.
* Structure: Si:Al ratio, R2O:RO, boron environment (X ratio), flux ratio,
  network connectivity proxies.
* Optical: chromophore molar concentration with per-oxide molar extinction
  weighting; opacifier volume fraction.

References
----------
COE coefficients and the correction factor are tabulated in
  "Glaze Thermal Expansion", Pottery: Experiments with Glazes
  (web.ncf.ca/bf250/glazeexpansion.html), which compiles:
    English & Turner, J. Am. Ceram. Soc. 10(8):551 (1927); 12:760 (1929).
    Hall, J. Am. Ceram. Soc. 13(3):182-190 (1930).
    Appen, cited in Vargin, "Technology of Enamels".
    Mayer & Havas, Sprechsaal 42:497; 44:188-207.
    Winkelmann & Schott, Ann. Phys. 51:730 (1894).
    Sankey, Glass Ind. (1921).
VFT viscosity model: Vogel (1921), Fulcher (1925), Tammann (1925).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- COE ---
# Linear thermal expansion coefficients (x 10^-6 / K) per oxide, per author.
# Each entry: oxide -> coefficient. Ranges of measurement differ per author,
# which is why we emit several models instead of trusting one.

COE_SANKEY = {
    "Li2O": 27.0, "Na2O": 39.5, "K2O": 46.5, "MgO": 3.4, "CaO": 13.0,
    "ZnO": 5.0, "SrO": 16.0, "BaO": 20.0, "SiO2": 3.8, "Al2O3": 4.2,
    "ZrO2": -6.0, "SnO2": -4.5, "P2O5": 14.0, "Fe2O3": 11.0,
}
COE_APPEN = {
    "Li2O": 27.0, "Na2O": 39.5, "K2O": 46.5, "MgO": 6.0, "CaO": 13.0,
    "ZnO": 5.0, "SrO": 16.0, "BaO": 20.0, "Al2O3": -3.0, "ZrO2": -2.9,
    "SnO2": -1.8, "P2O5": 14.0, "Fe2O3": 11.0,
}
COE_ENGLISH_TURNER = {
    "Li2O": 6.7, "Na2O": 38.0, "K2O": 30.0, "MgO": 8.9, "CaO": 14.0,
    "ZnO": 3.7, "SrO": 9.3, "BaO": 7.8, "PbO": 3.5, "SiO2": 3.8,
    "B2O3": -2.5, "Al2O3": -1.8, "Fe2O3": 4.1, "TiO2": 1.1, "P2O5": 5.9,
}
COE_HALL = {
    "Li2O": 54.0, "Na2O": 51.0, "K2O": 42.0, "MgO": 0.0, "CaO": 15.0,
    "ZnO": 7.0, "BaO": 9.1, "PbO": 7.5, "SiO2": 4.0, "B2O3": -6.5,
    "Al2O3": 1.4, "ZrO2": 2.3, "SnO2": 6.6, "Fe2O3": 13.3, "TiO2": 13.6,
    "P2O5": 6.7,
}
COE_WINKELMANN = {
    "Li2O": 33.0, "Na2O": 33.0, "K2O": 28.0, "MgO": 0.3, "CaO": 17.0,
    "ZnO": 6.0, "BaO": 10.0, "PbO": 10.0, "SiO2": 2.7, "B2O3": 0.3,
    "Al2O3": 17.0, "ZrO2": 2.3,
}
COE_Mayer_Havas = {
    "Li2O": 29.88, "Na2O": 38.0, "K2O": 30.0, "MgO": 2.0, "CaO": 11.0,
    "ZnO": 7.7, "BaO": 12.0, "PbO": 14.0, "SiO2": 0.5, "B2O3": 2.0,
    "Al2O3": 5.0, "ZrO2": 7.0,
}

COE_MODELS = {
    "sankey": COE_SANKEY,
    "appen": COE_APPEN,
    "english_turner": COE_ENGLISH_TURNER,
    "hall": COE_HALL,
    "winkelmann": COE_WINKELMANN,
    "mayer_havas": COE_Mayer_Havas,
}

# Appen's SiO2 term is concentration dependent
def _appen_sio2_coe(sio2_mol_fraction: float) -> float:
    """Appen: 3.8 for 0-0.67 molar SiO2, then linearly decreasing."""
    if not np.isfinite(sio2_mol_fraction):
        return 3.8
    if sio2_mol_fraction <= 0.67:
        return 3.8
    return max(0.0, 3.8 - 10.0 * (sio2_mol_fraction - 0.67))


def _appen_b2o3_coe(x_ratio: float) -> float:
    """Appen boron anomaly term: (1.25*(4-X)) - 5, X clamped to <= 4."""
    if not np.isfinite(x_ratio):
        return 0.0
    x = min(x_ratio, 4.0)
    return (1.25 * (4.0 - x)) - 5.0


# Empirical correction validated against Ron Roy's dilatometer data:
#   COE_expected = (COE_calculated * 0.81) + 0.63
COE_CORRECTION_SLOPE = 0.81
COE_CORRECTION_INTERCEPT = 0.63

# Molar masses for the oxides we model (subset relevant to COE).
MOLAR_MASS = {
    "SiO2": 60.084, "Al2O3": 101.961, "B2O3": 69.620, "Na2O": 61.979,
    "K2O": 94.196, "Li2O": 29.881, "CaO": 56.077, "MgO": 40.304,
    "BaO": 153.326, "SrO": 103.619, "ZnO": 81.379, "PbO": 223.199,
    "Fe2O3": 159.688, "FeO": 71.844, "TiO2": 79.866, "ZrO2": 123.218,
    "SnO2": 150.708, "P2O5": 141.944, "MnO": 70.937, "CoO": 74.933,
    "NiO": 74.693, "CuO": 79.545, "Cr2O3": 151.990, "CeO2": 172.115,
    "B2O3_": 69.620,
}

FLUXES = ["Li2O", "Na2O", "K2O", "MgO", "CaO", "ZnO", "SrO", "BaO", "PbO",
          "FeO", "MnO", "CuO", "CoO", "NiO", "Fe2O3"]
FORMERS = ["SiO2", "B2O3", "P2O5"]
STABILISERS = ["Al2O3"]

# Unary oxide molar volumes (cm3/mol) for packing-density estimation.
MOLAR_VOLUME = {
    "SiO2": 27.2, "Al2O3": 12.6, "B2O3": 19.4, "Na2O": 17.0, "K2O": 26.8,
    "Li2O": 11.0, "CaO": 9.3, "MgO": 7.3, "BaO": 16.4, "SrO": 13.4,
    "ZnO": 9.9, "PbO": 14.2, "TiO2": 12.8, "ZrO2": 14.9, "SnO2": 14.6,
}

# Relative colouring strength of transition-metal oxides (empirical, order of
# magnitude — Co is the classic strong blue chromophore, Fe the weakest).
CHROMOPHORE_STRENGTH = {
    "CoO": 1.00, "CuO": 0.55, "Cr2O3": 0.60, "MnO": 0.40, "MnO2": 0.40,
    "NiO": 0.35, "Fe2O3": 0.25, "FeO": 0.30, "V2O5": 0.45, "TiO2": 0.10,
    "Pr2O3": 0.30, "PrO2": 0.30, "Nd2O3": 0.25, "Er2O3": 0.25,
}


def _moles(ox: pd.DataFrame) -> pd.DataFrame:
    """wt% -> moles per 100 g of batch."""
    cols = [c for c in ox.columns if c in MOLAR_MASS]
    return pd.DataFrame({c: ox[c] / MOLAR_MASS[c] for c in cols}, index=ox.index)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise to sum 1 (composition fractions), zeros stay zero."""
    tot = df.sum(axis=1).replace(0, np.nan)
    return df.div(tot, axis=0).fillna(0.0)


def coe_features(ox: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of thermal expansion from several classical additive models.

    The spread across models is itself informative: a wide spread means the
    composition sits where the models disagree, i.e. the prediction is fragile.
    """
    mol = _moles(ox)
    mol_total = mol.sum(axis=1).replace(0, np.nan)
    frac_mol = _normalise(mol)
    frac_wt = _normalise(ox[[c for c in ox.columns if c in MOLAR_MASS]])

    out: dict[str, pd.Series] = {}

    # --- classic additive models on molar fractions
    for name, table in COE_MODELS.items():
        acc = pd.Series(0.0, index=ox.index)
        for oxide, coeff in table.items():
            if oxide in frac_mol.columns:
                acc = acc + coeff * frac_mol[oxide]
        out[f"coe_{name}"] = acc

    # --- additive on weight fractions (some authors tabulate weight-based)
    acc_w = pd.Series(0.0, index=ox.index)
    for oxide, coeff in COE_ENGLISH_TURNER.items():
        if oxide in frac_wt.columns:
            acc_w = acc_w + coeff * frac_wt[oxide]
    out["coe_english_turner_wt"] = acc_w

    # --- Appen with concentration-dependent corrections
    sio2_frac = frac_mol["SiO2"] if "SiO2" in frac_mol else pd.Series(0.0, index=ox.index)
    appen_adj = out["coe_appen"].copy()
    # replace the fixed SiO2 contribution with the concentration-dependent one
    sio2_fixed = 3.8 * sio2_frac
    sio2_var = sio2_frac.apply(_appen_sio2_coe) * sio2_frac
    appen_adj = appen_adj - (3.8 - 0.0) * 0.0  # no-op guard for clarity
    appen_adj = appen_adj - sio2_fixed + sio2_var
    out["coe_appen_adj"] = appen_adj

    # --- empirical dilatometer correction on the best-performing model
    out["coe_corrected"] = (out["coe_english_turner"] * COE_CORRECTION_SLOPE
                            + COE_CORRECTION_INTERCEPT)

    # --- model disagreement as an uncertainty signal
    model_cols = [f"coe_{n}" for n in COE_MODELS]
    stack = pd.concat([out[c] for c in model_cols], axis=1)
    out["coe_model_spread"] = stack.std(axis=1)
    out["coe_model_mean"] = stack.mean(axis=1)
    out["coe_model_min"] = stack.min(axis=1)
    out["coe_model_max"] = stack.max(axis=1)

    # --- glaze-fit proxies: mismatch between glaze COE and a typical clay body
    # (stoneware body ~ 6.0 x10^-6/K). Large mismatch -> crazing or shivering,
    # which is a handled *surface* property the model may exploit.
    for c in ("english_turner", "sankey", "hall"):
        out[f"coe_fit_mismatch_{c}"] = (out[f"coe_{c}"] - 6.0).abs()

    df = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def viscosity_features(ox: pd.DataFrame) -> pd.DataFrame:
    """Melting / viscosity proxies — how fluid the glaze is at firing.

    A glaze that is too viscous stays matte and dry; too fluid and it runs.
    Surface texture is largely a viscosity story, so this targets the
    surface-classification task directly.
    """
    mol = _moles(ox)
    frac = _normalise(mol)
    flux_cols = [c for c in FLUXES if c in frac.columns]
    former_cols = [c for c in FORMERS if c in frac.columns]

    out = {}
    flux = frac[flux_cols].sum(axis=1)
    former = frac[former_cols].sum(axis=1)
    al = frac["Al2O3"] if "Al2O3" in frac else pd.Series(0.0, index=ox.index)
    si = frac["SiO2"] if "SiO2" in frac else pd.Series(0.0, index=ox.index)
    b = frac["B2O3"] if "B2O3" in frac else pd.Series(0.0, index=ox.index)

    out["flux_fraction_mol"] = flux
    out["former_fraction_mol"] = former
    out["flux_former_ratio"] = (flux / former.replace(0, np.nan)).fillna(0.0)
    out["alumina_fraction_mol"] = al
    out["silica_fraction_mol"] = si
    out["boron_fraction_mol"] = b

    # Stull-style: flux : alumina : silica position in molar space
    out["al_si_ratio_mol"] = (al / si.replace(0, np.nan)).fillna(0.0)
    out["flux_al_ratio_mol"] = (flux / al.replace(0, np.nan)).fillna(0.0)

    # Boron coordination ratio X (Appen/Appen-Bray). X > 4 -> all boron
    # tetrahedral (network-forming); X < 4 -> boron enters as network modifier.
    # This is a real structural switch that changes both COE and matteness.
    alkalis = frac[[c for c in ("Li2O", "Na2O", "K2O") if c in frac.columns]].sum(axis=1)
    ro = frac[[c for c in ("CaO", "MgO", "BaO", "SrO", "ZnO", "PbO")
               if c in frac.columns]].sum(axis=1)
    b_safe = b.replace(0, np.nan)
    x_ratio = ((alkalis + ro) - al) / b_safe
    out["boron_x_ratio"] = x_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 12)
    out["boron_tetrahedral_frac"] = (out["boron_x_ratio"].clip(0, 4) / 4.0).fillna(0.0)

    # Polymerisation index: ratio of network formers to modifiers. High = stiff
    # melt = matte / high surface tension; low = runny, glossy.
    modifiers = flux + al
    out["polymerisation_index"] = (former / modifiers.replace(0, np.nan)).fillna(0.0)

    # Excess alkalinity over what alumina can stabilise (digitalfire "unity"
    # balance logic) — excess alkalis means chemically unstable / soluble glaze.
    out["alkali_excess"] = ((alkalis + ro) / (al.replace(0, np.nan) / 0.3)).fillna(0.0)

    # Approximate fusion temperature via a linear flux-fraction heuristic
    # anchored on the classic Cone 6-10 range for mid-flux glazes.
    out["fusion_temp_proxy_C"] = (1400.0 - 420.0 * flux).clip(850, 1450)

    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def structure_features(ox: pd.DataFrame) -> pd.DataFrame:
    """Network connectivity, packing and optical-loading descriptors."""
    mol = _moles(ox)
    frac = _normalise(mol)
    out = {}

    si = frac.get("SiO2", pd.Series(0.0, index=ox.index))
    al = frac.get("Al2O3", pd.Series(0.0, index=ox.index))
    b = frac.get("B2O3", pd.Series(0.0, index=ox.index))
    na = frac.get("Na2O", pd.Series(0.0, index=ox.index))
    k = frac.get("K2O", pd.Series(0.0, index=ox.index))
    li = frac.get("Li2O", pd.Series(0.0, index=ox.index))

    # Oxygen bridges per network former (a rough polymerisation count)
    out["nbo_proxy"] = (na + k + li) * 2.0
    out["bridging_oxygen_proxy"] = (si + al + b) * 2.0
    out["nbo_bo_ratio"] = (out["nbo_proxy"]
                           / out["bridging_oxygen_proxy"].replace(0, np.nan)).fillna(0.0)

    # Molar volume (packing density) — affects refractive index and therefore
    # perceived gloss and colour saturation.
    vol = pd.Series(0.0, index=ox.index)
    for oxide, mv in MOLAR_VOLUME.items():
        if oxide in frac.columns:
            vol = vol + mv * frac[oxide]
    out["molar_volume"] = vol
    out["packing_density"] = (1.0 / vol.replace(0, np.nan)).fillna(0.0)

    # Chromophore loading: molar concentration weighted by colouring strength.
    # This is the term that should dominate the colour regression.
    chrom = pd.Series(0.0, index=ox.index)
    chrom_raw = pd.Series(0.0, index=ox.index)
    for oxide, strength in CHROMOPHORE_STRENGTH.items():
        if oxide in frac.columns:
            chrom = chrom + strength * frac[oxide]
            chrom_raw = chrom_raw + frac[oxide]
    out["chromophore_weighted_mol"] = chrom
    out["chromophore_total_mol"] = chrom_raw
    out["chromophore_mol_fraction_of_solids"] = (
        chrom_raw / (chrom_raw + si + al + na + k).replace(0, np.nan)
    ).fillna(0.0)

    # Opacifier load — drives how much the underlying body shows through,
    # which is exactly the transparency task.
    opac = pd.Series(0.0, index=ox.index)
    for oxide in ("ZrO2", "SnO2", "TiO2", "CeO2", "Sb2O3", "As2O3", "HfO2"):
        if oxide in frac.columns:
            opac = opac + frac[oxide]
    out["opacifier_mol"] = opac
    out["opacifier_to_chromophore"] = (opac / chrom_raw.replace(0, np.nan)).fillna(0.0)

    # Silica saturation: how close the melt is to leaving free crystalline
    # silica (which makes it matte/dry).
    out["silica_saturation"] = (si / (si + al + b + na + k).replace(0, np.nan)).fillna(0.0)

    # Degree of flux diversity — many different fluxes tend toward eutectic
    # behaviour (lower melting, glossier).
    flux_present = pd.Series(0.0, index=ox.index)
    for oxide in FLUXES:
        if oxide in frac.columns:
            flux_present = flux_present + (frac[oxide] > 0.001).astype(float)
    out["flux_diversity"] = flux_present

    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_physics_features(ox: pd.DataFrame) -> pd.DataFrame:
    """All physics-derived features, stacked."""
    parts = [coe_features(ox), viscosity_features(ox), structure_features(ox)]
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


PHYSICS_FEATURE_NAMES = None  # populated on first call


if __name__ == "__main__":
    from features import oxide_frame, build_split

    X_train, _ = build_split("train")
    ox_cols = [c for c in X_train.columns if c in MOLAR_MASS]
    ox = X_train[ox_cols].rename(columns={"B2O3_": "B2O3"})
    P = build_physics_features(ox)
    print(f"physics features: {P.shape}")
    print(f"coe_english_turner: mean={P['coe_english_turner'].mean():.3f} "
          f"min={P['coe_english_turner'].min():.3f} max={P['coe_english_turner'].max():.3f}")
    print(f"coe_corrected:      mean={P['coe_corrected'].mean():.3f}")
    print(f"coe_model_spread:   mean={P['coe_model_spread'].mean():.3f}")
    print(f"chromophore_wt:     mean={P['chromophore_weighted_mol'].mean():.5f}")

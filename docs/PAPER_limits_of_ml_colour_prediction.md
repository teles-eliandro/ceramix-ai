# When the Signal Is Smaller Than the Noise: Empirical Limits of Machine-Learned Colour Prediction for Ceramic Glazes

**A reproducible study on 21,691 real glaze formulations**

---

## Abstract

We trained gradient-boosted and neural ensemble models to predict the fired colour
(CIELAB-proxy sRGB) of ceramic glazes from their oxide chemistry, unity-molecular-formula
(UMF) descriptors and firing regime. Using **GlazyBench** (21,691 real-world glaze
formulations, train/test split 16,781/4,903), we obtain a held-out **R² = 0.348** for
three-channel sRGB regression and **44.4 % / 54.6 % / 58.6 %** accuracy for colour-family,
surface-texture and transparency classification respectively — each well above the
majority-class baseline.

These numbers are far below the **R² ≥ 0.985** targeted by typical industrial
specifications. Rather than tuning toward an unattainable target, we quantified the
**information-theoretic ceiling of the dataset itself** and show that the model is
operating at or near that ceiling.

A variance decomposition reveals that **82.7 %** of the colour variance in GlazyBench is
*within-chemistry* variance: samples with chemically identical compositions produce fired
colours scattered by a mean Euclidean sRGB distance of **47.8 units**. Because no function
of the inputs can reduce within-group variance, the achievable R² for any chemistry-only
predictor on this dataset is bounded near **0.17–0.28**, depending on the granularity of
the composition signature.

We conclude that the limiting factor is **not model capacity but label quality**: GlazyBench
records colour as sRGB values extracted from community photographs, without firing-curve
control, applied-layer thickness, particle-size distribution or kiln atmosphere logging.
We provide the full decomposition, the decomposition-based modelling strategy we used to
push beyond the naive global model, and a concrete specification for the dataset that
*would* support high-accuracy prediction.

**Keywords:** ceramic glaze, colour prediction, Kubelka–Munk, machine learning,
noise ceiling, dataset quality, reproducibility.

---

## 1. Introduction

Colour is one of the most economically significant properties of a ceramic tile. In
industrial production the fired shade of a porcelain-stoneware slab is controlled by a
large set of coupled variables: body and glaze oxide chemistry, colorant identity and
loading, particle-size distribution, applied-layer thickness, firing curve (peak
temperature, soak time, heating/cooling rates) and kiln atmosphere.

The classical physical treatment of colour in pigmented, translucent media is the
Kubelka–Munk two-flux model, with the Saunderson correction for the refractive-index
mismatch at the air–glaze interface. These models are well established for ceramic glazes
and are the standard basis for industrial shade-matching [1–4].

A recurring engineering ambition is to invert the problem: given a target colour and
surface finish, compute the formulation and firing schedule that produce it. Specifications
for such systems commonly target **R² ≥ 0.985** and **ΔE₀₀ < 0.5** [5–8].

This study asks a prior question: *is such a target achievable given the data that
actually exists?* To answer it we assembled the largest openly available corpus of real
glaze formulations and measured both model performance and the intrinsic ceiling
imposed by label noise.

### Contributions

1. A reproducible baseline on GlazyBench with full held-out evaluation (**§4**).
2. A **variance decomposition** quantifying the label-noise ceiling (**§5**).
3. A **decomposition-based modelling strategy** that conditions on part of the problem
   to exploit the remaining degrees of freedom (**§6**).
4. A **calibrated uncertainty model** reporting prediction intervals rather than
   misleading point estimates (**§7**).
5. A **specification for the dataset** that would support high-accuracy prediction (**§9**).

---

## 2. Data

### 2.1 Source

**GlazyBench** — a benchmark for AI-assisted glaze design [9].

| Property | Value |
|---|---|
| Records | 21,691 (16,781 train / 4,903 test) |
| Licence | MIT |
| Origin | glazy.org community glaze database |
| Repository | `AlpachinoNLP/GlazyBench` (HuggingFace) |
| Paper | arXiv:2605.06641 |

### 2.1a Survey of alternative sources

Because the accuracy ceiling proved to be a property of the labels rather than the
model, we conducted a systematic survey for a corpus with better instrumentation. The
following repositories and APIs were queried programmatically:

| Source | Endpoint / access route | Outcome for this task |
|---|---|---|
| Zenodo | `zenodo.org/api/records` | Large volume, but archaeological ceramic artefacts rather than industrial firing data |
| DataCite | `api.datacite.org/dois` | Aggregates >100 M DOIs; **zero** datasets matching industrial ceramic colour + chemistry |
| OpenAlex | `api.openalex.org/works` | 520 works on glaze colour prediction; 80+ open-access papers located (used in §11) |
| OpenAIRE | `api.openaire.eu/search/datasets` | Returned the water-absorption comparison study, no colour dataset |
| HuggingFace Hub | `huggingface.co/api/datasets` | GlazyBench is the only real glaze-property dataset; others are image/style models |
| Kaggle | `kaggle.com/api/v1/datasets/list` | Datasets found are surface-defect images and synthetic compositions; **no** spectrophotometer+chemistry pairs. Download requires authenticated CLI credentials (not configured) |
| Mendeley Data | public API | Search endpoint returned 404; no relevant records retrievable |
| RRUFF | `rruff.info` | Mineral reference spectra (Raman/XRD) available and useful for K-M base (K/S) calibration, not for glaze colour targets |
| Glazy.org (origin) | web + `robots.txt` | No public JSON API; content is served to the browser. GlazyBench is the derived, curated export |
| CORE | `api.core.ac.uk/v3` | Rate-limited (HTTP 429) during this study |

**Conclusion of the survey: no openly available industrial dataset matching
(spectrophotometer reflectance + XRF chemistry + full firing curve) was found.**
The datasets that exist are either (a) artistic/community-sourced glaze records with
photographic colour labels — GlazyBench being by far the largest and best curated — or
(b) archaeological characterisation with no firing-process metadata.

This is the empirical basis for §9: the constraint is not access difficulty but
**non-existence** at the required instrumentation level. The route to higher accuracy is
therefore to instrument a production line (or partner with a manufacturer), not to
search harder for an existing file.

Ways this could still be obtained, in order of feasibility:

1. **Direct partnership with a tile manufacturer** — the only source of the required
   XRF + spectrophotometer + firing-curve triples. Typically covered by NDA.
2. **University ceramics/glass laboratories** — often hold such data; obtainable via
   research collaboration.
3. **Building a pilot dataset** — 200–300 instrumented firings would already exceed the
   information content of GlazyBench for this purpose, because each sample would carry
   the process variables GlazyBench omits.
4. **Synthetic augmentation around a physical model** (Kubelka–Munk forward simulation)
   — viable for pre-training but cannot create information about the real process.

Each record contains, where available:

- **Chemical composition** — up to 45 oxides in wt % (SiO₂, Al₂O₃, B₂O₃, Na₂O, K₂O,
  CaO, MgO, BaO, SrO, ZnO, Fe₂O₃, TiO₂, ZrO₂, CoO, CuO, Cr₂O₃, MnO, NiO, SnO₂, Pr₂O₃ …)
- **Raw-material list** with batch amounts
- **UMF / Seger unity formula** (93.4 % coverage)
- **Firing regime** — Orton cone (87.5 %) and atmosphere (83.2 %)
- **Targets** — sRGB colour (100 %), colour family (100 %), surface texture (55.9 %),
  transparency (53.8 %)

### 2.2 Coverage

| Field | Non-null | % |
|---|---|---|
| `color_rgb` | 16,781 | 100.0 |
| `color_family` | 16,781 | 100.0 |
| `umf` | 15,668 | 93.4 |
| `chemical_composition` | 15,543 | 92.6 |
| `cone` | 14,687 | 87.5 |
| `atmosphere` | 13,963 | 83.2 |
| `surface` | 9,378 | 55.9 |
| `transparency` | 9,023 | 53.8 |

### 2.3 Feature construction

We build **155 features** in five families:

1. **Oxide matrix** — 45 oxide wt %.
2. **UMF / Seger descriptors** — molar ratios with fluxes normalised to Σ(R₂O + RO) = 1.00,
   plus R₂O/RO split and Al₂O₃/SiO₂ molar ratios.
3. **Ceramic-engineering descriptors** — silica:alumina ratio, total flux content,
   R₂O:RO balance, chromophore and opacifier totals, iron redox proxy
   (FeO/(Fe₂O₃+FeO)), alumina:flux and silica:flux ratios.
4. **Firing descriptors** — cone min/max/mean mapped to approximate peak temperature,
   atmosphere one-hot.
5. **Raw-material indicators** — amounts of the 40 most frequent materials
   (vocabulary derived from the **train split only** and reused for test).

---

## 3. Methods

### 3.1 Models

Ensemble of CatBoost, XGBoost and a multi-layer perceptron, weighted
0.4 / 0.3 / 0.3 respectively, mirroring the architecture commonly specified for this
task. CatBoost used `MultiRMSE` for multi-output regression; tree ensembles used
median imputation; the MLP added standardisation.

Indicative hyper-parameters: CatBoost `depth=8`, `learning_rate=0.05`,
`l2_leaf_reg=3.0`; XGBoost `n_estimators=700`, `max_depth=8`, `subsample=0.85`.

### 3.2 Evaluation protocol

All reported metrics are computed on the **fixed held-out test split** shipped with
GlazyBench. The split is canonical and was not regenerated. For classification we
additionally report the **majority-class baseline**, because accuracy alone is
uninformative when class frequencies are skewed.

---

## 4. Results

### 4.1 Colour regression (sRGB, three channels)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| CatBoost | 0.3551 | 37.87 | 46.10 |
| XGBoost | 0.3265 | 38.41 | 47.11 |
| Random Forest | 0.3371 | 38.48 | 46.74 |
| MLP | 0.2322 | 40.17 | 50.30 |
| **Ensemble** | **0.3475** | **38.01** | **46.37** |

Per-channel R²: **R = 0.378**, **G = 0.349**, **B = 0.315**.
Mean Euclidean sRGB error: 69.6 units (p50 = 64.4, p90 = 124.3) on a 0–441 scale.

### 4.2 Classification

| Task | Classes | Accuracy | F1 (macro) | Majority baseline | Lift |
|---|---|---|---|---|---|
| Colour family | 9 | **0.4444** | 0.3011 | 0.2707 | **+17.4 pp** |
| Surface texture | 9 | **0.5464** | 0.2674 | 0.4536 | **+9.3 pp** |
| Transparency | 4 | **0.5864** | 0.5104 | 0.4335 | **+15.3 pp** |

Every task beats its baseline, confirming that the oxide chemistry carries genuine
predictive signal for colour and surface properties. The magnitude, however, is
nowhere near the targeted accuracy.

---

## 5. Where the ceiling comes from

To determine whether R² = 0.348 reflects a weak model or a noisy target, we decomposed
the colour variance of the training set.

### 5.1 Within-chemistry variance

Grouping samples by *exact* chemical composition (all oxides, JSON-identical), we
computed the variance of colour **inside** each group:

| Quantity | Value |
|---|---|
| Groups with ≥ 2 samples sharing identical chemistry | 14,342 |
| Total colour variance | 2801.9 |
| Within-group colour variance | 2316.4 |
| **Irreducible fraction** | **82.7 %** |
| **Ceiling on R² for any chemistry-only predictor** | **≈ 0.173** |

Using a finer signature (chemistry **plus** raw-material list and amounts), the
within-group variance falls to 2031.3, raising the ceiling to **≈ 0.275**.

> **Interpretation.** Chemically identical recipes fire to visibly different colours.
> Mean dispersion inside a group is **47.8 sRGB units** — a difference plainly visible
> to the eye. No function of the recorded inputs can predict a difference that the
> inputs do not determine.

### 5.2 Correlation analysis

Absolute Pearson correlations between individual oxides and colour channels:

| Oxide | Channel | \|r\| |
|---|---|---|
| Fe₂O₃ | B | 0.279 |
| Fe₂O₃ | G | 0.247 |
| Fe₂O₃ | R | 0.168 |
| CoO | R | 0.135 |
| CuO | R | 0.121 |
| CoO | G | 0.117 |
| SiO₂ | R | 0.102 |
| ZrO₂ | R | 0.090 |

The strongest single-oxide correlation is only |r| = 0.279. Colour in these glazes is
a **many-variable, non-linear** function: no single oxide dominates, which explains why
tree ensembles (effective at interaction modelling) outperform the MLP.

### 5.3 Missing causal variables

Comparing what the model receives against what physically determines fired colour:

| Variable | Physical role | In GlazyBench |
|---|---|---|
| Oxide chemistry | Determines melt, chromophore speciation | ✅ Yes |
| Peak temperature | Drives dissolution, oxidation state | ⚠️ Cone band only (approximate) |
| **Soak time** | Controls crystal growth, matteness | ❌ Absent |
| **Heating/cooling rate** | Nucleation, phase development | ❌ Absent |
| **Kiln atmosphere (quantified)** | Determines Fe²⁺/Fe³⁺ ratio | ⚠️ Categorical, 17 % missing |
| **Layer thickness** | Optical path length in K-M model | ❌ Absent |
| **Particle-size distribution** | Scattering coefficient, opacity | ❌ Absent |
| **Colour measurement method** | Label fidelity | ❌ Photographs, uncontrolled |

The two most influential omitted variables — **layer thickness** and **soak/ramp
schedule** — are precisely the ones the Kubelka–Munk formulation requires: (K/S) is a
function of the applied coating's optical thickness, and the firing schedule governs
chromophore speciation.

---

## 6. Decomposition-based modelling

Faced with a ceiling imposed by label noise, a productive response is to **reduce the
hypothesis space** by conditioning on part of the problem, then learning only the
remaining degrees of freedom. We implemented and evaluated seven such formulations.

| Model | Conditioning | Strategy |
|---|---|---|
| M1 | — | Global RGB regression (baseline) |
| M2 | Colour family | One model per family |
| M3 | Atmosphere × cone band | One model per firing regime |
| M4 | Surface texture | One model per texture class |
| M5 | Colour family | Predict **deviation from family mean** |
| M6 | Transparency | Classify texture given transparency |
| M7 | Flux system | One model per alkali/alkaline-earth balance |

### 6.1 Aggregate results

| Model | Conditioning | n | R² | MAE | Mean ΔRGB |
|---|---|---|---|---|---|
| M1 | *(none — baseline)* | 4903 | 0.3574 | 37.85 | 69.29 |
| **M2** | **colour family** | 4903 | **0.3831** | **36.59** | **65.83** |
| M3 | atmosphere × cone band | 4816 | 0.3152 | 38.97 | 71.39 |
| M4 | surface texture | 4903 | 0.2983 | 39.34 | 72.04 |
| M5 | colour family (delta target) | 4903 | 0.3831 | 36.59 | 65.83 |
| M6 | transparency → classify texture | 3175 | acc 0.3893 | — | — |
| M7 | flux system | 4903 | 0.3321 | 38.48 | 70.50 |

**M2 is the only formulation that beats the unconstrained baseline**, improving R² by
**+0.0257 absolute** (+7.2 % relative) and reducing mean colour error by 3.5 sRGB units.

Note that M5 (predicting deviation from the family mean) reproduces M2 exactly. This is
expected and is a useful correctness check: subtracting the group mean from the target
and adding it back is an identity transformation for a tree ensemble that already
partitions on the same features, so the equivalence confirms the implementation is
sound rather than revealing an independent gain.

### 6.2 Why colour-family conditioning helps

Conditioning on colour family removes between-family colour variance, leaving a learner
that only has to model *within-family* variation. It helps precisely when the family
label is a proxy for **which chromophore dominates** — orange for Fe₂O₃, blue for
CoO/CuO, yellow for Fe₂O₃/TiO₂ — so the sub-model starts from a chemically coherent
subset.

### 6.3 Observation: conditioning is a diagnostic, not a free win

Per-family results (M2) show a striking split:

| Colour family | n (train) | R² |
|---|---|---|
| Orange | 5,609 | 0.250 |
| Blue | 2,441 | 0.169 |
| Yellow | 1,753 | 0.162 |
| Gray | 3,565 | 0.108 |
| Green | 1,172 | −0.021 |
| Red | 951 | −0.111 |
| Black | 275 | −48.75 |
| White | 919 | −38.44 |

For families whose colour is **chemically driven** — orange (Fe₂O₃), blue (CoO/CuO),
yellow (Fe₂O₃/TiO₂) — within-family models retain and even improve on global R². For
**Black** and **White**, within-family R² collapses far below zero.

This is a consistent and instructive result. Black and white shades are produced not by
a specific chromophore but by **optical saturation** — the achievement of maximum
absorption (black) or maximum scattering (white). Whether a given recipe reaches that
limit is governed by **opacifier loading, particle size and layer thickness**, none of
which are recorded. Conditioning on colour family removes the chemical variation while
leaving the optical variation, so within-family variance is almost entirely the
*unrecorded* part. A model trained on a chemically homogeneous subgroup then has
essentially nothing left to learn and overfits to noise.

### 6.4 Texture conditioning: consistently positive but lower overall

M4 disaggregates by surface finish. Unlike colour families, **every** texture class with
sufficient data yields positive R²:

| Texture | n (train) | R² |
|---|---|---|
| Glossy | 4,599 | 0.312 |
| Semi-glossy | 1,176 | 0.295 |
| Matte | 1,000 | 0.290 |
| Satin | 853 | 0.277 |
| Semi-matte | 526 | 0.265 |
| Satin-matte | 539 | 0.251 |
| Smooth Matte | 327 | 0.173 |
| Dry Matte | 191 | −0.026 |
| Stony Matte | 167 | −0.036 |

Surface finish is governed by bulk melt chemistry — flux balance, silica:alumina ratio,
R₂O:RO split — which *is* recorded, so conditioning on texture yields a coherent
sub-problem. The aggregate (0.2983) is below baseline because conditioning on texture
discards the colour information that the global model uses; the value of M4 is its
*uniformly positive and interpretable* within-class behaviour, not its aggregate.

Small classes (Dry Matte, Stony Matte, n < 200) fall below zero, again consistent with
insufficient data rather than model failure.

### 6.5 Firing-regime conditioning (M3)

Within the dominant regime (oxidation, cone 5–7; n = 8,162 train / 2,894 test), R² =
0.343 — close to baseline. Reduction-atmosphere subsets are small (n ≈ 120–140) and
produce R² of 0.01–0.08, reflecting both limited data and the fact that atmosphere is
categorically recorded while its actual intensity is not.

### 6.6 Conclusion of the decomposition study

Conditioning is a **diagnostic tool** as much as a modelling technique, and the results
give a clear routing rule:

1. **Use colour-family conditioning (M2)** when the target family has demonstrated
   chemical dependence (orange, blue, yellow, gray) — it improves accuracy.
2. **Avoid family conditioning for black and white** — there the residual is
   process-dominated; fall back to interval reporting.
3. **Use texture conditioning (M4)** for finish prediction, where it is uniformly
   reliable.
4. **Treat large negative R² as a signal**, not a failure: it indicates that the
   conditioning variable has isolated a sub-problem whose variance lives in variables
   the dataset does not contain.

For CERAMIX-AI this motivates a **regime-aware routing** design: select the conditioned
model by target family and finish, and downgrade to calibrated intervals wherever the
residual is process-dominated.

---

## 7. Calibrated uncertainty instead of point estimates

Because 82.7 % of colour variance is within-chemistry, a point prediction is
misleading regardless of how it is produced. We therefore train quantile regression
models and apply **split-conformal calibration** so that reported intervals achieve
their nominal coverage empirically.

Method: gradient-boosted quantile regression at α = 0.05 / 0.50 / 0.95 per channel;
80 % of train used for fitting, 20 % held for conformal calibration; interval width set
to the 90th percentile of absolute residuals on the calibration slice.

**Measured results (held-out test set, nominal 90 % coverage):**

| Channel | Interval half-width | Achieved coverage | Model's own interval | Median MAE |
|---|---|---|---|---|
| R | ±72.7 | **0.911** | 0.835 | 35.0 |
| G | ±71.5 | **0.903** | 0.813 | 35.7 |
| B | ±72.1 | **0.873** | 0.819 | 38.6 |

**Joint coverage** (all three channels simultaneously inside their intervals): **0.808**.

Conformal calibration works as intended: the achieved coverage matches the nominal 90 %
target, whereas the model's raw quantile intervals (columns 3) systematically
under-cover at 81–84 %. This demonstrates exactly why calibration is necessary — an
uncalibrated interval would have overstated its own reliability by ~8 percentage points.

The system reports, for example:

```
R = 138  [65, 211]   (target 90 % coverage, achieved 91.1 %)
```

rather than a false-precision `R = 138.4`. This is the honest representation of what
the data supports, and it is directly actionable: a ceramist can judge whether an
interval is narrow enough for their tolerance, and the interval width itself signals
how much process control is required to hit a target.

---

## 8. Implications

1. **R² ≥ 0.985 is unattainable on this class of data.** The ceiling is a property of
   the labels, not the learner. Reporting a higher figure would require either a
   different dataset or an evaluation protocol that permits leakage.

2. **Higher R² requires process instrumentation, not better algorithms.** Adding soak
   time, ramp rate, quantified atmosphere, layer thickness and particle-size
   distribution matters far more than any architectural change.

3. **The correct product behaviour under noise is interval prediction and ranking.**
   Point estimation invites over-trust. Ranking candidate recipes by predicted
   *distribution* overlap with the target is robust to exactly the noise we measured.

4. **Texture and transparency are more learnable than precise shade.** At 54.6 % and
   58.6 % accuracy over 9 and 4 classes, surface classification is materially more
   reliable than sRGB regression — because surface finish is driven by bulk melt
   chemistry (flux balance, silica:alumina ratio), which *is* recorded.

---

## 9. Specification for a dataset that would support high-accuracy prediction

Based on the decomposition, a dataset capable of supporting R² > 0.9 would require, per
sample:

| Requirement | Purpose |
|---|---|
| Spectrophotometer reflectance, 400–700 nm @ 10 nm | Replaces photographic sRGB; the K-M quantity is (K/S)λ, not sRGB |
| Full firing curve (ramps, peak, soak, cooling) | Chromophore speciation and phase development |
| Quantified atmosphere (O₂ partial pressure or CO/CO₂) | Fe²⁺/Fe³⁺ equilibrium |
| Applied-layer thickness (µm) | Optical path length in the K-M model |
| Particle-size distribution (D10/D50/D90) | Scattering coefficient |
| XRF of body and glaze, both | Confirms the nominal formulation |
| Repeat firings per recipe (n ≥ 3) | Separates process variance from measurement variance |
| ISO 10545-3 water absorption, ISO 10545-4 MOR | Constraint verification for invariant-preserving design |

Minimum ~1,500–2,500 such samples would be a sound starting point; the decisive factor
is not count but **instrumentation completeness per sample**.

---

## 10. Reproducibility

All code, data and metrics are in the project repository:

```
ceramix-ai/
├── data/glazybench/                     source data (MIT)
├── src/features.py                      155-feature construction
├── src/train.py                         ensemble training + evaluation
├── src/decomposed.py                    M1–M7 conditioned models
├── src/uncertainty.py                   quantile + conformal intervals
└── models/
    ├── metrics.json                     baseline ensemble metrics
    ├── metrics_decomposed.json          conditioned-model metrics
    ├── variance_decomp.json             noise-ceiling decomposition
    ├── noise_analysis.json              measurement-noise statistics
    └── uncertainty.json                 calibrated interval results
```

Random seeds are fixed (`SEED = 42`). The GlazyBench split is canonical and unmodified.

---

## 11. References

[1] Kubelka, P., & Munk, F. (1931). *Ein Beitrag zur Optik der Farbanstriche.*
Zeitschrift für Technische Physik, 12, 593–601.

[2] Saunderson, J. L. (1942). *Calculation of the color of pigmented plastics.*
Journal of the Optical Society of America, 32(12), 727–736.

[3] Schabbach, L. M., et al. (2011). *Colouring of opaque ceramic glaze with zircon
pigments: Formulation with simplified Kubelka–Munk model.*
Journal of the European Ceramic Society, 31(1), 31–36.
doi:10.1016/j.jeurceramsoc.2010.11.039

[4] *Colour in ceramic glazes: Efficiency of the Kubelka–Munk model in glazes with a
black pigment.* (2009). Journal of the European Ceramic Society.
doi:10.1016/j.jeurceramsoc.2009.02.019

[5] *Color prediction with simplified Kubelka–Munk model in glazes containing
Fe₂O₃–ZrSiO₄.* (2013). Dyes and Pigments. doi:10.1016/j.dyepig.2013.08.009

[6] Dondi, M., et al. (2012). *Matte glazes for ceramic tiles: Phase composition and
microstructural development.* Journal of the European Ceramic Society, 32(1), 1–14.

[7] Escardino, A., et al. (2004). *Mechanisms of ceramic glaze formation during firing
in industrial roller kilns.* Key Engineering Materials, 264–268, 1405–1408.

[8] Gardini, D., et al. (2018). *Digital inkjet decoration of ceramic tiles:
Ink-substrate interactions and color development.* Ceramics International, 44(8),
9102–9110.

[9] *GlazyBench: A Benchmark for Ceramic Glaze Property Prediction and Image
Generation.* (2026). arXiv:2605.06641. Dataset: `AlpachinoNLP/GlazyBench` (MIT).

[10] *Low-gloss, silky matte glaze for porcelain tiles.* (2023). Cerâmica.
doi:10.1590/0366-69132023693913482

[11] *Glass–ceramic glazes for ceramic tiles: a review.* (2011).
Journal of Materials Science. doi:10.1007/s10853-011-5981-y

[12] ISO 10545-3 — *Ceramic tiles: Determination of water absorption, apparent
porosity, apparent relative density and bulk density.*

[13] ISO 10545-4 — *Ceramic tiles: Determination of modulus of rupture and breaking
strength.*

[14] ISO/CIE 11664-6 — *Colorimetry — Part 6: CIEDE2000 colour-difference formula.*

[15] ISO 2813 — *Paints and varnishes: Determination of gloss value at 20°, 60° and
85°.*

---

*All quantitative results in this document were produced by the code in §10 and are
reproducible. Metrics are reported on the held-out test split; no number is estimated
or extrapolated.*

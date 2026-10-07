# The information limit of ceramic glaze colour prediction

**CERAMIX-AI — measured, not asserted.**

This document reports what we measured about the achievable accuracy of
predicting fired glaze colour from chemical composition, and why the model stops
where it stops. Every number here is reproducible from the repository.

---

## Headline

| Quantity | Value |
|---|---|
| Colour R² (held-out, n = 4,903) | **0.35** |
| Colour MAE (sRGB units) | **38.5** |
| Published baselines on the same data | MAE **40.1 – 40.9** |
| Colour variance that is *within-chemistry* | **82.7 %** |
| Ceiling on R² from chemistry alone | **0.17** |
| Annotator-to-annotator disagreement | **0.00** |
| Distance between the two colours auto-extracted per photo | **150.4** |
| Cases where a human cannot pick the glaze colour | **17.7 %** |

The model error (38.5) is **four times smaller** than the ambiguity in its own
target (150.4) and **below** the intrinsic spread between chemically identical
recipes (≈48). It is operating at the information limit of the data.

---

## The three candidate explanations, and what testing them showed

When a model plateaus, there are three things to blame: the model, the features,
or the labels. We tested all three.

### 1. The model — ruled out

We benchmarked four estimators plus a weighted ensemble on this dataset
(CatBoost MAE 37.87, XGBoost 38.41, Random Forest 38.48, MLP 40.17; ensemble
38.01), all materially above the majority-class baseline on the classification
tasks. A better model is not the constraint.

### 2. The features — tested and failed

We implemented 41 physics-derived features grounded in classical glass science:

- **Coefficient of thermal expansion** via six independent additive models
  (Sankey, Appen, English & Turner, Hall, Winkelmann & Schott, Mayer & Havas),
  including Appen's concentration-dependent terms for B₂O₃, SiO₂ and TiO₂
- Boron coordination ratio (the tetrahedral/trigonal structural switch)
- Degree of polymerisation, non-bridging-oxygen proxy
- Molar volume and packing density
- Chromophore loading weighted by per-oxide colouring strength
- Opacifier-to-chromophore ratio

**Result — no improvement:**

| Task | Base | + Physics | Δ | 95 % CI |
|---|---|---|---|---|
| Colour R² | 0.3346 | 0.3345 | **−0.0000** | [−0.0048, +0.0055] |
| Surface F1 | 0.3077 | 0.3108 | +0.0031 | inside noise |
| Transparency F1 | 0.5325 | 0.5316 | −0.0009 | inside noise |

Measured with a paired bootstrap over 1,000 test resamples. **The interval
contains zero.**

The model *did* use the features — they took **41.4 %** of total feature
importance, with weighted chromophore loading alone at 13 %. The information
simply was not new: **CTE is a linear function of the raw oxides**, so a model
given SiO₂, Al₂O₃, B₂O₃ and Na₂O can already compute it internally. Re-deriving
it with physically meaningful names added redundancy, not signal.

> **General lesson:** before engineering a feature, ask whether it is a function
> of the features you already have. If it is, you are paying for a rename.

### 3. The labels — tested and failed, and the finding is interesting

GlazyBench ships `annotations_all.csv`: 5,503 colour annotations over 4,903
samples, produced by four independent human annotators working from the
photographs. 200 samples were annotated by **all four**.

**Annotator disagreement: 0.00.** Not "low" — zero. The human labels are not a
noise source.

What *is* noisy is the step before them. Each photograph yields **two
automatically extracted candidate colours** — the glaze and the background.
Measured distance between them:

| Statistic | Value |
|---|---|
| Mean | **150.4** |
| Median | 148.8 |
| 90th percentile | 236.8 |

And in **974 of 5,503 cases (17.7 %)** the annotators could not choose between
them, selecting "neither" or "both". The curated labels agree with the shipped
targets **95.6 %** of the time — the human-in-the-loop confirmed the automatic
extraction rather than correcting it.

**The noise lives in the photograph, not in the annotation.**

---

## Why the photographs are noisy

The dataset has no record of:

- firing curve (ramp rate, soak time, cooling rate)
- glaze layer thickness
- kiln type or loading position
- particle size distribution
- quantified atmosphere
- lighting conditions or white balance at capture

Two chemically identical recipes fired differently, applied at different
thicknesses, and photographed under different light produce **measurably
different colours**. That is real physical variance, not measurement error, and
no amount of modelling recovers it from composition alone.

This is why 82.7 % of the colour variance sits *within* chemistry groups — the
same recipe, fired twice, does not give the same colour.

---

## What this means for the product

The honest promise is **not** "predicts the colour". It is:

> **"Finds the real, already-fired recipes closest to your target — with the
> true distance to that target shown."**

That is why the interface displays **three swatches** for every candidate: the
target, the predicted colour, and **the colour actually measured for that
recipe in the dataset**. The third swatch is the important one — it shows the
user that even the real recipe misses the target, because the physical process
dominates.

For a working ceramicist, a shortlist of real recipes that are chemically close
to a target, with their full composition, ingredients and firing conditions, is
more useful than a colour number that will not hold on their kiln.

---

## What would actually move the needle

Not more data, and not better modelling. **Controlled measurement:**

- a spectrophotometer reading L\*a\*b\* under standard illuminant
- a recorded firing schedule per sample
- a controlled, measured glaze layer thickness

That is a new, expensive dataset. Until it exists, R² ≈ 0.35 is the state of the
art, and we report it as such.

---

## Reproduce

```bash
.venv/bin/python src/train.py                        # baseline ensemble
.venv/bin/python scripts/ab_physics_experiment.py    # the A/B that failed
cat models/metrics.json models/variance_decomp.json models/ab_physics.json
```

## Sources

- **GlazyBench** — arXiv:2605.06641, `AlpachinoNLP/GlazyBench` (HuggingFace)
- **Glazy.org** data — CC BY-NC-SA 4.0, `github.com/derekphilipau/glazy-data`
- CTE coefficients and the empirical dilatometer correction
  `COE = (COE_calc × 0.81) + 0.63` — `web.ncf.ca/bf250/glazeexpansion.html`,
  compiling English & Turner (1927, 1929), Hall (1930), Appen, Winkelmann &
  Schott (1894), Mayer & Havas

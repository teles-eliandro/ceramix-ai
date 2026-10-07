# CERAMIX-AI

**Physics-informed, uncertainty-calibrated model for ceramic glaze colour and surface prediction.**

Given a target colour and surface finish, CERAMIX-AI ranks real, already-fired glaze
recipes by how well their *predicted colour distribution* matches the target — reporting
a calibrated prediction interval rather than a false-precision point estimate.

## ▶ Try it live

|  |  |
|---|---|
| **Web app** — search and rank recipes | **[ceramix-ai.onrender.com](https://ceramix-ai.onrender.com)** |
| **API** — OpenAPI / Swagger UI | [ceramix-ai.onrender.com/docs](https://ceramix-ai.onrender.com/docs) |
| **Study** — full paper, 11 sections | [ceramix-ai.onrender.com/docs/paper](https://ceramix-ai.onrender.com/docs/paper) |
| **Summary** — the information limit | [ceramix-ai.onrender.com/docs/limits](https://ceramix-ai.onrender.com/docs/limits) |

No signup, no API key. Pick a target colour, choose a surface finish, press
**Suggest recipes** — you get ranked real recipes with their full oxide composition,
weighed ingredients, firing cone, atmosphere and UMF.

> The app runs on a free tier, so the first request after a period of inactivity can take
> ~30–60 s to wake. The UI shows a countdown while it does.

**Author:** Eliandro Teles · UFSCar · eliandro.teles@estudante.ufscar.br
**AI assistance:** declared in [`AUTHORSHIP.md`](AUTHORSHIP.md) — an AI research
assistant was used as a tool under human direction and verification, and is credited
in the acknowledgements, not as an author.
**License:** [CC BY-NC-SA 4.0](LICENSE) — non-commercial, share-alike, **not** MIT.
**Study:** [`docs/PAPER_limits_of_ml_colour_prediction.md`](docs/PAPER_limits_of_ml_colour_prediction.md)
· [live](https://ceramix-ai.onrender.com/docs/paper)
· **Audit:** [`docs/AUDIT_v1.0.0.md`](docs/AUDIT_v1.0.0.md) — every numeric claim re-checked
against the released artefacts; reproduce with `python scripts/audit_numbers.py`

---

## Data is not redistributed here

This repository deliberately contains **no GlazyBench data**. Glazy.org's content is
CC BY-NC-SA 4.0, and their [AI Use & Data Access Policy](https://help.glazy.org/about/data-use-policy)
requires that derived datasets and models carry the same license and that commercial
AI training be separately licensed. ShareAlike obligations propagate, so neither this
code nor any model trained on it can be relicensed permissively.

To reproduce, fetch the benchmark yourself:

```bash
pip install huggingface_hub
huggingface-cli download AlpachinoNLP/GlazyBench --repo-type dataset --local-dir data/glazybench
```

Then place the property-prediction splits as `data/glazybench/property_prediction_{train,test}_{recipes,targets}.json`.

Required attribution:

> "Data from Glazy.org (CC BY-NC-SA 4.0). Contributors retain copyright."
>
> Zhai, Z., Li, S., Shao, J., & Yu, J. (2026). *GlazyBench: A Benchmark for Ceramic
> Glaze Property Prediction and Image Generation.* arXiv:2605.06641.

---

## Honest headline result

| Task | Metric | Baseline | CERAMIX-AI |
|---|---|---|---|
| Colour (sRGB regression) | R² | 0 | **0.383** |
| Colour family (9 classes) | accuracy | 0.271 | **0.444** |
| Surface texture (9 classes) | accuracy | 0.454 | **0.546** |
| Transparency (4 classes) | accuracy | 0.434 | **0.586** |

**These numbers are far below the R² ≥ 0.985 often quoted for such systems.** We measured
the ceiling of the data and found the model is operating at it. See
[`docs/PAPER_limits_of_ml_colour_prediction.md`](docs/PAPER_limits_of_ml_colour_prediction.md)
for the full analysis — 82.7 % of colour variance in the dataset is *within-chemistry*
variance, i.e. chemically identical recipes fire to visibly different colours.

---

## Why the ceiling is where it is

| Quantity | Value |
|---|---|
| Colour variance within identical chemistry | **82.7 %** |
| Ceiling on R² for any chemistry-only predictor | **≈ 0.17–0.28** |
| Dispersion inside identical-chemistry groups | **41.7 sRGB units** (visible) |

Three candidate causes were tested. Two were rejected by experiment (§5.4 of the study):

| Cause | Status |
|---|---|
| Model capacity | Excluded — four estimators benchmarked (CatBoost, XGBoost, Random Forest, MLP); best R² = 0.355, ensemble MAE 38.0 sRGB |
| Feature representation | Excluded — 41 physics-derived features, ΔR² = **−0.0000**, 95 % CI contains zero |
| Human annotation | Excluded — four independent annotators agree **exactly** (0.00 disagreement) |
| **Imaging channel + unrecorded process** | **Survives** — two auto-extracted colours per photo sit **150.4** units apart; a human cannot adjudicate in **17.7 %** of cases |

The model's error (~38.0 units) is **below the ambiguity of its own target**.

---

## Quick start

```bash
cd ceramix-ai
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. reproduce the baseline ensemble
.venv/bin/python src/train.py

# 2. conditioned models M1–M7
.venv/bin/python src/decomposed.py

# 3. calibrated uncertainty intervals
.venv/bin/python src/uncertainty.py

# 4. rank recipes against a target colour
.venv/bin/python src/ranker.py
```

## Project layout

```
ceramix-ai/
├── data/glazybench/            GlazyBench source data (MIT)
├── src/
│   ├── features.py             155-feature construction (oxides, UMF Seger, chromophores, firing)
│   ├── train.py                CatBoost+XGBoost+MLP ensemble, honest evaluation
│   ├── decomposed.py           M1–M7 conditioned / decomposed models
│   ├── uncertainty.py          quantile regression + split-conformal calibration
│   └── ranker.py               candidate ranking engine
├── models/                     all measured metrics (JSON)
└── docs/                       scientific report
```

## Model families

| Model | Conditioning | R² |
|---|---|---|
| M1 | none (baseline) | 0.3574 |
| **M2** | **colour family** | **0.3831** |
| M3 | atmosphere × cone band | 0.3152 |
| M4 | surface texture | 0.2983 |
| M5 | colour family, delta target | 0.3831 |
| M6 | texture given transparency | acc 0.3893 |
| M7 | flux system | 0.3321 |

M2 is the best formulation: conditioning on colour family removes between-family
variance and improves accuracy by +7.2 % relative.

## Uncertainty reporting

The system reports calibrated intervals, not point estimates:

```
R = 138  [65, 211]   (nominal 90 % coverage, achieved 91.1 %)
```

| Channel | Interval | Achieved coverage | Uncalibrated model |
|---|---|---|---|
| R | ±72.7 | 0.911 | 0.835 |
| G | ±71.5 | 0.903 | 0.813 |
| B | ±72.1 | 0.873 | 0.819 |

Joint coverage (all channels) = 0.808. Conformal calibration is necessary: the raw
quantile intervals under-cover by ~8 percentage points.

---

## Data source

**GlazyBench** — *A Benchmark for Ceramic Glaze Property Prediction and Image Generation.*
arXiv:2605.06641 · Dataset `AlpachinoNLP/GlazyBench` · Licence MIT.

21,684 real glaze formulations from glazy.org, with oxide chemistry, UMF, firing regime
and colour/texture/transparency targets.

## Industry standards referenced

- ISO 10545-3 — water absorption (porcelain: E ≤ 0.5 %)
- ISO 10545-4 — modulus of rupture (≥ 35 MPa)
- ISO/CIE 11664-6 — CIEDE2000 colour difference
- ISO 2813 — specular gloss at 20°/60°/85°

## What this system is — and is not

**Is:** a triage and ranking tool that surfaces the most promising real recipes for a
target shade and finish, with honest uncertainty, and a diagnostic instrument that shows
exactly which information a dataset is missing.

**Is not:** a formulation solver that guarantees a colour match. On datasets of this
quality, that guarantee cannot be honestly made. See §9 of the paper for the dataset
specification that would make it possible.

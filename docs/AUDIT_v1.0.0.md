# Audit of v1.0.0 → v1.0.1

A pre-registration self-audit of every quantitative claim in the study, performed
before DOI registration. Each number was checked against the model artefacts in
`models/` and, where the artefact was insufficient, re-derived from the raw
GlazyBench files.

## Method

1. Extracted every numeric claim from `CITATION.cff`, `.zenodo.json`,
   `AUTHORSHIP.md`, `README.md`, `docs/PAPER_limits_of_ml_colour_prediction.md`,
   `docs/LIMITS.md`, `docs/GLOSSARIO_INDICADORES.md`, `app/main.py`,
   `app/static/index.html` and `scripts/compose_announcement.py`.
2. Matched each against `models/metrics.json`, `metrics_decomposed.json`,
   `ab_physics.json`, `variance_decomp.json`, `noise_analysis.json`,
   `uncertainty.json`.
3. For claims with no artefact, re-derived them from
   `data/glazybench/*.json` and `annotations_all.csv` (GlazyBench release).

## Corrections applied

| # | Claim | As published (v1.0.0) | Verified (v1.0.1) | Basis |
|---|---|---|---|---|
| 1 | Dataset size | 21,691 | **21,684** | `train 16,781 + test 4,903`. The 21,691 figure is `total_unique_samples` in `source_records_summary.json`; it exceeds `html_metadata_count` (21,684) by exactly the 7 records that carry no HTML metadata and therefore no usable colour target. The evaluated dataset is 21,684. |
| 2 | Ensemble MAE | 38.5 | **38.0** | `metrics.json → tasks.rgb.ensemble.mae = 38.0114`. The 38.5 figure came from `ab_physics.json → rgb.base.mae = 38.563`, an experimental **comparison arm**, not the reported model. |
| 3 | Dispersion within identical chemistry | 47.8 | **41.7** | `noise_analysis.json → identical_chem_dispersion = 41.703`. The 47.8 figure is not reproducible from any artefact or from the raw data under any of the six dispersion definitions tested (mean distance to centroid = 41.70; mean pairwise excluding diagonal = 78.50; pooled within sd = 50.79; √within-variance = 48.13). |
| 4 | Group count in §5.1 | 14,342 (labelled "groups with ≥ 2 samples") | **14,342 distinct signatures; 661 with ≥ 2 samples** | Re-derived from `chemical_composition` in the training recipes. 14,342 is the number of distinct signatures; the within-group variance can only be measured over the 661 groups that contain more than one sample. The two figures were not in conflict — the table label was wrong. |
| 5 | Baseline comparison | "ahead of the dataset authors' own baselines (40.1–40.9)" and "exceeds published baselines" | **"all four single-model baselines evaluated here"** | The 40.1–40.9 range is `mlp 40.17 / xgboost 38.41 / rf 38.48` from `metrics.json` — the study's **own** models, not baselines published by the GlazyBench authors. The claim of superiority over an external baseline was unsupported and has been removed everywhere. |

## Claims verified as correct

Re-derived from source and confirmed — no change needed:

| Claim | Published | Reproduced |
|---|---|---|
| 82.7 % within-chemistry variance | 82.7 % | `variance_decomp.irreducible_fraction = 0.826728` ✓ |
| R² ceiling from chemistry | ≈ 0.173 | `variance_decomp.r2_ceiling_from_chemistry = 0.173272` ✓ |
| ΔR² from physics features | −0.0000 | `ab_physics.delta_r2 = −4.74e−05` ✓ |
| 95 % CI | [−0.0048, +0.0055] | `[−0.004768, +0.005514]` ✓ |
| Base features | 155 | `ab_physics.n_features_base = 155` ✓ |
| Physics features added | 41 | `ab_physics.physics_added = 41` ✓ |
| Total features | 196 | `ab_physics.n_features_phys = 196` ✓ |
| Test set | 4,903 | `metrics.json.dataset.test_n = 4903` ✓ |
| Ensemble R² | 0.348 | `0.3475` ✓ |
| Inter-annotator disagreement | 0.00 | reproduced = 0.0 ✓ |
| Candidate-colour distance | 150.4 (median 148.8, p90 236.8) | reproduced: 150.415 / 148.772 / 236.757 ✓ |
| Annotators undecided | 17.7 % (974 / 5,503) | reproduced: 0.176994 ✓ |
| Curated vs shipped labels | 95.6 % | reproduced: 0.955842 ✓ |

The label-analysis figures (150.4, 0.00, 17.7 %, 95.6 %) were the ones most at
risk of being unverifiable, because `annotations_all.csv` is not redistributed in
this repository. They reproduce exactly from the GlazyBench release using the
script printed in §10 of the paper.

## Authorship

v1.0.0 listed the author as affiliated with **Universidade Federal de São Carlos
(UFSCar)**, with a UFSCar student e-mail. This was incorrect and has been
corrected: the author holds a Materials Engineering degree from the
**Universidade Federal da Paraíba (UFPB)** and did not complete the Master's
programme at UFSCar. Institutional affiliation is a formal declaration, so a
non-enrolled institution was not defensible. The affiliation is now
**UFPB** — the degree actually awarded — and the ORCID field is left empty.
The UFSCar affiliation has been removed from `CITATION.cff`, `.zenodo.json` and
`AUTHORSHIP.md`.

## Reproducing this audit

```bash
python scripts/audit_numbers.py     # re-checks every claim above
```

Any deviation from the table above is a regression.

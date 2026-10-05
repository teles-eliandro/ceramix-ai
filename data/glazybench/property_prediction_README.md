# Property Prediction Benchmark

This subfolder contains the canonical fixed split for the glaze property prediction benchmark.

## Files

- `train/targets.json`
- `train/recipes.json`
- `train/metadata.json`
- `test/targets.json`
- `test/recipes.json`
- `test/metadata.json`
- `test_ids.json`

## Tasks

### Transparency

- Type: 4-class classification
- Labels: `Opaque`, `Semi-opaque`, `Translucent`, `Transparent`

### Surface

- Type: 9-class classification
- Labels: `Glossy`, `Semi-glossy`, `Satin`, `Satin-matte`, `Matte`, `Semi-matte`, `Smooth Matte`, `Dry Matte`, `Stony Matte`

### Color family

- Type: 9-class classification
- Labels: `Black`, `Blue`, `Gray`, `Green`, `Orange`, `Purple`, `Red`, `White`, `Yellow`

### Color RGB

- Type: 3-target regression
- Fields: `color_rgb.r`, `color_rgb.g`, `color_rgb.b`

## Current canonical counts

- Train total: 16,781
- Test total: 4,903
- Train transparency labels: 9,023
- Test transparency labels: 3,322
- Train surface labels: 9,378
- Test surface labels: 3,730
- Train color family labels: 16,781
- Test color family labels: 4,903

## Notes

- This export intentionally uses `targets.json` as the canonical target file.
- In this shared package, `train/recipes.json` and `train/metadata.json` are filtered to the canonical target IDs so the split is internally aligned.
- Historical or intermediate variants are excluded from this package.
- The split is fixed and should not be regenerated when reporting benchmark results.
- `metadata.json` is included for traceability and optional multimodal extensions, but the benchmark labels live in `targets.json`.
- A minimal executable reference is provided in `../baselines/property_prediction_baseline.py`.
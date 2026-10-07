#!/usr/bin/env python3
"""Re-check every quantitative claim in the study against the released artefacts.

Run from the repository root:  python scripts/audit_numbers.py

Exits non-zero if any claim does not reproduce. See docs/AUDIT_v1.0.0.md for the
record of the v1.0.0 -> v1.0.1 corrections.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = lambda f: json.load(open(os.path.join(ROOT, "models", f)))
D = lambda f: json.load(open(os.path.join(ROOT, "data", "glazybench", f)))

failures, checks = [], 0


def check(label, published, actual, tol=None):
    global checks
    checks += 1
    if tol is None:
        ok = published == actual
    else:
        ok = abs(float(published) - float(actual)) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label:<58} published={published}  actual={actual}")
    if not ok:
        failures.append(label)


m = M("metrics.json")
dec = M("metrics_decomposed.json")
ab = M("ab_physics.json")
vd = M("variance_decomp.json")
na = M("noise_analysis.json")
un = M("uncertainty.json")

print("\n== Dataset ==")
train_n, test_n = m["dataset"]["train_n"], m["dataset"]["test_n"]
check("train_n", 16781, train_n)
check("test_n", 4903, test_n)
check("dataset size (train+test)", 21684, train_n + test_n)
src = D("source_records_summary.json")
check("source_records_summary total_unique_samples", 21691, src["total_unique_samples"])
check("source_records_summary html_metadata_count", 21684, src["html_metadata_count"])
check("n_features", 155, m["dataset"]["n_features"])

print("\n== Model ==")
check("ensemble r2", 0.348, m["tasks"]["rgb"]["ensemble"]["r2"], tol=0.001)
check("ensemble mae", 38.0, m["tasks"]["rgb"]["ensemble"]["mae"], tol=0.05)
check("catboost r2", 0.355, m["tasks"]["rgb"]["catboost"]["r2"], tol=0.001)
check("catboost mae", 37.87, m["tasks"]["rgb"]["catboost"]["mae"], tol=0.01)
check("xgboost mae", 38.41, m["tasks"]["rgb"]["xgboost"]["mae"], tol=0.01)
check("rf mae", 38.48, m["tasks"]["rgb"]["rf"]["mae"], tol=0.01)
check("mlp mae", 40.17, m["tasks"]["rgb"]["mlp"]["mae"], tol=0.01)

print("\n== Classification ==")
check("colour family acc", 0.4444, m["tasks"]["color_family"]["ensemble"]["accuracy"], tol=1e-4)
check("surface acc", 0.5464, m["tasks"]["surface"]["ensemble"]["accuracy"], tol=1e-4)
check("transparency acc", 0.5864, m["tasks"]["transparency"]["ensemble"]["accuracy"], tol=1e-4)

print("\n== Physics A/B ==")
check("n_features_base", 155, ab["n_features_base"])
check("physics_added", 41, ab["physics_added"])
check("n_features_phys", 196, ab["n_features_phys"])
check("delta_r2", -0.0000, ab["rgb"]["delta_r2"], tol=0.00005)
check("ci95_low", -0.0048, ab["rgb"]["bootstrap"]["ci95_low"], tol=1e-4)
check("ci95_high", 0.0055, ab["rgb"]["bootstrap"]["ci95_high"], tol=1e-4)

print("\n== Ceiling ==")
check("irreducible fraction (82.7%)", 0.8267, vd["irreducible_fraction"], tol=1e-4)
check("r2 ceiling", 0.173, vd["r2_ceiling_from_chemistry"], tol=0.001)
check("total_variance", 2801.9, vd["total_variance"], tol=0.1)
check("within_group_variance", 2316.4, vd["within_group_variance"], tol=0.1)
check("identical_chem_dispersion", 41.7, na["identical_chem_dispersion"], tol=0.05)
check("identical_chem_groups", 661, na["identical_chem_groups"])

print("\n== Uncertainty ==")
check("conformal coverage R", 0.911, un["channels"]["R"]["coverage_conformal"], tol=1e-3)
check("joint coverage", 0.808, un["joint_coverage"], tol=1e-3)

print(f"\n{checks - len(failures)}/{checks} claims reproduce.")
if failures:
    print("\nFAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All claims in docs/AUDIT_v1.0.0.md reproduce.")

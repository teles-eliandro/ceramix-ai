#!/usr/bin/env python3
"""Fetch GlazyBench and rebuild the deploy bundle from scratch.

Why this exists
---------------
The repository does not ship the GlazyBench data or the derived
`models/deploy_bundle.pkl`. Glazy.org content is licensed CC BY-NC-SA 4.0 and
their AI Use & Data Access Policy requires derived datasets to carry the same
licence and to preserve per-contributor attribution; the bundle additionally
embeds 4,903 complete recipe records, so committing it would be redistribution.

Both artifacts are therefore rebuilt here instead of being version-controlled.
The build command in `render.yaml` calls this script.

What it does
------------
1. Downloads the GlazyBench property-prediction split from HuggingFace into
   `data/glazybench/`, under the canonical filenames the rest of the code expects.
2. Runs `scripts/export_deploy_bundle.py` to train the models and write
   `models/deploy_bundle.pkl`.

If the data is already present and the bundle already exists, it exits early —
so it is safe to run on every deploy.

Licence and attribution
-----------------------
Data: "Data from Glazy.org (CC BY-NC-SA 4.0). Contributors retain copyright."
Benchmark: Zhai, Z., Li, S., Shao, J., & Yu, J. (2026). GlazyBench: A Benchmark
for Ceramic Glaze Property Prediction and Image Generation. arXiv:2605.06641.

Usage
-----
    python scripts/fetch_and_build.py            # fetch if needed, then build
    python scripts/fetch_and_build.py --force    # ignore the bundle cache
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "glazybench"
BUNDLE = ROOT / "models" / "deploy_bundle.pkl"
HF_REPO = "AlpachinoNLP/GlazyBench"

# Canonical names used by src/features.py, mapped to their paths in the HF repo.
SPLITS = {
    "property_prediction_train_recipes.json": "property_prediction/train/recipes.json",
    "property_prediction_train_targets.json": "property_prediction/train/targets.json",
    "property_prediction_test_recipes.json": "property_prediction/test/recipes.json",
    "property_prediction_test_targets.json": "property_prediction/test/targets.json",
}


def _have_data() -> bool:
    return all((DATA / n).exists() for n in SPLITS)


def fetch_data() -> None:
    """Download the benchmark splits from HuggingFace, under canonical names."""
    if _have_data():
        print("[fetch] GlazyBench data already present — skipping download")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[fetch] huggingface_hub missing. Install with:")
        print("        pip install huggingface_hub")
        sys.exit(1)

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] downloading GlazyBench from {HF_REPO} (~30 MB)")
    for local_name, repo_path in SPLITS.items():
        dest = DATA / local_name
        if dest.exists():
            print(f"[fetch]   {local_name} present")
            continue
        try:
            cached = hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                     filename=repo_path)
        except Exception as exc:  # noqa: BLE001 - report and abort clearly
            print(f"[fetch] FAILED on {repo_path}: {exc}")
            print("[fetch] If the file layout changed, list the repo contents at")
            print(f"        https://huggingface.co/datasets/{HF_REPO}/tree/main")
            sys.exit(1)
        shutil.copyfile(cached, dest)
        print(f"[fetch]   {local_name}  ({dest.stat().st_size/1e6:.1f} MB)")

    print("[fetch] done. Data from Glazy.org (CC BY-NC-SA 4.0); contributors retain copyright.")


def build_bundle(force: bool = False) -> None:
    """Train and export the self-contained deploy artifact."""
    if BUNDLE.exists() and not force:
        print(f"[build] {BUNDLE.name} already exists — skipping (use --force to rebuild)")
        return
    script = ROOT / "scripts" / "export_deploy_bundle.py"
    print(f"[build] running {script.relative_to(ROOT)} (takes ~5 min)")
    proc = subprocess.run([sys.executable, "-u", str(script)], cwd=ROOT)
    if proc.returncode != 0:
        print(f"[build] export failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    if not BUNDLE.exists():
        print("[build] export reported success but no bundle was written")
        sys.exit(1)
    print(f"[build] wrote {BUNDLE.name}  ({BUNDLE.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild the bundle even if it already exists")
    ap.add_argument("--fetch-only", action="store_true",
                    help="download the data and stop")
    args = ap.parse_args()

    fetch_data()
    if not args.fetch_only:
        build_bundle(force=args.force)


if __name__ == "__main__":
    main()

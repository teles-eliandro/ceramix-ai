"""
CERAMIX-AI — FastAPI service.

Endpoints
---------
GET  /health                  liveness + model status
GET  /api/meta                dataset + metric metadata (for the UI)
POST /api/predict             recipe -> predicted colour + interval + finish
POST /api/suggest             target colour + finish -> ranked candidate recipes
POST /api/colour              colour conversion helpers (hex <-> rgb <-> lab)

The service loads the trained models once at startup.

Reference: GlazyBench, arXiv:2605.06641 (MIT).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CERAMIX-AI", version="0.1.0",
              description="Uncertainty-calibrated ceramic glaze colour prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {}
STATE["ready"] = False
STATE["error"] = None


# ------------------------------------------------------------------ models --
class RecipeIn(BaseModel):
    """A glaze recipe: oxide wt% and optionally raw materials."""
    oxides: dict[str, float] = Field(default_factory=dict,
                                     description="oxide symbol -> wt%")
    ingredients: list[dict] = Field(default_factory=list,
                                    description="[{material, amount}]")
    cone: str | None = Field(None, description="e.g. '6' or '5-7'")
    atmosphere: str | None = Field(None, description="Oxidation|Reduction|Neutral")


class SuggestIn(BaseModel):
    colour: str = Field(..., description="hex (#RRGGBB), rgb(r,g,b) or 'r,g,b'")
    surface: str | None = Field(None, description="target texture, e.g. Matte")
    transparency: str | None = Field(None, description="Opaque|Translucent|...")
    top_k: int = Field(12, ge=1, le=50)


class ColourIn(BaseModel):
    colour: str


# ------------------------------------------------------------- helpers -----
def _ensure_loaded():
    if "ranker" not in STATE:
        raise HTTPException(503, "models not loaded yet")


@app.on_event("startup")
def _startup():
    try:
        from ranker import RecipeRanker, rgb_to_lab
        STATE["ranker"] = RecipeRanker().train()
        STATE["meta"] = _load_meta()
        STATE["ready"] = True
        print("[ceramix] models loaded")
    except Exception as e:  # pragma: no cover
        STATE["ready"] = False
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[ceramix] model load failed: {e}")


def _load_meta() -> dict:
    out = {"dataset": None, "metrics": None, "decomposed": None, "uncertainty": None}
    for key, fn in (("metrics", "metrics.json"),
                    ("decomposed", "metrics_decomposed.json"),
                    ("uncertainty", "uncertainty.json")):
        p = MODELS / fn
        if p.exists():
            out[key] = json.loads(p.read_text())
    out["dataset"] = {
        "name": "GlazyBench",
        "arxiv": "arXiv:2605.06641",
        "licence": "MIT",
        "repository": "AlpachinoNLP/GlazyBench",
        "records": 21691,
    }
    return out


def _parse_colour(s: str) -> tuple[int, int, int]:
    from ranker import hex_to_rgb
    s = s.strip()
    if s.startswith("#"):
        return hex_to_rgb(s)
    if s.lower().startswith("rgb"):
        nums = "".join(c if (c.isdigit() or c == ",") else " " for c in s)
        parts = [p for p in nums.replace(" ", ",").split(",") if p]
        if len(parts) >= 3:
            return tuple(int(float(p)) for p in parts[:3])  # type: ignore
    if "," in s:
        parts = s.split(",")
        if len(parts) >= 3:
            return tuple(int(float(p.strip())) for p in parts[:3])  # type: ignore
    raise HTTPException(400, f"cannot parse colour: {s!r}. Use #RRGGBB or 'r,g,b'")


def _recipe_frame(recipes: list[RecipeIn]):
    """Turn API recipes into the frame the feature builders expect."""
    import pandas as pd
    rows = []
    for r in recipes:
        rows.append({
            "chemical_composition": r.oxides,
            "ingredients": [{"material": i.get("material"),
                             "amount": i.get("amount")} for i in r.ingredients],
            "cone": r.cone,
            "atmosphere": r.atmosphere,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- routes -----
@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("ready") else "loading",
            "ready": bool(STATE.get("ready")),
            "error": STATE.get("error")}


@app.get("/api/meta")
def meta():
    _ensure_loaded()
    return STATE["meta"]


@app.post("/api/predict")
def predict(recipes: list[RecipeIn]):
    """Predict colour + interval + finish for one or more recipes."""
    _ensure_loaded()
    ranker = STATE["ranker"]
    import pandas as pd
    from features import (firing_features, ingredient_features, oxide_frame,
                          derived_features, umf_features)
    rec = _recipe_frame(recipes)
    ox = oxide_frame(rec.to_dict("records"))
    X = pd.concat([ox, umf_features(ox), derived_features(ox),
                   firing_features(rec), ingredient_features(rec, vocab=ranker.vocab)],
                  axis=1)
    X = X.reindex(columns=ranker.cols, fill_value=0.0).fillna(0)
    P = ranker.predict(X)

    onnx_like = []
    for i in range(len(X)):
        onnx_like.append({
            "rgb": [int(round(v)) for v in P["rgb_median"][i]],
            "hex": f"#{''.join(f'{int(max(0,min(255,round(v)))):02X}' for v in P['rgb_median'][i])}",
            "interval": {
                "low": [int(round(v)) for v in P["rgb_low"][i]],
                "high": [int(round(v)) for v in P["rgb_high"][i]],
                "nominal_coverage": 0.90,
            },
            "surface": str(P["surface"][i]) if "surface" in P else None,
            "transparency": str(P["transparency"][i]) if "transparency" in P else None,
        })
    return {"count": len(onnx_like), "predictions": onnx_like}


@app.post("/api/suggest")
def suggest(body: SuggestIn):
    """Rank real recipes against a target colour + finish."""
    _ensure_loaded()
    import pandas as pd
    ranker = STATE["ranker"]
    tgt = _parse_colour(body.colour)

    from features import build_split
    cand_path = MODELS / "candidate_cache.parquet"
    if cand_path.exists():
        cand = pd.read_parquet(cand_path)
    else:
        Xte, _ = build_split("test", ing_vocab=ranker.vocab)
        cand = Xte.reindex(columns=ranker.cols, fill_value=0.0).fillna(0)
        try:
            cand.to_parquet(cand_path)
        except Exception:
            pass

    res = ranker.rank(tgt, target_surface=body.surface,
                      target_transparency=body.transparency,
                      candidates=cand, top_k=body.top_k)
    from ranker import rgb_to_hex, rgb_to_lab
    return {
        "target": {"hex": rgb_to_hex(tgt), "rgb": list(tgt),
                   "lab": [round(float(v), 2) for v in rgb_to_lab(tgt)]},
        "requested": {"surface": body.surface, "transparency": body.transparency},
        "candidates": res,
        "disclaimer": ("Prediction intervals reflect measured label noise "
                       "(82.7% of colour variance is within-chemistry). "
                       "Candidates are real fired recipes from GlazyBench, "
                       "ranked by distribution overlap — not guaranteed matches."),
    }


@app.post("/api/colour")
def colour(body: ColourIn):
    from ranker import rgb_to_hex, rgb_to_lab
    rgb = _parse_colour(body.colour)
    return {"hex": rgb_to_hex(rgb), "rgb": list(rgb),
            "lab": [round(float(v), 2) for v in rgb_to_lab(rgb)]}


@app.get("/api/surfaces")
def surfaces():
    """The texture classes the model can predict (from GlazyBench)."""
    return {"surfaces": ["Glossy", "Semi-glossy", "Satin", "Satin-matte", "Matte",
                         "Semi-matte", "Smooth Matte", "Dry Matte", "Stony Matte"],
            "transparency": ["Opaque", "Semi-opaque", "Translucent", "Transparent"]}


# ------------------------------------------------------------- frontend ----
@app.get("/")
def index():
    """Serve the single-page UI."""
    f = STATIC / "index.html"
    if not f.exists():
        return {"service": "CERAMIX-AI", "docs": "/docs", "ui": "not built"}
    return FileResponse(f)


# mount last so /api/* keeps priority
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

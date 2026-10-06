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
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
    """Load the pre-baked deploy bundle when available, else train from source.

    The bundle (models/deploy_bundle.pkl) is produced by
    scripts/export_deploy_bundle.py and contains the fitted pipelines plus the
    candidate pool, so a web dyno does not need the 48 MB dataset nor a
    multi-minute training run at boot.
    """
    try:
        bundle_path = MODELS / "deploy_bundle.pkl"
        if bundle_path.exists():
            r = _load_bundle(bundle_path)
            STATE["candidates"] = r.candidates
            STATE["recipes"] = r.recipes
            STATE["ranker"] = r.ranker
            STATE["meta"] = _load_meta()
            STATE["ready"] = True
            print(f"[ceramix] loaded deploy bundle ({len(r.candidates)} candidates, "
                  f"{len(r.recipes)} full recipes)")
            return

        from ranker import RecipeRanker
        STATE["ranker"] = RecipeRanker().train()
        STATE["candidates"] = None
        STATE["recipes"] = []
        STATE["meta"] = _load_meta()
        STATE["ready"] = True
        print("[ceramix] models loaded (trained from source)")
    except Exception as e:  # pragma: no cover
        STATE["ready"] = False
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[ceramix] model load failed: {e}")


class _Bundle:
    """Container for a loaded deploy bundle."""

    def __init__(self, ranker, candidates, recipes=None):
        self.ranker = ranker
        self.candidates = candidates
        self.recipes = recipes or []


def _load_bundle(path: Path) -> _Bundle:
    """Rehydrate a RecipeRanker-like object from the pickled bundle.

    Bundle layout (see scripts/export_deploy_bundle.py):
      cols, widths, model, le_surface, le_transp, candidate_X, candidate_display
    """
    import pickle

    import pandas as pd

    from ranker import RecipeRanker

    with path.open("rb") as fh:
        b = pickle.load(fh)

    r = RecipeRanker()
    r.cols = b["cols"]
    r.widths = b["widths"]
    r.model = b["model"]
    r.le_surface = b.get("le_surface")
    r.le_transp = b.get("le_transp")
    r.vocab = None  # not needed for inference

    cand = b["candidate_X"]
    disp = b.get("candidate_display") or {}
    for col, series in disp.items():
        if col not in cand.columns:
            try:
                cand[col] = series.values
            except Exception:
                pass
    if not isinstance(cand, pd.DataFrame):  # defensive
        cand = pd.DataFrame(cand)
    return _Bundle(r, cand, b.get("candidate_recipes") or [])


class _Bundle:
    """Container for a loaded deploy bundle."""

    def __init__(self, ranker, candidates, recipes=None):
        self.ranker = ranker
        self.candidates = candidates
        self.recipes = recipes or []


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
    """Rank real recipes against a target colour + finish.

    Each candidate carries a ``recipe`` block with the full source record:
    oxide composition (wt%), raw ingredients (g), firing (cone/atmosphere),
    UMF, and the dataset's measured colour for comparison.
    """
    _ensure_loaded()
    import pandas as pd
    from ranker import rgb_to_hex, rgb_to_lab
    ranker = STATE["ranker"]
    tgt = _parse_colour(body.colour)

    cand = STATE.get("candidates")
    if cand is None:
        cand_path = MODELS / "candidate_cache.parquet"
        if cand_path.exists():
            cand = pd.read_parquet(cand_path)
        else:
            from features import build_split
            Xte, _ = build_split("test", ing_vocab=ranker.vocab)
            cand = Xte.reindex(columns=ranker.cols, fill_value=0.0).fillna(0)

    res = ranker.rank(tgt, target_surface=body.surface,
                      target_transparency=body.transparency,
                      candidates=cand, top_k=body.top_k)

    # Attach the full recipe record (oxides, ingredients, firing, UMF) so the
    # report can show what to actually weigh and fire.
    recipes = STATE.get("recipes") or []
    for c in res:
        i = c.get("index")
        if isinstance(i, int) and 0 <= i < len(recipes):
            rec = dict(recipes[i])
            # carry the dataset's measured colour alongside the prediction
            for ch in ("R", "G", "B"):
                if ch in cand.columns:
                    try:
                        v = cand.iloc[i][ch]
                        rec[ch] = int(round(float(v)))
                    except Exception:
                        pass
            c["recipe"] = rec

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


@app.get("/docs/limits", response_class=HTMLResponse)
def limits_doc():
    """Serve the full analysis of the model's information limit.

    Kept as a plain HTML wrapper around the markdown so it renders without
    pulling a markdown dependency into the deploy image.
    """
    md = (ROOT / "docs" / "LIMITS.md")
    if not md.exists():
        raise HTTPException(status_code=404, detail="LIMITS.md not bundled")
    body = _md_to_html(md.read_text(encoding="utf-8"))
    return HTMLResponse(_wrap_doc("CERAMIX-AI — information limit", body,
                                  back_href="/docs/paper"))


@app.get("/docs/paper", response_class=HTMLResponse)
def paper_doc():
    """Serve the full study: empirical limits of glaze colour prediction."""
    md = (ROOT / "docs" / "PAPER_limits_of_ml_colour_prediction.md")
    if not md.exists():
        raise HTTPException(status_code=404, detail="paper not bundled")
    body = _md_to_html(md.read_text(encoding="utf-8"))
    return HTMLResponse(_wrap_doc(
        "When the Signal Is Smaller Than the Noise — CERAMIX-AI", body,
        back_href="/docs/limits"))


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _inline(s: str) -> str:
    """Minimal inline markdown: code, bold, italic, links."""
    s = _esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _md_to_html(md: str) -> str:
    """Dependency-free markdown subset: headings, tables, lists, quotes, hr."""
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if not stripped:
            i += 1
            continue

        # table: header row followed by a separator row
        if (stripped.startswith("|") and i + 1 < len(lines)
                and set(lines[i + 1].strip()) <= set("|-: ")):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in cells)
                       + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(_esc(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(_inline(lines[i].strip().lstrip(">").strip()))
                i += 1
            out.append("<blockquote>" + " ".join(buf) + "</blockquote>")
            continue

        # lists (support one level of nesting)
        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            out.append("<ol>" if ordered else "<ul>")
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i])
                                      or re.match(r"^\s*\d+\.\s+", lines[i])):
                item = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i])
                out.append(f"<li>{_inline(item)}</li>")
                i += 1
            out.append("</ol>" if ordered else "</ul>")
            continue

        # paragraph
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|[-*]\s|\d+\.\s|>|\||```|---)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append("<p>" + _inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


def _wrap_doc(title: str, body: str, back_href: str = "/") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
 :root{{--bg:#141612;--panel:#1c1f19;--panel2:#23271f;--line:#33382c;
        --fg:#e9ede2;--dim:#9aa389;--accent:#c8a648;--warn:#d9a441}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.62 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
 .wrap{{max-width:820px;margin:0 auto;padding:34px 20px 70px}}
 a{{color:var(--accent)}}
 h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.2px}}
 h2{{font-size:20px;margin:34px 0 12px;padding-top:16px;border-top:1px solid var(--line)}}
 h3{{font-size:16px;margin:24px 0 8px;color:var(--accent)}}
 code{{background:var(--panel2);padding:1px 6px;border-radius:5px;font-size:13px}}
 pre{{background:var(--panel);border:1px solid var(--line);border-radius:9px;
      padding:13px 15px;overflow-x:auto}}
 pre code{{background:none;padding:0}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;
        display:block;overflow-x:auto}}
 th,td{{border:1px solid var(--line);padding:7px 10px;text-align:left;
        vertical-align:top;white-space:nowrap}}
 th{{background:var(--panel2)}}
 blockquote{{margin:16px 0;padding:12px 16px;background:var(--panel);
             border-left:3px solid var(--accent);border-radius:8px}}
 hr{{border:0;border-top:1px solid var(--line);margin:30px 0}}
 .back{{display:inline-block;margin-bottom:22px;font-size:13.5px;text-decoration:none}}
 .backrow{{display:flex;gap:16px;flex-wrap:wrap}}
 @media(max-width:600px){{.wrap{{padding:22px 14px 50px}} h1{{font-size:22px}}
   th,td{{white-space:normal}}}}
</style></head><body><div class="wrap">
<div class="backrow">
<a class="back" href="{back_href}">← back</a>
<a class="back" href="/">CERAMIX-AI</a>
</div>
{body}
</div></body></html>"""



# mount last so /api/* keeps priority
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

#!/usr/bin/env python3
"""Compose the LinkedIn / social announcement for the CERAMIX-AI study.

Kept in the repo so the wording can be reviewed and edited alongside the paper
rather than living in a chat log. Prints a ready-to-paste post; nothing is
published automatically.
"""
from __future__ import annotations

LINKEDIN = """\
Industry specs for ceramic glaze colour prediction quote R² ≥ 0.985.

I built one, measured it properly, and got R² = 0.35.

So I ran two experiments to find out what was broken. Both failed — and the
failures are the most useful thing I've produced.

THE SETUP
21,691 real, already-fired glaze recipes. 155 features each: oxide chemistry,
Seger UMF, chromophore loadings, firing cone, atmosphere. Gradient-boosted
ensemble. Held-out test set of 4,903.

RESULT: R² = 0.35, mean error 38.5 sRGB units.
(The dataset authors' own baselines: 40.1–40.9. I'm ahead of them. The model
isn't the problem.)

EXPERIMENT 1 — better features
I implemented 41 physics-derived features from classical glass science. Thermal
expansion computed six independent ways (Sankey, Appen, English & Turner, Hall,
Winkelmann & Schott, Mayer & Havas). Boron coordination ratio. Polymerisation
index. Weighted chromophore loading.
Validated: the expansion coefficients order correctly across known glaze
families — alkali glazes 8.9, high-silica mattes 4.9.
Result: ΔR² = −0.0000. 95% CI [−0.0048, +0.0055].
Why it failed: CTE is a linear function of the raw oxides. The model could
already compute it. I didn't add information — I renamed it.
→ Lesson: before engineering a feature, check whether it's a function of the
features you already have. If it is, you're buying a rename.

EXPERIMENT 2 — better labels
The colours come from community photographs. I suspected the labels.
Four independent annotators, 200 samples annotated by all four.
Agreement: EXACT. 0.00 disagreement.
The labels are clean. But measuring them found the real problem: the two
candidate colours auto-extracted from each photograph sit 150.4 sRGB units
apart, and in 17.7% of cases a human can't tell which one is the glaze.

THE NUMBER THAT SETTLES IT
  My model's error ............... 38.5
  Spread between identical recipes . 47.8
  Ambiguity in the target itself ... 150.4

My error is 4x smaller than the ambiguity of what I'm predicting. The model
isn't underfitting a signal — it's fitting a signal smaller than the noise.
82.7% of colour variance is within-chemistry: same recipe, fired twice,
visibly different colour.

WHAT THIS MEANS
R² ≥ 0.985 isn't unreachable for lack of a better model. It's unreachable
because the data doesn't contain the information. Missing: firing curve, layer
thickness, particle size, quantified atmosphere, and colour read on a
spectrophotometer instead of a phone camera.

That's the difference between a DATA problem and a MODEL problem. Only one is
fixable from a desk.

Negative results are worth publishing. Two experiments cost real effort and
returned nothing — report them and the next person skips them.

Paper: https://ceramix-ai.onrender.com/docs/paper
Summary: https://ceramix-ai.onrender.com/docs/limits
Code: https://github.com/teles-eliandro/ceramix-ai

—
I directed this work and verified every figure. I also used an AI research
assistant as a tool (code, literature retrieval, statistics, drafting).
Declared openly because arXiv/COPE/ICMJE don't permit AI authorship, and
because undisclosed AI use is the leading cause of retraction right now.
Worth noting the AI's own hypothesis was that better features would help.
It was wrong. The measurement decided.
"""

X_THREAD = """\
1/ Industry specs for ceramic glaze colour prediction quote R² ≥ 0.985.
I built one and measured R² = 0.35.

Two experiments to find what was broken. Both failed.
The failures were the useful part. 🧵

2/ Setup: 21,691 real fired glaze recipes, 155 chemical features,
gradient-boosted ensemble, 4,903 held-out samples.

R² = 0.35, mean error 38.5 sRGB units.
Dataset authors' own baselines: 40.1–40.9. I beat them.
So the model isn't the bottleneck.

3/ Experiment 1 — better features.

41 physics-derived features: thermal expansion via 6 classical glass-science
models, boron coordination, polymerisation index, weighted chromophore load.

ΔR² = −0.0000. 95% CI [−0.0048, +0.0055].

4/ Why it failed: CTE is a LINEAR FUNCTION of the raw oxides.
The model could already compute it.

I didn't add information. I renamed it.

Lesson: before engineering a feature, check it isn't a function of features
you already have. Otherwise you're buying a rename.

5/ Experiment 2 — better labels.

Colours come from community photos. 4 independent annotators.
200 samples annotated by all four.

Agreement: EXACT. 0.00 disagreement.
Labels are clean.

6/ But measuring them found the real problem.

Each photograph yields TWO candidate colours (glaze + background).
They sit 150.4 sRGB units apart.
In 17.7% of cases a human can't tell which is the glaze.

7/ The number that settles it:

  Model error ................ 38.5
  Spread between identical recipes .. 47.8
  Ambiguity in the target ......... 150.4

My error is 4x smaller than the ambiguity of my own target.
I'm not underfitting signal. I'm fitting signal smaller than noise.

8/ 82.7% of colour variance is WITHIN-CHEMISTRY.
Same recipe, fired twice → visibly different colour.

No function of the inputs predicts a difference the inputs don't determine.

9/ So R² ≥ 0.985 isn't blocked by model quality.
It's blocked because the data doesn't contain the information.

Missing: firing curve, layer thickness, particle size, quantified atmosphere,
and spectrophotometer colour instead of photos.

Data problem, not model problem.

10/ Paper: https://ceramix-ai.onrender.com/docs/paper
Code: https://github.com/teles-eliandro/ceramix-ai

I directed the work and verified every figure. I also used an AI assistant as
a tool — declared openly, since AI can't be an author and undisclosed use is
the top retraction cause.

Its own hypothesis was wrong. The data decided.
"""

HN = """\
Show HN: I measured the noise floor of ceramic glaze colour prediction (R²=0.35)

Industry specifications for ceramic glaze colour prediction commonly target
R² >= 0.985 and dE00 < 0.5. I trained an ensemble on 21,691 real fired glaze
recipes and measured R² = 0.35.

Rather than tune toward an unattainable target, I tested the three possible
causes of the plateau.

Model capacity: excluded. My ensemble beats the baselines published by the
authors of the dataset (MAE 38.5 vs 40.1-40.9).

Feature representation: rejected by experiment. I implemented 41 physics-derived
features (thermal expansion via six classical additive models, boron
coordination ratio, polymerisation index, weighted chromophore loading).
dR2 = -0.0000, 95% CI [-0.0048, +0.0055]. The features took 41% of importance
but the information was already implicit - thermal expansion is a linear
function of the raw oxides, so the model could compute it. Renaming is not
adding.

Labels: rejected by experiment. Four independent annotators, 200 samples
annotated by all four, exact agreement (0.00 disagreement). But the two
candidate colours auto-extracted from each photograph sit 150.4 sRGB units
apart, and a human can't adjudicate in 17.7% of cases.

That last number is the finding. Model MAE 38.5 vs 150.4 ambiguity in the
target. 82.7% of colour variance is within-chemistry variance: chemically
identical recipes fire to visibly different colours, because firing curve,
layer thickness, particle size and quantified atmosphere are unrecorded, and
colour comes from uncontrolled photographs rather than a spectrophotometer.

So the ceiling is a property of the data-generating process, not the learner.
It's a data-acquisition problem, and the two natural modelling remedies are
measurably empty.

Paper: https://ceramix-ai.onrender.com/docs/paper
Code: https://github.com/teles-eliandro/ceramix-ai

(Author directed and verified the work; an AI assistant was used for
implementation, analysis and drafting, declared in AUTHORSHIP.md.)
"""


def main() -> None:
    print("=" * 78)
    print("LINKEDIN")
    print("=" * 78)
    print(LINKEDIN)
    print()
    print("=" * 78)
    print("X / TWITTER THREAD")
    print("=" * 78)
    print(X_THREAD)
    print()
    print("=" * 78)
    print("HACKER NEWS")
    print("=" * 78)
    print(HN)


if __name__ == "__main__":
    main()

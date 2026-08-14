# Improving SURDS `depth`: converting pass@k into pass@1

**Status:** analysis + method shortlist, nothing launched yet.
**Date:** 2026-08-12
**Eval basis:** heldout SURDS val, 333 `depth` examples, arm `rl_init_cp896` (our 8B SFT
student = SURDS+Mulberry consolidation SFT, the RL warm-start), 16 samples @ T=0.8.
Committed aggregate for arm `full`: `research/eval/heldout_metrics/metrics_aggregate.parquet`.

---

## 1. The task

Estimate an object's distance from the camera and pick one of **3 offered ranges**:

> *"How far is the vertical distance of the white car in the picture from the camera?"*
> Options: `Between 13 and 17 meters` / `Between 7 and 11 meters` / `Between 19 and 25 meters`

Scored by `score_surds.score_one(tt="depth")`: parse all numbers, take the **range midpoint**,
count correct if the prediction is **in-range OR within 4 m of the midpoint**. So it is a
continuous estimate graded with a tolerance band, but *presented* as a 3-way multiple choice —
a distinction that matters a great deal below.

---

## 2. Measured state

| metric | value |
|---|---|
| pass@1 (greedy) | **0.535** |
| pass@2 / pass@4 / pass@8 / pass@16 | 0.682 / 0.802 / 0.892 / 0.934 |
| maj@16 | **0.559** |
| chance floor (3-way) | 0.333 |
| chance-corrected pass@1 — `(acc − 1/3)/(1 − 1/3)` | **0.303** |

Committed arm `full` is slightly higher (pass@1 0.556, pass@8 0.856). The SURDS paper reports
**69.84** on depth vs our **55.6** — depth is the one task where they are genuinely ahead.

**We already beat both teachers on depth** (235B pass@1 0.408, 32B 0.126). So teacher
distillation is *not* a lever here — any improvement must be self-generated or rule-derived.

### 2.1 CAUTION: the pass@k gap on this task is largely an artifact

The obvious read — *"pass@8 0.89 vs pass@1 0.53, so there's +36 pp of latent capability to
harvest with RL"* — **does not survive scrutiny on a 3-way multiple choice.**

- A **uniform 3-way guesser** scores pass@8 = **0.961** and pass@16 = **0.998**.
- Our pass@16 is **0.934 — below the uniform-guess ceiling.**
- The model emits **all 3 distinct options on 47.4%** of questions (mean 2.94 distinct
  answers per 16 samples).

So a large share of "pass@16 success" is just *the model having tried every option*, not a
latent competence waiting to be selected. **Do not size the depth opportunity from pass@k.**
The trustworthy signals are **pass@1 vs the 0.333 floor** (chance-corrected 0.303) and the
error structure in §3.

> This is the same lesson that corrected our `fb` read: on a binary task a coin-flip scores
> pass@8 = 0.996, which is why `orig_thinking`'s "superior" fb pass@8 of 0.910 was really a
> high-entropy near-random sampler (pass@1 0.258). **Always compare pass@k against the
> chance ceiling for that task's option count.**

---

## 3. The real finding: a systematic, distance-dependent UNDER-estimation bias

This is robust, large, and directly actionable.

**Errors are almost perfectly one-sided.** On greedy midpoint error (n=315 parseable):

| statistic | value |
|---|---|
| over-estimates | **0.6 %** |
| under-estimates | **44.4 %** |
| mean signed error | **−3.78 m** |
| median \|error\| | 0.00 m |
| within tolerance (≤4 m) | 0.559 |
| within 2× tolerance (≤8 m) | **0.908** |

The model essentially **never over-estimates**. When wrong it is wrong in one direction, and
91% of predictions land within 8 m — so these are **near-misses to the adjacent (nearer)
range**, not random picks.

**Accuracy collapses monotonically with true distance:**

| true distance | n | greedy accuracy |
|---|---|---|
| < 10 m | 55 | **0.964** |
| 10–20 m | 123 | 0.659 |
| 20–30 m | 67 | 0.328 |
| > 30 m | 70 | **0.286** |

Near-field depth is **solved** (0.96). Beyond 20 m the model is at or below the 0.333 chance
floor. This is textbook monocular **depth foreshortening** — perspective compresses far
distances, and the model reads compressed pixel evidence as "nearer".

**Self-consistency cannot fix it:** maj@16 (0.559) ≈ pass@1 (0.535). The bias dominates the
sample distribution, so voting re-elects it. Same signature as yaw — but with a crucial
difference: this error is a **monotone scalar bias**, not an unstable categorical confusion,
which makes it far more correctable.

**Conclusion: depth is not a "select the right sample" problem, it is a
*calibration* problem — specifically far-field distance compression.**

---

## 4. Methods, ranked by expected value

### T1 — Reference-scale primer *(cheapest; do first)*

Give the model metric anchors and an explicit anti-compression warning, in the same
prompt-primer pattern that worked for yaw (+4.6 pp greedy) and is being tested for fb.

Content: known real-world scales (lane width ≈ 3.5 m, sedan length ≈ 4.5 m, bus ≈ 12 m,
typical lamp-post spacing), a procedure that requires the model to *name a reference object
and count how many fit* between camera and target, plus a calibration line: *"distances beyond
20 m are commonly under-estimated; if your estimate exceeds 20 m, re-check against a reference
object before answering."*

- **Cost:** one ~15 min A/B job, no training. Same harness as `yaw_coord_primer_ab.sh`.
- **Why it may work:** it converts an unconstrained perceptual guess into a countable chain,
  and directly counter-weights the measured one-sided bias.
- **Read the result stratified by the §3 distance bands** — the >20 m rows are the target;
  overall accuracy will under-report the effect.
- **Risk:** unlike the yaw/fb conventions, this is *not* a definitional error the model is
  mis-reading — it is a perceptual bias, so a stated rule may not move it. This is exactly
  what the A/B is for, and it is cheap enough to be worth the information either way.

### T2 — Ground-plane geometric grounding *(highest ceiling; the principled fix)*

For any object resting on the road, depth is **determined** by the image row of its
ground-contact point, given camera intrinsics + a flat-ground assumption:

```
Z = f_y * h_cam / (v_contact - v_horizon)
```

All SURDS images are nuScenes `CAM_FRONT` at a fixed 1600×900 with **known, constant
intrinsics and camera height** — so this is computable, not approximated. Critically, **our
model already emits a `<grounding>` block with per-object image points** (see
`grounding_presence_rate` in the metrics), so the input to this formula is already being
produced.

Two ways to use it, in increasing strength:

1. **Rule-verified depth SFT** — synthesize CoT that reasons *through* the ground-contact
   chain ("contact point at y≈740 → below horizon by … → Z ≈ …"), with the arithmetic
   verified against gold. This teaches a *procedure* whose error does not grow with distance,
   attacking the far-field collapse at its source. Same generator pattern as the planned
   rule-verified yaw SFT, and again **no teacher needed** (we beat both teachers on depth).
2. **Grounding-consistency reward in RL** — reward agreement between the emitted contact
   point and the answered depth. Verifiable and cheap (text-parse only).

- **Cost:** medium (build the derivation + validate intrinsics against nuScenes calibration).
- **Caveat:** breaks for non-ground objects and on slopes; needs a fallback path. Validate the
  formula's own accuracy against gold on the training set **before** generating any traces —
  if the geometric estimate is not itself accurate, this whole line is dead.

### T3 — RL with a dense, asymmetric depth reward

RL is *viable* here even though maj@16 ≈ pass@1, because verifiable-reward RL upweights the
**correct** rollout regardless of whether it was the majority — unlike self-consistency, which
is what maj@16 actually measures. Two design requirements from §3:

- **Dense, not binary:** reward on |midpoint error| (e.g. Gaussian or piecewise-linear), so
  partial improvement produces gradient. A binary tolerance reward wastes the near-miss
  structure (91% within 2× tol) and produces dead all-fail groups on the far bands.
- **Asymmetric:** penalize under-estimation more than over-estimation, to counteract a bias
  that is 44.4% one-way vs 0.6% the other.

Also **stratify or re-weight the curriculum toward >20 m examples** — that is where all the
error mass is, and the current pool skews near-field.

- **Cost:** high (full RL run). **Do it after T1/T2** — RL sharpens what is present; if T2
  installs a better procedure first, RL has something worth sharpening.
- This is also the right home for the pending **SAPO/CISPO** variants: unlike the saturated
  binaries where GRPO and DAPO tied, depth has genuine reward signal to differentiate them.

### T4 — Test-time crop-zoom on far objects

Far objects occupy few pixels; a targeted crop-and-rescale before the depth call restores
resolution exactly where accuracy collapses (>20 m). Attacks a plausible root cause of the
foreshortening, and is training-free.

- **Cost:** medium (inference-loop change, ~2× forward passes on far objects).
- Best treated as a **diagnostic first**: if oracle-cropping far objects lifts accuracy, that
  proves the bottleneck is input resolution rather than reasoning, which then redirects T2/T3.

### T5 — Not worth pursuing

- **Teacher distillation** — we beat both teachers (235B 0.408, 32B 0.126). No signal to
  distill.
- **Self-consistency / majority voting** — maj@16 ≈ pass@1; the bias survives voting.
- **More pass@k chasing** — see §2.1; the metric is not measuring what it appears to.

---

## 5. Recommended sequence

1. **T1 reference-scale primer A/B** — 15 min, cheap information, stratify results by distance band.
2. **T4 crop-zoom oracle probe** — determines whether the far-field collapse is a *resolution*
   or a *reasoning* failure. This finding redirects everything downstream, so it is worth
   running early even though the full method is medium-cost.
3. **T2 ground-plane rule-verified SFT** — the principled fix; validate the geometric estimator
   against gold before generating traces.
4. **T3 dense asymmetric RL** — last, on top of whatever T2 installs; also the venue for
   SAPO/CISPO.

## 6. Evaluation hygiene (applies beyond depth)

- Always report **pass@k against the chance ceiling** for the task's option count
  (`1 − ((k_opts−1)/k_opts)^k`). On 3-way depth, pass@8 = 0.961 is *chance*; on binary fb,
  pass@8 = 0.996 is *chance*.
- Report **chance-corrected accuracy** `(acc − p_chance)/(1 − p_chance)` alongside raw.
- For depth specifically, always report **stratified by true-distance band** — the aggregate
  hides a 0.96 → 0.29 collapse.
- Track **signed** midpoint error, not just |error|; the one-sidedness is the whole diagnosis.

---

## 7. Provenance

- Metrics: `research/eval/heldout_metrics/metrics_aggregate.parquet` (arm `full`, pass@1/8/maj@8).
- 16-sample generations: `$DATA_ROOT/eval_runs/dapo_vs_grpo_gen/rl_init_cp896.parquet`.
- Scoring: `research/eval/score_surds.py::score_one(tt="depth")` — midpoint ±4 m or in-range.
- Related: `research/eval/gen_val_ablation.py` (`--primers`, `--only-templates`) is the harness
  for T1; `slurm_scripts/yaw_coord_primer_ab.sh` and `slurm_scripts/fb_convention_primer_ab.sh`
  are the templates for a primer A/B.

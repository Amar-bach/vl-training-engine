"""Consolidate the GRPO-accuracy-stall diagnosis data into a single committed JSON
that notebooks/grpo_accuracy_stall_diagnosis.ipynb loads (keeps the 64MB
completions.jsonl + scratch artifacts out of the notebook).

Sources:
  * subagent_research/surds-grpo-accuracy-stall/T1_rollout_analysis/T1_raw_numbers.json
  * subagent_research/surds-grpo-accuracy-stall/T4_calibration/T4_raw_numbers.json
  * rl_runs/.../A_grpo_v3/L2_direct/v1-20260617-011220/logging.jsonl   (training trajectory)
  * rl_runs/.../completions.jsonl                                       (curated example rollouts)

Output: notebooks/data/grpo_accuracy_stall_diagnosis_data.json

Run from anywhere:
    python research/notebook_builders/_prep_grpo_accuracy_stall_data.py
"""
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/data4/shasta/amar.amarjyoti/research_data")
SUB = REPO / "subagent_research" / "surds-grpo-accuracy-stall"
RUN = DATA_ROOT / "rl_runs/bakeoff/A_grpo_v3/L2_direct/v1-20260617-011220"
OUT = REPO / "notebooks" / "data" / "grpo_accuracy_stall_diagnosis_data.json"

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_NUM_RE = re.compile(r"-?\d+\.?\d*")

# SURDS xy2d frame constants. Curriculum `solution` gold = absolute PIXELS on the native
# 1600x900 nuScenes frame; model prediction = 0-1000 NORMALISED. Reconcile into pixels
# before the L2 (repo CLAUDE.md "SURDS xy2d coordinate frames").
XY_W, XY_H, XY_TOL_PX = 1600, 900, 50.0


def parse_answer(text):
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def think_snippet(text, n=320):
    """First n chars of the reasoning (after stripping the <think> tag)."""
    t = re.sub(r"</?think>", "", text or "", flags=re.I).strip()
    t = re.sub(r"<answer>.*", "", t, flags=re.S | re.I).strip()
    return (t[:n] + " …") if len(t) > n else t


# --------------------------------------------------------------------------
# 1. T1 + T4 precomputed numbers
# --------------------------------------------------------------------------
t1 = json.loads((SUB / "T1_rollout_analysis/T1_raw_numbers.json").read_text())
t4 = json.loads((SUB / "T4_calibration/T4_raw_numbers.json").read_text())

# --------------------------------------------------------------------------
# 2. Training trajectory from logging.jsonl
# --------------------------------------------------------------------------
traj = {k: [] for k in ["step", "binacc", "densebin", "reward", "kl",
                        "entropy", "clip", "len", "grad"]}
KEYMAP = {
    "binacc": "rewards/SurdsAccuracy/mean", "densebin": "rewards/SurdsDenseBinary/mean",
    "reward": "reward", "kl": "kl", "entropy": "entropy/mean",
    "clip": "clip_ratio/region_mean", "len": "completions/mean_length", "grad": "grad_norm",
}
for line in (RUN / "logging.jsonl").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "rewards/SurdsAccuracy/mean" not in d:
        continue
    gs = d.get("global_step/max_steps", "")
    step = int(str(gs).split("/")[0]) if "/" in str(gs) else d.get("step")
    traj["step"].append(step)
    for k, src in KEYMAP.items():
        traj[k].append(d.get(src))

# --------------------------------------------------------------------------
# 3. Curated example rollouts from completions.jsonl
#    Each row = a stringified list of 16 rollouts for one prompt-group.
# --------------------------------------------------------------------------
def _groups(sols):
    """Yield (start, end) index spans of consecutive identical solutions (=one prompt)."""
    i = 0
    n = len(sols)
    while i < n:
        j = i
        while j < n and sols[j] == sols[i]:
            j += 1
        yield i, j
        i = j


def find_examples(path, want, max_rows=6000):
    """Scan rows; collect one illustrative example per requested kind.

    completions.jsonl fields are native JSON lists of length = prompts*generations
    (e.g. 128 = 8 prompts x 16). Consecutive identical `solution` entries = one prompt.
    """
    import math
    found = {}
    with open(path) as f:
        for li, line in enumerate(f):
            if li > max_rows or len(found) == len(want):
                break
            try:
                row = json.loads(line)
                comps, accs, sols = row["completion"], row["SurdsAccuracy"], row["solution"]
            except Exception:
                continue
            if not isinstance(comps, list) or not isinstance(sols, list):
                continue
            for s, e in _groups(sols):
                gold = str(sols[s])
                gl = gold.lower()
                gc, ga = comps[s:e], accs[s:e]

                # yaw south-bias: gold DIAGONAL, a wrong rollout predicts a SOUTH dir
                if "yaw_southbias" in want and gl in ("northeast", "northwest"):
                    for c, a in zip(gc, ga):
                        pred = (parse_answer(c) or "").lower()
                        if a == 0 and ("southeast" in pred or "southwest" in pred or pred == "south"):
                            found["yaw_southbias"] = {
                                "family": "yaw", "gold": gold, "pred": parse_answer(c),
                                "think": think_snippet(c)}
                            break

                # depth: SAME prompt has BOTH a correct and a wrong rollout
                if "depth_mixed" in want and ("meter" in gl or "between" in gl):
                    cor = next((c for c, a in zip(gc, ga) if a == 1), None)
                    wro = next((c for c, a in zip(gc, ga) if a == 0), None)
                    if cor and wro:
                        found["depth_mixed"] = {
                            "family": "depth", "gold": gold,
                            "correct_pred": parse_answer(cor), "correct_think": think_snippet(cor),
                            "wrong_pred": parse_answer(wro), "wrong_think": think_snippet(wro)}

                # xy2d near-miss: a wrong rollout, scored in the CORRECT PIXEL frame.
                # Curriculum `solution` gold is absolute PIXELS (1600x900); the model
                # prediction is 0-1000 NORMALISED -> rescale pred to px before the L2
                # (the old code compared norm-vs-pixel directly -> bogus ~8x-tol "random
                # miss"; see repo CLAUDE.md "SURDS xy2d coordinate frames").
                if "xy2d_miss" in want and ("[" in gold and "," in gold):
                    gp = re.findall(r"-?\d+\.?\d*", gold)
                    for c, a in zip(gc, ga):
                        pred = parse_answer(c) or ""
                        pp = re.findall(r"-?\d+\.?\d*", pred)
                        if a == 0 and len(gp) >= 2 and len(pp) >= 2:
                            pxx, pyy = float(pp[0]) * XY_W / 1000.0, float(pp[1]) * XY_H / 1000.0
                            d_px = math.hypot(pxx - float(gp[0]), pyy - float(gp[1]))
                            # prefer a genuine near-miss (within 2x tol) to illustrate the
                            # corrected story; fall back to the first wrong rollout.
                            cand = {
                                "family": "xy2d", "gold": gold, "pred": pred,
                                "l2_px": round(d_px, 1), "over_tol": round(d_px / XY_TOL_PX, 2),
                                "think": think_snippet(c)}
                            if d_px <= 2 * XY_TOL_PX or "xy2d_miss" not in found:
                                found["xy2d_miss"] = cand
                            if d_px <= 2 * XY_TOL_PX:
                                break
    return found


examples = find_examples(RUN / "completions.jsonl",
                         want={"yaw_southbias", "depth_mixed", "xy2d_miss"})


# --------------------------------------------------------------------------
# 3b. CORRECTED xy2d near-miss in PIXEL space (overrides the frame-bugged T1 block).
#     T1's analysis_3_nearmiss.xy2d compared 0-1000 norm pred vs ~1600 pixel gold
#     (tol_median 38.5 norm) -> bogus err_over_tol_median 8.78x / 8.9% within 2x tol,
#     i.e. the "xy2d is random" artifact. Rescaling pred to pixels (tol 50px) shows the
#     wrong points are a median 1.98x tol out and ~half the WRONG rollouts (and ~64% of
#     ALL rollouts) sit within 2x tol -> a NEAR-MISS distribution, not random.
# --------------------------------------------------------------------------
def recompute_xy2d_nearmiss_px(path, max_rows=200000):
    import math
    import numpy as np
    errs, n_corr, n_tot = [], 0, 0
    with open(path) as f:
        for li, line in enumerate(f):
            if li > max_rows:
                break
            try:
                row = json.loads(line)
                comps, accs, sols = row["completion"], row["SurdsAccuracy"], row["solution"]
            except Exception:
                continue
            if not isinstance(comps, list):
                continue
            for c, a, s in zip(comps, accs, sols):
                g = str(s)
                gp = _NUM_RE.findall(g)
                if not (g.strip().startswith("[") and len(gp) >= 2):
                    continue  # xy2d gold only (yaw/depth excluded)
                n_tot += 1
                if a == 1:
                    n_corr += 1
                    continue
                pp = _NUM_RE.findall(parse_answer(c) or "")
                if len(pp) < 2:
                    continue
                pxx, pyy = float(pp[0]) * XY_W / 1000.0, float(pp[1]) * XY_H / 1000.0
                errs.append(math.hypot(pxx - float(gp[0]), pyy - float(gp[1])))
    e = np.array(sorted(errs)) if errs else np.array([0.0])
    return {
        "n": int(len(errs)),
        "frame": "pixels_1600x900",
        "err_median": round(float(np.median(e)), 1),
        "err_p25": round(float(np.percentile(e, 25)), 1),
        "err_p75": round(float(np.percentile(e, 75)), 1),
        "tol_median": XY_TOL_PX,
        "frac_within_tol": round(float((e <= XY_TOL_PX).mean()), 4),
        "frac_within_2x_tol": round(float((e <= 2 * XY_TOL_PX).mean()), 4),
        "err_over_tol_median": round(float(np.median(e) / XY_TOL_PX), 2),
        "err_over_tol_p75": round(float(np.percentile(e, 75) / XY_TOL_PX), 2),
        "err_over_tol_p95": round(float(np.percentile(e, 95) / XY_TOL_PX), 2),
        # whole-rollout-pool context (correct + within-2x-of-wrong)
        "student_acc_within_tol": round(n_corr / n_tot, 3) if n_tot else None,
        "frac_all_within_2x_tol": round(
            (n_corr + int((e <= 2 * XY_TOL_PX).sum())) / n_tot, 3) if n_tot else None,
        "n_rollouts_total": int(n_tot),
    }


_xy_px = recompute_xy2d_nearmiss_px(RUN / "completions.jsonl")
t1["analysis_3_nearmiss"]["xy2d_norm_BUGGED"] = t1["analysis_3_nearmiss"].get("xy2d")
t1["analysis_3_nearmiss"]["xy2d"] = _xy_px
print(f"  xy2d near-miss recomputed in PIXEL frame: median {_xy_px['err_over_tol_median']}x tol, "
      f"{_xy_px['frac_within_2x_tol']*100:.0f}% of wrong within 2x, student "
      f"{_xy_px['student_acc_within_tol']*100:.0f}% correct")

# --------------------------------------------------------------------------
# 4. RFT-yield numbers (from T6 red-team) — embedded constants
# --------------------------------------------------------------------------
rft_yield = {
    "header": ["family", "prompts_with_correct", "prompts_total", "unique_correct_cots"],
    "rows": [
        ["yaw", 363, 430, 2934],
        ["fb", 163, 206, 1417],
        ["depth+distance", 215, 224, 1591],
        ["xy2d", 16, 20, 202],
        ["lr", 10, 21, 101],
    ],
    "note": "yaw correct rollouts are south-bias luck (NE 7.7% / NW 10.7% per-class, below 12.5% chance) — RFT on them would entrench the bias. yaw+fb = ~70% of harvest.",
}

root_causes = [
    ["yaw perception capability gap (axis mirror-flip)", 45,
     "yaw (~46% of L2) is an axis mirror-flip ambiguity (E->W 50%, NE->SE 65%): the model "
     "gets the orientation axis, wrong sign. RL only sharpens present capability, so it "
     "reshuffles a wrong perceptual prior -> flat yaw acc + 40x KL drift. This is the real "
     "unsharpenable family and it dominates the average."],
    ["xy2d is a CONSOLIDATION target, not a capability gap", 20,
     "Corrected (was 'capability absent / reward inactive'): in PIXEL space wrong points are "
     "a median 1.98x tol out (not 8.8x) and ~64% of rollouts fall within 2x tol = NEAR-MISS; "
     "student greedy ~30%, the 235B teacher is 76%, and the Gaussian dense reward (sigma~50px) "
     "is ACTIVE at ~100px error. xy2d belongs with depth/distance (latent, distillable), not "
     "with yaw. The earlier 'random / reward de-facto inactive' read was a coord-FRAME bug."],
    ["Credit dilution + long-CoT attention tax", 15,
     "Outcome reward spread over long CoT; long CoT reduces visual-token attention."],
    ["On-policy / exploration limits", 12,
     "num_iterations=1, clip inert, temp-1.0-only exploration -> no pressure to discover new grounding."],
    ["Data/band weighting (yaw=46%)", 8,
     "Aggregate dominated by the unsharpenable yaw family; depth +8.3pp / distance +4.7pp gains "
     "get averaged away. Down-weighting yaw surfaces the real per-family progress."],
]

solution_plan = [
    ["1", "GATE: frozen-SFT per-template pass@1 vs pass@16 (held-out, greedy+n16, cardinal/diagonal yaw split, leakage check, re-baseline vs greedy SFT)",
     "discriminate exploitable vs capability-capped", "~1 GPU-hr", "near-free, do first"],
    ["2", "Teacher-distillation consolidation SFT for xy2d + yaw: harvest 235B-teacher-CORRECT "
          "pixel-frame xy2d (76%) and class-balanced yaw traces, continue-SFT from B1 (job pretrain_model_35)",
     "lift xy2d near-miss -> hit (teacher 76% >> student 30%) and rebalance yaw classes", "~1 day 8xGPU", "highest-ROI move (in flight)"],
    ["3", "RFT self-distillation for depth/distance/fb (latent, pass@16>>pass@1)",
     "consolidate present capability cheaply", "~4-6 GPU-hr", "supporting"],
    ["4", "Curriculum re-weight: down-weight yaw so depth/distance/xy2d gains aren't averaged away",
     "efficiency / stop yaw averaging-down", "free split", "supporting"],
    ["5", "yaw perception track: audit teacher diagonal-yaw (NE/NW flip is shared), crop-zoom tool-use, try Instruct base",
     "raise the yaw CEILING (the one genuine capability gap)", "8-16 GPU-hr", "separate track"],
]

# --------------------------------------------------------------------------
# 5. Write consolidated JSON
# --------------------------------------------------------------------------
payload = {
    "meta": {
        "run": "A_grpo_v3 L2-direct (job 1064116)",
        "adapter": str(RUN / "checkpoint-710"),
        "config": "GRPO, Qwen3-VL-8B-Thinking SFT, LoRA r128, 16 gen/prompt, beta=0.01, lr 5e-6, "
                  "reward=1*binary+0.20*Gaussian-dense, L2 band 710 steps",
        "greedy_sft_baseline": 0.6583,
        "overall_hot_acc": t1["metadata"]["overall_acc"],
        "template_counts": t1["metadata"]["template_counts"],
    },
    "t1_trajectory": t1["analysis_1_trajectory"],
    "t1_structure": t1["analysis_2_structure"],
    "t1_nearmiss": t1.get("analysis_3_nearmiss", {}),
    "t1_group_variance": t1.get("analysis_4_group_variance", t1.get("analysis_4", {})),
    "t4_histograms": t4["histograms"],
    "t4_yaw": t4["yaw_analysis"],
    "training_trajectory": traj,
    "examples": examples,
    "rft_yield": rft_yield,
    "root_causes": root_causes,
    "solution_plan": solution_plan,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"  trajectory points: {len(traj['step'])}")
print(f"  examples found: {list(examples.keys())}")

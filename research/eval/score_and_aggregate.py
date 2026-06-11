#!/usr/bin/env python
"""
score_and_aggregate.py — score the SURDS×Mulberry ablation generations and
aggregate per-arm / per-(arm × template_type) metrics for the conference paper.

WHAT IT DOES
------------
1. Auto-discovers every ``<arm>.parquet`` under ``--eval-dir`` (globbing — arms are
   NOT hardcoded, so an 8th ``full`` arm appearing later is picked up automatically).
2. Joins each arm to ``val_meta.parquet`` on ``idx``.
3. For every (arm, idx) scores BOTH the greedy generation and each of the 8 samples
   with ``score_surds.score_one`` (answer extracted via ``parse_answer``;
   ``template_type`` + ``image_wh`` supplied from val_meta / ``get_image_wh``).
4. Writes:
     - metrics_per_question.parquet  (one row per (arm, idx))
     - metrics_aggregate.parquet     (per arm AND per arm × template_type)
     - metrics_summary.json          (tidy nested dict, same numbers)
   and prints a compact summary table.

PASS@K ESTIMATOR
----------------
Unbiased HumanEval estimator. For a question with n samples and c correct,
    pass@k = 1 - C(n-c, k) / C(n, k)   (and 1.0 if c==0? no: 0.0 if c==0; 1.0 if c>=... )
implemented numerically-stably as
    pass@k = 1 - prod_{i = n-c+1 .. n} (1 - k / i)
(==0 when c==0, ==1 when k > n-c). Averaged over questions per cell.

MAJ@8 (self-consistency)
------------------------
- categorical templates: majority vote over the 8 sample answers, normalized via
  score_surds._canon_categorical; the modal canonical token's representative raw
  answer is then scored against gold with score_one.
- continuous templates (xy2d, depth): majority vote is ill-defined, so we take the
  *median* of the parsed numeric predictions (component-wise median point for xy2d;
  median of the range-midpoints for depth), format it back, and score that with
  score_one. Documented in metrics_summary.json under ``maj8_continuous_method``.

ROBUSTNESS
----------
- a missing/empty arm parquet is skipped with a warning.
- truncated / unparseable generations score as incorrect; parse-fail rate tracked.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

# import the scoring module that sits next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_surds as ss  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EVAL_DIR = "/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs/ablation_val1k"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAL_META = os.path.join(HERE, "val_meta.parquet")
BASELINE_ARM = "baseline_thinking"

# ordered template_type slugs (used for stable output ordering)
TEMPLATE_ORDER = ["lr", "distance", "fb", "yaw", "xy2d", "depth"]

# regexes for reasoning-trace features
_GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.S | re.I)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_OBJ_RE = re.compile(r"<obj\d+>", re.I)


# ---------------------------------------------------------------------------
# pass@k — unbiased HumanEval estimator (numerically stable)
# ---------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased estimator: probability >=1 of k drawn (without replacement) is correct.

    pass@k = 1 - C(n-c, k) / C(n, k)  ==  1 - prod_{i=n-c+1..n} (1 - k/i).
    Returns 0.0 if c==0; 1.0 if (n-c) < k (i.e. impossible to draw all-wrong).
    """
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # stable product form
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


# ---------------------------------------------------------------------------
# Reasoning-trace features (computed on the greedy generation)
# ---------------------------------------------------------------------------
def trace_features(text):
    """Return think_char_len, think_word_len, n_grounding_objs, has_grounding.

    Robust to the Qwen3-VL-Thinking eval setup where the chat template SEEDS the
    opening ``<think>`` in the prompt, so the *generation* starts mid-reasoning with
    no opening tag (it still emits ``</think><answer>``). We therefore define the
    reasoning span as everything before ``</think>`` or ``<answer>`` (whichever comes
    first), after removing any ``<grounding>`` block and stray think tags — so
    reasoning length is comparable across thinking and instruct arms.
    """
    if not text:
        return 0, 0, 0, False
    # grounding objects (inside a <grounding> block, else anywhere inline)
    mg = _GROUNDING_RE.search(text)
    if mg:
        n_objs = len(_OBJ_RE.findall(mg.group(1)))
        has_grounding = True
    else:
        n_objs = len(_OBJ_RE.findall(text))
        has_grounding = n_objs > 0
    # reasoning span
    mt = _THINK_RE.search(text)
    if mt:
        think = mt.group(1)
    else:
        body = text
        ma = re.search(r"<answer>", body, re.I)
        if ma:
            body = body[:ma.start()]
        mte = re.search(r"</think>", body, re.I)
        if mte:
            body = body[:mte.start()]
        body = _GROUNDING_RE.sub("", body)            # don't count grounding as reasoning
        body = re.sub(r"</?think>", "", body, flags=re.I)
        think = body
    return len(think), len(think.split()), n_objs, has_grounding


# ---------------------------------------------------------------------------
# Majority-vote answer (maj@8)
# ---------------------------------------------------------------------------
def majority_answer(sample_answers, template_type):
    """Compute the consensus answer string over the 8 sample <answer> bodies.

    categorical -> modal canonical token (return a representative raw answer).
    continuous  -> median numeric (xy2d: component-wise median point;
                   depth: median of midpoints, formatted as a 'X meters' scalar).
    Returns None if nothing parseable.
    """
    tt = (template_type or "").strip().lower()
    answers = [a for a in sample_answers if a is not None and str(a).strip() != ""]
    if not answers:
        return None

    if tt in ss.CONTINUOUS_TEMPLATES:
        if tt == "xy2d":
            pts = [ss._parse_point(a) for a in answers]
            pts = [p for p in pts if p is not None]
            if not pts:
                return None
            xs = np.median([p[0] for p in pts])
            ys = np.median([p[1] for p in pts])
            return f"[{xs:.1f}, {ys:.1f}]"
        else:  # depth
            mids = [ss._depth_mid(a) for a in answers]
            mids = [m for m in mids if m is not None]
            if not mids:
                return None
            return f"{float(np.median(mids)):.2f} meters"

    # categorical: modal canonical token, return a representative raw answer
    canon = [ss._canon_categorical(a) for a in answers]
    pairs = [(c, a) for c, a in zip(canon, answers) if c]
    if not pairs:
        return answers[0]
    counts = Counter(c for c, _ in pairs)
    top_canon, _ = counts.most_common(1)[0]
    for c, a in pairs:
        if c == top_canon:
            return a
    return answers[0]


# ---------------------------------------------------------------------------
# Per-arm scoring
# ---------------------------------------------------------------------------
def score_arm(df, meta):
    """Score one arm's parquet (already joined to meta). Returns per-question rows."""
    rows = []
    for _, r in df.iterrows():
        tt = r["template_type"]
        gold = r["gold_answer"]
        img_path = r["image_path_meta"] if "image_path_meta" in r and pd.notna(r["image_path_meta"]) else r.get("image_path")
        image_wh = None
        if tt in ss.CONTINUOUS_TEMPLATES and tt == "xy2d" and img_path:
            try:
                image_wh = ss.get_image_wh(img_path)
            except Exception:
                image_wh = None

        # --- greedy ---
        greedy_text = r.get("greedy_text")
        g_ans = ss.parse_answer(greedy_text)
        g_res = ss.score_one(g_ans, gold, tt, image_wh=image_wh)
        greedy_correct = bool(g_res["correct"])
        answer_parse_ok = bool(g_res["parse_ok"])

        # --- samples ---
        samples = r.get("samples")
        if samples is None or (isinstance(samples, float) and pd.isna(samples)):
            samples = []
        samples = list(samples)
        sample_answers = [ss.parse_answer(s) for s in samples]
        sample_results = [ss.score_one(a, gold, tt, image_wh=image_wh) for a in sample_answers]
        n_samples = len(sample_results)
        n_correct = sum(1 for res in sample_results if res["correct"])
        n_parse_fail = sum(1 for res in sample_results if not res["parse_ok"])

        # --- maj@8 ---
        maj_ans = majority_answer(sample_answers, tt)
        maj_res = ss.score_one(maj_ans, gold, tt, image_wh=image_wh)
        maj_correct = bool(maj_res["correct"])

        # --- reasoning features (greedy trace) ---
        tcl, twl, n_obj, has_g = trace_features(greedy_text)

        rows.append(dict(
            arm=r["arm"],
            idx=int(r["idx"]),
            template_type=tt,
            answer_kind=r["answer_kind"],
            greedy_correct=greedy_correct,
            answer_parse_ok=answer_parse_ok,
            n_samples=n_samples,
            n_samples_correct=int(n_correct),
            n_sample_parse_fail=int(n_parse_fail),
            maj_answer=maj_ans,
            maj_correct=maj_correct,
            think_char_len=tcl,
            think_word_len=twl,
            n_grounding_objs=n_obj,
            has_grounding=has_g,
        ))
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_cell(g):
    """Aggregate metrics for a group of per-question rows (one arm or arm×template)."""
    n = len(g)
    out = {"n": int(n)}
    if n == 0:
        return out

    # pass@1 (greedy) and sampled accuracy
    out["pass@1"] = float(g["greedy_correct"].mean())
    # per-sample mean accuracy (across the 8), averaged over questions
    with np.errstate(invalid="ignore"):
        per_q_sampled = np.where(g["n_samples"] > 0,
                                 g["n_samples_correct"] / g["n_samples"].replace(0, np.nan),
                                 np.nan)
    out["pass@1_sampled"] = float(np.nanmean(per_q_sampled)) if np.isfinite(np.nanmean(per_q_sampled)) else 0.0

    # pass@k for k=1..8 (use the max available sample count, capped at 8)
    nmax = int(g["n_samples"].max()) if n else 0
    nmax = min(nmax, 8)
    for k in range(1, 9):
        if k > nmax or nmax == 0:
            out[f"pass@{k}"] = float("nan")
            continue
        vals = []
        for _, r in g.iterrows():
            ns = int(r["n_samples"])
            if ns < k:
                continue
            vals.append(pass_at_k(ns, int(r["n_samples_correct"]), k))
        out[f"pass@{k}"] = float(np.mean(vals)) if vals else float("nan")

    # maj@8 (self-consistency)
    out["maj@8"] = float(g["maj_correct"].mean())

    # parse-fail rates
    out["greedy_parse_fail_rate"] = float(1.0 - g["answer_parse_ok"].mean())
    tot_samp = int(g["n_samples"].sum())
    out["sample_parse_fail_rate"] = float(g["n_sample_parse_fail"].sum() / tot_samp) if tot_samp else float("nan")

    # reasoning aggregates (greedy trace)
    out["mean_think_char_len"] = float(g["think_char_len"].mean())
    out["mean_think_word_len"] = float(g["think_word_len"].mean())
    out["mean_n_grounding_objs"] = float(g["n_grounding_objs"].mean())
    out["grounding_presence_rate"] = float(g["has_grounding"].mean())

    # correctness vs think-length breakdown
    corr = g[g["greedy_correct"]]
    inc = g[~g["greedy_correct"]]
    out["mean_think_word_len_correct"] = float(corr["think_word_len"].mean()) if len(corr) else float("nan")
    out["mean_think_word_len_incorrect"] = float(inc["think_word_len"].mean()) if len(inc) else float("nan")
    return out


def build_aggregate(per_q):
    """Return a long-form aggregate DataFrame: rows for each arm (template_type='ALL')
    and each (arm × template_type)."""
    recs = []
    for arm, g_arm in per_q.groupby("arm"):
        agg = aggregate_cell(g_arm)
        agg.update(arm=arm, template_type="ALL")
        recs.append(agg)
        for tt, g_tt in g_arm.groupby("template_type"):
            agg_tt = aggregate_cell(g_tt)
            agg_tt.update(arm=arm, template_type=tt)
            recs.append(agg_tt)
    df = pd.DataFrame(recs)
    # column order
    front = ["arm", "template_type", "n"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def add_deltas(agg, baseline_arm=BASELINE_ARM):
    """Add Δ-vs-baseline columns for every numeric metric, per (arm, template_type)."""
    metric_cols = [c for c in agg.columns if c not in ("arm", "template_type", "n")]
    base = agg[agg["arm"] == baseline_arm].set_index("template_type")
    if base.empty:
        warnings.warn(f"baseline arm '{baseline_arm}' not found; skipping Δ columns")
        return agg
    delta_rows = []
    for _, r in agg.iterrows():
        d = {}
        tt = r["template_type"]
        if tt in base.index and r["arm"] != baseline_arm:
            for m in metric_cols:
                bval = base.loc[tt, m]
                d[f"delta_{m}"] = float(r[m] - bval) if pd.notna(r[m]) and pd.notna(bval) else float("nan")
        else:
            for m in metric_cols:
                d[f"delta_{m}"] = float("nan")
        delta_rows.append(d)
    return pd.concat([agg.reset_index(drop=True), pd.DataFrame(delta_rows)], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                    help="dir containing <arm>.parquet generation files")
    ap.add_argument("--glob", default="*.parquet",
                    help="glob (within --eval-dir) for arm parquets")
    ap.add_argument("--val-meta", default=DEFAULT_VAL_META,
                    help="val_meta.parquet with idx/template_type/gold_answer/etc")
    ap.add_argument("--out-dir", default=HERE,
                    help="where to write metrics_*.parquet / metrics_summary.json")
    ap.add_argument("--baseline-arm", default=BASELINE_ARM)
    args = ap.parse_args()

    meta = pd.read_parquet(args.val_meta)
    # rename meta image_path to avoid colliding with the generation's image_path
    meta = meta.rename(columns={"image_path": "image_path_meta"})

    pattern = os.path.join(args.eval_dir, args.glob)
    files = sorted(glob.glob(pattern))
    # exclude *_meta.json siblings & any non-arm parquet by name convention is unnecessary
    files = [f for f in files if f.endswith(".parquet")]
    if not files:
        print(f"[WARN] no arm parquets matched {pattern}", file=sys.stderr)
        sys.exit(2)

    print(f"[score_and_aggregate] discovered {len(files)} arm parquet(s) in {args.eval_dir}")
    all_rows = []
    for f in files:
        arm = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            warnings.warn(f"arm '{arm}': failed to read ({e}); skipping")
            continue
        if df.empty:
            warnings.warn(f"arm '{arm}': empty parquet; skipping")
            continue
        if "arm" not in df.columns:
            df["arm"] = arm
        merged = df.merge(meta, on="idx", how="inner")
        if merged.empty:
            warnings.warn(f"arm '{arm}': no idx overlap with val_meta; skipping")
            continue
        if len(merged) < len(df):
            warnings.warn(f"arm '{arm}': {len(df)-len(merged)} rows dropped (no meta match)")
        print(f"  - {arm:18s} {len(merged):>5d} questions")
        all_rows.extend(score_arm(merged, meta))

    if not all_rows:
        print("[ERROR] no rows scored; aborting", file=sys.stderr)
        sys.exit(2)

    per_q = pd.DataFrame(all_rows)
    per_q_path = os.path.join(args.out_dir, "metrics_per_question.parquet")
    per_q.to_parquet(per_q_path, index=False)

    agg = build_aggregate(per_q)
    agg = add_deltas(agg, baseline_arm=args.baseline_arm)
    agg_path = os.path.join(args.out_dir, "metrics_aggregate.parquet")
    agg.to_parquet(agg_path, index=False)

    # tidy nested summary json
    summary = {
        "baseline_arm": args.baseline_arm,
        "n_arms": int(per_q["arm"].nunique()),
        "arms": sorted(per_q["arm"].unique().tolist()),
        "maj8_continuous_method": (
            "continuous templates (xy2d, depth) have no well-defined modal answer; "
            "maj@8 uses the component-wise median predicted point (xy2d) / median of "
            "range-midpoints (depth), re-scored with score_surds.score_one."
        ),
        "pass_at_k_estimator": "1 - C(n-c, k)/C(n, k) = 1 - prod_{i=n-c+1..n}(1 - k/i), averaged over questions",
        "by_arm": {},
    }
    for arm, g in agg.groupby("arm"):
        summary["by_arm"][arm] = {}
        for _, r in g.iterrows():
            cell = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                    for k, v in r.items() if k not in ("arm", "template_type")}
            summary["by_arm"][arm][r["template_type"]] = cell
    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)

    # ---- compact summary table ----
    print("\n=== OVERALL (template_type=ALL) ===")
    overall = agg[agg["template_type"] == "ALL"].copy()
    show_cols = ["arm", "n", "pass@1", "pass@1_sampled", "pass@8", "maj@8",
                 "grounding_presence_rate", "mean_think_word_len",
                 "greedy_parse_fail_rate"]
    show_cols = [c for c in show_cols if c in overall.columns]
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}", "display.width", 200):
        print(overall[show_cols].to_string(index=False))

    print("\n=== Δ vs baseline (pass@1, template_type=ALL) ===")
    if "delta_pass@1" in overall.columns:
        d = overall[overall["arm"] != args.baseline_arm][["arm", "delta_pass@1", "delta_pass@8", "delta_maj@8"]]
        with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 200):
            print(d.to_string(index=False))

    print(f"\nwrote:\n  {per_q_path}\n  {agg_path}\n  {summary_path}")


if __name__ == "__main__":
    main()

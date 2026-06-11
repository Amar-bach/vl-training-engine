#!/usr/bin/env python3
"""
compare_stage_c.py — score the Stage-C enrichment SFT checkpoint and report its
Δ vs the Stage-B1 winner (``baseline_thinking``) on BOTH SURDS eval sets used for
the Mulberry ablation: in-distribution ``ablation_val1k`` and ``heldout_surdsval``.

Prereq: generate ``stage_c.parquet`` in BOTH eval dirs first (see stage_c_eval_README.md):
    eval_runs/ablation_val1k/stage_c.parquet      (val = sft_stageB/val_1k.jsonl)
    eval_runs/heldout_surdsval/stage_c.parquet    (val = heldout_surdsval.jsonl)

This driver re-runs score_and_aggregate (which auto-discovers every <arm>.parquet and
adds Δ-vs-baseline columns) on each eval dir with its matching val_meta, then prints a
compact STAGE-C-vs-BASELINE_THINKING overall table for each set. It does not submit jobs.

Usage:
    python compare_stage_c.py            # both sets
    python compare_stage_c.py --only indist|heldout
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs"

SETS = {
    "indist": {
        "eval_dir": os.path.join(DATA, "ablation_val1k"),
        "val_meta": os.path.join(HERE, "val_meta.parquet"),
        "out_dir":  os.path.join(HERE, "indist_metrics"),
        "label":    "IN-DISTRIBUTION (ablation_val1k)",
    },
    "heldout": {
        "eval_dir": os.path.join(DATA, "heldout_surdsval"),
        "val_meta": os.path.join(HERE, "heldout_val_meta.parquet"),
        "out_dir":  os.path.join(HERE, "heldout_metrics"),
        "label":    "HELD-OUT (heldout_surdsval)",
    },
}

# Stage-C's scientific question is "did Stage-C enrichment beat the Stage-B1 winner?",
# so we contrast against baseline_thinking (= SURDS-SFT Thinking = the Stage-B1 winner).
# NOTE: the global aggregation Δ reference is now the zero-shot orig_thinking, so we compute
# the Stage-C-vs-winner contrast directly from raw metrics here rather than from delta_* fields.
ARM = "stage_c"
BASELINE = "baseline_thinking"   # the Stage-B1 winner (relabeled "SURDS-SFT (Thinking)")


def run_aggregate(cfg):
    stage_c_pq = os.path.join(cfg["eval_dir"], f"{ARM}.parquet")
    if not os.path.exists(stage_c_pq):
        print(f"[skip] {cfg['label']}: missing {stage_c_pq} "
              f"(generate it first — see stage_c_eval_README.md)", file=sys.stderr)
        return None
    cmd = [
        sys.executable, os.path.join(HERE, "score_and_aggregate.py"),
        "--eval-dir", cfg["eval_dir"],
        "--val-meta", cfg["val_meta"],
        "--out-dir",  cfg["out_dir"],
        "--baseline-arm", BASELINE,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return os.path.join(cfg["out_dir"], "metrics_summary.json")


def overall_row(summary, arm):
    """Pull the template_type=ALL row for one arm: summary['by_arm'][arm]['ALL']."""
    return summary.get("by_arm", {}).get(arm, {}).get("ALL")


def per_template(summary, arm):
    """{template_type: row} for one arm, excluding ALL."""
    d = summary.get("by_arm", {}).get(arm, {})
    return {tt: row for tt, row in d.items() if tt != "ALL"}


def report(cfg, summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    base = overall_row(summary, BASELINE)
    sc = overall_row(summary, ARM)
    print(f"\n================  {cfg['label']}  ================")
    if base is None or sc is None:
        print(f"  could not locate overall rows (base={base is not None}, "
              f"stage_c={sc is not None}); inspect {summary_path}")
        return
    # Contrast Stage-C against the Stage-B1 winner directly from raw metrics (the global
    # delta_* fields now reference the zero-shot orig_thinking, not this winner).
    metrics = ["pass@1", "pass@1_sampled", "pass@8", "maj@8"]
    print(f"  OVERALL (all 6 template families pooled)  —  Stage-C vs Stage-B1 winner")
    print(f"  {'metric':18s} {'SURDS-SFT(Thinking)':>20s} {'stage_c':>10s} {'Δ (pts)':>10s}")
    for m in metrics:
        b = base.get(m); s = sc.get(m)
        if b is None or s is None:
            continue
        print(f"  {m:18s} {100*b:20.2f} {100*s:10.2f} {100*(s-b):+10.2f}")
    # per-template-family greedy pass@1 delta
    bt = per_template(summary, BASELINE); st = per_template(summary, ARM)
    print(f"\n  per-family greedy pass@1 (baseline -> stage_c, Δ pts):")
    for tt in ["lr", "distance", "fb", "yaw", "xy2d", "depth"]:
        if tt in bt and tt in st:
            b = bt[tt].get("pass@1"); s = st[tt].get("pass@1")
            if b is not None and s is not None:
                print(f"    {tt:10s} {100*b:7.2f} -> {100*s:7.2f}  ({100*(s-b):+.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["indist", "heldout"], default=None)
    args = ap.parse_args()
    keys = [args.only] if args.only else ["indist", "heldout"]
    for k in keys:
        cfg = SETS[k]
        sp = run_aggregate(cfg)
        if sp:
            report(cfg, sp)
    print("\nNote: the per-(arm × template_type) breakdown and full Δ columns are in each "
          "out-dir's metrics_aggregate.parquet / metrics_summary.json.")


if __name__ == "__main__":
    main()

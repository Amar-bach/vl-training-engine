#!/usr/bin/env python
"""
log_offline_eval_wandb.py — push the OFFLINE per-checkpoint SURDS held-out eval
(yaw / xy2d / overall answer accuracy) into the EXISTING SFT wandb run, so the
"did the yaw+xy2d consolidation SFT close the gap" question is answered right
next to the training curves.

WHY THIS EXISTS
---------------
The SFT run (job 1064290, wandb run 5qmwzryf, entity samarjyo) finished BEFORE the
sample-trace callback was patched to score real answer accuracy, so it only logged
eval/loss + eval/token_acc (teacher-forced, NOT answer correctness). This script
backfills the real per-template answer accuracy for the 4 saved checkpoints by
RESUMING that run and logging a fresh `offline/*` metric family + a summary Table.

INPUT
-----
A `metrics_aggregate.parquet` produced by score_and_aggregate.py over a gen dir
containing the 4 consolidation checkpoints plus the reference arms. Rows are
per (arm, template_type) with columns pass@1 (greedy), pass@8, maj@8, n, delta_*.

The consolidation run was INITIALISED from the `full` arm (pretrain_model_15,
+Full Mulberry) — so `full` is the step-0 "before" point. `teacher_235b` is the
ceiling; `baseline_thinking` / `orig_thinking` are extra references.

STEP AXIS NOTE
--------------
We resume a FINISHED run, so wandb's internal step continues monotonically from the
last training step — logging the offline points at the checkpoints' own (smaller)
training steps with an explicit step= would be SILENTLY DROPPED ("step must be
monotonically increasing"). Instead we define a custom x-axis `offline/ckpt_step`
and log the offline curve against it; the internal step just auto-increments.
"""
import argparse
import math
import os
import sys

import pandas as pd

# arm -> (training global_step, display role). The 4 checkpoints are the consolidation
# run; `full` is its init (step 0). References get role tags for the Table.
CKPT_ARMS = [
    ("full",                  0,    "init (+Full Mulberry, pretrain_model_15)"),
    ("stageb_yawxy_cp896",    896,  "consolidation ckpt (epoch 1)"),
    ("stageb_yawxy_cp1792",   1792, "consolidation ckpt (epoch 2)"),
    ("stageb_yawxy_cp2688",   2688, "consolidation ckpt (epoch 3)"),
    ("stageb_yawxy_cp3584",   3584, "consolidation ckpt (epoch 4)"),
]
REFERENCE_ARMS = [
    ("teacher_235b",      "teacher ceiling (Qwen3-VL-235B zero-shot)"),
    ("baseline_thinking", "plain SURDS-SFT (Thinking)"),
    ("orig_thinking",     "zero-shot Qwen3-VL-8B-Thinking (floor)"),
]


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="metrics_aggregate.parquet from score_and_aggregate.py")
    ap.add_argument("--run-id", default=os.environ.get("RESUME_RUN_ID", "5qmwzryf"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "samarjyo"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "qwenvl-sft-stageb"))
    args = ap.parse_args()

    agg = pd.read_parquet(args.metrics)
    have = set(agg["arm"].unique())
    print(f"[log_offline_eval] arms in metrics: {sorted(have)}")

    def cell(arm, tt, metric):
        """Return a metric value for (arm, template_type) or None if absent."""
        m = agg[(agg["arm"] == arm) & (agg["template_type"] == tt)]
        if m.empty or metric not in m.columns:
            return None
        return _num(m.iloc[0][metric])

    def arm_metrics(arm):
        return {
            "yaw_greedy":     cell(arm, "yaw", "pass@1"),
            "yaw_pass8":      cell(arm, "yaw", "pass@8"),
            "yaw_maj8":       cell(arm, "yaw", "maj@8"),
            "n_yaw":          cell(arm, "yaw", "n"),
            "xy2d_greedy":    cell(arm, "xy2d", "pass@1"),
            "xy2d_pass8":     cell(arm, "xy2d", "pass@8"),
            "xy2d_maj8":      cell(arm, "xy2d", "maj@8"),
            "n_xy2d":         cell(arm, "xy2d", "n"),
            "overall_greedy": cell(arm, "ALL", "pass@1"),
            "overall_pass8":  cell(arm, "ALL", "pass@8"),
        }

    import wandb

    run = wandb.init(id=args.run_id, entity=args.entity, project=args.project,
                     resume="must")
    print(f"[log_offline_eval] resumed run {run.entity}/{run.project}/{run.id} "
          f"(name={run.name})")

    # Custom x-axis so the offline curve plots against the checkpoint's TRAINING step
    # rather than the (continuing, monotonic) internal wandb step.
    wandb.define_metric("offline/ckpt_step")
    wandb.define_metric("offline/*", step_metric="offline/ckpt_step")

    # ---- offline accuracy curve over the consolidation checkpoints --------------
    for arm, step, _role in CKPT_ARMS:
        if arm not in have:
            print(f"[log_offline_eval] WARN: arm {arm!r} missing; skipping curve point")
            continue
        m = arm_metrics(arm)
        payload = {"offline/ckpt_step": step}
        for k, v in m.items():
            if k.startswith("n_") or v is None:
                continue
            payload[f"offline/{k}"] = v
        wandb.log(payload)
        print(f"[log_offline_eval] step={step:>5} {arm:24s} "
              f"yaw greedy={m['yaw_greedy']} pass8={m['yaw_pass8']} | "
              f"xy2d greedy={m['xy2d_greedy']} pass8={m['xy2d_pass8']}")

    # ---- summary Table: every checkpoint + reference arm ------------------------
    cols = ["arm", "role", "ckpt_step", "n_yaw", "yaw_greedy", "yaw_pass8", "yaw_maj8",
            "n_xy2d", "xy2d_greedy", "xy2d_pass8", "xy2d_maj8",
            "overall_greedy", "overall_pass8"]
    table = wandb.Table(columns=cols)
    for arm, step, role in CKPT_ARMS:
        if arm not in have:
            continue
        m = arm_metrics(arm)
        table.add_data(arm, role, step, m["n_yaw"], m["yaw_greedy"], m["yaw_pass8"],
                       m["yaw_maj8"], m["n_xy2d"], m["xy2d_greedy"], m["xy2d_pass8"],
                       m["xy2d_maj8"], m["overall_greedy"], m["overall_pass8"])
    for arm, role in REFERENCE_ARMS:
        if arm not in have:
            continue
        m = arm_metrics(arm)
        table.add_data(arm, role, None, m["n_yaw"], m["yaw_greedy"], m["yaw_pass8"],
                       m["yaw_maj8"], m["n_xy2d"], m["xy2d_greedy"], m["xy2d_pass8"],
                       m["xy2d_maj8"], m["overall_greedy"], m["overall_pass8"])
    wandb.log({"offline/per_checkpoint_table": table})

    # ---- headline scalars into run.summary --------------------------------------
    init = arm_metrics("full") if "full" in have else {}
    best = arm_metrics(CKPT_ARMS[-1][0]) if CKPT_ARMS[-1][0] in have else {}
    ceil = arm_metrics("teacher_235b") if "teacher_235b" in have else {}
    summ = {}
    for tag, d in (("init", init), ("final_ckpt", best), ("teacher_ceiling", ceil)):
        for k in ("yaw_greedy", "yaw_pass8", "xy2d_greedy", "xy2d_pass8",
                  "overall_greedy"):
            if d.get(k) is not None:
                summ[f"offline_summary/{tag}_{k}"] = d[k]
    # deltas final vs init (the "did SFT move yaw/xy2d" answer)
    for k in ("yaw_greedy", "xy2d_greedy", "overall_greedy"):
        if init.get(k) is not None and best.get(k) is not None:
            summ[f"offline_summary/delta_{k}_final_vs_init"] = best[k] - init[k]
    run.summary.update(summ)
    print("[log_offline_eval] summary:", {k: round(v, 4) for k, v in summ.items()})

    wandb.finish()
    print("[log_offline_eval] done.")


if __name__ == "__main__":
    sys.exit(main())

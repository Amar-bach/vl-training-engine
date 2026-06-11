# Stage-C enrichment eval — how to compare vs the Stage-B1 winner

Goal: measure the marginal value of **Stage-C (DeepSeek) enrichment** over the
**Stage-B1 winner** (`baseline_thinking` = pretrain_model_14 Qwen3-VL-8B-Thinking
SURDS-20k) on the SAME two SURDS eval sets used for the Mulberry ablation:

- **in-distribution:** `eval_runs/ablation_val1k/`   (val = `sft_stageB/val_1k.jsonl`)
- **held-out:**        `eval_runs/heldout_surdsval/`  (val = `heldout_surdsval.jsonl`)

`baseline_thinking.parquet` already exists in BOTH dirs — we only generate the new
`stage_c` arm and re-aggregate. NOTE: the global aggregation Δ reference is now the
zero-shot `orig_thinking` (see `baselines_and_teachers_README.md`); the Stage-C-vs-Stage-B1
contrast is computed directly from raw metrics in `compare_stage_c.py` (vs `baseline_thinking`,
the Stage-B1 winner = "SURDS-SFT (Thinking)").

## 0. Prereq
Stage-C checkpoint from `pretrain_model_17.sh`. After training, find the final ckpt:
```
ls -d /mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_17_stagec_*/v0-*/checkpoint-*
```
Set `CKPT=<that checkpoint dir>` below. (3 epochs × 20,154 / 32 ≈ 1890 steps, same as
pretrain_model_14, so the last checkpoint is typically `checkpoint-1890`.)

## 1. Generate `stage_c.parquet` on BOTH sets  (you submit these — sxm5, 8 GPU)
Reuses `research/eval/eval_ablation.sh` (SLURM job-name stays generic `pretrain_model_16`).

```bash
CKPT=<stage_c_checkpoint_dir>

# (a) in-distribution val_1k  -> eval_runs/ablation_val1k/stage_c.parquet
CKPT="$CKPT" ARM=stage_c BASE=thinking \
  sbatch /mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/research/eval/eval_ablation.sh

# (b) held-out SURDS val      -> eval_runs/heldout_surdsval/stage_c.parquet
CKPT="$CKPT" ARM=stage_c BASE=thinking \
  VAL_JSONL=/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs/heldout_surdsval/heldout_surdsval.jsonl \
  OUT_DIR=/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs/heldout_surdsval \
  sbatch /mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/research/eval/eval_ablation.sh
```

## 2. Score + compare (login node, no GPU)
```bash
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda
cd /mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/research/eval
python compare_stage_c.py          # both sets; prints overall + per-family Δ vs baseline_thinking
```
This re-runs `score_and_aggregate.py` on each eval dir (auto-discovers the new
`stage_c.parquet`, recomputes Δ-vs-`baseline_thinking`), writing:
- `indist_metrics/`  (ablation_val1k)
- `heldout_metrics/` (heldout_surdsval)
each with `metrics_per_question.parquet`, `metrics_aggregate.parquet`, `metrics_summary.json`,
then prints the focused Stage-C-vs-baseline overall + per-template-family table.

## Interpreting
`Δ (pts)` > 0 ⇒ Stage-C enrichment helped on that metric/family vs the Stage-B1 winner.
Headline metrics: greedy `pass@1`, `pass@8` (sampler upper bound), `maj@8` (self-consistency).
Because the two arms share ids/images/questions and (where no usable Stage-C trace existed)
the same Stage-B traces, the Δ isolates the value of the cleaned Stage-C reasoning on the
~86% of points it touched.

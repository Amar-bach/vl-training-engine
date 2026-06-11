# Baselines reframe + zero-shot teacher (32B / 235B) evals

## What changed (presentation / Δ reference)
The **baselines** are now the off-the-shelf, **no-SFT** checkpoints:
- `orig_instruct`, `orig_thinking`  (Qwen3-VL-8B, zero-shot)
- `teacher_32b`, `teacher_235b`     (Qwen3-VL-32B / -235B-A22B Thinking, zero-shot) — when generated

The SURDS-only SFT runs (`baseline_instruct`, `baseline_thinking`) are **relabeled
`SURDS-SFT (Instruct/Thinking)`** — they are *trained arms*, not baselines.

Two references are used in the analysis (notebook `visionr1_ablation_analysis.ipynb`):
- **Headline Δ = vs `orig_thinking`** (zero-shot) — "training lift over the untrained model".
  This is the `--baseline-arm orig_thinking` aggregation (already re-run for both eval sets).
- **Ablation Δ = vs `baseline_thinking`** (SURDS-SFT) — the help/hurt of *adding a Mulberry
  domain*. Computed in-notebook (`cell_delta_vs` / `overall_delta_vs`) because the ±1-pt
  ablation effect would be invisible against the +30-pt lift-over-zero-shot.

### Why zero-shot Thinking < Instruct (it was mostly a truncation artifact)
At a 2048-token budget zero-shot Thinking emitted ~800–990 `<think>` words and **33–48% of
greedy decodes never reached a closed `<answer>`** (truncated mid-reasoning → scored wrong);
pass@8 in-dist was ~tied (88.5 vs 89.0). So the deficit was schema non-compliance +
token-budget truncation, not a capability gap. We therefore now run the long-reasoning
zero-shot/teacher arms at an **8192-token budget** (see below). The SURDS-SFT / Mulberry arms
are terse — verified <0.5% near the 2048 ceiling, max ~1780 words — so they are
**budget-invariant** and keep their existing 2048 parquets. See notebook §1b.

## Zero-shot evals at 8k (you submit — sxm5, 8 GPU; greedy + n=8 @ temp 0.8, **8192 tok**, tp 8)
Script: `slurm_scripts/pretrain_model_20.sh` (generic job-name `pretrain_model_20`). Defaults
are now `MAXTOK=8192 MAXLEN=12288`. Writes `<ARM>.parquet` into BOTH eval dirs. Submit all
four zero-shot arms — including re-running the two **8B originals** so the whole zero-shot
group is uniform at 8k (`orig_thinking` is the Δ reference, so it MUST be re-run):

```bash
cd /mnt/sandbox/amar.amarjyoti/slurm_scripts

# 8B zero-shot Thinking (the Δ reference)
MODEL=/mnt/data4/shasta/amar.amarjyoti/research_data/models/Qwen3-VL-8B-Thinking \
ARM=orig_thinking BASE=thinking sbatch pretrain_model_20.sh

# 8B zero-shot Instruct
MODEL=/mnt/data4/shasta/amar.amarjyoti/research_data/models/Qwen3-VL-8B-Instruct \
ARM=orig_instruct BASE=instruct sbatch pretrain_model_20.sh

# 32B (dense Thinking)
MODEL=/mnt/data4/shasta/amar.amarjyoti/research_data/models/Qwen3-VL-32B-Thinking \
ARM=teacher_32b sbatch pretrain_model_20.sh

# 235B-A22B (FP8 MoE Thinking) — enforce-eager for the flashinfer MoE kernel
MODEL=/mnt/data4/shasta/amar.amarjyoti/research_data/models/Qwen3-VL-235B-A22B-Thinking-FP8 \
ARM=teacher_235b ENFORCE_EAGER=1 GPU_MEM_UTIL=0.92 sbatch pretrain_model_20.sh
```

Both teachers are **Thinking-only** (no Instruct variant exists locally) — the 8k budget is
what makes their greedy scores reflect capability rather than truncation.

### Final re-aggregation (avoid the per-job race)
Each job re-aggregates at the end, but if two finish at once they can race writing the same
`*_metrics/` files, and a job that aggregates *before* the new `orig_thinking` lands would use
the stale 2048 reference. So after **all four** jobs finish, run ONE clean aggregation:
```bash
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda
cd /mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/research/eval
ER=/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs
python score_and_aggregate.py --eval-dir $ER/ablation_val1k  --val-meta val_meta.parquet         --out-dir indist_metrics  --baseline-arm orig_thinking
python score_and_aggregate.py --eval-dir $ER/heldout_surdsval --val-meta heldout_val_meta.parquet --out-dir heldout_metrics --baseline-arm orig_thinking
```

## After teacher parquets land
Re-render the notebook (auto-discovers the new arms):
```bash
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda
cd /mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine
python research/notebook_builders/_build_visionr1_ablation_analysis.py
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --output visionr1_ablation_analysis.executed.ipynb \
  notebooks/visionr1_ablation_analysis.ipynb
```

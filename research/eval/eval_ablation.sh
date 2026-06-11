#!/bin/bash
# eval_ablation.sh — SLURM sbatch wrapper for gen_val_ablation.py
#
# Evaluates one Qwen3-VL-8B checkpoint on the spatial-reasoning val_1k set.
# Parametrize via env vars; one submission per arm.
#
# ---------------------------------------------------------------------------
# SUBMISSION COMMANDS (one per arm)
# Root: /mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs
# ---------------------------------------------------------------------------
#
#   # baseline_instruct
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_14_Qwen3-VL-8B-Instruct_1063158/v0-20260608-173039/checkpoint-1890 \
#   ARM=baseline_instruct \
#   BASE=instruct \
#   sbatch eval_ablation.sh
#
#   # baseline_thinking
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_14_Qwen3-VL-8B-Thinking_1063159/v0-20260608-173042/checkpoint-1890 \
#   ARM=baseline_thinking \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
#   # geometry_math
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_15_surdsXmulberry_geometry_math_1063203/v0-20260608-233103/checkpoint-9126 \
#   ARM=geometry_math \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
#   # chart_plot
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_15_surdsXmulberry_chart_plot_1063204/v0-20260608-233355/checkpoint-8160 \
#   ARM=chart_plot \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
#   # science_diagram
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_15_surdsXmulberry_science_diagram_1063205/v0-20260609-044629/checkpoint-3786 \
#   ARM=science_diagram \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
#   # doc_text
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_15_surdsXmulberry_doc_text_1063206/v0-20260609-061937/checkpoint-9189 \
#   ARM=doc_text \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
#   # general_vqa
#   CKPT=/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/pretrain_model_15_surdsXmulberry_general_vqa_1063207/v0-20260609-071451/checkpoint-6858 \
#   ARM=general_vqa \
#   BASE=thinking \
#   sbatch eval_ablation.sh
#
# ---------------------------------------------------------------------------

#SBATCH --job-name=pretrain_model_16
#SBATCH --output=/mnt/sandbox/amar.amarjyoti/outputs/%j-%x.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=sxm5
#SBATCH --mem-per-gpu=120G
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amar.amarjyoti@bluerivertech.com

set -ex

# ---------------------------------------------------------------------------
# Required env vars (must be set at submission time)
# ---------------------------------------------------------------------------
CKPT="${CKPT:?ERROR: set CKPT=<path-to-checkpoint-dir>}"
ARM="${ARM:?ERROR: set ARM=<arm-name>}"

# Optional env vars with defaults
BASE="${BASE:-thinking}"   # thinking | instruct
DATA_ROOT=/mnt/data4/shasta/amar.amarjyoti/research_data
VAL_JSONL="${VAL_JSONL:-$DATA_ROOT/vlm_cot_distill/sft_stageB/val_1k.jsonl}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/eval_runs/ablation_val1k}"
OUT="${OUT:-$OUT_DIR/${ARM}.parquet}"

# ---------------------------------------------------------------------------
# Conda environment (rlvr_conda) — mirrors pretrain_model_15.sh
# ---------------------------------------------------------------------------
module load miniforge
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda

export HF_HOME=$DATA_ROOT/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/stubs' | paste -sd:)"
export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# vLLM tensor-parallel workers must use 'spawn' (not fork) or CUDA re-init fails in workers.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[ -d "$CKPT" ]      || { echo "ERROR: CKPT not found: $CKPT";         exit 1; }
[ -f "$VAL_JSONL" ] || { echo "ERROR: VAL_JSONL not found: $VAL_JSONL"; exit 1; }

mkdir -p "$OUT_DIR"

echo "--- GPU visibility check ---"
nvidia-smi -L
python -c "import torch; print('torch CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

echo "--- Run params ---"
echo "ARM:  $ARM"
echo "BASE: $BASE"
echo "CKPT: $CKPT"
echo "VAL:  $VAL_JSONL"
echo "OUT:  $OUT"

# ---------------------------------------------------------------------------
# Resolve gen_val_ablation.py by ABSOLUTE path. SLURM copies this script to
# /var/spool/slurmd/... before running, so $BASH_SOURCE does NOT point at the
# repo copy — hardcode the repo location (override with GEN_SCRIPT=... if moved).
# ---------------------------------------------------------------------------
GEN_SCRIPT="${GEN_SCRIPT:-/mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/research/eval/gen_val_ablation.py}"
[ -f "$GEN_SCRIPT" ] || { echo "ERROR: gen_val_ablation.py not found at $GEN_SCRIPT"; exit 1; }

# ---------------------------------------------------------------------------
# Run generation
# ---------------------------------------------------------------------------
python "$GEN_SCRIPT" \
    --ckpt       "$CKPT" \
    --arm        "$ARM" \
    --val        "$VAL_JSONL" \
    --out        "$OUT" \
    --base       "$BASE" \
    --n-sample   8 \
    --temp       0.8 \
    --max-tokens 2048 \
    --tp         8

echo "eval_ablation done. Output: $OUT"

# VLM Training Handoff — ms-swift session

Written 2026-05-15 by the OpenRLHF-prorl-research session. Read this first.

## Why you're here

VLM student training (SFT → RLVR → RLAIF) for autonomous-driving VQA moved
from `OpenRLHF-prorl-research` to this repo because:

1. OpenRLHF VLM path forces a separate full reference-model copy (no LoRA
   adapter-disable trick) — wastes GPU memory.
2. ms-swift has Day-0 Qwen3-VL examples + worked GRPO+LoRA+vLLM colocate
   recipe: `examples/train/grpo/internal/vllm_lora_qwenvl72b.sh`.
3. Plugin-based rewards fit both RLVR (rule verifier) and RLAIF (judge call).

## Goal sequence

1. **SFT** Qwen3-VL-8B with LoRA-64 on Stage B rejection-sampling winners
   (gated, image-grounded traces from a Qwen3-VL-235B teacher).
2. **RLVR** with GRPO, rule-based verifier reward (answer-match against gold).
3. **RLAIF** with GRPO, judge-model reward plugin.

All three stages: Qwen3-VL-8B base, LoRA rank 64, one 8×H200 sxm5 node.

## Data — where Stage B output lives

Data root (the ONLY permitted path under /mnt/data4):
`/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/`

Files needed for the SFT formatter (3-file join keyed by `id`):

| Role | Path |
|---|---|
| **Source QAs** (image path + question + gold) | `_train_qa_for_cot.jsonl` |
| **Stage A traces** (per `(id, sample_idx)`: `grounding`, `thinking`, `answer`) | `cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl` |
| **Stage B judge** (per `id`: `best_idx`, `scores[]`, `scene_description`) | `judge_1058662_qwen32b.jsonl` (and sibling `judge_*_qwen32b.jsonl` — 40,645 records total) |

Stage C polish (DeepSeek text-only rederivation) is still running in the
OpenRLHF repo; **not required** for the initial SFT run — Stage B winners
are sufficient. Stage C unlocks a separate **text-only** SFT split later.

## Stage D formatter — to be written here

Pure JSONL transform. Output ms-swift multimodal schema (`messages` + `images`).

**Join logic:**
- For each Stage B record `r`, look up `(r.id, r.best_idx)` in Stage A → get
  winning `grounding`, `thinking`, `answer`.
- Join `r.id` with `_train_qa_for_cot.jsonl` → get `image_path`, `question`,
  `gold_answer`.

**Image-grounded gate (D1, no C1):**
```
parse_ok_b1[best_idx]
AND best_score.answer_correctness == 1
AND best_score.hallucination       >= 4
AND best_score.visual_grounding    >= 4
AND best_score.reasoning_quality   >= 3
```

**Open decision the user has NOT made yet:**
- Should the assistant target include the `<obj1>…[x,y]</obj1>` grounding block
  before `<reasoning>`? Two options:
  - (a) `<reasoning>thinking</reasoning><answer>answer</answer>` — clean
  - (b) `{grounding}\n<reasoning>thinking</reasoning><answer>answer</answer>`
        — student learns point-style grounding too (closer to teacher)
  Ask the user before generating the SFT file.

- Train/val split: hold out a slice of SURDS train, OR use the existing
  `surds_val_stratified_3k.jsonl` (different ids, image-anchored eval).

## Compute & infra

- **Cluster:** SLURM. Partitions: `sxm5` = H200 (use this), `gen5` = A40,
  `gen3/gen4` = A6000.
- **Job naming:** `pretrain_model_N.sh` pattern (privacy convention).
- **sbatchw watcher:** `~/slurm_watcher/sbatchw.sh script.sh` opts a job into
  auto-diagnose + retry (cap 3) on failure. Use for production jobs.
- **Conda env:** `rlvr_conda` — activate before any HF/python work.
- **vLLM+H200 gotchas:** for FP8 MoE specifically, need `CUDA_HOME`,
  header symlinks, `--enforce-eager`, `lib64` symlink. (Not needed for
  Qwen3-VL-8B bf16.)

## Feasibility on 1×H200 node (8 GPUs) for Qwen3-VL-8B + LoRA-64

- **SFT:** trivial; zero2 + LoRA is enough.
- **GRPO with vLLM colocate + `sleep_level 1`:** comfortable headroom. Crank
  `num_generations` to 16–32 and `max_completion_length` to 4–8k.
- **LoRA ref-model:** ms-swift handles adapter-disable for ref natively →
  no separate ref copy. This is the reason we're here.
- **RLAIF judge:** if judge is co-located, must be ≤ ~16B for comfort. For a
  Qwen3-VL-32B judge, serve on a second node and call over HTTP. For
  Qwen3-VL-235B judge, definitely separate node.

## Hard rules (from project memory — do not violate)

1. **No `sbatch` without explicit per-turn approval.** Every SLURM submission
   requires the user to OK that specific submission. Iterating during a
   running job on free sxm5 is the only exception.
2. **Propose edits first.** Show diffs in chat, wait for approval before
   Write/Edit. Iterating fast on incremental fixes during active runs is the
   exception.
3. **No Claude attribution.** Strip "Co-Authored-By: Claude" / "🤖 Generated
   with Claude Code" from commits, PRs, and any output the user might
   forward.
4. **Only high-impact questions.** Decide routine choices yourself; ask only
   on architectural, indecisive forks.
5. **Data > 100 MB → `/mnt/data4/shasta/amar.amarjyoti/research_data/`.**
   Never search elsewhere under /mnt/data4.
6. **In-session journaling.** Append durable bullets to today's session
   memory file in real time; don't post-process.

## Recipes to crib from in this repo

| Need | File |
|---|---|
| Qwen3-VL SFT (transformers backend) | `examples/models/qwen3_vl/transformers.sh` |
| Qwen3-VL SFT (deepspeed zero3) | `examples/models/qwen3_vl/zero3.sh` |
| Qwen3-VL SFT (mcore parallel) | `examples/models/qwen3_vl/mcore.sh` |
| GRPO + LoRA + vLLM colocate (Qwen2.5-VL-72B template) | `examples/train/grpo/internal/vllm_lora_qwenvl72b.sh` |
| Reward plugin pattern | `examples/train/grpo/plugin/plugin.py` |
| LoRA SFT generic | `examples/train/lora_sft.sh` |

## First moves for the new session

1. Read this file, then read `MEMORY.md` if it exists in this repo's memory
   dir (it won't on first invocation — create one as you go).
2. Confirm with the user: SFT target format (with/without grounding block),
   train/val split choice.
3. Draft the Stage D formatter as `tools/build_sft_image_stageB.py` (or
   wherever fits this repo's layout) — show diff, wait for approval.
4. Then draft `pretrain_model_13.sh` (or next free N — check with user) for
   the SFT run on Qwen3-VL-8B + LoRA-64 on sxm5.
5. Do NOT sbatch anything without explicit approval that turn.

## Cross-repo pointers

- OpenRLHF-prorl-research project memory:
  `/home/amar.amarjyoti/.claude/projects/-mnt-sandbox-amar-amarjyoti-research-code-OpenRLHF-prorl-research/memory/`
- Original CoT pipeline plan (Stages A/B/C/D):
  `/mnt/sandbox/amar.amarjyoti/research_code/OpenRLHF-prorl-research/cot_enrichment_plan.md`
- Stage B implementation (reference for join schema):
  `/mnt/sandbox/amar.amarjyoti/research_code/OpenRLHF-prorl-research/vlm_cot_distill/stage_b_judge/qwen_judge.py`

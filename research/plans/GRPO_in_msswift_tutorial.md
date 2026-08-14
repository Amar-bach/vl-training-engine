# GRPO in this vendored ms-swift — an exhaustive, repo-grounded tutorial

This is a hands-on tutorial for running **GRPO** (Group Relative Policy Optimization) and its
algorithmic relatives inside the ms-swift copy vendored in this repo
(`/mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine`). It is written for an ML
researcher who knows RL basics (policy gradients, PPO clipping, KL regularization) but has
never read this codebase.

Every claim is anchored to a `file:line` so you can jump straight into the source. All paths
are relative to the repo root unless stated otherwise.

> Quick orientation: GRPO is launched via `swift rlhf --rlhf_type grpo ...`. The trainer lives
> in `swift/rlhf_trainers/grpo_trainer.py`, the rollout machinery in
> `swift/rlhf_trainers/rollout_mixin.py`, the reward functions in `swift/rewards/orm.py`, and
> the CLI args in `swift/arguments/rlhf_args.py` + `swift/rlhf_trainers/args_mixin.py`. Runnable
> examples are in `examples/train/grpo/`.

---

## Table of contents

1. [What GRPO is & how ms-swift implements it](#1-what-grpo-is--how-ms-swift-implements-it)
2. [The data format](#2-the-data-format)
3. [Reward functions](#3-reward-functions)
4. [The vLLM rollout](#4-the-vllm-rollout)
5. [The algorithm family as flags](#5-the-algorithm-family-as-flags)
6. [End-to-end runnable example](#6-end-to-end-runnable-example)
7. [Hyperparameter guidance & gotchas](#7-hyperparameter-guidance--gotchas)
8. [Where to look in the code](#8-where-to-look-in-the-code)

---

## 1. What GRPO is & how ms-swift implements it

### 1.1 The idea in one paragraph

GRPO (DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) is a critic-free
policy-gradient method. For each prompt you sample a **group** of `G` completions
(`num_generations`). You score each completion with one or more reward functions, sum them into
a scalar reward `r_i`, and form the **advantage** as the reward minus the *group mean*:

```
A_i = r_i - mean_over_group(r)          # optionally divided by the group std
```

That group mean is the "baseline" — it replaces PPO's learned value network (the *critic*).
Because the baseline comes for free from the group, there is no value head and no GAE. The
policy is then updated with a PPO-style clipped surrogate using these advantages, optionally
with a KL penalty toward a frozen reference model.

The whole family (DAPO, GSPO, CISPO, RLOO, REINFORCE++, SAPO, Dr.GRPO, REAL, CHORD) is built by
swapping out three pieces: **how the advantage is computed** (`advantage_estimator`,
`scale_rewards`), **at what granularity importance sampling happens**
(`importance_sampling_level`), and **the exact loss/normalization** (`loss_type`). Section 5 is
the table.

### 1.2 The trainer class chain

The GRPO trainer is assembled by multiple inheritance — `grpo_trainer.py:83`:

```python
class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
```

- `HFGRPOTrainer` is `trl.GRPOTrainer` (HuggingFace TRL ≥ 0.20 is asserted — `rlhf_args.py:489`).
- `RolloutTrainerMixin` (`rollout_mixin.py:82`) provides vLLM generation, weight syncing, and
  CPU offload.
- `SwiftMixin` provides the swift-specific model/template/checkpoint plumbing.

### 1.3 The concrete training step (where to put your breakpoints)

The heart of one GRPO step is `_generate_and_score_completions` — `grpo_trainer.py:233`. Read it
top to bottom; it is the spine of the algorithm:

1. **Generate** — `inputs = self._generate_completions(inputs)` (`grpo_trainer.py:238`, defined
   at `:213`). This strips the last assistant turn and asks the model (via vLLM or a
   Transformers engine) to regenerate it. See §4.
2. **Score** — `total_rewards_per_func = self._score_completions(inputs)`
   (`grpo_trainer.py:239`, defined at `:305`). This calls every reward function and gathers the
   results across processes into a `Tensor[N, num_reward_funcs]`. The actual per-function loop
   that calls `reward_func(completions, **reward_kwargs)` is in `_compute_rewards_per_func`
   (`grpo_trainer.py:337`, call site `:368`).
3. **Dynamic sampling** (DAPO) — if `dynamic_sample` is set, groups with reward std = 0 are
   dropped and resampled (`grpo_trainer.py:242-244`).
4. **Advantages** — `total_advantages = self._compute_advantages(...)`
   (`grpo_trainer.py:248`, defined at `:406`). See §1.4.
5. Advantages are attached to each micro-batch (`grpo_trainer.py:262-265`) and prompts +
   completions are logged for wandb/swanlab (`grpo_trainer.py:267-300`).

Then, separately, the optimizer step calls **`compute_loss`** (`grpo_trainer.py:997`) →
`_compute_loss` (`:1008`) → **`_compute_loss_and_metrics`** (`:1028`), which dispatches on
`self.loss_type`. See §1.5.

### 1.4 `_compute_advantages` — the group-relative math

`grpo_trainer.py:406-577` (default grouped mode; a "request-aware" mode for multi-turn starts at
`:579`). The key steps:

- **Aggregate reward across functions** with `reward_weights` (`grpo_trainer.py:475`):
  ```python
  rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
  ```
  Note the `nansum`: a reward function may return `None`/`NaN` for a row and it is simply skipped
  (a `None` becomes `torch.nan`, `grpo_trainer.py:369`).
- **KL-in-reward** (RLOO / REINFORCE++ style): if `kl_in_reward` and `beta != 0`, the per-token
  KL is summed per sequence and subtracted from the reward *before* baselining
  (`grpo_trainer.py:477-491`).
- **Reshape into groups** `[-1, num_generations]` and subtract the group mean
  (`grpo_trainer.py:499-520`):
  - `grpo`/`reinforce_plus_plus`: `A_i = r_i - group_mean`.
  - `rloo`: leave-one-out, `A_i = r_i·K/(K-1) - group_mean·K/(K-1)` (`:514-517`).
- **Normalization** depends on `scale_rewards` and `advantage_estimator`
  (`grpo_trainer.py:522-569`):
  - `grpo`/`rloo` divide the advantage by the **reward** std (group, batch, or none).
  - `reinforce_plus_plus` divides by the **advantage** std (`:523-542`).
  - `gdpo` is a special per-reward-function normalization + global whitening (`:555-565`).
  - `none` leaves advantages un-normalized (this is the "Dr.GRPO" / unbiased choice).

So GRPO's advantage is literally "reward minus group mean, optionally over group-std" — no
critic anywhere.

### 1.5 `_compute_loss_and_metrics` — the surrogate loss

`grpo_trainer.py:1028`. The flow:

1. Recompute current-policy per-token logprobs (`:1033`).
2. Optional **overlong filter**: mask out truncated completions from the loss
   (`:1059-1064`).
3. Optional **KL-in-loss** (GRPO style, `beta != 0 and not kl_in_reward`): a k3 KL estimator
   `exp(Δ) - Δ - 1` (`:1068-1071`).
4. **Importance-sampling ratio** at the configured level (`:1110-1127`):
   - `token`: per-token log-ratio (vanilla GRPO/PPO).
   - `sequence`: a single sequence-mean log-ratio (this is **GSPO**, `:1113-1117`).
   - `sequence_token`: GSPO-token hybrid (`:1119-1121`).
5. **Loss dispatch** on `self.loss_type` (`:1129-1218`):
   - `grpo`/`bnpo`/`dr_grpo`/`dapo`: PPO clipped surrogate `-min(coef_1·A, coef_2·A)` with
     `coef_2 = clamp(coef_1, 1-ε_low, 1+ε_high)` (`:1142-1149`).
   - `cispo`: clipped-IS-weight × logprob, MiniMax-style (`:1129-1131`).
   - `sapo`: soft sigmoid gating instead of hard clipping (`:1132-1139`).
   - `real`: a logsumexp ranking loss over positive/negative groups (`:1140-1141` and
     `:1180-1212`).
6. **Normalization** of the token loss (`:1172-1218`) differs per `loss_type`:
   - `grpo`: per-sequence mean then batch mean (`:1174`).
   - `bnpo`: sum over all tokens / total tokens (`:1176`).
   - `dr_grpo`: sum / `(batch_size × max_completion_length)` — the unbiased Dr.GRPO normalizer
     (`:1178-1179`).
   - `cispo`/`dapo`: sum / global token count (`:1213-1216`).

That is the entire algorithm. Everything else is plumbing (vLLM, offload, logging, multi-turn).

---

## 2. The data format

### 2.1 Required and optional columns

GRPO consumes a chat-style dataset. The schema is:

| Column | Type | Required? | Role |
|--------|------|-----------|------|
| `messages` | `List[Dict]` (OpenAI format) | **yes** | The conversation. The **last assistant turn is stripped at rollout** and the model regenerates it. |
| `solution` | `str` | for `accuracy`/most rewards | The gold answer. Reward functions read it via their `solution` kwarg. |
| `images` | `List[str/PIL]` | multimodal only | Image paths/objects; passed through to vLLM and to reward kwargs. |
| `videos` / `audios` | list | multimodal only | Same, for other modalities. |
| *(any extra column)* | anything | no | Auto-forwarded into every reward function's `**kwargs`. |

The mechanism that makes "any extra column" appear in your reward function is
`RowPreprocessor.rows_to_batched(reward_inputs)` (`grpo_trainer.py:349`). It batches every
dataset column into a list keyed by column name, so `solution`, `images`, `difficulty`,
`template_type`, `verification_info`, … all arrive as `kwargs['<colname>']` (a list aligned with
`completions`).

In addition to dataset columns, the trainer injects:
- `trainer_state` — the HF `TrainerState` (so you can read `global_step`, used by curriculum
  rewards; e.g. `plugin.py:651`).
- `response_token_ids` — token ids of each completion (used by `cosine`/`soft_overlong`,
  `orm.py:159`, `:226`).
- `request_id`, `trajectory_inputs` — for multi-turn rollouts (`grpo_trainer.py:347-348`).

### 2.2 How the gold answer reaches the reward function

The completion text and all batched columns are handed to each reward function at
`grpo_trainer.py:368`:

```python
output_reward_func = reward_func(completions, **reward_kwargs)
```

So the builtin `accuracy` ORM signature `__call__(self, completions, solution, **kwargs)`
(`orm.py:78`) receives `completions` (list of generated strings) and `solution` (list of gold
strings) zipped 1-to-1. **The column must literally be named `solution`** for the builtin
accuracy/cosine rewards to find it.

### 2.3 Worked example rows

**Text (math) row** — for `--reward_funcs accuracy format`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math assistant. Put your final answer in <answer></answer>."},
    {"role": "user", "content": "What is 12 * 9?"},
    {"role": "assistant", "content": "<think>12*9=108</think><answer>108</answer>"}
  ],
  "solution": "<answer>108</answer>"
}
```

At rollout the assistant turn is removed; the model regenerates it. `MathAccuracy` (`orm.py:69`)
extracts the `<answer>…</answer>` from both the completion and `solution`, parses them with
`math_verify`, and returns `1.0`/`0.0`. `Format` (`orm.py:123`) checks the
`^<think>…</think>\s*<answer>…</answer>$` regex.

**Multimodal (VL) row** — for a Qwen2.5-VL run with `--reward_funcs external_r1v_acc format`:

```json
{
  "messages": [
    {"role": "user", "content": "<image>How many red cubes are in the image?"},
    {"role": "assistant", "content": "<think>...</think><answer>3</answer>"}
  ],
  "images": ["/data/clevr/img_000123.png"],
  "solution": "<answer>3</answer>"
}
```

The `images` column flows into vLLM (so the model actually sees the image during rollout) and
also into the reward function's `**kwargs`. The `<image>` tag in the text marks where the image
embedding is spliced. `MultiModalAccuracyORM` (`plugin.py:95`) compares completion vs `solution`
by symbolic parse first, then exact `<answer>` string match.

> Datasets used by the example scripts: `AI-MO/NuminaMath-TIR` (text math),
> `AI-ModelScope/chartqa_digit_r1v_format`, `AI-ModelScope/clevr_cogen_a_train`,
> `lmms-lab/multimodal-open-r1-8k-verified` (multimodal). These already have `messages` +
> `solution` (+ `images`) in the expected schema.

---

## 3. Reward functions

### 3.1 Builtin reward functions (the `orms` registry)

The registry is at the bottom of `swift/rewards/orm.py:455-464`:

```python
orms = {
    'toolbench':     ReactORM,
    'math':          MathORM,
    'accuracy':      MathAccuracy,     # orm.py:69
    'format':        Format,           # orm.py:123
    'react_format':  ReActFormat,      # orm.py:132
    'cosine':        CosineReward,     # orm.py:141
    'repetition':    RepetitionPenalty,# orm.py:176
    'soft_overlong': SoftOverlong,     # orm.py:216
}
```

What they do:

| Key | Class | Signature | Behaviour |
|-----|-------|-----------|-----------|
| `accuracy` | `MathAccuracy` | `(completions, solution, **kw)` | Extracts `<answer>`, parses with `math_verify`, returns 1.0/0.0. Requires `pip install math_verify` (asserted at `orm.py:74`). |
| `format` | `Format` | `(completions, **kw)` | 1.0 iff completion matches `^<think>…</think>\s*<answer>…</answer>$` (`orm.py:127`). |
| `react_format` | `ReActFormat` | `(completions, **kw)` | ReAct `Action:/Action Input:` format check. |
| `cosine` | `CosineReward` | `(completions, solution, **kw)` | Length-scaled reward: correct answers rewarded more when *short*, wrong answers penalized less when *short*. Reads `response_token_ids` (`orm.py:159`). Tunable via `--cosine_*` args. |
| `repetition` | `RepetitionPenalty` | `(completions, **kw)` | Negative penalty for repeated n-grams (`--repetition_n_grams`, `--repetition_max_penalty`). |
| `soft_overlong` | `SoftOverlong` | `(completions, **kw)` | Soft length penalty in `[L_max - L_cache, L_max]`. Requires `--soft_max_length`/`--soft_cache_length` (asserted at `orm.py:220`; `soft_max_length` auto-defaults to `max_completion_length`, `rlhf_args.py:357-362`). |

`cosine`'s parameters live in `args_mixin.py:352-356` (`cosine_min/max_len_value_wrong/correct`,
`cosine_max_len`); `repetition`'s at `:358-359`.

### 3.2 The ORM signature (verbatim)

The base classes are at `swift/rewards/orm.py:16-66`:

```python
class ORM:
    """Base class for synchronous outcome reward models (ORM)."""

    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous reward models (I/O-bound, run with asyncio.gather)."""

    def __init__(self, args=None, **kwargs):
        self.args = args

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError
```

Contract:
- `__call__` is **batched**: `completions` is `List[str]`; every dataset column arrives as a
  same-length list via `**kwargs`.
- It must return a `List[float]` of length `len(completions)`. Return `None` for a row to skip it
  (becomes `NaN`, ignored in the weighted nansum — `grpo_trainer.py:369`).
- `self.args` is the full `GRPOConfig`, so you can read any hyperparameter (e.g.
  `args.max_completion_length`) inside your reward.
- Use `AsyncORM` for network/API rewards (LLM judges, sandbox execution); they are gathered in
  parallel with `asyncio.gather` (`grpo_trainer.py:374-391`).

### 3.3 Writing a custom reward plugin

**Step 1 — define + register** in a plugin file (this is exactly the pattern in
`examples/train/grpo/plugin/plugin.py:24-36`):

```python
# my_plugin.py
from typing import List
from swift.rewards import ORM, orms


class SurdsAccuracy(ORM):
    """Reward = 1.0 if the gold answer string appears in the completion, else 0.0."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        return [1.0 if str(s).strip() in c else 0.0 for c, s in zip(completions, solution)]


orms['surds_accuracy'] = SurdsAccuracy   # the key you pass to --reward_funcs
```

A curriculum-style reward that reads the training step (as in `plugin.py:649-651`):

```python
class StepAwareFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        step = kwargs['trainer_state'].global_step          # injected by the trainer
        weight = 1.0 if step >= 30 else 0.5
        import re
        ok = [bool(re.match(r'^<think>.*?</think>\s*<answer>.*?</answer>$', c, re.DOTALL))
              for c in completions]
        return [weight if o else 0.0 for o in ok]

orms['step_aware_format'] = StepAwareFormat
```

**Step 2 — wire it on the command line.** `--external_plugins` imports your file (which runs the
`orms[...] = ...` registration); `--reward_funcs` selects keys; `--reward_weights` weights them:

```bash
swift rlhf --rlhf_type grpo \
  --external_plugins /abs/path/my_plugin.py \   # imported → registers into `orms`
  --reward_funcs surds_accuracy format \        # keys looked up in `orms`
  --reward_weights 1.0 0.2 \                     # len must == #reward_funcs (+#reward_models)
  ...
```

Each selected key is instantiated as `orms[key](args=grpo_config)`. The final scalar reward is
`Σ_k weight_k · reward_k` via the weighted `nansum` at `grpo_trainer.py:475`. If
`--reward_weights` is omitted, every function gets weight `1.0` (`rlhf_args.py` docstring,
`:140-143`).

> Reference custom rewards already in `plugin.py`: `external_countdown` (`:40`),
> `external_r1v_acc` (multimodal accuracy, `:95`), `external_code_reward` (E2B sandbox, `:185`),
> `async_genrm` (LLM-judge over an API, `:463`, an `AsyncORM`). The runnable wiring is
> `examples/train/grpo/plugin/run_external_reward_func.sh`.

### 3.4 Reward *models* (nn.Module) vs reward *functions*

Besides functions there are **reward model plugins** (`rm_plugins` registry, `plugin.py:902-911`)
for using an actual scoring model: pass `--reward_model <path>` and `--reward_model_plugin
<key>`. A reward model is appended as the **last** reward source, so its weight is the last entry
in `--reward_weights` (`rlhf_args.py:140-143`). The nn.Module path is called at
`grpo_trainer.py:359-362`. This tutorial focuses on reward *functions*; see `plugin.py:914` and
`swift/rewards/rm_plugin/` for the GenRM path.

---

## 4. The vLLM rollout

GRPO is rollout-dominated: most wall-clock is spent generating `G` completions per prompt. ms-swift
uses vLLM for this, in one of two modes.

### 4.1 Colocate vs server mode

Set with `--vllm_mode` (required when `--use_vllm true`; enforced at `rlhf_args.py:396-397`):

- **`colocate`** — vLLM runs **in the same processes** as training, sharing the GPUs. Before each
  generation the training weights are pushed into the vLLM engine, vLLM generates, then it sleeps
  to free memory for the backward pass. This is the common single-node setup
  (`vllm_vl7b.sh:9-12`). Generation is in `_colocate_rollout` (`rollout_mixin.py:1181`).
- **`server`** — a **separate** vLLM server (launched with `swift rollout`) does generation; the
  trainer talks to it over HTTP via `self.vllm_client.infer` (`rollout_mixin.py:1129`,
  `:1213`). Required for `--async_generate true` (`rlhf_args.py:404-405`). See `real.sh` for a
  server-mode example (it launches `swift rollout --model ...` then `swift rlhf --vllm_mode
  server --vllm_server_host 127.0.0.1 --vllm_server_port 8000`).

### 4.2 Weight syncing (colocate)

Each rollout the engine is woken, fresh weights are copied in, generation runs, then the engine
sleeps:

- Wake: `self.engine.engine.wake_up(...)` (`rollout_mixin.py:1057-1073`).
- Sync: `_move_model_to_vllm` (`rollout_mixin.py:442`) pushes current weights into the vLLM
  engine — full weights for full-parameter training, or just the LoRA deltas for LoRA training.
- Sleep: `self.engine.engine.sleep(level=args.sleep_level)` (`rollout_mixin.py:1095`).

The engine itself is built in `_prepare_vllm_engine` (`rollout_mixin.py:258`).

### 4.3 Memory knobs: `sleep_level`, offload, `gpu_memory_utilization`

- **`--sleep_level {0,1,2}`** — how aggressively vLLM releases GPU memory between rollouts.
  `0` keeps everything resident (fastest, most memory); `1`/`2` free progressively more. The
  example scripts use `--sleep_level 1` almost everywhere. **Note:** GRPO auto-resets
  `sleep_level = 0` when `async_generate`, server mode, or `not use_vllm`
  (`rlhf_args.py:341-342`).
- **`--offload_model true` / `--offload_optimizer true`** — move the **training** weights /
  optimizer states to CPU RAM during vLLM generation so vLLM has the GPU; moved back for the
  backward pass. Offload logic: `offload_model` (`rollout_mixin.py:1313`), `offload_optimizer`
  (`:1348`). Used in nearly every example with TP > 1 or big models.
- **`--move_model_batches N`** — when offloading, move the model back to GPU in `N` chunks to
  smooth the memory spike (`args_mixin.py:140-142`, `:186`). The 72B VL example uses
  `--move_model_batches 40` (`vllm_lora_qwenvl72b.sh`).
- **`--vllm_gpu_memory_utilization`** (default `0.9`, `args_mixin.py:37`) — fraction of GPU vLLM
  may use. In colocate you must leave room for training, so examples drop it to `0.4`–`0.6`
  (`vllm_vl7b.sh:11`, `chord.sh`, `gspo.sh`).
- **`--vllm_tensor_parallel_size`** (default `1`) — shard the rollout model across GPUs; the VL
  examples use `4` (`vllm_vl7b.sh:12`).
- **`--vllm_max_model_len`** — the vLLM KV-cache context length; set it ≥ prompt + completion.

### 4.4 Multimodal knobs

- **`MAX_PIXELS` env var** — caps the per-image pixel budget the vision tower / vLLM ingests.
  `vllm_vl7b.sh:1` sets `MAX_PIXELS=1003520`; the 3B/72B examples use `602112`. Larger = more
  visual detail but more tokens/memory. This is an **environment variable**, not a `--flag`.
- **`--vllm_limit_mm_per_prompt`** — JSON cap on media per prompt, e.g. `'{"image": 1}'` or
  `'{"image": 5, "video": 2}'`. Parsed from JSON at `args_mixin.py:59-60` and forwarded to the
  engine as `limit_mm_per_prompt` (`args_mixin.py:79`). Multimodality is detected via
  `model.model_meta.is_multimodal`.

### 4.5 `num_generations`, `steps_per_generation`, `num_iterations`, `generation_batch_size`

These control the rollout/optimizer cadence and are the single most error-prone part of GRPO:

- **`--num_generations G`** (default `8`, `rlhf_args.py:158`) — completions per prompt; the group
  size for the advantage baseline. Larger `G` ⇒ lower-variance baseline but more rollout cost.
- **`--generation_batch_size`** — total samples produced in one vLLM call. If unset it is derived
  (`args_mixin.py:200-202`):
  ```
  steps_per_generation = gradient_accumulation_steps           # default
  generation_batch_size = global_batch_size * steps_per_generation
  ```
  where `global_batch_size = per_device_train_batch_size × num_processes`. **It must be divisible
  by `num_generations`** (validated, `args_mixin.py:203-`).
- **`--steps_per_generation`** — how many optimizer micro-steps **reuse** one rollout batch
  (defaults to `gradient_accumulation_steps`, `args_mixin.py:201`). Setting it `> grad_accum`
  amortizes rollout cost over more updates (GSPO uses `4`, SAPO uses `32`). Only one of
  `steps_per_generation` / `generation_batch_size` should be set (`args_mixin.py:155-156`).
- **`--num_iterations µ`** (default `1`, `rlhf_args.py:148-149`, GRPO's "K") — number of PPO-style
  inner epochs over the *same* rollout batch. `µ > 1` makes training off-policy within the batch
  (the clipping then actually bites); `µ = 1` is on-policy and the trainer skips computing
  `old_per_token_logps` (`grpo_trainer.py:1079-1080`).

---

## 5. The algorithm family as flags

All of these are still `--rlhf_type grpo`; they differ only in the three knobs from §1.1. Every
flag set below was cross-checked against an example `.sh` or an `AdvancedResearch` doc (citation
in the last column).

| Variant | Critical flags | Verified against |
|---------|----------------|------------------|
| **GRPO** (vanilla) | *(defaults)* `--loss_type grpo --advantage_estimator grpo --scale_rewards group --importance_sampling_level token --beta 0.04` | `vllm_vl7b.sh`, defaults in `rlhf_args.py:351,380-381` |
| **DAPO** | `--loss_type dapo --epsilon_high 0.28 --dynamic_sample true --max_resample_times 3 --overlong_filter true --reward_funcs soft_overlong --soft_cache_length 4096` | `AdvancedResearch/DAPO.md:82-89` |
| **GSPO** | `--importance_sampling_level sequence --epsilon 3e-4 --epsilon_high 4e-4 --beta 0.0 --steps_per_generation 4` | `internal/gspo.sh` |
| **CISPO** | `--loss_type cispo --epsilon_high 5.0` | `AdvancedResearch/CISPO.md:56-57` |
| **RLOO** | `--advantage_estimator rloo --kl_in_reward true` (`scale_rewards` auto → `none`) | `internal/rloo.sh` |
| **REINFORCE++** | `--advantage_estimator reinforce_plus_plus --scale_rewards batch --kl_in_reward true --importance_sampling_level sequence` | `internal/reinforce_plus_plus.sh` |
| **SAPO** | `--loss_type sapo --tau_pos 1 --tau_neg 1.05` | `internal/sapo.sh` |
| **REAL** | `--loss_type real --real_tau 0.5` (forces `scale_rewards=none`, server mode) | `internal/real.sh`, `rlhf_args.py:372-377` |
| **Dr.GRPO** | `--loss_type dr_grpo` (unbiased length normalization, typically `--scale_rewards none`) | `grpo_trainer.py:1177-1179`, `AdvancedResearch/DAPO.md:45` |
| **BNPO** | `--loss_type bnpo` (token-level batch normalization) | `grpo_trainer.py:1175-1176`, `DAPO.md:45` |
| **CHORD** | `--chord_sft_dataset <ds> --chord_sft_per_device_train_batch_size 1 --chord_mu_warmup_steps 0 --chord_mu_decay_steps 200 --chord_mu_peak 0.9 --chord_mu_valley 0.05 --chord_enable_phi_function false` | `internal/chord.sh` |

**Important:** GSPO is *not* a `loss_type`. It is `importance_sampling_level=sequence` plus a
small asymmetric epsilon and `beta=0` — the sequence-level IS branch at `grpo_trainer.py:1113-1117`
*is* the GSPO objective. (Confirmed by `gspo.sh` header comments citing arXiv:2507.18071.)

### The orthogonal knobs explained

- **`--beta`** — KL weight toward the frozen reference model. Default `0.04` for GRPO
  (`rlhf_args.py:351`). **`beta=0.0` drops the reference model entirely** (`rlhf_args.py:301-302`
  sets `ref_model=None`), saving memory; GSPO/CHORD use `beta=0`, while the LoRA examples use a
  tiny `--beta 0.001`.
- **`--kl_in_reward`** — if true, KL is subtracted from the *reward* (RLOO/REINFORCE++ style,
  `grpo_trainer.py:477-491`); if false, KL is added to the *loss* (GRPO style,
  `:1068-1071`). Auto-defaults: `False` for grpo, `True` for rloo/reinforce++
  (`rlhf_args.py:364-368`).
- **`--epsilon` / `--epsilon_high`** — lower/upper PPO clip. `epsilon_high` defaults to `epsilon`
  if unset; DAPO's "Clip-Higher" decouples them (`0.2`/`0.28`). For CISPO, `epsilon_high` is the
  IS-weight cap and is *large* (`5.0`).
- **`--scale_rewards {group,batch,none,gdpo}`** — advantage normalization (§1.4). Auto-default is
  tied to the estimator: `group` (grpo), `none` (rloo), `batch` (reinforce++)
  (`rlhf_args.py:379-387`). `none` = the unbiased Dr.GRPO choice.
- **`--advantage_estimator {grpo,rloo,reinforce_plus_plus}`** — chooses the baseline math
  (`args_mixin.py:410`).
- **`--importance_sampling_level {token,sequence,sequence_token}`** — token (GRPO), sequence
  (GSPO), sequence_token (GSPO-token) (`args_mixin.py:402`).
- **`--dynamic_sample` / `--max_resample_times`** — DAPO: drop reward-std-0 groups and resample up
  to N times (`args_mixin.py:386-387`, applied at `grpo_trainer.py:242-244`).
- **`--overlong_filter`** — drop truncated completions from the loss (`args_mixin.py:388`, applied
  at `grpo_trainer.py:1059-1064`).
- **`--delta`** — INTELLECT-2 two-sided clip upper bound (`args_mixin.py:348`, applied at
  `grpo_trainer.py:1144-1145`); incompatible with the liger kernel (`rlhf_args.py:498-499`).

### When to use which (one paragraph)

Start with **vanilla GRPO** (`group` scaling, `token` IS, `beta` small). If you see gradient
collapse because too many groups have identical rewards (binary correctness on easy data), reach
for **DAPO** — dynamic sampling kills std-0 groups and `epsilon_high=0.28` keeps exploration
alive. If training is unstable on long generations or large MoE models, **GSPO**
(sequence-level IS) is much more robust to per-token ratio noise — it is the default recommendation
for long-CoT and big models. **CISPO** (MiniMax) avoids dropping clipped tokens entirely and tends
to be stable at scale with a large `epsilon_high`. **RLOO** and **REINFORCE++** are leaner
baselines (RLOO = leave-one-out, REINFORCE++ = batch-whitened advantage) when you want fewer moving
parts and KL folded into the reward. **SAPO** replaces hard clipping with a smooth gate (fewer
dead gradients). **Dr.GRPO** (`scale_rewards none` + `loss_type dr_grpo`) removes GRPO's length and
std biases when you care about unbiased estimates. **CHORD** interleaves an SFT loss on expert data
with GRPO to stabilize from a weak start. For VL specifically, the shipped examples default to
plain GRPO or RLOO/REINFORCE++/SAPO with `external_r1v_acc + format` rewards.

---

## 6. End-to-end runnable example

### 6.1 Qwen2.5-VL-7B LoRA GRPO — every line annotated

This is `examples/train/grpo/internal/vllm_vl7b.sh` with each line explained:

```bash
MAX_PIXELS=1003520 \                       # env: cap pixels per image fed to the vision tower
NPROC_PER_NODE=8 \                         # env: 8 training processes (= 8 GPUs); must == num GPUs for vLLM
swift rlhf \
    --rlhf_type grpo \                     # select GRPO (vs dpo/ppo/gkd/...)
    --model Qwen/Qwen2.5-VL-7B-Instruct \  # base policy (multimodal)
    --tuner_type lora \                    # LoRA fine-tuning (no ref_model needed; see §7)
    --dataset AI-ModelScope/chartqa_digit_r1v_format \  # has messages + solution + images
    --load_from_cache_file true \          # reuse preprocessed dataset cache
    --use_vllm true \                      # use vLLM for rollout generation
    --vllm_mode colocate \                 # vLLM shares the training GPUs (vs server)
    --vllm_gpu_memory_utilization 0.5 \    # leave half the GPU for training weights/activations
    --vllm_tensor_parallel_size 4 \        # shard the rollout model across 4 GPUs
    --torch_dtype bfloat16 \               # bf16 weights
    --system examples/train/grpo/prompt.txt \  # system prompt asking for <think>/<answer>
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \      # micro-batch per GPU (prompts, pre-expansion)
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \                 # canonical GRPO LR (see §7)
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \      # → steps_per_generation defaults to 1
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 1024 \         # max generated tokens per completion
    --reward_funcs accuracy format \       # builtin MathAccuracy + Format (weights default to 1.0 each)
    --num_generations 8 \                  # G = 8 completions per prompt → group size 8
    --sleep_level 1 \                      # release vLLM GPU memory between rollouts
    --temperature 1.0 \                    # sampling temperature for rollout
    --top_p 0.85                           # nucleus sampling for rollout
```

Batch math for this script: `global_batch = per_device(1) × procs(8) = 8`;
`generation_batch_size = 8 × steps_per_generation(=grad_accum=1) = 8`; that is divisible by
`num_generations=8`, i.e. exactly **one prompt** sampled 8 times per device-group per step.

To run it (do not auto-submit — follow the repo's job policy):

```bash
bash examples/train/grpo/internal/vllm_vl7b.sh
```

### 6.2 Text-only GRPO (Qwen2.5-7B, full-parameter) — the differences

Take §6.1 and change:

```bash
    --model Qwen/Qwen2.5-7B-Instruct \     # text LLM
    --tuner_type full \                    # FULL-parameter (NOT lora)
    --dataset AI-MO/NuminaMath-TIR \       # text math; messages + solution, no images
    --reward_funcs accuracy \              # no need for image-aware reward
    --deepspeed zero3 \                    # full-param at 7B needs ZeRO-3 sharding
    --offload_model true \                 # offload weights to CPU during vLLM rollout
    --offload_optimizer true \             # offload optimizer states too
    # (drop MAX_PIXELS, --tuner_type lora, --vllm_tensor_parallel_size if single-GPU rollout)
```

**LoRA vs full-parameter — the key differences:**
- LoRA (`--tuner_type lora`): **no `ref_model` is loaded** for the KL term (the base model with
  adapters disabled serves as reference); weight-sync to vLLM pushes only LoRA deltas; much lower
  memory; `beta` can be a small `0.001`.
- Full (`--tuner_type full`): a separate frozen `ref_model` is created (defaults to `--model`,
  `rlhf_args.py:303-306`) when `beta != 0`; needs DeepSpeed ZeRO-2/3 + offload; full weights are
  synced to vLLM each rollout.

### 6.3 VL full-parameter / 72B reference

For a large VL full-parameter run, see `examples/train/grpo/internal/vllm_lora_qwenvl72b.sh`
(72B, LoRA, TP=4, `--move_model_batches 40`, `--reward_weights 1 0.1`) and
`examples/train/grpo/plugin/run_external_reward_func.sh` (3B, full, custom plugin rewards,
`--epsilon 0.2 --epsilon_high 0.28`).

---

## 7. Hyperparameter guidance & gotchas

### 7.1 Learning rate

Every example uses **`--learning_rate 1e-6`** (the `real.sh` server example uses `2e-6`). GRPO is
extremely sensitive to LR; `1e-6` is the safe canonical value. Do not start higher.

### 7.2 `beta` / KL

- GRPO default `beta = 0.04` (`rlhf_args.py:351`). The LoRA examples deliberately use a *tiny*
  `--beta 0.001` to stay close to on-policy without strong anchoring.
- **`--beta 0.0` removes the reference model** (`rlhf_args.py:301-302`) — saves a whole model's
  worth of memory. GSPO and CHORD set `beta=0`. Use it when you want pure reward maximization or
  are memory-constrained, and trust the clip to control drift.
- `kl_in_reward` toggles whether KL hits the reward (rloo/reinforce++) or the loss (grpo). You
  rarely set it manually — let the estimator default decide.

### 7.3 `num_generations` vs the batch math (the #1 footgun)

The constraint, enforced at `args_mixin.py:200-236`:

```
global_batch_size       = per_device_train_batch_size × num_processes
generation_batch_size   = global_batch_size × steps_per_generation       # if not set explicitly
require: generation_batch_size % num_generations == 0
```

Concrete numbers from the example scripts:
- **`vllm_vl7b.sh`**: `pdtb=1`, procs=8, grad_accum=1, `G=8` → gen_batch = `1·8·1 = 8`, `8 % 8 =
  0` ✓ (1 unique prompt × 8 generations).
- **`gspo.sh`**: `pdtb=2`, procs=8, grad_accum=8, `steps_per_generation=4`, `G=16` → global =
  `2·8 = 16`, gen_batch = `16·4 = 64`, `64 % 16 = 0` ✓ (4 unique prompts × 16 generations).
- **`chord.sh`**: `pdtb=4`, procs=8, grad_accum=8, `G=8`, `steps_per_generation=4` → global = 32,
  gen_batch = `32·4 = 128`, header comment notes total GRPO batch = `32 prompts × 8 = 256` across
  the full grad-accum window. (See the `chord.sh` header for the worked arithmetic.)

If you get *"generation_batch_size must be divisible by num_generations"*, fix
`per_device_train_batch_size`, `num_processes`, `steps_per_generation`, or `num_generations` until
the product divides evenly.

### 7.4 Memory: offload / sleep_level / gpu_mem_util

Typical colocate recipe (from the examples):
- `--vllm_gpu_memory_utilization 0.4–0.6` (not the `0.9` default — you must leave room for the
  trainer).
- `--sleep_level 1`, `--offload_model true`, `--offload_optimizer true`.
- For very large models add `--move_model_batches 40` to chunk the GPU↔CPU transfer.
- Full-parameter ⇒ `--deepspeed zero2` or `zero3`.

Remember `sleep_level` is auto-forced to `0` in server/async/non-vLLM mode
(`rlhf_args.py:341-342`), so offload becomes your only memory lever there.

### 7.5 Common errors

- **`GRPO with vLLM is not compatible with device_map` / set `NPROC_PER_NODE = num_processes`** —
  raised at `rlhf_args.py:491-493`. You launched with model-parallel `device_map`. Set
  `NPROC_PER_NODE` equal to the number of GPUs (one process per GPU).
- **`Your current version of trl is outdated … pip install -U trl`** — TRL ≥ 0.20 is required
  (`rlhf_args.py:489`).
- **`The math_verify package is required`** — `accuracy`/`cosine` need `pip install math_verify`
  (`orm.py:74`).
- **`soft_cache_length must be set when using soft overlong rewards`** — set
  `--soft_cache_length` (and optionally `--soft_max_length`) (`rlhf_args.py:357-359`).
- **`cached_dataset is not supported for GRPO`** — GRPO needs the raw dataset to re-roll
  completions (`rlhf_args.py:337-338`).
- **All-rewards-None warning** — every reward returned `None`/`NaN` for a row; check your column
  names (`grpo_trainer.py:394-402`). Most often the dataset's gold column is not literally named
  `solution`.
- **Multimodal left-truncation shape mismatch** — for VL models, `truncation_strategy='left'` can
  cut image tokens; prefer `--truncation_strategy delete` (which resamples), per the warning at
  `rlhf_args.py:150-156`.
- **liger kernel restrictions** — `--use_liger_kernel` disallows `delta`, sequence parallel,
  padding-free, entropy mask, non-grpo estimators (`rlhf_args.py:494-515`).

---

## 8. Where to look in the code

| Path | What it is |
|------|------------|
| `swift/rlhf_trainers/grpo_trainer.py` | The GRPO trainer. Key methods: `_generate_and_score_completions` (`:233`), `_compute_rewards_per_func` (`:337`, reward call site `:368`), `_compute_advantages` (`:406`), `compute_loss`→`_compute_loss_and_metrics` (`:997`/`:1028`, loss dispatch `:1129-1218`). |
| `swift/rlhf_trainers/rollout_mixin.py` | vLLM rollout + weight sync + offload. `_prepare_vllm_engine` (`:258`), `_move_model_to_vllm` (`:442`), `_fast_infer` (`:1050`), `_server_rollout` (`:1129`), `_colocate_rollout` (`:1181`), `offload_model`/`offload_optimizer` (`:1313`/`:1348`). |
| `swift/rewards/orm.py` | Builtin reward functions + `ORM`/`AsyncORM` base classes (`:16-66`) and the `orms` registry (`:455-464`). |
| `swift/arguments/rlhf_args.py` | The `RLHFArguments` dataclass: `rlhf_type`, `beta`, `ref_model`, `max_completion_length`, the GRPO defaulting logic in `_init_grpo` (`:334`), rollout init `_init_rollout` (`:389`), validation `_check_grpo` (`:482`). |
| `swift/rlhf_trainers/args_mixin.py` | `GRPOArgumentsMixin`: all the algorithm knobs — `epsilon`/`epsilon_high`/`delta` (`:346-348`), `cosine_*`/`repetition_*` (`:352-359`), `chord_*` (`:365-372`), `dynamic_sample`/`overlong_filter`/`soft_*` (`:386-390`), `scale_rewards` (`:394`), `importance_sampling_level` (`:402`), `tau_pos`/`tau_neg`/`real_tau` (`:406-415`), `advantage_estimator`/`kl_in_reward` (`:410-412`), and all `vllm_*` + offload + `generation_batch_size`/`steps_per_generation` fields. |
| `examples/train/grpo/internal/*.sh` | Runnable recipes: `vllm_vl7b.sh` (VL LoRA GRPO), `gspo.sh`, `sapo.sh`, `chord.sh`, `rloo.sh`, `reinforce_plus_plus.sh`, `real.sh`, `vllm_lora_qwenvl72b.sh`, `moe_*`, `qlora.sh`, `transformers.sh` (no-vLLM). |
| `examples/train/grpo/plugin/plugin.py` | Reference custom rewards/schedulers: `external_countdown`, `external_r1v_acc`, `external_code_reward`, `async_genrm`, plus reward-model and multi-turn-scheduler examples. |
| `examples/train/grpo/plugin/run_external_reward_func.sh` | How to wire `--external_plugins` + `--reward_funcs`. |
| `docs/source_en/Instruction/GRPO/AdvancedResearch/*.md` | Per-variant deep dives: `DAPO.md`, `GSPO.md`, `CISPO.md`, `SAPO.md`, `RLOO.md`, `REINFORCEPP.md`, `REAL.md`, `CHORD.md`, `entropy_mask.md`, `training_inference_mismatch.md`. |

### Minimal checklist to launch a GRPO run

1. Dataset has `messages` (+ `solution`, + `images` if VL) — §2.
2. Pick reward functions: builtins via `--reward_funcs accuracy format`, or a plugin via
   `--external_plugins file.py --reward_funcs my_key` (+ `--reward_weights`) — §3.
3. `--use_vllm true --vllm_mode colocate` + memory knobs (`--vllm_gpu_memory_utilization 0.5
   --sleep_level 1 --offload_model true`) — §4.
4. Set `--num_generations`, `--per_device_train_batch_size`, `--gradient_accumulation_steps`,
   `--steps_per_generation` so the divisibility constraint holds — §7.3.
5. Pick the variant flags from the §5 table (or just defaults for vanilla GRPO).
6. `--learning_rate 1e-6`, choose `--beta` (0.04 default, 0.001 for LoRA, 0.0 to drop the ref
   model) — §7.

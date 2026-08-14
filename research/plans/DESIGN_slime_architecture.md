# slime — Architecture & Code-Reading Guide (focus: multi-turn agentic RL)

Purpose: read this once, then open the slime source and recognise almost everything. Every
claim is anchored to `file:line` in the cloned repo at
`/mnt/sandbox/amar.amarjyoti/research_code/slime` (paths below are relative to that root).
The **multi-turn / agentic RL** path (§8) is the centerpiece — the rest is the scaffolding you
need to understand it.

---

## 1. Mental model — three planes glued by Ray

slime is an RL post-training framework built on **Megatron (training) + SGLang (rollout) + Ray
(orchestration)**. There are exactly three planes, and one manager object that bridges two of them:

```
                 ┌──────────────────────────────────────────────────────┐
   driver        │  train.py / train_async.py   (the rollout→train loop) │
   (1 process)   └───────────────┬───────────────────────┬──────────────┘
                                 │ generate()            │ async_train() + update_weights()
                 ┌───────────────▼─────────────┐  ┌───────▼──────────────────────────┐
   ROLLOUT plane │ RolloutManager (Ray actor)  │  │ RayTrainGroup (N Megatron actors)│ TRAIN plane
                 │  ├ data source / buffer      │  │  └ MegatronTrainRayActor × world │
                 │  └ SGLang engines + router   │  │     (policy, +ref/critic/teacher)│
                 └───────────────┬─────────────┘  └───────▲──────────────────────────┘
                                 │ Sample objects         │ weights pushed each sync
                                 └────────────────────────┘ (CUDA-IPC if colocate, else NCCL)
```

- **Driver** (`train.py`): a thin sequential loop — *generate a rollout → train on it → push new
  weights to the rollout engines → repeat*. On-policy.
- **Rollout plane** = `RolloutManager` (`slime/ray/rollout.py:422`), a Ray actor that owns (a) the
  **data source / buffer** (the prompt feed + over-generation buffer) and (b) the **SGLang engines**
  behind a router. It produces `Sample` objects.
- **Train plane** = `RayTrainGroup` (`slime/ray/actor_group.py:10`), one `MegatronTrainRayActor`
  (`slime/backends/megatron_utils/actor.py:46`) per training GPU. Consumes batches of `Sample`s,
  runs Megatron forward/backward, computes GRPO/PPO loss.
- The two planes never talk directly: rollout `Sample`s are converted to a per-DP-rank tensor dict,
  `ray.put()`-boxed, and handed to the train actors; trained weights are pushed back to SGLang.

Training backend is **Megatron-only** (`--train-backend` accepts only `megatron`,
`slime/utils/arguments.py` validation). Rollout backend is **SGLang-only**, by design.

---

## 2. Repo map — where to read what

| You want to understand… | Open… |
|---|---|
| The central data object (everything is a `Sample`) | `slime/utils/types.py:93` |
| The top-level loop | `train.py`, `train_async.py` |
| GPU layout, colocate vs disaggregate | `slime/ray/placement_group.py` |
| The rollout manager + SGLang engines + router | `slime/ray/rollout.py`, `slime/backends/sglang_utils/sglang_engine.py` |
| The prompt feed + over-generation buffer | `slime/rollout/data_source.py` |
| **Default** (single-turn) rollout function | `slime/rollout/sglang_rollout.py:618` |
| **Custom multi-turn** rollout (the agentic path) | `examples/geo3k_vlm_multi_turn/rollout.py:315` |
| The tool/environment interface | `examples/geo3k_vlm_multi_turn/{base_env,env_geo3k}.py` |
| Reward dispatch (rule-based + custom) | `slime/rollout/rm_hub/__init__.py` |
| The Megatron training actor + train step | `slime/backends/megatron_utils/actor.py`, `.../model.py` |
| GRPO advantage + PPO-clip loss + loss-mask reduction | `slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py`, `.../cp_utils.py` |
| Weight sync train→SGLang | `slime/backends/megatron_utils/update_weight/*` |
| Model build + HF↔Megatron conversion | `slime/backends/megatron_utils/model_provider.py`, `.../checkpoint.py`, `.../megatron_to_hf/` |
| Every CLI arg + every customization hook | `slime/utils/arguments.py` |

**Suggested reading order:** §3 (`Sample`) → §4 (driver loop) → §8 (multi-turn — the part you
care about) → then §5–§7 only as deep as you need.

---

## 3. The `Sample` object — the one type that flows everywhere

`slime/utils/types.py:93` — a `@dataclass`. Rollout produces it, the buffer stores it, reward fills
it, training consumes it. Learn these fields and 80% of the codebase reads itself.

Response-region fields (the ones that must stay length-aligned — this is the heart of multi-turn):
- `tokens: list[int]` (`:109`) — **prompt ids + ALL response/observation ids**, grows turn by turn.
- `response_length: int` (`:116`) — number of response-region tokens.
- `loss_mask: list[int]` (`:119`) — per-response-token, **1 = trainable, 0 = masked** (tool/obs).
- `rollout_log_probs: list[float]` (`:121`) — SGLang logprobs for trainable tokens, `0.0` for masked.
- `response: str` (`:115`) — decoded response region (filled at finalize).
- `reward: float | dict` (`:118`), `label: str` (`:117` — gold answer for rule-based RMs).

Multimodal fields:
- `multimodal_inputs: dict` (`:110`) — **raw** images/videos (what SGLang gets as `image_data`).
- `multimodal_train_inputs: dict` (`:111`) — **processed** tensors (`pixel_values`, `image_grid_thw`)
  that the Megatron forward consumes.

Identity/grouping:
- `group_index` (`:97`), `index` (`:98`), `rollout_id` (`:106`). **`rollout_id` matters for multi-turn
  fan-out**: if one rollout returns `list[Sample]` (subagent/segment split), every sibling must
  share the same `rollout_id` so loss is averaged within the rollout, not over-counted (`:99-106`).
- `status: Status` (`:140`) — `PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED` (`:130-138`).

The key mutator — **`append_response_tokens(...)`** (`:253-314`): appends tokens, extends
`loss_mask` with `[1 if trainable else 0]*len` (`:292`) and `rollout_log_probs` (zeros for
non-trainable, `:281`). It **enforces** that trainable tokens carry logprobs and masked ones do not
(`:272-281`), and validates `len(loss_mask) == len(rollout_log_probs) == response_length` (`:384-389`).
This invariant is what makes a hand-rolled multi-turn loop safe — you can't silently misalign masks.

Helper: `effective_response_length = sum(loss_mask)` (`:249-251`).

---

## 4. The driver loop — `train.py`

`train.py:9-98`. Setup order matters: **rollout manager is created first** so it can derive
`num_rollout` from epoch count (`:17`), then the training models (`:20`), then an initial weight
push so the SGLang engines start from the loaded checkpoint (`:26`).

The loop (`train.py:63-95`), three phases per step:
```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))   # 1. ROLLOUT
    ...
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))            # 2. TRAIN
    ...
    actor_model.update_weights()                                             # 3. WEIGHT SYNC
```
Strictly sequential ⇒ on-policy. Under `--colocate`, the SGLang engines sit on the *same* GPUs and
are offloaded during the train phase.

`train_async.py` is the same but **prefetches rollout N+1 while training on N** (`:31-49`),
forbids `--colocate` (`:11`), and syncs weights only every `--update-weights-interval` steps
(`:65-69`) — trading on-policy-ness for throughput (rollouts become slightly stale/off-policy,
which is why the TIS importance-sampling correction in §7 exists).

---

## 5. Ray orchestration & GPU placement — `slime/ray/`

- **Placement groups** (`placement_group.py`): one Ray PG of `{GPU:1,CPU:1}` bundles, strategy
  `PACK` (`:47-48`), bundles sorted by `(node_ip, gpu_id)` for deterministic rank order (`:21-39`).
- **colocate vs disaggregate** is one decision in `_get_placement_group_layout` (`:100-117`):
  - `--colocate` → `total = max(actor_gpus, rollout_gpus)`, `rollout_offset = 0` — **actor and
    rollout share the same bundles** (`:114-115`). Forces CUDA-IPC weight transfer.
  - default → `total = actor_gpus + rollout_gpus`, `rollout_offset = actor_gpus` — rollout GPUs come
    *after* actor GPUs (`:117`). Allows NCCL weight broadcast and async overlap.
  - Both share **one PG object**; only the bundle slice differs (`create_placement_groups`, `:120-137`).
- **Train actors**: `RayTrainGroup._allocate_gpus_for_actor` (`actor_group.py:48-119`) spawns
  `world_size` `MegatronTrainRayActor`s pinned to bundles (`PlacementGroupSchedulingStrategy`,
  `:109-116`), each `num_gpus=1` but fractional CPU so a colocated SGLang engine can share. Group
  methods fan out `ray.get([actor.X.remote() …])`: `async_train` (`:131`), `update_weights` (`:155`),
  `offload`/`onload` (`:159`).
- **Rollout actors** (`rollout.py`): `SGLangEngine` (one per engine) → `ServerGroup` (homogeneous
  engines, one tp_size) → `RolloutServer` (one model behind one router) → `RolloutManager` (owns all
  of it + the data source + a lock actor). `RolloutManager.__init__` (`:425-472`) calls
  `start_rollout_servers` (`:1089`) which launches a router per model and registers engines to it.

---

## 6. Rollout plane internals

### 6.1 Data source / buffer — `slime/rollout/data_source.py`
The "data buffer" **is** the data source. Default class `RolloutDataSourceWithBuffer` (`:168`).
- `RolloutDataSource.get_samples(n)` (`:90-118`) slices `n` prompts and, for each, builds a **group of
  `n_samples_per_prompt` deep-copies** (`:108-117`) — this is where GRPO's "n responses per prompt"
  grouping is born.
- `RolloutDataSourceWithBuffer` adds an in-memory `buffer` (`:171`) drained first via
  `pop_first` (`:225`) for over-generation/partial-rollout reuse; `add_samples` (`:198-211`) validates
  each group has exactly `n_samples_per_prompt` and re-enqueues leftovers.

### 6.2 Default (single-turn) rollout function — `slime/rollout/sglang_rollout.py`
- `generate_rollout(args, rollout_id, data_source, evaluation=False)` (`:618-640`) — sync entry; runs
  the async over-generation driver `generate_rollout_async` (`:375-467`), which keeps pulling groups
  from the data source until `rollout_batch_size` is met, applying dynamic filters.
- `generate(args, sample, sampling_params)` (`:153-220`) — the low-level single call: POSTs to the
  **router** `http://{sglang_router_ip}:{sglang_router_port}/generate` (`:159`) with
  `return_logprob: True`; for VL it sends `image_data` + `text` (`:183-188`) so SGLang expands image
  placeholders; unpacks `meta_info["output_token_logprobs"]` and calls `append_response_tokens`.
- `generate_and_rm` (`:223-286`) wraps generation + reward; **this is the dispatch point where a
  custom multi-turn generate function is substituted** (see §8.1).

### 6.3 SGLang engines + router — `slime/backends/sglang_utils/sglang_engine.py`
Each `SGLangEngine` Ray actor launches an SGLang HTTP server subprocess (`launch_server_process`,
`:51`) and **registers to the router** (`/add_worker` or `/workers`, `:190-218`). Generation is not
called on the engine directly — rollout code POSTs `/generate` to the **router**, which load-balances
to workers. The engine actor exposes control endpoints: `flush_cache`, `pause/continue_generation`,
and the `update_weights_from_{tensor,distributed,disk}` endpoints used by weight sync (§7.4).

### 6.4 Reward hub — `slime/rollout/rm_hub/__init__.py`
- `async_rm(args, sample)` (`:55-96`) priority: per-sample `sample.custom_rm_path` → global
  `args.custom_rm_path` → built-in `rm_type` (`math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`,
  `remote_rm`, `random`). `"math"` = `grade_answer_verl(response, label)` (`:83`).
- `batched_async_rm` (`:99-110`) — group mode; a custom RM may receive the **whole list** and return a
  list. **A custom RM must handle both a single `Sample` and a `list[Sample]`.**
- Custom RM signature: `async def custom_rm(args, sample_or_list, **kwargs) -> float | list[float]`.

---

## 7. Train plane internals — Megatron + GRPO/PPO

### 7.1 The training actor
`MegatronTrainRayActor` (`actor.py:46`). Builds model+optimizer (`:48`), and keeps an in-memory
**multi-role weight store** `weights_backuper` (`:108-117`) tagged `actor`/`ref`/`teacher`/`old_actor`.
`_switch_model(tag)` (`:276`) swaps a tagged param set into the live model — this is how slime runs
ref/teacher/old-policy forward passes on one set of GPUs instead of hosting separate models.

`train()` (`:378`) → `train_actor()` (`:428`): pull batch → (optional) ref/teacher/old-policy logprob
forwards (`:442-491`) → `compute_advantages_and_returns` (`:507`) → Megatron `train()` (`:522`) →
refresh the `"actor"` CPU backup (`:540`).

### 7.2 GRPO advantage (two stages)
1. **Group normalization happens on the rollout side**, not in the loss. `_post_process_rewards`
   (`rollout.py:686-711`): reshape rewards to `(-1, n_samples_per_prompt)`, subtract the group mean
   (`:702-703`), optionally divide by `std+1e-6` if `grpo_std_normalization` (`:705-707`).
2. **Broadcast to tokens** on the train side: `compute_advantages_and_returns` (`loss.py:657`) →
   for GRPO, `get_grpo_returns` (`ppo_utils.py:327`) just does `ones_like(kl) * reward` (`:333`) — the
   scalar group-normalized advantage is repeated across every response token.

### 7.3 PPO-clip loss and — critically — how `loss_mask` gates multi-turn
- Core math `compute_policy_loss` (`ppo_utils.py:124-148`): `ratio = exp(log_probs - old_log_probs)`,
  standard clipped surrogate, optional dual-clip.
- `old_log_probs` = rollout logprobs if `--use-rollout-logprobs` else a recomputed forward pass
  (`loss.py:908`); for strictly on-policy single-microbatch steps it falls back to
  `log_probs.detach()` (`:924-925`).
- **The loss-mask reduction is the multi-turn linchpin.** `get_sum_of_sample_mean` (`cp_utils.py:47`)
  builds the reducer every loss term passes through:
  ```python
  # cp_utils.py:73-81 (cp_size==1)
  def sum_of_sample_mean(x):
      return sum((x_i * loss_mask_i).sum() / clamp_min(denom, 1)
                 for x_i, loss_mask_i, denom in zip(x.split(response_lengths), loss_masks, sample_denoms))
  ```
  Masked-out tokens (tool/observation, `loss_mask==0`) are multiplied by 0 **and** excluded from the
  denominator ⇒ they contribute zero gradient. The mask originates per-`Sample` (`rollout.py:747`),
  is padded/CP-sliced into `full_loss_masks` (`data.py:120-148`), passed to the model forward
  (`model.py:624`), and consumed here. **This is exactly why a multi-turn trajectory can be trained as
  one sequence while only the model's own tokens get gradient.**
- Reference-model KL: loaded as the `ref` tag (`actor.py:120`), forwarded via `_switch_model("ref")`;
  enters either the advantage (`kl_coef`) or the loss (`use_kl_loss + kl_loss_coef`, `loss.py:1049`);
  skipped entirely when `kl_coef==0`.
- **TIS (Truncated Importance Sampling)** (`loss.py:827-848`): when rollout logprobs ≠ train logprobs
  (the async/off-policy case), `tis = exp(train_logp - rollout_logp)` clamped and multiplied into the
  PG loss — the correction that makes `train_async.py` sound.

### 7.4 Weight sync (train → SGLang) — `update_weight/`
`update_weights()` (`actor.py:580`) picks a transport at init (`:139-159`):
- `--colocate` → `UpdateWeightFromTensor` (**CUDA-IPC**, same GPUs): iterate HF weight chunks
  (`update_weight_from_tensor.py:165-174`), serialize via `MultiprocessingSerializer`, hand to the
  colocated engine's `update_weights_from_tensor` endpoint.
- disaggregated → `UpdateWeightFromDistributed` (**NCCL** broadcast from train rank 0 to all engines).
The HF naming for SGLang is produced by the **same** Megatron↔HF mapping used for checkpoints (the
`HfWeightIterator`, bridge or raw), so the synced tensors are HF-named and SGLang-loadable.

### 7.5 Megatron coupling you can't avoid
- Parallelism args (TP/PP/CP/EP/ETP) set on the model provider (`model_provider.py:95-107`).
- Checkpoints: HF→Megatron load via **Megatron-Bridge** `AutoBridge.from_hf_pretrained` +
  `load_hf_weights` (`checkpoint.py:129-141`); Megatron→HF via per-arch converters in
  `megatron_to_hf/` (e.g. `qwen3_vl.py`). This conversion step is the main friction tax vs ms-swift.

---

## 8. ★ Multi-turn / agentic RL — the centerpiece ★

This is what you'd build on for SURDS. The whole thing is implemented as a **custom generate
function** that replaces the default single-turn `generate`, plus a pluggable **environment** that
runs the tool. The runnable reference is `examples/geo3k_vlm_multi_turn/` — a multimodal,
multi-turn, tool-calling GRPO example on Qwen3-VL.

### 8.1 The dispatch contract — how your loop gets called
`generate_and_rm` (`sglang_rollout.py:250-261`) chooses your function over the default:
```python
custom_func_path = sample.generate_function_path or args.custom_generate_function_path
if custom_func_path:
    custom_generate_func = load_function(custom_func_path)
    sample = await custom_generate_func(args, sample, sampling_params)   # (+evaluation= if accepted)
```
Signature contract (`docs/en/get_started/customization.md:79`):
```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]
```
Your function must fill `tokens`, `response_length`, `loss_mask`, `status`, and the multimodal
tensors. **It must NOT set `reward`** — after it returns, the manager calls the default RM because
`sample.reward is None` (`sglang_rollout.py:282-284`). So in geo3k the env's per-turn score is just
*in-context feedback to the model*; the **training reward is the math RM** on the final `sample.response`
(launcher sets `--rm-type math`). (You can override with `--custom-rm-path` instead.)

Returning `list[Sample]` = trajectory split into multiple trainable segments (subagent / context
compaction); all siblings must share `rollout_id` (`customization.md:91`, `types.py:99-106`).

### 8.2 The turn loop — `examples/geo3k_vlm_multi_turn/rollout.py:315`
`generate` (`:315`) forbids partial rollout, loads the env from `args.rollout_interaction_env_path`,
reads `args.max_turns`, then runs (`:324-382`):

```
prepare start state: tokenize prompt (+process images), seed budget, response_tokens=[]
for turn_idx in range(max_turns):
    cap max_new_tokens to remaining budget
    response = SGLang /generate(input_ids=sample.tokens, image_data=current_images, return_logprob=True)
    append model tokens          → loss_mask=1, real logprobs     # _append_to_sample(..., loss_mask_val=1)
    budget -= len(model tokens)
    if finish_reason length/abort: break (TRUNCATED/ABORTED)
    obs, done, info = env.step(response_text)                      # run the tool
    if done: status=COMPLETED; break
    obs_ids, obs_images, ... = encode(env.format_observation(obs)) # tool output → tokens(+images)
    append observation tokens    → loss_mask=0, logprobs=[0.0]*n   # _append_to_sample(..., loss_mask_val=0)
    budget -= len(obs tokens)
    accumulate images across turns
finalize: merge per-turn image tensors, decode response, set status
```

Why each piece matters:
- **Full-prefix re-feed**: each turn calls SGLang with the entire `sample.tokens` (prompt + all prior
  turns), so the model attends to the whole conversation; SGLang prefix-caching makes this cheap
  (`_run_inference_step`, `:192-209`).
- **Per-token logprobs come straight from the engine** (`output["meta_info"]["output_token_logprobs"]`,
  `:203-205`) — the exact sampled ids are preserved, never re-tokenized. Trainable tokens *must* carry
  these (the `Sample` invariant).
- **Loss mask is set at append time**: `loss_mask_val=1` for the model's generations (`:343`),
  `loss_mask_val=0` with zeroed logprobs for tool/observation tokens (`:362-363`). After the loop, one
  `Sample` carries a perfectly aligned `tokens / loss_mask / rollout_log_probs` triple — and §7.3's
  reducer trains only the `1`s.
- **Budget** (`:184-188`) = `rollout_max_context_len - len(prompt)`, decremented by *both* model and
  observation tokens, capping `max_new_tokens` each turn so the context never overflows.

### 8.3 The environment interface — `base_env.py` / `env_geo3k.py`
`BaseInteractionEnv` (`base_env.py`, 26 lines) is tiny: `reset()`, `step(response_text)`, `close()`,
and `format_observation(observation) -> chat message`. The geo3k env's `step` (`env_geo3k.py:182-238`)
returns a **3-tuple `(observation_dict, done, info)`** — there is no separate reward. It parses the
**last** `<tool_call>{...}</tool_call>` block (regex `:22`, parse `:57-86`), runs the tool, and packs
a feedback string into the observation.

**★ The per-turn image-injection primitive (this is the SURDS-zoom enabler) ★** —
`base_env.format_observation` (`base_env.py:15-25`):
```python
def format_observation(self, observation):
    content = []
    for _, images in (observation.get("multi_modal_data") or {}).items():
        for image in images:
            content.append({"type": "image", "image": image})     # ← inject image into next turn
    content.append({"type": "text", "text": observation.get("obs_str", "")})
    return {"role": "user", "content": content}
```
So **an env whose `step` returns `{"obs_str": ..., "multi_modal_data": {"image": [zoomed_pil]}}` gets
its image fed into the next turn automatically** — no changes to `rollout.py`. Downstream,
`_encode_observation_for_generation` (`rollout.py:54-112`) runs `process_vision_info([message])` to
pull those images into tensors and `encode_image_for_rollout_engine` to re-encode them for SGLang, and
it **trims the chat-template preamble** (`:71-107`) so only the new observation's tokens are appended.
Cross-turn image tensors are concatenated at finalize (`_merge_multimodal_train_inputs`, `:115-138`).

### 8.4 Config & launcher
- `geo3k_vlm_multi_turn_config.yaml` — **two knobs only**: `max_turns: 3` and
  `rollout_interaction_env_path: examples.geo3k_vlm_multi_turn.env_geo3k`. Loaded via `--custom-config-path`.
- `run_geo3k_vlm_multi_turn.py:54-70` wires it:
  `--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate`,
  `--custom-config-path …`, `--label-key answer` (→ `sample.label`, the RM target + env ground truth),
  `--multimodal-keys '{"image":"images"}'`, `--rm-type math`, `--apply-chat-template`,
  `--advantage-estimator grpo`, `--n-samples-per-prompt 8`. Runs on 4 GPUs, `--colocate`, TP=4
  (`:130-132`) — a real single-node path.

### 8.5 Other agentic examples (one line each, to see the range)
- `examples/search-r1/` — text-only multi-turn RAG: custom generate calls a retrieval server between
  turns, injects results as text observations.
- `examples/tau-bench/` — tool-calling agent via an OpenAI-style tool adapter; returns one loss-masked `Sample`.
- `examples/coding_agent_rl/` — full SWE agent in an E2B sandbox; reward = tests pass; trajectory
  **fans out into `list[Sample]`** segments sharing one `rollout_id` (the canonical multi-sample case).

---

## 9. Customization hooks — the supported extension surface

All are `load_function`-loaded path strings (imported in `RolloutManager.__init__`, `rollout.py:438-450`).
The ones that matter for an agentic VL task:

| Hook (CLI arg) | What you implement | arguments.py |
|---|---|---|
| `--custom-generate-function-path` | the multi-turn loop (§8.2) | `:453` |
| `--custom-config-path` | YAML with `max_turns`, env path (§8.4) | — |
| `--rollout-interaction-env-path` | your tool/env (`build_env` + `step` + `format_observation`) | (custom config) |
| `--custom-rm-path` | reward, if not using a built-in `rm_type` | `:1327` |
| `--multimodal-keys` | map your image column into `sample.multimodal_inputs` | `:627` |
| `--rollout-function-path` | replace the *whole* outer loop (rarely needed) | `:307` |
| `--rollout-data-postprocess-path` | mutate loss masks after logprobs | `:515` |
| `--custom-loss-function-path` / `--custom-advantage-function-path` | swap the RL objective | `:892` / `:935` |

Full list (~18 hooks) in `docs/en/get_started/customization.md:9-30`.

---

## 10. What this means for a SURDS image-zoom agent (concrete mapping)

To port the §0 SURDS zoom idea onto slime you write **one env module + one config**, no framework
changes:
1. `surds_zoom_env.py` exposing `build_env(sample, args)` and a `Geo3kEnv`-shaped class whose
   `step(response_text)` parses the model's zoom `<tool_call>` (bbox), crops the 1600×900 `.webp`, and
   returns `{"obs_str": "...", "multi_modal_data": {"image": [cropped_pil]}}` → the base
   `format_observation` injects the crop into the next turn (§8.3). `done` when the model emits a final
   `<answer>` or `max_turns` is hit.
2. `surds_zoom_config.yaml`: `max_turns: 3`, `rollout_interaction_env_path: …surds_zoom_env`.
3. Reward: either compute SURDS score inside the env and set it via `--custom-rm-path` (port your
   existing `score_surds`), or keep the env's score as feedback and grade the final `<answer>` with a
   custom RM. **Mind the xy2d coordinate frame** (curriculum gold = pixels, Qwen pred = 0–1000) — the
   same trap as in ms-swift, plus the *separate* bbox-frame question for the zoom crop.
4. Launch like `run_geo3k_vlm_multi_turn.py` but with the Qwen3-VL-8B SURDS-SFT checkpoint
   (HF→Megatron via Bridge), `--multimodal-keys`, your data, `--advantage-estimator grpo`.

The loss-mask machinery (§7.3) and per-turn image injection (§8.3) — the two hardest parts of an
agentic VL loop — are **already built**; you supply tool logic + reward + data.

---

## 11. slime vs ms-swift — the one-paragraph reminder
slime's multi-turn VL path runs on **Megatron** (scales to MoE/large), is a clean pluggable
env+hook contract with strong correctness tooling, and has a near-exact reference
(`geo3k_vlm_multi_turn`). The cost is the Megatron + Ray + SGLang + Megatron-Bridge stack and
HF↔Megatron checkpoint conversion. ms-swift gives you a working SURDS GRPO pipeline *today* with a
one-command launch, but its multi-turn agentic path does **not** run on Megatron — so it can't follow
you to large scale. Use ms-swift for the cheap first probe; treat slime as the long-term home if
agentic-VL becomes the main direction. (Full analysis: this was covered in the prior comparison.)

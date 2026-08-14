# RL Framework Choice for Agentic SURDS Training: ms-swift vs slime vs NVIDIA Molt

*Date: 12 August 2026. Companion to `2026-08-12_deepeyesv2_tool_use_agent_on_surds.md`.*
*Question: for the DeepEyesV2-style tool-use agent, do we keep modifying this vendored ms-swift, or
move to slime or Molt?*

---

## 0. Verdict

**Stay on ms-swift for Phase 1 (zoom-only probe). Read Molt's `geo3k` agent as a reference
implementation before building the sandbox. Re-decide at the Phase 3 boundary, against the explicit
triggers in §8.**

The reasoning in one paragraph: ms-swift's multi-turn multimodal path is *real and verified working
in this checkout* (§3.1) — including the per-turn image injection that is the hard part — and it is
the only one of the three that supports LoRA, which every existing SURDS RL arm uses and which our
8-GPU budget effectively requires. Migrating now costs 2–3 weeks and forfeits comparability with the
whole bake-off. But ms-swift has one genuine correctness defect in the agentic path (§5) that the
other two frameworks were explicitly built to eliminate, and it cannot follow us past one node
because its multi-turn path does not run on Megatron. So the answer is "yes for now", not "yes".

The one thing that would make me reconsider immediately: **Molt ships a multi-turn VLM agent that
executes model-generated Python in a sandboxed subprocess and grades the final `<answer>` block**
(their `geo3k` recipe). That is not *like* the DeepEyesV2 tool — it is the same object. It does not
change the verdict, because of LoRA and migration cost, but it means the sandbox work in Phase 0/3
should start from reading their implementation rather than from a blank file.

---

## 1. What is actually being decided

Not "which framework is best". The decision is: **where does the agentic SURDS work live for the next
3–6 months**, given that we have (a) a working single-turn GRPO pipeline with six trained arms whose
numbers we want to stay comparable to, (b) one node of 8 GPUs, and (c) a build list (sandbox,
multi-turn eval harness, cold-start data pipeline) that is mostly framework-independent.

Weighted criteria, in the order they actually matter here:

| # | Criterion | Why it dominates |
|---|---|---|
| 1 | **Multi-turn multimodal support that works today** | The whole task is image observations fed back per turn. Everything else is secondary |
| 2 | **LoRA support** | Every RL arm in the bake-off is LoRA r128; 8 GPUs shared with vLLM |
| 3 | **Comparability with existing arms** | A framework switch confounds every delta we measure |
| 4 | **Token/loss-mask correctness in the agentic path** | Silent-corruption risk; see §5 |
| 5 | **Cost of the migration** | Reward plugin, eval, wandb, SLURM, data format all re-wire |
| 6 | **Scale headroom** | Matters at the *next* project, not this one |

Note that 6 is where slime and Molt win decisively and where ms-swift loses decisively — and it is
the criterion with the least weight for a one-node experiment. That asymmetry is the whole analysis.

---

## 2. The three frameworks, grounded

### 2.1 ms-swift (this checkout)

Verified by inspection today and in the preceding session:

- **Training**: HF `Trainer`/accelerate + DeepSpeed. No custom parallelism layer.
- **Rollout**: vLLM, two modes. *Colocate* (what all `grpo_bakeoff_*.sh` use: TP=8, `sleep_level 1`,
  `offload_model/optimizer true`) runs vLLM in the training processes and time-slices the GPUs.
  *Server* (`swift rollout`) is a FastAPI service forking `vllm_data_parallel_size` worker processes.
- **Weight sync**: `PyNcclCommunicator` over a `StatelessProcessGroup`, HTTP only for handshake and
  metadata (`swift/rlhf_trainers/vllm_client.py:186-215`). LoRA-aware: pushes adapter tensors when the
  rollout engine has LoRA enabled, otherwise merges and pushes the full model.
- **"Async"**: a single `ThreadPoolExecutor(max_workers=1)` plus a queue (`rollout_mixin.py:93`,
  `1552-1573`) giving exactly one step of off-policy overlap. Not a distributed scheduler.
- **Ray**: present (`swift/ray/`) as an optional device-group placement layer, inert unless
  `--use_ray true`. Not used by our GRPO path.
- **Multi-turn**: works in **both** modes — `_colocate_multi_turn_infer` (`rollout_mixin.py:884`) and
  the server-side scheduler (`swift/pipelines/infer/rollout.py:292`). Megatron path raises
  `NotImplementedError`.

### 2.2 slime (THUDM)

Already documented in depth at `research/plans/DESIGN_slime_architecture.md` (409 lines); not re-derived here.

- **Three planes glued by Ray**: Megatron training actor, SGLang rollout engines + router, shared data
  buffer.
- **Weight sync**: CUDA-IPC when colocated, NCCL broadcast when disaggregated; HF-named tensors via the
  same Megatron↔HF mapping used for checkpoints.
- **Agentic model**: you write a `custom_generate` function plus a tiny `BaseInteractionEnv`
  (`reset`/`step`/`close`/`format_observation`, 26 lines). ~18 documented extension hooks.
- **The reference that matters**: `examples/geo3k_vlm_multi_turn/` — multimodal, multi-turn,
  tool-calling GRPO on Qwen3-VL, running on **4 GPUs colocated with TP=4**. Per-turn image injection
  is a base-class primitive: an env whose `step` returns
  `{"obs_str": ..., "multi_modal_data": {"image": [pil]}}` gets that image into the next turn with no
  framework changes.
- **Loss masking is explicit and correct by construction**: model tokens appended with
  `loss_mask_val=1` and real per-token logprobs from the engine; observation tokens with
  `loss_mask_val=0` and zeroed logprobs.
- **v0.3.0** (2026) added fully-async training as a first-class path, variable global batch size for
  long-tail agentic trajectories, and delta weight sync.
- **Cost**: Megatron + Ray + SGLang + Megatron-Bridge, and an HF↔Megatron checkpoint conversion tax on
  every model in and out.

### 2.3 NVIDIA Molt (`NVIDIA-NeMo/labs-molt`, arXiv:2607.21653, released 22 July 2026)

The newest of the three and explicitly designed for the workload we are about to do.

- **Three components**: Ray (placement + async queue), vLLM (rollout, *unforked*), NeMo AutoModel +
  **FSDP2** (training, with TP/EP/CP). No Megatron.
- **Size as a design goal**: ~8.6K lines for the whole RL path, vs ~62K for verl and ~25K for slime.
  The stated target is a codebase a researcher — or a coding assistant — can hold entirely in context.
- **Two agent forms on one data path**: `Env` (Gym-style, framework drives the loop) and `ChatAgent`
  (user-owned loop; stock OpenAI/Anthropic SDK calls train as-is via a loopback capture server).
- **The headline guarantee**: *"Molt never trains on a token it did not generate."* Prompts enter as
  token IDs, completions return as token IDs plus per-token logprobs, and text never passes through a
  tokenizer mid-episode. See §5 — this is the substantive claim.
- **Async protocol**: partial rollout — in-flight requests are *not* discarded at a weight update;
  engines pause, actor shards broadcast, retained requests resume. Staleness handled by per-token
  importance correction with a sequence-level gate, and the framework **refuses to run partial rollout
  without the correction enabled**.
- **Algorithms**: REINFORCE/++/RLOO, GRPO, Dr. GRPO, GSPO, GAE+critic, DAPO filtering, on-policy
  distillation — as plain functions selected by name.
- **VLM + multi-turn tool use are first-class**, with shipped recipes: Qwen3.6-35B-A3B on geo3k VLM,
  and **`geo3k.py`, a multi-turn Python-tool agent that runs model-generated code in a sandboxed
  subprocess, feeds stdout back as the next turn, and grades the final `<answer>` block**.
- **Scale**: tested to 700B–1T-class MoE at EP256. Single-node 8-GPU quick-starts exist.
- **Maturity**: Apache-2.0, ~904 stars, 190 commits, 4 open issues at time of writing. Docker image
  `hijkzzz/molt:latest`.
- **Limitations the paper states itself**: single backend by design; MoE routing consistency between
  rollout and training is *mitigated* (routing replay) not eliminated; and — importantly —
  **convergence-parity validation "awaits upstream correction"**, because a MoE routing mismatch on
  the benchmark checkpoint filtered sequences, so their headline step-time numbers measure
  "throughput without an effective policy update."
- **LoRA: not supported / not mentioned** anywhere in the paper or repo docs. FSDP2 full-parameter.

---

## 3. Head-to-head on the criteria that decide this

| | ms-swift (ours) | slime | Molt |
|---|---|---|---|
| Multi-turn VL working today | **Yes**, verified in this checkout (§3.1) | Yes, `geo3k_vlm_multi_turn` on 4 GPUs | Yes, `geo3k` VLM recipe |
| Multi-turn **+ code sandbox** reference | No — must build | `coding_agent_rl` (E2B, text) | **Yes — `geo3k.py`, sandboxed Python, image tasks** |
| **LoRA** | **Yes** (adapter-only weight sync) | Yes (Megatron-side) | **No** — FSDP2 full-param |
| Trains on generated token IDs | **No** (§5) — re-templated unless the scheduler returns token IDs | Yes, by construction | **Yes, enforced** |
| Loss-mask discipline for tool tokens | Available but opt-in per scheduler | Explicit, in the reference example | Framework-owned |
| Async | 1 step off-policy, single thread | Fully async (v0.3.0) | Partial rollout + importance correction |
| Training backend | DeepSpeed/accelerate | Megatron (+Bridge conversion) | AutoModel + FSDP2 |
| Rollout engine | vLLM | SGLang | vLLM (unforked) |
| Scale ceiling for *multi-turn* | **1 node** (no Megatron path) | MoE / multi-node | 700B–1T MoE |
| Our reward/eval/wandb/SLURM wiring | **Already done** | Re-wire | Re-wire |
| Comparability with existing 6 arms | **Preserved** | Broken | Broken |
| Codebase size to reason about | large, vendored | ~25K | ~8.6K |
| Maturity for our use | proven here | proven, mature | 3 weeks old; parity unvalidated |

### 3.1 The ms-swift capability I verified, because it is what the decision turns on

Per-turn image injection works in **both** rollout modes. The scheduler's `step()` mutates
`infer_request.images` in place (which carries the crop into the next generation call), and separately
returns `rollout_infos['images']`, which `_postprocess_rollout_outputs`
(`swift/rlhf_trainers/rollout_mixin.py:1279-1285`) uses to override the multimodal columns of the
training batch:

```python
if output.rollout_infos:
    multi_modal_keys = ['images', 'videos', 'audios']
    for key in multi_modal_keys:
        if key in output.rollout_infos:
            input_data[key] = output.rollout_infos[key]
```

That path is shared by colocate and server mode. So the single hardest piece of an agentic-VL loop —
getting a tool-produced image into both the next rollout turn *and* the training batch — is already
solved here. This is the main empirical reason the verdict is "stay".

---

## 4. What migration would actually cost

Framework-independent (must be built regardless, ~60 % of the effort): the sandbox service, the
cold-start data pipeline, the SURDS geometry helpers, and the faithfulness instrumentation.

Framework-specific, i.e. what a migration burns:

| Item | Status on ms-swift | Re-work on slime | Re-work on Molt |
|---|---|---|---|
| SURDS reward (`surds_reward_plugin.py` → `score_surds`) | done, 3 ORMs | port to `--custom-rm-path` | port to a Python reward fn (easy — reward is arbitrary Python) |
| Per-template accuracy logger, cognitive-behavior logger | done | rebuild | rebuild |
| Data format (`solution`/`template_type`/`image_path` columns) | done | `Sample` + `--multimodal-keys` | Env-owned |
| Checkpoint in/out | HF native | **HF↔Megatron via Bridge** | HF native (AutoModel) |
| SLURM + wandb-from-`.env` | done | adapt | adapt |
| LoRA r128 arms | supported | supported | **not supported — arms become full-param** |
| Comparability with the 6 existing arms | preserved | lost | lost |

Realistic estimate: **2–3 weeks** to re-reach today's starting line on either alternative, plus the
LoRA problem on Molt. Against a total project estimate of 3–4 weeks, that is not a rounding error.

The LoRA point deserves emphasis because it is easy to wave away. Going full-parameter on Molt is not
just a memory question (8B full-param RL on 8×H100 with vLLM colocated is tight but arguably
feasible) — it changes the intervention. Every SURDS RL result we have is LoRA r128 all-linear. A
full-param agentic arm that beats a LoRA single-turn arm has confounded tool use with tuner capacity,
and A1 (the compute-matched no-tool control) would have to be re-run full-param too, doubling the
control cost.

---

## 5. The one real correctness argument against ms-swift

Molt's central claim — that a framework should never train on a token the model did not generate —
sounds like marketing until you look at what ms-swift does in the multi-turn path.

In `_postprocess_rollout_outputs`, the training input's `response_token_ids` is set **only if the
scheduler returned it**:

```python
if output.response_token_ids:
    input_data['response_token_ids'] = output.response_token_ids
    if output.response_loss_mask:
        input_data['response_loss_mask'] = output.response_loss_mask
else:
    if not self.multi_turn_scheduler:
        input_data['response_token_ids'] = output.response.choices[0].token_ids
```

Read the `else` branch carefully: when a multi-turn scheduler *is* active and did *not* return token
IDs, no `response_token_ids` is set at all — the training tokens are re-derived by applying the chat
template to the assembled `messages` list. That is a **retokenization** of text the model emitted as
tokens, which is precisely the divergence class Molt is built to refuse.

**This is not hypothetical for us.** The vendored `VisualToolBoxScheduler`
(`examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py:370-406`) returns only
`{'infer_request': ..., 'rollout_infos': ...}` — no `response_token_ids`, no `response_loss_mask`. The
generic `ToolCallScheduler` (`plugin.py:1194-1219`) *does* the right thing, extending
`response_token_ids` and appending zeros to `response_loss_mask` for tool output. So the correct
pattern exists in the repo; the DeepEyes scheduler we were planning to copy does not follow it.

There is a guard rail: at `rollout_mixin.py:977-984` the code checks that the number of rollout
logprobs matches the count of `loss_mask == 1` tokens. That catches gross misalignment, but only when
logprobs are being collected.

**Consequence for the plan, regardless of framework choice:** the SURDS scheduler must return
`response_token_ids` + `response_loss_mask` explicitly — model tokens masked 1, injected
`<tool_response>` tokens masked 0 — following `ToolCallScheduler`, not `VisualToolBoxScheduler`. This
is a small change and it removes the strongest technical argument for migrating. It should go into
Phase 0.

This is also, honestly, the most valuable thing this comparison produced: reading Molt's design
surfaced a latent defect in the ms-swift path we were about to inherit by copy-paste.

---

## 6. The scale argument, and why it doesn't bind yet

ms-swift's multi-turn path is DeepSpeed-only; `swift/megatron/trainers/grpo_trainer.py` wires a
scheduler but the multi-turn branch raises `NotImplementedError`. So ms-swift caps out at what fits on
one node with LoRA. slime and Molt both scale to MoE across many nodes.

That ceiling is irrelevant to this project — we have exactly one node and an 8B student. It becomes
binding the moment the direction is "train a large agentic VL model", at which point ms-swift is not a
candidate at all. Which is why §8 sets triggers rather than declaring ms-swift adequate forever.

---

## 7. Where Molt's `geo3k` agent should be used *without* migrating

The strongest concrete finding is that Molt ships the exact artifact our Phase 0/3 has to build: a
multi-turn agent that executes model-generated Python in a sandboxed subprocess, feeds stdout back as
the next turn's observation, handles image payloads, and grades a final `<answer>` block.

Read it for: sandbox process isolation and timeout handling; how stdout/stderr and rendered images are
packaged into an observation; how the code block is extracted and errors surfaced back to the model;
turn-budget accounting when observations are variable-length. None of that is framework-coupled — it
is the tool-server design, which is our largest unknown. Adopting the *design* while staying on
ms-swift captures most of the value at none of the migration cost.

Same for slime's `BaseInteractionEnv`: its `format_observation` contract (return
`{"obs_str": ..., "multi_modal_data": {"image": [...]}}`) is a cleaner tool-interface boundary than
ms-swift's `step()`-mutates-`infer_request`, and it is worth imitating *inside* our ms-swift scheduler
so that a later port is mechanical.

---

## 8. Recommendation and migration triggers

**Now (Phases 0–2): ms-swift.** Reasons, in order: multi-turn VL verified working here (§3.1); LoRA
supported; comparability with six existing arms preserved; reward/eval/SLURM/wandb wiring already
done; the one correctness defect is a ~20-line fix (§5) rather than a framework limitation.

**Do these three things regardless:**
1. Fix the token-identity issue — SURDS scheduler returns `response_token_ids` + `response_loss_mask`
   (§5). Phase 0.
2. Structure the tool as an env-shaped object (`step` → `{obs_str, images}`) behind a thin adapter, so
   the tool logic is portable and a later migration is mechanical rather than a rewrite.
3. Read Molt's `geo3k.py` and slime's `env_geo3k.py` before writing the sandbox (§7).

**Re-decide at the Phase 3 boundary.** Migrate if any of these fires:

| Trigger | Move to |
|---|---|
| We need a model that does not fit one node with LoRA | slime (Megatron, proven) or Molt |
| Rollout throughput is the binding constraint and 1-step-off-policy overlap is not enough | Molt (partial rollout + importance correction) or slime (fully async) |
| Debugging ms-swift's rollout/training seam is eating more time than the research | Molt — 8.6K lines vs a large vendored tree is the whole point of its design |
| Full-parameter RL becomes necessary anyway | Molt (LoRA objection evaporates) |
| We want DeepEyesV2's actual recipe at their scale (32+ GPUs, full-param, DAPO) | slime or Molt |

**Do not migrate for**: "Molt is newer", "slime is what the big labs use", or a general sense that the
stack is nicer. At one node with LoRA and six comparable arms already on disk, those are not reasons.

---

## 9. What would make this analysis wrong

Stated plainly, since the verdict is a judgement call:

- **If the sandbox turns out to be deeply entangled with the rollout loop** rather than a separate
  service, Molt's ownership of that loop becomes worth much more than I have credited, and the
  "adopt the design, not the framework" move in §7 fails.
- **If ms-swift's colocate multi-turn has correctness problems beyond §5** that only appear under load
  (image-tensor accumulation across turns, KV-cache interaction with `sleep_level`), the fix cost
  could exceed the migration cost. Phase 1 is partly a test of this, and it should be watched for
  explicitly rather than assumed away.
- **If Molt adds LoRA**, the single strongest objection disappears and the calculus changes
  substantially. Worth re-checking their releases at the Phase 3 boundary.
- **I have not run either alternative.** slime's assessment is grounded in the repo's own 409-line
  architecture doc (a code read, not a run); Molt's is grounded in the paper and repo documentation
  only. Neither has been executed on this cluster, and single-node behaviour is where frameworks
  usually disappoint.
- Molt's own paper says convergence parity is not yet validated (§2.3). Adopting it now means being
  the ones who find out.

---

## References

1. NVIDIA NeMo. *Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement
   Learning.* arXiv:2607.21653, July 2026. https://arxiv.org/abs/2607.21653 ·
   code: https://github.com/NVIDIA-NeMo/labs-molt
2. THUDM. *slime — an LLM post-training framework for RL scaling.* https://github.com/THUDM/slime ·
   v0.3.0 release notes (fully-async path, variable global batch size, delta weight sync).
3. ms-swift, this checkout — `swift/rlhf_trainers/rollout_mixin.py`,
   `swift/rlhf_trainers/vllm_client.py`, `swift/pipelines/infer/rollout.py`, `swift/ray/`,
   `swift/rollout/multi_turn.py`, `examples/train/grpo/plugin/`.

**Internal:** `research/plans/DESIGN_slime_architecture.md` (the 409-line slime code-reading guide; §8 and
§10 map directly onto this decision), `research/plans/GRPO_in_msswift_tutorial.md`,
`research/plans/2026-08-12_deepeyesv2_tool_use_agent_on_surds.md` (the plan this choice serves).

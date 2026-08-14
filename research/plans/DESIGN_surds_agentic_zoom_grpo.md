# Design: Agentic (multi-turn) GRPO for SURDS — image-zoom tool

Status: **design only, nothing launched.** Author target: a *basic first* experiment that
answers one question — **can the SURDS student learn anything from a multi-turn
look-closer interaction?** — before investing in a richer tool suite.

The whole design reuses machinery that already exists in this repo. The only genuinely new
code is one scheduler subclass and a system prompt. Everything else (reward, eval, data,
job-script skeleton, wandb wiring) is adapted from what is already committed.

---

## 0. The one-paragraph version

Turn 1: the student sees the full 1600×900 nuScenes frame + the SURDS question, thinks, and
*optionally* emits a `image_zoom_in_tool` call with a bbox. The scheduler crops that bbox out
of the `.webp`, feeds the zoomed crop back as a new image. Turn 2 (3): the student answers
`<answer>...</answer>`. Reward = the **existing** `surds_*` reward on the final answer, plus a
small bonus for *correctly* using the zoom. Trained with the standard GRPO trainer (not
Megatron — multi-turn is unsupported there) on the external async vLLM rollout server.

This is exactly the DeepEyes recipe (`examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py`),
re-pointed at SURDS data + SURDS reward.

---

## 1. Why image-zoom, and why this is the right "basic first"

- The committed diagnosis ([[project-grpo-accuracy-stall-diagnosis]],
  [[project-surds-grpo-v4-pertemplate]]) is that **xy2d** is the weakest subtask — the student
  produces ~30 % *near-miss* points (right object, point lands just outside the 50 px
  tolerance). A look-closer-then-refine loop attacks exactly that failure mode: zoom into the
  candidate region, re-point on the magnified crop, get the pixel precise.
- It needs **one** tool, no sandbox, no external services beyond the rollout/verify vLLM that
  GRPO already runs. Minimum new surface ⇒ cleanest attribution of any gain.
- It is a strict superset test: if a 2-turn zoom loop cannot move xy2d, a heavier
  code-interpreter/Program-of-Thought tool almost certainly won't either, so we'd stop early.

Scope the first run to **xy2d only** (optionally + `lr`/`fb`, which are also localization). Hold
out depth/distance/yaw — zoom doesn't obviously help those and they'd dilute the signal.

---

## 2. Components and exactly what changes

| Component | Reuse | New work |
|---|---|---|
| Multi-turn scheduler | `MultiTurnScheduler` base (`swift/rollout/multi_turn.py`), pattern from `VisualToolBoxScheduler` | **NEW** `SurdsZoomScheduler` — a ~40-line subclass |
| Reward | `surds_reward_plugin.py` (`surds_accuracy` / `surds_dense` / `surds_dense_binary`) | optional thin wrapper adding a **tool-use bonus** (DeepEyes-style) |
| Rollout backend | external async vLLM (`swift rollout`, `vllm_mode=server`, `vllm_use_async_engine=true`) | none |
| Trainer | standard GRPO (`swift rlhf --rlhf_type grpo`) | none |
| Data | `processed/surds/curriculum/L{1,2,3}.jsonl` (already has `solution`, `template_type`, `image_path`) | filter to xy2d (+lr/fb); add tool-format system prompt |
| Eval | `research/eval/score_surds.py`, `score_and_aggregate.py` | **multi-turn eval harness** (see §7 — the real gap) |
| Job script | `slurm_scripts/grpo_bakeoff_*.sh` skeleton + wandb-from-.env pattern | new `pretrain_model_N.sh` |

### 2.1 The scheduler (new file: `examples/train/grpo/plugin/surds_zoom_plugin.py`)

Mirror `VisualToolBoxScheduler` almost verbatim. Key methods:

- `check_finished(...)`: stop if `super().check_finished()` (length / max_turns) OR the last
  completion contains **no** `<tool_call>` (i.e. the model went straight to `<answer>`). Same
  logic DeepEyes uses.
- `step(...)`: parse the `<tool_call>` JSON → `image_zoom_in_tool` with `bbox_2d`; crop via
  `img.crop(bbox)` after `maybe_resize_bbox`; append `<tool_response><image>...</tool_response>`
  as a **user** turn and append the cropped PIL image to `infer_request.images`; return
  `{'infer_request': ..., 'rollout_infos': {'images': infer_request.images}}`. The image-list
  override via `rollout_infos['images']` is the documented multimodal mechanism.

Register `multi_turns['surds_zoom_scheduler'] = SurdsZoomScheduler`.

Loss masking: **do not** use `--loss_scale last_round`. We want gradient on the *tool-call
decision* turn too, not only the final answer. The injected `<tool_response>` text is a *user*
message, so it is already outside the assistant loss — no manual masking needed. (This differs
from the math multi-turn examples, which deliberately train only the last round.)

`max_turns = 3` (allow: answer-direct / one-zoom / two-zoom). Start there; can drop to 2.

---

## 3. ⚠️ The coordinate-frame trap — TWO different frames, do not conflate

This is the single highest-risk part and the thing most likely to silently break the run.
There are **two independent** coordinate questions here:

1. **The `<answer>` xy2d point** — unchanged from current GRPO. Qwen emits **0–1000 normalised**;
   curriculum gold is **pixels**; the reward plugin already reconciles this by passing
   `image_wh` for xy2d (`score_one(..., image_wh=(1600,900))`, default `gold_space='pixels'`).
   This is the repo CLAUDE.md rule and is **already correct** — do not touch it.

2. **The zoom `bbox_2d`** — a NEW frame question the existing rule does *not* cover. The
   DeepEyes `step()` crops with `img.crop(bbox)` in the **pixel space of the fetched image**,
   where `img = fetch_image(...)` has been `smart_resize`d by `qwen_vl_utils` to fit
   `max_pixels`. So the bbox the model emits must be in *that resized pixel grid*, not 0–1000,
   and not the native 1600×900. **This must be verified empirically before trusting any run**,
   because if the model emits the bbox in a different frame than `crop()` expects, every crop is
   garbage and the tool is pure noise (training may still "work" but the tool adds nothing —
   the worst kind of silent failure, exactly the class of bug the CLAUDE.md xy2d section warns
   about).

   **Validation step (cheap, do first):** take 20 xy2d curriculum samples, run the un-trained
   student once with the zoom prompt, dump `(emitted bbox, fetched-image WH, crop)`, and eyeball
   whether the crop actually contains the queried object. Decide the bbox frame from data, then
   hard-code the matching transform in `maybe_resize_bbox`. Write the conclusion into the repo
   CLAUDE.md alongside the existing xy2d frame note so it doesn't get re-discovered next session.

---

## 4. Reward design

Final-turn answer is scored by the **existing** plugin — no new scoring logic. Two additions:

- **Tool-use bonus** (DeepEyes idea, adapted): `+δ` (e.g. 0.2–0.5) **only when** the rollout
  actually called zoom (>1 image in the message history) **and** the final answer is correct.
  This rewards *useful* tool use, not tool use for its own sake, and avoids the degenerate
  "always zoom" or "never zoom" attractors. Implement as a thin `SurdsZoomReward(ORM)` wrapper
  that calls the existing `_score_sample` then adds the bonus by counting images in
  `kwargs['messages']` — same pattern as `DeepEyesReward.compute_score`.
- Keep a **`format` reward** (already a registered ORM) at low weight (~0.2) so the
  `<think>/<tool_call>/<answer>` structure stays well-formed.

For the xy2d-focused run use **`surds_dense_binary`** as the base (sharp step for correct +
small Gaussian pull on near-misses) — it is the one built for the near-miss regime.

Reward funcs: `--reward_funcs surds_zoom format --reward_weights 1.0 0.2` (or
`surds_dense_binary` directly + a separate tool-bonus reward if we keep them decoupled).

---

## 5. Warm start & the format-priming question (decision point)

- Base checkpoint: the consolidation SFT **cp896** ([[project-surds-grpo-v4-pertemplate]]).
  Strong on SURDS answers but **never trained to emit `<tool_call>`**.
- Risk: pure-RL-from-cp896 may rarely emit a well-formed tool call, so the policy never
  *sees* a zoomed crop and never discovers the tool is useful → collapses to single-turn.
- Two options:
  - **(A) Pure RL + format reward** (DeepEyes does this from an *instruct* model). Cheapest;
    relies on Qwen3-VL-Thinking's native tool-call prior surviving the SURDS SFT.
  - **(B) Tiny format-warmup SFT**: a few hundred synthetic 2-turn traces (zoom on the gold
    region → correct point) to teach the schema, then RL. More robust, one extra small job.
- **Recommendation:** try (A) first as the basic probe; if tool-use rate at step ~50 is near
  zero, fall back to (B). This is a go/no-go we read from wandb, not a guess up front.

---

## 6. Metrics — how we decide it worked

Primary (vs the single-turn GRPO xy2d arm, everything else frozen):
- **xy2d pass@1** on val_1k — did it move?
- **pass@1 ↔ pass@8 gap** — the capability-vs-sharpening signal we already track.

Diagnostic (these tell us *why*, and catch the silent-tool-failure):
- **Tool-use rate**: fraction of rollouts that emit ≥1 valid zoom.
- **Conditional accuracy**: acc | zoomed vs acc | not-zoomed. If zoom doesn't lift conditional
  accuracy, the crop frame is probably wrong (→ §3) even if reward rises.
- **Mean turns**, **invalid-bbox rate** (format health).

---

## 7. The real implementation gap: multi-turn EVAL

Training is well-supported; **eval is not multi-turn today.** `score_and_aggregate.py` runs a
single-turn vLLM generate. For an honest comparison the agentic model must be *evaluated with
the same zoom loop it was trained with* (otherwise we test a tool-trained policy with no tool).
Options:
- Reuse the rollout server + `SurdsZoomScheduler` to generate val_1k completions, then feed the
  final-turn text into the existing `score_surds` scoring. This is the cleanest and reuses the
  scheduler — but needs a small driver script.
- Cheaper interim: report **train-time rollout reward** trends + tool-use rate from wandb for
  the go/no-go, and only build the full multi-turn val harness once the probe looks alive.

Flagging this now because it's the part with no existing code, and it gates the headline metric.

---

## 8. Run plan (nothing here is launched without explicit go-ahead)

0. **Frame validation** (§3) — 20-sample dump, decide bbox transform. *No job.*
1. Write `surds_zoom_plugin.py` (scheduler + optional reward wrapper) + system prompt.
2. Build the xy2d(+lr/fb) curriculum subset with the tool-format system prompt.
3. `swift rollout` server: `--vllm_use_async_engine true --multi_turn_scheduler
   surds_zoom_scheduler --max_turns 3`.
4. `swift rlhf --rlhf_type grpo --vllm_mode server --external_plugins surds_zoom_plugin.py
   --reward_funcs surds_zoom format` from cp896, LoRA (match bake-off config).
5. Read tool-use rate + reward at ~50 steps → go/no-go on option (A) vs (B).
6. If alive: build multi-turn val harness (§7), eval, compare to single-turn arm.

### Constraints honored
- Megatron path is out (multi-turn `NotImplementedError`) — use the standard GRPO trainer.
- SLURM job-name stays generic `pretrain_model_N`; real name only in `WANDB_NAME`.
- wandb → **personal `samarjyo`** from `.env`, xtrace-suppressed; **confirm destination before
  launch** ([[feedback-wandb-destination-confirm]]).
- One `pretrain_*` job at a time; validate this one job to a real step before anything else
  ([[feedback-validate-one-job-first]], [[feedback-one-job-and-user-controls-execution]]).
- Large files (data subset, rollout logs, ckpts) → `/mnt/data4/.../research_{data,logs}`.

---

## 9. Open decisions for the user
1. **Subtask scope**: xy2d-only (cleanest) vs xy2d+lr+fb (more data, noisier attribution)?
2. **Format priming**: pure-RL probe (A) first, or build the small warmup SFT (B) up front?
3. **Eval**: build the multi-turn val harness now, or gate it on a positive train-time probe?

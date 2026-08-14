# DeepEyesV2-style Agentic Harness — Architecture

*Date: 13 August 2026. Scope: crop tool + Python/numpy REPL tool on SURDS. Molt is the sole external
reference framework; DeepEyesV2 is a source of artifacts, not architecture.*
*Companions: `2026-08-12_deepeyesv2_tool_use_agent_on_surds.md` (the research plan),
`2026-08-12_rl_framework_choice_msswift_vs_slime_vs_molt.md` (why we stay on ms-swift).*

---

## 0. Summary of decisions

| Decision | Call | Basis |
|---|---|---|
| Adopt DeepEyesV2's harness with search disabled? | **No** | §1 — their released code is not fit to build on, and there is no working search to disable |
| Is DeepEyesV2's code optimal/fast? | **No** | §1.2 — synchronous lock-step rollout, tool work on one rank, unpooled blocking HTTP, shared-kernel session bug, zero tests |
| Framework | **ms-swift**, colocate mode | prior doc; reinforced by §6.4 |
| External reference | **Molt** | §2 — its `Env`/`Result`, TITO trajectory, loss mask and sandbox are directly liftable |
| Core architectural rule | **One tool implementation, used identically by training rollout and eval** | §3 |
| Tools for v1 | `image_crop` + `python_repl` (numpy/PIL, no network) | user decision |

Two things I got wrong in the 12 Aug docs and correct here: the ms-swift token-identity problem is
worse and more specific than described (§6.1), and the crop coordinate frame is now pinned to exact
numbers rather than flagged as unknown (§5).

---

## 1. Should we take the DeepEyesV2 harness and switch off search?

Cloned to `/mnt/sandbox/amar.amarjyoti/research_code/DeepEyesV2` (23 MB) and audited in full.
The answer is no, for three independent reasons.

### 1.1 "Switching off search" is not the question — their search does not exist

`verl/workers/agent/envs/deepeyesv2/search_utils.py:18-46` is a **stub that returns fabricated
results** (`"This is a placeholder snippet for query: …"`). The real search API is not in the repo.
The same is true of the inference demo. The eval path (`vlmeval/api/vllm_api.py`) is already
code-only. So the tool set we would inherit is: one code-execution tool — which is what we want —
plus dead code around it.

Mechanically, disabling search is shallow (7 files, listed in the audit) but *mandatory*, because
`search_utils.py:7-14` does a module-level `open()` of a hardcoded relative path to an MMSearch-R1
cache; the env cannot be imported without it. The prompt that mentions search lives inside their HF
parquet, not the repo, so it would have to be regenerated anyway.

### 1.2 Is their code optimal and fast? Honestly, no

Findings that bear directly on throughput and correctness:

- **The rollout loop is synchronous and lock-step.** `parallel_env.py:111-319` is a plain
  `for step in range(max_turns)` that generates for *all* active sequences, waits for the entire
  batch, then executes tools. With `max_turns=9` and a 200 s code timeout, one slow trajectory stalls
  the whole batch for that turn. No async engine, no per-sequence continuation.
- **Tool execution runs on TP rank 0 only** (`parallel_env.py:198-203`); every other rank blocks in
  `pg.broadcast_object`. On a TP=8 job that is 7 idle GPUs for the entire tool phase.
- **Blocking, unpooled HTTP.** `requests.post` per call with no `Session` — fresh TCP handshake every
  tool call (`deepeyesv2.py:333`). Concurrency is a 2-thread pool. **No retry**: one exception returns
  `None` and kills the trajectory with `done=True` (`:148-150`), silently truncating the episode and
  biasing the GRPO group.
- **The session model is broken.** `session_id` is a *class attribute* evaluated once at import
  (`deepeyesv2.py:76`); `__init__` never sets a per-instance one. Every rollout in a worker process
  shares **one Jupyter kernel**, so each `reset()` overwrites `image_1` and all user variables in a
  namespace other trajectories are using. Their eval path does it correctly per-request
  (`vllm_api.py:487`), so this reads as a regression in the RL path. At `rollout.n=16` this is
  cross-trajectory state corruption.
- **Import-time side effects.** `reward_score/deepeyesv2.py:16-26` builds OpenAI clients and issues an
  unguarded `requests.get(.../models)` **at import**; the module fails to import if the judge is down.
  `api_base` at `:24` is a leaked loop variable, so with multiple endpoints every client points at the
  last one.
- **`fix_python_indentation` (`deepeyesv2.py:53-70`) rewrites the model's code** with a
  `line.endswith(':')` heuristic plus `autopep8 --aggressive`, which corrupts multi-line and
  triple-quoted strings. This is actively harmful and we must not carry it.
- **Dead/broken code**: `reward_score/__init__.py:110` imports a `benchmark` module that does not
  exist (all eight search/simpleqa eval sources raise ImportError); `compute_score_acc` would crash if
  reached; `tool_reward` computed and never used; unconditional full-code prints at 256×16 rollouts.
- **Zero tests** for anything they authored.

### 1.3 The vendoring is the real cost

`reinforcement_learning/verl/` is verl `0.4.0.dev` **vendored in-tree** as a single squashed commit
with no upstream remote — unrebasable. It pins `vllm==0.8.2` / `transformers==4.51.3`
(`scripts/install_deepeyes.sh`), which is Qwen2.5-VL-era and **will not run Qwen3-VL**, our student.
Their launch script assumes `nnodes=4 × 8 = 32 GPUs` for a 7B; we have 8.

Against all that, the genuinely load-bearing authored logic is roughly **900 lines**
(`parallel_env.py` 513 + `deepeyesv2.py` 410). Adopting the harness means inheriting an unmaintainable
fork and a framework migration to obtain 900 lines we would have to fix anyway.

**Verdict: lift artifacts, not architecture.** The specific artifacts are in §2.

---

## 2. What we take from each repo

### 2.1 From Molt (`/mnt/sandbox/amar.amarjyoti/research_code/molt`) — the design

| Artifact | Location | Use |
|---|---|---|
| Python sandbox | `examples/python/tools/python_executor.py` (151 L) | **Lift near-verbatim.** `subprocess.run(["python3","-I","-c",script])` in a `TemporaryDirectory`, `RLIMIT_AS` 1 GiB, wall-clock timeout (default 10 s), output truncated to 2048 chars, **never raises** — errors become observation text |
| Tool interface | same file, `schema` + `execute(arguments)` | Our two tools implement exactly this pair |
| Agent contract | `molt/agents/base.py:208-239` — `async step(state) -> Result` | Shape our tool-loop state object on `Result(reward, observation, terminated, info, images)` |
| Loss mask | `molt/trainer/rollout/samples_generator.py:37-56` `_build_action_token_mask` | 20 pure lines; the mask-over-tool-observations design |
| TITO accumulator | `molt/agents/base.py:122-176` `Trajectory.append_action / append_feedback` | The invariant we must reproduce in ms-swift terms (§6) |
| VLM plumbing | `molt/utils/vlm_utils.py` | `estimate_vllm_input_expansion_delta`, `accumulate_mm_inputs`, `merge_mm_train_inputs` — torch/PIL/transformers only |
| Never trust server placeholders | `router.py:166-193` `_align_features_to_canonical` | The *idea*: recompute image-token offsets from your own canonical ids |
| Eval = same loop | `samples_generator.py:159-183` → same runner as training | §7's central rule |
| Practices | `session_id` per rollout; `asyncio.to_thread` for blocking tools; tool schema via the model's own chat template from a dataset `tools` column | adopt all four |

Two Molt caveats we inherit as risks: `Result.images` plumbing exists but **no shipped example ever
returns images** — the crop-tool path is unexercised there too; and `geo3k.py` hardcodes raw ChatML
control tokens rather than going through `apply_chat_template`, which is correct-but-brittle.

### 2.2 From DeepEyesV2 — artifacts only

| Artifact | Location | Why |
|---|---|---|
| **Code-only system prompt + tool format** | `evaluation/VLMEvalKit/vlmeval/api/prompt.py:20-74` (`AGENT_CONFIG`) | The single best thing in the repo: a validated, search-free system prompt, user template, `<code>```python … ```</code>` format, return template and stop tokens — already externalised as config |
| Sandbox wire format + kernel bootstrap | `vlmeval/api/agent_utils.py:143-181`, `:21-24,183-185` | `image_1 = Image.open(BytesIO(b64))` bootstrap; ~90 useful lines. Fix the `/root/...` paths, add `Session` + retry |
| Qwen-safe resize / white-image filter | `agent_utils.py:42-55,101-110,68-72` | Returned crops must satisfy Qwen's ≥28–32 px constraint |
| Observation-token bookkeeping | `parallel_env.py:236-276`, `_merge_multi_modal_inputs :54-80`, `get_rope_index` recompute `:288-300` | **Read closely, re-derive.** `action_mask=0` / `attention_mask=1` on observation tokens plus `image_grid_thw` merging is the subtle correctness core of interleaved-image multi-turn RL |
| Stop-token trick | `parallel_env.py:117-128` | `custom_stop=["</code>"]` with `include_stop_str_in_output=True`, then force-prefill `<think>` on the resumed assistant turn |

**Explicitly rejected**: the vendored verl tree; their reward module; the 190-line metaclass tool
registry (a dict of callables replaces it); `search_utils.py`; `fix_python_indentation`; the
class-attribute `session_id` pattern.

---

## 3. Architectural principle: one tool core, two drivers

The single rule everything else follows from:

> The tool implementations, the prompt, the parsing, and the observation format live in **one
> framework-independent package**. Training rollout and evaluation are two thin drivers over it.

Rationale: the 12 Aug plan identified "no multi-turn eval harness" as the gating risk, and the
literature's recurring failure is evaluating a tool-trained policy without its tools. If eval and
training share a code path, that class of error becomes impossible rather than merely discouraged.
Molt gets this right structurally (`generate_eval_samples` calls the same runner); DeepEyesV2 does
not (separate VLMEvalKit implementation of the same loop, with different timeouts and retry).

```
        ┌────────────────────────── research/agentic/ (no swift, no torch) ──────────────────────────┐
        │  tools/base.py     Tool protocol: .schema (JSON) + .execute(args, ctx) -> Observation      │
        │  tools/crop.py     image_crop      — bbox -> PIL crop           (pure, in-process)         │
        │  tools/pyrepl.py   python_repl     — code -> stdout/stderr/imgs (sandbox client)           │
        │  sandbox/server.py FastAPI exec service   sandbox/client.py  pooled Session + retry        │
        │  frames.py         pinned coordinate transforms (§5)                                       │
        │  prompts.py        system prompt + tool schemas + stop strings                             │
        │  loop.py           run_episode(generate_fn, sample) -> Episode   ← THE shared turn loop    │
        └───────────────┬──────────────────────────────────────────────────┬─────────────────────────┘
                        │                                                  │
      generate_fn = vLLM engine (in-process)              generate_fn = offline vLLM LLM.generate
                        │                                                  │
  examples/train/grpo/plugin/surds_agent_plugin.py         research/eval/gen_val_agentic.py
   · SurdsAgentScheduler(MultiTurnScheduler)                · drives run_episode over val/heldout
   · returns token ids + loss mask + images                 · writes the SAME parquet schema as
   · SurdsAgentReward(ORM) -> score_surds                     gen_val_ablation.py
                        │                                                  │
              swift rlhf --rlhf_type grpo                    research/eval/score_and_aggregate.py
                                                                (unchanged)
```

`loop.py:run_episode` is the contract seam. It takes a `generate_fn(messages, images, stop) ->
(text, token_ids, logprobs)` and knows nothing about ms-swift, vLLM server mode, or GRPO. Everything
that decides *agent behaviour* — when to stop, how to parse, what the observation looks like — lives
there and is therefore identical in training and eval by construction.

---

## 4. The two tools

### 4.1 `image_crop`

Pure, in-process, no sandbox. Parses a bbox, crops, returns a PIL image.

- **Input**: `{"bbox_2d": [x1, y1, x2, y2]}`.
- **Frame**: see §5. Non-negotiable, and the highest-risk item in the build.
- **Validation** (ported from `VisualToolBoxScheduler.maybe_resize_bbox`, which is sound): clamp to
  bounds, reject inverted/degenerate boxes, reject aspect ratio > 100:1, expand to ≥28 px per side
  centred on the original box.
- **Failure mode**: never raise. Return an observation containing the error text, exactly as Molt's
  executor does. A malformed bbox is a *training signal*, not an exception.

### 4.2 `python_repl`

- **Isolation**: Molt's model — `subprocess.run(["python3","-I","-c",script], cwd=tempdir,
  preexec_fn=set_rlimits, timeout=T, capture_output=True)`. `RLIMIT_AS` 1 GiB, `RLIMIT_CPU`, no
  network (we drop Molt's silence on this and explicitly block sockets in the preamble).
- **Not a Jupyter kernel.** DeepEyesV2 uses a persistent kernel per session; their own session bug is
  a direct consequence. We use a **stateless executor with an explicit preamble**: each call gets a
  fresh process whose namespace is seeded with `img` (the frame), `numpy as np`, `PIL`, and the
  geometry helpers. Persistence across turns is provided by re-executing the accumulated code, not by
  a live kernel. This trades a little compute for the elimination of an entire class of
  cross-trajectory contamination, and it makes a rollout exactly reproducible from its transcript.
- **Preamble**: `img` loaded from the frame path; `np`; `K` (camera intrinsics) and `project` /
  `unproject` / `ground_plane_depth` helpers per the 12 Aug plan §4.3. Never nuScenes 3-D boxes.
- **Outputs**: `stdout`, `stderr`, and images produced via an explicit `show(pil_or_array)` helper
  (avoids depending on matplotlib backend behaviour). Truncate text to 2048 chars; cap images per
  round (DeepEyesV2 uses 10 — we start at 2, since each costs ~1–2k visual tokens).
- **Timeout**: 10 s, not DeepEyesV2's 200 s. A 200 s timeout in a lock-step batch is the throughput
  bug in §1.2; numpy geometry on one image does not need it.

### 4.3 Prompt and call format

Take DeepEyesV2's `AGENT_CONFIG` format (`<code>```python … ```</code>` + `<answer>…</answer>`), which
is validated and pairs with the `custom_stop=["</code>"]` trick. Adopt Molt's practice of shipping
tool schemas through the model's own chat template rather than hand-writing a system prompt, *if*
Qwen3-VL's template renders them cleanly — to be verified in Phase 0, else fall back to the literal
prompt.

---

## 5. Coordinate frames — now pinned (Phase 0 result)

Measured today, not assumed:

- Every SURDS frame is **1600×900** (verified across a 60-file sample of 27,152 training images).
- At our production `MAX_PIXELS=1003520`, `qwen_vl_utils.fetch_image` resizes it to **1316×728** —
  exactly 47×26 patches of 28 px.
- Scale factors are **anisotropic**: `sx = 1316/1600 = 0.8225`, `sy = 728/900 = 0.8089`. They differ
  because each dimension is independently rounded to a multiple of 28. **A single scalar rescale is
  wrong.**

So there are four frames, and `frames.py` owns all conversions:

| # | Frame | Size | Who lives here |
|---|---|---|---|
| 1 | Qwen normalised | 0–1000 | everything the model emits (points, and presumably bboxes — to confirm) |
| 2 | Fetched/model pixel | **1316×728** | what `img.crop()` operates on inside the scheduler |
| 3 | Native SURDS pixel | 1600×900 | curriculum/source-QA gold; the 50 px xy2d tolerance |
| 4 | Sandbox-internal | whatever we hand it | the `img` we seed the REPL with — **we choose; seed with native 1600×900 and document it** |

`VisualToolBoxScheduler.step` crops in frame 2. If the model emits a bbox in frame 1, `crop()` reads
0–1000 as pixels on a 1316×728 canvas — taking roughly the top-left 76 % of width and clipping at
728 in height. Every crop wrong, nothing crashes, training proceeds, the tool contributes nothing.
That is the exact silent failure the plan warned about, and it is now quantified.

**Still to do (needs a live model):** the 20-sample dump to determine which frame the model actually
emits a *bbox* in. Points are 0–1000 by Qwen convention; bboxes are assumed to be but not verified.
Once determined, hard-code the transform in `frames.py` and write the conclusion into the repo
`CLAUDE.md` beside the existing xy2d section.

---

## 6. Token accounting and loss masking — the correctness core

### 6.1 What ms-swift actually does (corrected)

In colocate multi-turn (`swift/rlhf_trainers/rollout_mixin.py:884-1000`):

- For the **final** turn, ms-swift builds `response_token_ids` / `response_loss_mask` itself from
  `response_choice.token_ids`, masking model tokens 1 (`:944-967`).
- For **intermediate** turns it records them **only if the scheduler returns them** (`:1006-1020`).
- `VisualToolBoxScheduler` returns neither. So intermediate turns' model tokens *and* the tool
  observations are both absent, and the `elif not response_token_ids[index]` branch (`:956`) leaves
  training with **only the last turn's tokens**.
- Meanwhile `rollout_logprobs` accumulates every turn (`:1029-1037`), so the count check at
  `:977-984` fails and **rollout importance sampling is silently disabled**.

This is worse than the retokenization I described on 12 Aug, and it is specific to the scheduler we
were about to copy. `ToolCallScheduler` (`plugin.py:1194-1219`) does it correctly — but only for
**text** observations.

### 6.2 The requirement

`SurdsAgentScheduler.step` must return, every turn:

```python
{
  'infer_request':       ...,   # with observation appended as a user turn + images extended
  'response_token_ids':  [...], # this turn's model tokens  +  observation tokens
  'response_loss_mask':  [...], # 1 * len(model tokens)     +  0 * len(observation tokens)
  'rollout_infos':       {'images': infer_request.images, ...},
}
```

### 6.3 Getting image tokens into the observation token ids

This is the hard part, and the reason it is hard is that an observation containing a crop expands to
hundreds of `<|image_pad|>` tokens that must be present, masked 0, and consistent with the
`image_grid_thw` the trainer computes.

The mechanism exists: `Qwen3VLTemplate` extends `Qwen2VLTemplate`
(`swift/template/templates/qwen.py:299,530`) with `placeholder_tokens = ['<|image_pad|>', …]` and a
`replace_tag` (`:321-334`) that expands `<image>` to
`<|vision_start|><|image_pad|><|vision_end|>`; `template.encode()` (`swift/template/base.py:575`)
drives it. So we encode the observation message through the template to obtain its expanded ids and
mask them 0 — the ms-swift analogue of Molt's `_tokenize_feedback` (`molt/agents/base.py:427-452`).

**Note a trap found while reading this**: `replace_tag` calls `fetch_image` *inside* the template
(`qwen.py:331`), so a returned crop is smart_resized **again** on the training side. `max_pixels` is
read from an env var at template init (`qwen.py:1209`, `get_env_args('max_pixels', …)`). In colocate
the rollout and trainer share a process, so they agree. **In server mode they are separate processes
and a `MAX_PIXELS` divergence would silently change the image token count between rollout and
training** — the exact class of mismatch Molt's `_align_features_to_canonical` exists to prevent.
This is a concrete argument for colocate (§6.4), and if we ever move to server mode we must assert
the two agree at startup.

Also inherited from DeepEyesV2's audit: after inserting images mid-sequence, **`get_rope_index` must
be recomputed** (`parallel_env.py:288-300`). Whether ms-swift does this on the multi-turn path is an
open Phase 0 verification item — if it does not, positions are wrong for every post-crop token.

### 6.4 Rollout mode

**Start in colocate.** It matches the bake-off arms (TP=8, `sleep_level 1`, offload), avoids the
`MAX_PIXELS` divergence above, and is the only mode where `completion_length_limit_scope='total'`
works — `multi_turn_completion_length_context` explicitly no-ops in server mode. Note the default is
`'per_round'` (`args_mixin.py:382`), so with `max_turns=4` an unset budget lets a trajectory reach 4×
`max_completion_length`; set `'total'` deliberately.

Colocate's risk is DeepEyesV2's bottleneck: a blocking sandbox call stalling training ranks. Two
mitigations, both from Molt: run tool calls off the event loop
(`asyncio.to_thread` / a thread pool) and keep the timeout at 10 s. Server mode is the escape hatch
if Phase 0's latency measurement says otherwise.

---

## 7. Eval harness

`research/eval/gen_val_agentic.py` — a driver over `loop.run_episode`, replacing
`gen_val_ablation.py`'s single-turn `run_vllm_pass`.

Non-negotiables:
1. **Same `loop.py`, same tools, same prompt** as training. The only difference is `generate_fn`.
2. **Identical output parquet schema** to `gen_val_ablation.py`, so `score_and_aggregate.py` and every
   existing figure/notebook work unchanged and the agentic arm sits in the same table as the six
   existing arms.
3. **A tools-disabled switch** — arm A5 in the plan (evaluate the tool-trained policy without tools)
   is nearly free and is the attribution control.
4. Per-episode logging of: tool-use rate, tool type distribution, turn count, exec-failure rate,
   crop-contains-gold-point (the automatic faithfulness check for xy2d), visual tokens.

Scoring stays `score_surds.score_one` with `gold_space='norm'` on val_1k/heldout — unchanged, and the
repo `CLAUDE.md` rule still governs.

---

## 8. File plan

**Convention: all new work for this project lives under `research/`.**

```
research/agentic/           # framework-independent core (no swift, no torch)
  frames.py           [built] pinned constants + native<->fetched<->norm transforms (§5)
  tools/base.py       [built] Tool protocol, Observation, ToolContext
  tools/crop.py       [built] image_crop
  sandbox/executor.py [built] stateless sandboxed exec (Molt-derived) + geometry preamble
  tests/              [built] 43 tests green
  prompts.py          [todo]  system prompt, tool schemas, stop strings
  tools/pyrepl.py     [todo]  python_repl Tool wrapper over sandbox/executor
  sandbox/server.py   [todo]  optional HTTP service + pooled client, if in-process is too slow
  loop.py             [todo]  run_episode(generate_fn, sample, tools, max_turns) -> Episode
examples/train/grpo/plugin/
  surds_agent_plugin.py [todo] SurdsAgentScheduler + SurdsAgentReward (wraps surds_dense_binary)

research/eval/
  gen_val_agentic.py  [todo]  multi-turn eval driver; emits the SAME parquet schema as
                              gen_val_ablation.py so score_and_aggregate.py and every
                              existing figure keep working unchanged
slurm_scripts/
  pretrain_model_N.sh [todo]  nothing submitted without go-ahead
```

This sits alongside the existing `research/eval/`, `research/cot_lib/`, `research/data_scripts/`
and `research/notebook_builders/`, which is the established layout — `surds_reward_plugin.py`
already resolves `research/eval/score_surds.py` by relative path from the plugin directory, and the
new harness resolves the same way.

Tests are listed because both reference repos' failures are the kind unit tests catch: DeepEyesV2 has
none for its authored code, and Molt's `tests/unit/test_chat_server.py:184,446` and
`test_python_executor.py:35` read as executable specifications worth mirroring.

---

## 9. Phase 0 checklist

| # | Item | Status |
|---|---|---|
| 1 | Clone Molt + DeepEyesV2, audit both | **done** |
| 2 | Pin the fetched-image frame (1316×728, anisotropic) | **done** (§5) |
| 3 | Confirm uniform 1600×900 across the dataset | **done** — 27,152 files, 60-file sample |
| 4 | Locate warm-start ckpt + its config | **done** — cp896 at `sft_runs/stageb_yawxy_1064290/v0-20260619-025339/checkpoint-896`, `max_length=4096` (must be raised) |
| 5 | Determine which frame the model emits a **bbox** in — 20-sample dump | **open**, needs a live model |
| 6 | Verify `get_rope_index` recomputation on ms-swift's multi-turn path (§6.3) | **open** |
| 7 | Sandbox spike + latency under simulated rollout concurrency | **open** |
| 8 | Verify Qwen3-VL chat template renders tool schemas cleanly (§4.3) | **open** |
| 9 | Build `frames.py` + `tools/` + sandbox with unit tests | **done** — 43 tests green |
| 10 | `loop.py` (shared turn loop) + `surds_agent_plugin.py` + `gen_val_agentic.py` | **next** |

### 9.1 Findings from building it

Two bugs the build surfaced that no amount of reading would have, both in the
silent-failure class this harness is designed around:

- **`RLIMIT_AS = 1 GiB` makes `import numpy` hang on this cluster.** Molt's executor
  defaults to a 1 GiB address-space cap. This node has **255 cores**; OpenBLAS reserves
  per-core thread arenas at import, blows the VA cap, and *hangs* rather than failing —
  every tool call would have burned its full 10 s timeout and returned an error. Molt
  never hit it because their geo3k preamble imports only `math`; ours imports numpy,
  which is the entire point of the tool. Fixed two ways: pin BLAS to one thread in the
  child (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` — the actual fix, and correct anyway
  for 16 concurrent rollouts) and raise the default cap to 4 GiB. Test suite went from
  186 s to 7 s.
- **`qwen_vl_utils.fetch_image` defaults to `max_pixels=12845056`**, which yields
  **1596×896** on a SURDS frame, not the 1316×728 the transforms are computed for.
  `VisualToolBoxScheduler._fetched_image` calls it with no `max_pixels`. Every crop
  would have been taken on a canvas 21 % larger than the coordinate math assumed —
  wrong region, no exception. `FrameSpec` now carries `max_pixels`, reads it from the
  `MAX_PIXELS` env var so it cannot disagree with ms-swift's template, and the crop tool
  asserts the fetched size matches the spec.

A third, smaller one: `python3` from `PATH` is the *system* interpreter under conda and
has no numpy, so the sandbox must run `sys.executable`.

Kill condition from the plan stands: if sandbox latency makes a training step ~10× the single-turn
step, redesign before writing more.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Bbox frame wrong ⇒ silent no-op tool (§5) | item 5 above; assert-on-frame in `frames.py`; conditional-accuracy metric would expose it |
| Observation image tokens mis-encoded ⇒ train/rollout mismatch (§6.3) | colocate; unit test that the scheduler's ids round-trip against `template.encode`; assert `MAX_PIXELS` parity |
| `get_rope_index` not recomputed | Phase 0 item 6; if absent, patch or fall back to text-only observations for the REPL |
| Sandbox stalls training ranks (DeepEyesV2's bottleneck) | thread-offload, 10 s timeout, retry with bounded backoff, latency measured in Phase 0 |
| `Result.images`-equivalent path unexercised in *both* reference repos | our crop tool is the first real user; over-test it |
| Context blow-up: 4 turns × crops at `max_length=4096` | raise `max_length`; cap images/round at 2; `completion_length_limit_scope='total'` |
| Molt's ChatML strings are hardcoded; ours must not be | build observations via `template.encode`, never string splicing |

---

## References

- Molt — `/mnt/sandbox/amar.amarjyoti/research_code/molt`; arXiv:2607.21653; `NVIDIA-NeMo/labs-molt`.
- DeepEyesV2 — `/mnt/sandbox/amar.amarjyoti/research_code/DeepEyesV2`; arXiv:2511.05271;
  `Visual-Agent/DeepEyesV2`.
- ms-swift, this checkout — `swift/rlhf_trainers/rollout_mixin.py`,
  `swift/template/templates/qwen.py`, `swift/template/base.py`,
  `examples/train/grpo/plugin/{plugin.py,deepeyes/deepeyes_plugin.py,surds_reward_plugin.py}`.
- Internal — `2026-08-12_deepeyesv2_tool_use_agent_on_surds.md`,
  `2026-08-12_rl_framework_choice_msswift_vs_slime_vs_molt.md`, `DESIGN_surds_agentic_zoom_grpo.md`,
  repo `CLAUDE.md` (xy2d coordinate frames).

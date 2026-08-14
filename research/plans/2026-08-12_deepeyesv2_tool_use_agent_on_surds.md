# Research: A DeepEyesV2-style Agentic Tool-Use Model Trained on SURDS

*Date: 12 August 2026. Status: **research / design only — nothing implemented, nothing launched.***
*Author-facing scope: what it would take, what the papers actually say, what this repo already has,
and what the honest experiment + evaluation plan looks like.*

---

## 0. Executive summary

**The ask.** Reproduce the DeepEyesV2 recipe — a VLM that interleaves reasoning with *executable
code* and *search* tool calls, trained cold-start-SFT → RL — and train it on SURDS driving
spatial-reasoning data.

**The five load-bearing conclusions from this research:**

1. **Drop web search; keep code execution.** DeepEyesV2's two tool families are code execution and
   web search. Search is *structurally useless* on SURDS: the questions are metric-geometric queries
   about a single nuScenes frame (where is this object in pixels, how far, which way is it facing).
   No external corpus answers them. Porting DeepEyesV2 verbatim would burn most of the engineering
   budget on the half of the system that cannot help. The interesting adaptation is to replace
   "search" with a **geometry tool** — the code sandbox pre-loaded with the frame's camera intrinsics
   and a small projection library — which is the SURDS-native analogue of "active knowledge seeking".

2. **The cold-start stage is not optional, and this reverses an earlier repo decision.** The existing
   `research/plans/DESIGN_surds_agentic_zoom_grpo.md` recommends trying pure RL first (its option A) and
   falling back to a format-warmup SFT only if tool-use rate collapses. DeepEyesV2's headline
   negative result is precisely that this fails, and fails *silently in a reward-hacking direction*:
   under direct RL the model converged to emitting exactly one code block per query containing
   non-executable placeholder comments. That is a stronger reason than "the format never emerges" —
   it means the go/no-go signal (tool-use rate > 0) would read as *healthy* while the tool did no
   work. Recommendation: build the cold-start set up front (option B).

3. **The single biggest engineering gap is the sandbox — this repo has none.** `grep -rl
   'sandbox|code_interpreter|python_executor' swift/` returns **nothing**. DeepEyesV2 runs a
   Dockerised Jupyter sandbox as a separate fleet of servers (their README warns you need several
   per GPU node or you saturate bandwidth and time out during rollout). Everything else — multi-turn
   scheduler, GRPO trainer, async vLLM rollout server, SURDS reward — already exists here.

4. **Compute is ~4× short of the reference recipe and forces a LoRA design.** DeepEyesV2 trains
   Qwen2.5-VL-7B full-parameter with "no less than 32 GPUs" for RL. Our SLURM scripts are uniformly
   `--nodes=1 --gres=gpu:8` on `sxm5`. With one node we must (a) keep LoRA (as the existing bake-off
   arms do), (b) shrink rollouts per prompt from 16, and (c) accept that the code sandbox competes
   with training for CPU on the same node.

5. **The SURDS subtask profile says exactly which tool to build first.** From this repo's own
   held-out eval (`research/eval/heldout_metrics/metrics_summary.json`, pass@1):

   | arm | ALL | xy2d | depth | yaw | lr | fb | distance |
   |---|---|---|---|---|---|---|---|
   | Qwen3-VL-8B-Thinking zero-shot (`orig_thinking`) | 0.329 | 0.081 | 0.096 | 0.153 | 0.688 | 0.258 | 0.700 |
   | Qwen3-VL-8B-Instruct zero-shot (`orig_instruct`) | 0.445 | 0.051 | 0.453 | 0.204 | 0.826 | 0.342 | 0.793 |
   | 235B teacher (`teacher_235b`) | 0.622 | 0.727 | 0.408 | 0.444 | 0.832 | 0.529 | 0.793 |
   | best SFT arm (`full`) | **0.662** | 0.703 | 0.556 | 0.502 | 0.832 | 0.571 | 0.811 |

   `lr` and `distance` are saturated (0.81–0.83) — a tool cannot help and they will dilute any
   measured effect. The headroom is **yaw (0.502)**, **depth (0.556)**, **fb (0.571)**, and the
   near-miss structure of **xy2d (0.703)**. Those four map onto two distinct tool affordances
   (crop-and-re-measure; project-and-compute), which is the natural scoping of the first experiment.

**Effort estimate:** ~3–4 weeks of build before the first honest number, of which the sandbox and the
multi-turn *evaluation* harness are roughly half. See §9.

---

## 1. Problem statement

### 1.1 The capability gap

A single-turn VLM answering a SURDS question must, in one forward pass at a fixed input resolution,
(i) locate an object in a 1600×900 nuScenes frame, (ii) recover a metric quantity about it (depth in
metres, yaw in degrees), and (iii) verbalise the answer. The repo's accumulated diagnosis says all
three sub-steps fail in characteristic, *separable* ways:

- **xy2d is a precision failure, not a recognition failure.** ~30 % of predictions are *near-misses* —
  right object, point lands just outside the 50 px tolerance
  ([[project-grpo-accuracy-stall-diagnosis]]). The model knows where the thing is; it cannot resolve
  the pixel at the encoder's working resolution.
- **yaw is a convention failure plus a perception failure.** A 180° convention gap accounts for ~⅓ of
  errors and is fixable by a prompt-level coordinate primer (+4.6 pp greedy); the residual is a 90°
  perception gap that prompting does not touch ([[project-yaw-coord-primer]]). The teachers share the
  flip, so distillation cannot fix it.
- **depth/distance are arithmetic-over-perception failures.** The model must estimate a range and
  compare it to a threshold, entirely in-head, with no ability to check its own estimate.

Each of these is the textbook motivation for tool use. The near-miss failure is what an image crop
fixes; the convention and arithmetic failures are what an executable geometry check fixes. Critically,
**neither is fixed by more RL on the current single-turn setup** — the repo's own conclusion from the
GRPO bake-off is that "RL sharpens, it does not teach" ([[project-grpo-accuracy-stall-diagnosis]]).

### 1.2 The research question

> Does giving a SURDS-tuned VLM an executable, geometry-aware tool interface — and training it, via
> cold-start SFT then RLVR, to invoke that interface *selectively* — produce accuracy gains on the
> metric subtasks (yaw, depth, fb, xy2d) that single-turn RL has been unable to produce; and are
> those gains *faithful*, i.e. attributable to the tool rather than to extra test-time compute?

The second clause is not decoration. It is the failure mode the 2025–26 literature is loudest about
(§2.5) and the one this repo has already been bitten by in a different guise (the xy2d coordinate-frame
bug, which made an entire teacher audit read 0.8 % when the truth was 76 %).

### 1.3 Non-goals

- Web/search tooling. Justified in §0.1; revisit only if the task distribution ever includes
  open-world knowledge (e.g. traffic-sign semantics by jurisdiction).
- GUI/computer use, video, multi-frame. Out of scope.
- Beating SURDS SOTA. The paper's own GRPO-aligned model scores 40.80 overall on its metric; our
  numbers are on a different, held-out split with different tolerances and are not comparable. The
  comparison that matters is **against our own best single-turn arm at matched data**.

---

## 2. Papers read

Five papers were read for this report. Notes are restricted to what is actually load-bearing for the
design; a broader survey of 32 works already exists at
`research/plans/lit_review_multimodal_agentic_tool_use.md` (25 July 2026) and is not duplicated here.

### 2.1 DeepEyesV2: Toward Agentic Multimodal Model — arXiv:2511.05271 (primary)

*Hong, Zhao, Zhu, Lu, Xu, Yu. ICLR 2026 poster. Code: github.com/Visual-Agent/DeepEyesV2.*

**Tools.** Two families. (a) **Python code execution** in a sandboxed Jupyter environment — the paper
groups the emergent uses as *crop, numerical analysis, mark (region annotation), other*; the sandbox
returns stdout/stderr plus any images produced by `plt.show()`, so PIL/matplotlib/numpy outputs come
back into the context as new image observations. (b) **Search** — image search via SerpAPI, text
search via Google. Both tool families are interleavable inside one trajectory, which is the paper's
central claim to novelty over single-tool predecessors.

**Training pipeline — the two-stage argument.** The paper's most transferable result is a *negative*
one. Pioneer experiments with direct RL from an instruct model "fail to induce robust tool-use
behavior": the policy converged to emitting exactly one code block per query, whose content was
non-executable placeholder comments — i.e. it learned to satisfy the format reward without ever
paying the cost of real tool use. This motivates a mandatory cold-start stage whose function is
*establishing the tool-use pattern*, with RL then *refining when to invoke*.

**Cold-start data construction.** Worth copying in structure:
1. **Source pooling** across three task families — perception (V*, ArxivQA, Pixmo Counting, TallyQA,
   SeekWorld), reasoning (ReVisual, MathCoder, ReTool), search (MMSearch-R1, VGR, Chain-of-Focus,
   VLM-R3) — plus generic long-CoT data.
2. **Difficulty filtering by baseline failure**: run Qwen2.5-VL-7B on each candidate and *retain only
   instances it answers correctly at most twice*. This is the mechanism that guarantees the retained
   set is one where tools plausibly help.
3. **Tool-benefit classification** splits the pool: tool-solvable examples → RL set; harder unsolved
   cases → cold-start set.
4. **Trace generation by strong teachers** (Gemini 2.5 Pro, GPT-4o, Claude Sonnet 4) producing
   reasoning with explicit tool-invocation markers; **only trajectories with a correct final answer
   AND error-free code are retained**.

**RL stage.** DAPO. Reward is simply `R = R_acc + R_format` — note there is *no* tool-use bonus term,
unlike DeepEyes v1. Batch size 256, 16 rollouts/prompt, max response 16,384 tokens, lr 1e-6, KL
coefficient 0.0, clip ratios 0.30/0.20 (DAPO's decoupled clipping). Framework: LLaMA-Factory for the
SFT stage, VeRL for RL, Qwen2.5-72B-Instruct served on vLLM as an LLM judge.

**Base models.** Qwen2.5-VL-7B primary (a 32B variant is supported). RL "requires no less than 32
GPUs for 7B training", Ray-orchestrated.

**Results (7B vs Qwen2.5-VL-7B baseline).** MathVerse 52.7 vs 45.6; MathVision 28.9 vs 25.6; ChartQA
88.4 vs 86.2; V* 81.8 vs 78.5; HRBench-4K 77.9 vs 71.6; MMSearch 63.7 vs 12.8. On their own
**RealX-Bench** (300 QA pairs across Daily Life / Media / Sports / Knowledge / Games, 24 % of which
need perception + reasoning + search simultaneously) DeepEyesV2 reaches 22.5 % average vs 13.5 %.

**Behavioural findings that shape our design.**
- *Task-adaptive tool choice emerges*: cropping for real-world perception, marking + numerical
  computation for OCR, arithmetic for charts, math for reasoning benchmarks, search for search tasks.
- *RL reduces tool invocation rate.* Post-cold-start the model over-invokes; RL teaches it to solve
  directly when tools are unnecessary. **This is the diagnostic to watch**: a healthy run should show
  tool-use rate *falling* during RL while accuracy rises.
- Ablation (their Table 5): cold-start needs perception + reasoning + long-CoT components together;
  dropping any one degrades results.

**Note on the excluded half.** MMSearch +50.9 is by far their largest delta and comes entirely from
the search tool. Removing search removes the most impressive number in the paper. Our expected effect
sizes should be calibrated against their *non-search* deltas: **+2 to +7 points**, not +50.

### 2.2 DeepEyes (v1): Incentivizing "Thinking with Images" via RL — arXiv:2505.14362

*Zheng et al., Xiaohongshu / Xi'an Jiaotong. Already vendored in this repo (§3.1).*

One tool: `image_zoom_in_tool`, a bbox crop. The capability "emerges natively, leveraging the model's
own grounding capability as an intrinsic function rather than relying on external specialized models
or APIs" — i.e. no external detector. Trained **end-to-end with RL and no cold-start SFT**, which the
paper presents as a feature, guided instead by tailored data selection and a reward with an explicit
**tool bonus**. Reports "a distinct evolution of active perception from initial exploration to
efficient and accurate exploitation."

**Why both papers matter here.** v1 and v2 disagree about cold start, and the disagreement is
informative rather than contradictory: v1's action space (emit a bbox) is *already latent* in a
grounding-pretrained backbone, so RL only has to re-route an existing capability; v2's action space
(write arbitrary Python) is genuinely novel and cannot be bootstrapped. **Our design straddles both**
— the crop tool is v1-like and might survive pure RL, the geometry/code tool is v2-like and will not.
That asymmetry is a real argument for the staged plan in §9.

### 2.3 SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios — arXiv:2411.13112

*Guo, Zhang et al.* 41,080 training VQA instances and 9,250 evaluation samples built on nuScenes,
across six categories: orientation (yaw), depth estimation, pixel-level localization (xy2d), pairwise
distance, lateral ordering (lr), and front-behind (fb). Proposes a GRPO alignment scheme with
"spatially grounded reward signals — capturing both perception-level accuracy (location) and
reasoning consistency (logic)", plus answer-correctness and format terms. Their GRPO-aligned model
scores 40.80 overall vs GPT-4o 13.30 and Gemini-2.0-flash 35.71.

**Design-relevant reading:** the benchmark is built from nuScenes annotations, which means the
*generative* metadata (3-D boxes, camera calibration, ego pose) exists upstream of every question.
That is what makes a geometry tool possible at all — see §4.3.

### 2.4 & 2.5 Secondary reads, via this repo's own literature review

Two findings from `research/plans/lit_review_multimodal_agentic_tool_use.md` are directly load-bearing and
are restated here with attribution (identifiers verified in that document, not re-verified today):

- **Pixel Reasoner** (Wang, Haozhe et al., 2025) documents *tool reluctance*: a competent text
  reasoner actively avoids a newly offered visual operation because text reasoning is locally cheaper,
  and introduces a **curiosity-driven reward** to counteract it at the reward level rather than via
  SFT. This is the failure mode on the opposite side from DeepEyesV2's placeholder-code hacking, and
  our reward design has to sit between the two (§4.5).
- **CodeV / TAPO** (Hou et al., 2025) supplies the field's most important negative result: "high
  final-answer accuracy often hides unfaithful visual reasoning — models may invoke tools on
  irrelevant regions or ignore tool outputs entirely, yet still guess the correct answer." Its fix is
  process rewards defined on the **tool interface** (was the returned crop sufficient?) rather than on
  the unverifiable CoT. This directly motivates the faithfulness protocol in §7.3, which is the part
  of the evaluation plan I would least want to cut.

---

## 3. What this codebase already provides (and what it does not)

Verified by inspection today.

### 3.1 Present and reusable

| Component | Location | Notes |
|---|---|---|
| Multi-turn rollout base class | `swift/rollout/multi_turn.py` (802 lines) | `MultiTurnScheduler` with `check_finished` / `step`; registry `multi_turns[...]` |
| **DeepEyes v1 zoom scheduler, vendored** | `examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py` | `VisualToolBoxScheduler` — bbox parse, `maybe_resize_bbox`, `img.crop`, `<tool_response><image>` injection, image-list override via `rollout_infos['images']`. Registered as `deepeyes_scheduler` |
| DeepEyes v1 reward, vendored | same file, `DeepEyesReward` | `0.8·acc + 0.2·format + 1.2·tool`, where `tool = 1` iff >1 image in history **and** answer correct. Uses an LLM judge over an OpenAI-compatible endpoint |
| Text-tool scheduler with loss masking | `examples/train/grpo/plugin/plugin.py:1221` `ToolCallScheduler` | Shows the pattern for **text** observations: extend `response_token_ids`, append `0`s to `response_loss_mask`. This is the mechanism a code sandbox needs |
| SURDS reward | `examples/train/grpo/plugin/surds_reward_plugin.py` | `surds_accuracy`, `surds_dense`, `surds_dense_binary`; delegates to `research/eval/score_surds.score_one` |
| SURDS scorer with frame safety rails | `research/eval/score_surds.py` | `score_one(..., image_wh, gold_space=)`, `XY2D_TOL_PX=50`, `DEPTH_TOL_M=4`, `NORM_XY_TOL≈38.5` |
| GRPO trainer + async vLLM rollout server | `swift rlhf --rlhf_type grpo`, `swift rollout` | `vllm_mode=server`, `vllm_use_async_engine=true` |
| SLURM + wandb-from-`.env` skeleton | `slurm_scripts/grpo_bakeoff_*.sh` | xtrace-suppressed key load, `WANDB_ENTITY=samarjyo` |
| Prior design doc for the 1-tool case | `research/plans/DESIGN_surds_agentic_zoom_grpo.md` | Zoom-only GRPO; §3 of it documents the bbox coordinate-frame trap |

### 3.2 Absent — this is the build list

1. **A code-execution sandbox. There is none anywhere in `swift/`.** This is the dominant new
   engineering item: a network service that accepts Python + image payloads, executes with a timeout
   in an isolated environment, and returns stdout/stderr plus rendered images. DeepEyesV2 ships this
   as a separate Docker repo and explicitly warns that a single server per node will time out under
   rollout load.
2. **A geometry/calibration tool layer** (our substitute for search) — see §4.3. No analogue exists.
3. **A multi-turn evaluation harness.** `research/eval/score_and_aggregate.py` runs a single-turn
   vLLM generate. A tool-trained policy evaluated without its tools measures nothing. This is already
   flagged as "the real implementation gap" in the existing zoom design doc and is *more* acute here.
4. **A cold-start trajectory synthesis pipeline** — teacher prompting, sandbox execution during
   generation, correct-answer-and-error-free-code filtering.
5. **Faithfulness instrumentation** (§7.3).
6. **A token-identity fix in the scheduler.** The vendored `VisualToolBoxScheduler` returns no
   `response_token_ids` / `response_loss_mask`, so training tokens are re-derived by re-templating the
   message list — a retokenization of what the model emitted as tokens. Follow `ToolCallScheduler`
   (`plugin.py:1194-1219`) instead: return explicit token IDs with model tokens masked 1 and injected
   `<tool_response>` tokens masked 0. See
   `research/plans/2026-08-12_rl_framework_choice_msswift_vs_slime_vs_molt.md` §5.

### 3.3 Hard constraints inherited from this environment

- **Megatron path is out.** `swift/megatron/trainers/grpo_trainer.py` wires a multi-turn scheduler but
  the multi-turn path raises `NotImplementedError`; use the standard GRPO trainer.
- **One node, 8 GPUs.** All `slurm_scripts` use `--nodes=1 --gres=gpu:8 --partition=sxm5`,
  `--mem-per-gpu=120G`, `--cpus-per-gpu=8`. Against DeepEyesV2's ≥32 GPUs for a 7B.
- **Image-token budget.** Existing runs set `MAX_PIXELS=1003520`. Every tool observation appends an
  image. A 3-turn trajectory with two returned crops is ~3× the visual tokens of the single-turn
  baseline — this interacts directly with the 8-GPU constraint and caps `max_turns`.
- Operational policy: generic `pretrain_model_N` SLURM job names, wandb → personal `samarjyo` with
  destination confirmed before launch, one `pretrain_*` job at a time, large artifacts under
  `/mnt/data4/.../research_{data,logs}`.

---

## 4. Method / architecture

### 4.1 Overview

A 3-stage pipeline over a Qwen3-VL-8B student, warm-started from the existing consolidation SFT
checkpoint (cp896) rather than from the raw instruct model — we already paid for SURDS answer
competence and should not re-learn it.

```
                 ┌──────────────────────────────────────────────┐
   nuScenes +    │  Stage 0: data construction                  │
   SURDS QA  ──► │   • baseline-failure filter (≤2/8 correct)   │
   + calib       │   • tool-benefit split → {cold-start, RL}    │
                 │   • teacher traces, sandbox-executed, filtered│
                 └───────────────┬──────────────────────────────┘
                                 ▼
   cp896  ────►  Stage 1: cold-start SFT on tool trajectories  ────►  π_cold
                                 │
                                 ▼
                 Stage 2: multi-turn GRPO/DAPO with SURDS reward ───►  π_RL
                                 │
      rollout loop:  policy ──<tool_call>──► [ sandbox ] ──<tool_response>──► policy
                                                  │
                                    crop / mark / project / compute
```

### 4.2 Tool 1 — `image_zoom_in_tool` (v1-style, near-free)

Straight port of `VisualToolBoxScheduler`. Targets the xy2d near-miss and `fb`. This is the cheapest
possible first tool because the code exists; the only new work is re-pointing it at SURDS data and
resolving the bbox frame question.

> **Coordinate-frame warning, carried forward and expanded.** There are now *three* frames in play,
> not two. (i) The `<answer>` xy2d point: Qwen emits 0–1000 normalised, gold frame varies by dataset
> (curriculum = pixels, SFT/val_1k/heldout = normalised) — handled correctly today by
> `score_one(..., gold_space=)`, do not touch. (ii) The zoom `bbox_2d`: `VisualToolBoxScheduler.step`
> crops in the pixel space of the **`smart_resize`d** image returned by `qwen_vl_utils.fetch_image`,
> which is neither 0–1000 nor native 1600×900. (iii) **New with this design** — any coordinate the
> model computes *inside the sandbox* is in whatever frame the sandbox handed it. All three must be
> pinned empirically (20-sample dump, eyeball the crops) before any run is trusted, and the
> conclusion written into the repo `CLAUDE.md`. A wrong frame here does not crash — it silently
> converts the tool into noise while every dashboard still looks alive.

### 4.3 Tool 2 — `python` sandbox with a SURDS geometry preamble (the substantive contribution)

This is the DeepEyesV2 code tool, plus the domain adaptation that replaces search.

**Interface.** Model emits Python inside a delimiter; the scheduler POSTs `{code, image_refs,
session_id}` to a sandbox service; the service returns `{stdout, stderr, images[]}`; the scheduler
injects a `<tool_response>` user turn carrying the text and any images, with the text tokens
**loss-masked** (the `ToolCallScheduler` pattern) and the images appended via `rollout_infos['images']`
(the `VisualToolBoxScheduler` pattern). Session state persists across turns within a trajectory —
Jupyter semantics, as in DeepEyesV2.

**Preloaded namespace.** `numpy`, `PIL`, `matplotlib`, plus:
- `img` — the frame as a PIL image, in a *documented* frame;
- `K`, `R`, `t` — the camera intrinsics/extrinsics for this sample, from nuScenes calibration;
- `project(xyz) -> (u,v)` and `unproject(u, v, depth) -> xyz`, `ground_plane_depth(v)` — thin helpers.

**Why this is the right substitution for search.** DeepEyesV2's search tool exists to *acquire
information not present in the image*. On SURDS the missing information is not semantic but
*calibrative*: the ego-frame geometry that turns a pixel into a metre. Handing the model `K` and a
projection helper is the same move — an external oracle that supplies what perception alone cannot —
and it targets the three weakest subtasks directly:
- **depth (0.556)**: pick the object's ground-contact pixel, call `ground_plane_depth`, get metres
  instead of guessing them;
- **yaw (0.502)**: draw the ego-frame axes onto the crop with matplotlib and *look* at the result,
  which is a perceptual fix for the 90° gap and a mechanical fix for the 180° convention gap the
  coord-primer only partially closed;
- **fb (0.571)**: a sign test on a projected coordinate rather than a verbal judgement.

**Honest risk.** This tool is *powerful enough to be a shortcut*. If `ground_plane_depth` is accurate,
the model can in principle solve depth without perceiving anything beyond one pixel — which is a
capability gain for the system but not evidence about the VLM. §7.3 exists partly to detect this, and
it is a genuine confound the writeup must state rather than bury.

**Explicitly out of scope for v1 of the tool:** anything that reads nuScenes 3-D box annotations.
That is the label. Exposing it is answer leakage, not tool use.

### 4.4 Turn budget and termination

`max_turns = 4` (direct answer / one tool call / two / three). Termination when the last completion
contains no tool call, or the cap is hit — the v1 `check_finished` logic. Mini-o3's over-turn masking
(train at a low cap, extrapolate at inference) is worth adopting later but is not a v1 requirement.

Loss masking: gradient **on** the tool-call decision tokens (that is the behaviour we are training),
**off** on injected tool output. Do not use `--loss_scale last_round`.

### 4.5 Reward design

DeepEyesV2 uses only `R_acc + R_format`. DeepEyes v1 adds a `1.2 × tool` bonus. Pixel Reasoner adds a
curiosity bonus. These three positions bracket a real tension, and the resolution depends on which
stage we are in:

**Proposed reward:**

```
R = 1.0 · R_acc            # existing surds_dense_binary, unchanged
  + 0.2 · R_format         # existing `format` ORM
  + β  · R_tool            # β = 0.3 initially, ANNEALED TO 0
  − γ  · R_exec_fail       # γ = 0.2, penalty for code that raises
```

- `R_acc`: **reuse `surds_dense_binary` unchanged.** It was built for the near-miss regime (sharp step
  for correct, small Gaussian pull otherwise) and is the reason a tool that *narrows* the miss gets
  credit before it starts converting misses into hits. Reusing it verbatim also keeps the agentic arm
  numerically comparable to every existing single-turn arm — do not invent a new accuracy reward.
- `R_tool = 1` iff a tool was invoked *and* the final answer is correct (the v1 conditional form,
  which avoids both "always call" and "never call" attractors). **Annealed to zero** over training —
  this is the synthesis: the bonus exists to defeat tool reluctance early (Pixel Reasoner's problem),
  and must vanish so the model can learn the *not*-calling decision that DeepEyesV2 observes emerging
  post-RL. Keeping β fixed would prevent the paper's headline behavioural result from ever appearing.
- `R_exec_fail`: a direct counter to the placeholder-code hack. Code that does not run is worse than
  no code.

Deliberately **not** proposed for v1: CodeV-style dense process rewards on tool outputs. Correct in
principle (§2.5) but requires a per-sample notion of "sufficient crop" that SURDS does not define. It
belongs in the faithfulness *evaluation* (§7.3) first; promote it to a reward only if the evaluation
shows unfaithfulness is the binding problem.

### 4.6 Cold-start data construction

Mirror DeepEyesV2's structure with our assets:

1. **Difficulty filter.** Run the current best single-turn arm (`full`, ALL 0.662) at 8 samples over
   the SURDS training pool; retain items answered correctly ≤2/8. On the observed profile this will
   concentrate hard on yaw/depth/fb/xy2d and will strip most of `lr`/`distance` automatically — the
   filter does the subtask scoping for us, which is more principled than hand-picking templates.
2. **Tool-benefit split.** Held-out portion of the retained set → RL; the rest → cold start.
3. **Trace generation.** We already have a 235B teacher in-pipeline (`teacher_235b`, xy2d 0.727) and
   the infrastructure to run it. Prompt it with the tool schema + sandbox, execute its code for real,
   and keep only trajectories with **correct final answer and error-free code**. Reusing our own
   teacher instead of Gemini/GPT-4o/Claude keeps this self-contained and avoids an external-API
   dependency in the data path.
4. **Composition.** DeepEyesV2's ablation says the cold-start mix needs perception + reasoning +
   long-CoT together. Our analogue: crop-style trajectories (xy2d/fb) + compute-style trajectories
   (depth/yaw) + a slice of *tool-free* long-CoT SURDS traces from the existing SFT set. That last
   component is the one most likely to be dropped for convenience and the one their ablation says not
   to drop — it is presumably what teaches the model that not calling a tool is a legal action.

### 4.7 Optimiser

DAPO, matching DeepEyesV2 and already available in the repo's RL bake-off. Their hyperparameters
(lr 1e-6, KL 0.0, clip 0.30/0.20) are a reasonable starting point; batch 256 × 16 rollouts is not
affordable on one node and must come down (§6.3). LoRA r128 all-linear, matching the existing
bake-off arms so the comparison stays controlled.

---

## 5. Data plan

| Set | Source | Purpose | Est. size |
|---|---|---|---|
| Difficulty-filtered pool | SURDS train (41,080 instances) filtered ≤2/8 by the `full` arm | parent pool | ~8–15 k (rate to be measured) |
| Cold-start SFT | teacher-generated, sandbox-executed, correctness+no-error filtered | Stage 1 | target 3–6 k trajectories |
| RL prompts | tool-solvable half of the filtered pool | Stage 2 | 4–8 k |
| val_1k | existing | in-training monitoring | 1 k |
| held-out eval | existing `heldout_val_meta.parquet` | headline numbers | 1,998 |

Sizing note: DeepEyesV2 does not publish per-category cold-start counts; the community reference
point for code-tool cold start is Thyme's 500 k, which is two orders of magnitude beyond us. We are
teaching a *narrow* action space (four helper functions, one image), so a few thousand verified
trajectories is a defensible target — but this is the single least-grounded number in the plan and
should be revisited after the first cold-start run's tool-format success rate is measured.

All artifacts → `/mnt/data4/.../research_data/`.

---

## 6. Experiments

### 6.1 Arms

Everything frozen except the intervention; all arms warm-start from cp896 and use LoRA r128.

| # | Arm | Purpose |
|---|---|---|
| A0 | `full` single-turn SFT (existing) | the incumbent to beat |
| A1 | single-turn GRPO on the filtered pool | **compute-matched no-tool control** — isolates "RL on hard data" from "tools" |
| A2 | cold-start SFT only (π_cold), tools at inference | how much is format acquisition alone? |
| A3 | **cold-start + RL (π_RL)** — the full recipe | headline |
| A4 | RL directly from cp896, no cold start | replicates DeepEyesV2's negative result on our data; cheap, and the answer is genuinely informative either way |
| A5 | A3 evaluated with tools **disabled** | attribution: how much of A3's gain survives without the tool? |

A1 and A5 are the two arms that make this a controlled experiment rather than a demo, and they are
the two most likely to be dropped under time pressure. A5 in particular is nearly free — same
checkpoint, different eval config.

### 6.2 Ablations (in priority order, only if A3 > A1)

1. **Tool set**: zoom only / geometry only / both. Answers which affordance carries the effect.
2. **β annealing schedule**: fixed β vs annealed. Tests the §4.5 synthesis directly.
3. **Cold-start composition**: drop the tool-free long-CoT slice (their Table 5 ablation on our data).
4. **`max_turns`** 2 / 4 / 6 against token cost.

### 6.3 Compute plan

One node, 8×GPU, `sxm5`. Per arm, roughly: rollout server on 2 GPUs, training on 6, sandbox workers on
the node's CPUs (`--cpus-per-gpu=8` → 64 cores; sandbox concurrency must be sized against rollout
concurrency or it becomes the bottleneck — DeepEyesV2's README is explicit that this is where their
setup falls over). Rollouts/prompt reduced from 16 → 8; batch reduced proportionally. Expect
significantly longer wall-clock per step than the single-turn bake-off arms because every tool call is
a synchronous round-trip *and* adds visual tokens.

Sequencing respects the standing one-`pretrain_*`-job-at-a-time rule; A4 (cheapest, most diagnostic)
should probably run first.

---

## 7. Evaluation

### 7.1 Primary metrics

On the existing 1,998-item held-out split, scored with `score_surds` **unchanged** (`gold_space='norm'`
for this split — the latent footgun documented in the repo `CLAUDE.md`):

- pass@1 and pass@8, **per template**, with `xy2d / depth / yaw / fb` as the pre-registered targets
  and `lr / distance` reported but excluded from the headline (saturated).
- The pass@1 ↔ pass@8 gap, which this repo already uses as its capability-vs-sharpening signal.

**Pre-registration matters here.** With six templates and multiple arms, post-hoc selection of "the
subtask that improved" is the obvious way to fool ourselves. Targets are fixed by §0.5 before any run.

### 7.2 Agentic-behaviour metrics (from wandb + rollout dumps)

- **Tool-use rate** overall and per template. Expected trajectory: high after cold start, **falling**
  during RL — DeepEyesV2's signature of adaptive invocation. A flat-and-high rate suggests the bonus
  is not annealing; a flat-and-zero rate suggests tool reluctance.
- **Task-adaptive tool choice**: distribution over {crop, mark, project, compute} per template. The
  DeepEyesV2 result predicts crop-heavy on xy2d/fb and compute-heavy on depth/yaw. If the
  distribution is uniform, the model is calling tools ritually.
- **Execution failure rate** — the placeholder-code-hack detector.
- **Conditional accuracy**: acc | tool-used vs acc | not-used, per template. If tool use does not lift
  conditional accuracy, the tool is decorative *even if reward rises*.
- Mean turns, mean visual tokens/trajectory.

### 7.3 Faithfulness protocol (the part not to cut)

Motivated by CodeV: high accuracy routinely coexists with tools invoked on irrelevant regions or
outputs ignored. SURDS is unusually well suited to checking this cheaply, because for xy2d we have a
gold point:

- **Crop containment**: does the emitted bbox contain the gold xy2d point? Fully automatic, no judge.
  Report `P(correct | crop contains gold)` vs `P(correct | crop misses gold)`. If those are equal, the
  crop is doing nothing.
- **Output dependence**: re-run a sample of trajectories with the tool response corrupted (crop
  replaced by a random region; sandbox stdout perturbed) and measure accuracy drop. No drop ⇒ the
  model is ignoring its own tool.
- **Leakage audit** for the geometry tool (§4.3): what fraction of correct depth answers are
  reproducible from `ground_plane_depth` alone with no visual grounding? This bounds how much of the
  gain is the tool answering rather than the model reasoning.

### 7.4 Cost accounting

Report tokens and wall-clock per question for every arm. A multi-turn agent spends 3–10× the
inference compute of a single-turn baseline; "tools beat no tools" is not a result unless it survives
compute matching. A1 is the compute-matched control; if A3 beats A0 but not A1, the honest headline is
"hard-data RL helped, tools did not."

### 7.5 The gating implementation risk

There is **no multi-turn evaluation harness today.** Until one exists, every number above is
unmeasurable, and the temptation will be to report train-time rollout reward instead — which is not
comparable to any existing arm. Building this (drive `swift rollout` + the scheduler over the held-out
split, feed final-turn text into the existing `score_surds`) is on the critical path, not a follow-up.

---

## 8. Risks

| Risk | Severity | Signal | Mitigation |
|---|---|---|---|
| **Silent coordinate-frame error** (3 frames now, §4.2) | **Highest** — has already bitten this project twice | conditional accuracy flat despite rising reward | 20-sample crop dump before any run; assert-on-frame in the scheduler; write the conclusion into `CLAUDE.md` |
| Sandbox is the throughput bottleneck | High | rollout step time ≫ single-turn | size workers against rollout concurrency; short timeouts; measure before scaling |
| Placeholder-code reward hacking (their result) | High | high tool-use rate + high exec-failure rate + flat conditional accuracy | cold start; `R_exec_fail`; A4 measures it directly |
| Tool reluctance (Pixel Reasoner's result) | Medium | tool-use rate → 0 in early RL | β bonus early, annealed |
| Geometry tool leaks the answer | Medium | depth gain ≫ all other gains | §7.3 leakage audit; never expose 3-D box labels |
| 8 GPUs insufficient for stable DAPO at reduced batch | Medium | reward variance, collapse | LoRA, fewer rollouts, gradient accumulation; fall back to GRPO |
| Effect size is small (+2–7 pts, per §2.1) and the split is noisy | Medium | overlapping CIs | pre-registered targets; bootstrap CIs on 1,998 items; report pass@8 too |
| Visual-token blow-up at `max_turns=4` | Low–Medium | OOM / truncation | cap `MAX_PIXELS` on tool-returned crops; start at `max_turns=2` |

---

## 9. Phased plan and effort estimate

Nothing here is launched without explicit go-ahead, and each phase has a kill condition.

**Phase 0 — frame validation + sandbox spike (3–4 days).** 20-sample bbox dump against the vendored
`VisualToolBoxScheduler`; stand up a minimal sandbox service and measure round-trip latency under
simulated rollout concurrency. *Kill condition: if sandbox latency makes a step 10× the single-turn
step, redesign before writing anything else.*

**Phase 1 — zoom-only agentic GRPO (~1 week).** Essentially the existing
`DESIGN_surds_agentic_zoom_grpo.md`, now with the cold-start recommendation reversed. One tool, no
sandbox, no new data pipeline. *This is the cheapest real number in the whole plan* and it gates the
rest: if a crop loop cannot move xy2d, a heavier code tool will not either.

**Phase 2 — multi-turn eval harness (3–5 days).** Blocking for any honest comparison; can overlap
Phase 1's training time.

**Phase 3 — cold-start data pipeline (1 week).** Difficulty filter, teacher trace generation with live
sandbox execution, correctness + no-error filtering.

**Phase 4 — full recipe (A2/A3/A4) + ablations (1–2 weeks).**

**Total: ~3–4 weeks to the first defensible headline number**, of which roughly half is sandbox +
eval-harness infrastructure rather than modelling.

---

## 10. Open questions for the user

1. **Search: confirmed out?** §0.1 argues it is structurally useless on SURDS. If the intent is to
   reproduce DeepEyesV2 *as a system* (including search) rather than to improve SURDS, the design
   changes substantially and Phase 3 grows.
2. **Is the geometry tool acceptable, or does it feel like cheating?** It is the most promising and
   the most confounded component. A defensible middle ground is to ship it but treat depth/yaw gains
   as "system" results and xy2d/fb gains as "model" results.
3. **Phase 1 first, or straight to the full recipe?** Phase 1 costs a week and could save four.
4. **Backbone: stay on cp896, or take the opportunity to move?** Staying keeps comparability with
   every existing arm; that is the recommendation.

---

## References

1. J. Hong, C. Zhao, C. Zhu, W. Lu, G. Xu, X. Yu. *DeepEyesV2: Toward Agentic Multimodal Model.*
   arXiv:2511.05271, 2025 (v4). ICLR 2026 poster. https://arxiv.org/abs/2511.05271 ·
   code: https://github.com/Visual-Agent/DeepEyesV2
2. Z. Zheng et al. *DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning.*
   arXiv:2505.14362, 2025. https://arxiv.org/abs/2505.14362
3. X. Guo, Zhang et al. *SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios
   with Vision Language Models.* arXiv:2411.13112, 2024. https://arxiv.org/abs/2411.13112
4. H. Wang et al. *Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven
   Reinforcement Learning.* 2025 — via `research/plans/lit_review_multimodal_agentic_tool_use.md` [#20].
5. Z. Hou et al. *CodeV / TAPO: process rewards on visual tool inputs and outputs; faithfulness of
   visual tool use.* 2025 — via `research/plans/lit_review_multimodal_agentic_tool_use.md` [#8].
6. Q. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025 — via the same
   review [#28]. The RL algorithm DeepEyesV2 uses.

**Internal documents this builds on:** `research/plans/DESIGN_surds_agentic_zoom_grpo.md` (the one-tool
precursor; its §3 coordinate-frame warning is extended in §4.2 here),
`research/plans/lit_review_multimodal_agentic_tool_use.md` (32-work survey, 25 July 2026),
`research/eval/heldout_metrics/metrics_summary.json` (the baseline table in §0.5), repo `CLAUDE.md`
(xy2d coordinate frames — mandatory reading before touching any scoring path).

# State of the Art: Multimodal Agentic Reasoning, Tool Use, and RL for Spatial/Metric Tasks

**Written:** 2026-07-05 · **Scope:** what's SOTA *now* in VLM tool-calling + RL, judged specifically against our
SURDS problem (Qwen3-VL-8B, nuScenes 1600×900, weak subtasks = **yaw** and **depth**).

**Provenance & completeness.** Assembled from a parallel literature survey (4 threads; 3 returned). The
RL-reward-shaping thread did not complete, so **§5 is thinner than the rest** and a few named papers were
never chased (VLM-R1, Visual-RFT, Perception-R1, and the pass@k "does RL incentivize capacity beyond base"
analysis, arXiv 2504.13837). **§8 lists every claim that is single-extraction or unverified — read it before
citing anything from this doc externally.**

---

## 0. Executive summary — the five things that change our plan

1. **An SFT-only method beats SURDS's own GRPO model on SURDS, by a lot.** arXiv **2603.06985** (Mar 2026)
   reports **overall 68.07 vs 40.80**, and **yaw 49.11 vs 20.97**, using *Visual Reference Tokens* and
   explicitly **no RL**. On our exact benchmark. This is the single most important datapoint in the survey.
2. **On metric depth, GRPO ≈ SFT — at 8–16× the compute.** DepthLM (ICLR 2026 **Oral**, Meta FAIR,
   arXiv 2509.25413) trains metric depth on nuScenes among others and finds SFT and GRPO "perform similarly
   well," and that **CoT provides zero benefit** ("the model shares similar reasoning traces for all inputs
   after GRPO training"). Metric depth is *perception*, not reasoning.
3. **Tool use has now been directly tested against tool-free controls — and largely fails.** Three independent
   2026 papers: **2606.02357** (*"93% of DeepEyesV2's tool-solved problems and 96% of Thyme's are also solved by
   at least one non-tool setting… agents learn tool-calling patterns more reliably than tool-contributed
   capabilities"*); **MED, 2602.01334** (ICML'26) which decomposes the gain and finds **tool contribution ratio
   0.22 on Qwen3-VL-8B — our exact model — i.e. 78% of the learning is intrinsic**; and DriveAgent-R1's own
   tool-vs-no-tool row at **+0.15 pp**. Most damning: **2606.00096** (ICML'26) runs on **3DSRBench orientation
   and CV-Bench-3D depth** — our subtasks — and finds a **2%→100% swing in tool-call rate buys +0.7 pp, while
   plain entropy regularization buys +3.7 pp** (see §5.5).
4. **We now have a *mechanism* for our 180° yaw flip.** COMFORT (ICLR 2025 **Oral**) shows VLMs default to a
   **"reflected" egocentric frame — left/right preserved, front/back reversed** — and are far weaker on the
   sagittal axis than the lateral. That is axis-correct/sign-wrong, exactly our failure. It also explains why
   the **teacher shares the flip** (shared language prior, not shared perception error) and why our **primer
   only fixed ~⅓** (COMFORT: explicit frame instruction drops all models to chance).
5. **Our metrics aren't comparable to the published SURDS table.** Published SURDS scores **yaw as 4-way
   cardinal classification** and **depth as binned classification**; we run metric depth in meters (±4 m) and
   hard 50 px L2. **We are running a strictly harder continuous variant.** That's a contribution, but it means
   no published baseline exists for our numbers.

**Bottom line for the "move to DeepEyes/REPL agent?" question: no.** This is no longer a judgment call — it's
tested. Reward design, input representation, and a convention-consistent SFT are all better-evidenced levers.
Details in §7. **Two cheap, high-value experiments fall out immediately: FlipEval on the yaw set (§3.4) and
switching pixel reference from text coordinates to a drawn marker (§2).**

---

## 1. The tool-calling landscape (post-DeepEyes)

Tool-calling VLMs for spatial reasoning is now a crowded subfield. The ones worth knowing:

| Paper | arXiv / venue | Base | Tools | Method | Headline |
|---|---|---|---|---|---|
| **TIGeR** | 2510.07181 | GLM-4.1V-Thinking | intrinsics, extrinsics, **metric depth**, SAM2, 2D→3D box, projection, **code executor** | SFT → RFT on TIGeR-300K | 79.30% avg, +5.83 over Gemini-2.5-Pro |
| **SpaceTools** | 2512.04069 (CVPR'26, NVlabs) | Qwen2.5-VL-**3B** | SAM2, DepthPro, RoboRefer+Molmo, GraspGen | **DIRL** (double interactive RL) | RoboSpatial-Home 79.38; **DIRL +12% over SFT, +16% over vanilla RL** |
| **DriveAgent-R1** | 2507.20879 (ICLR'26) | Qwen2.5-VL-3B | **RoI zoom**, Depth Anything V2, DetAny3D, history retrieval | DM-SFT → FCM-RL → AMS-RL (**MP-GRPO**) | nuScenes 47.10% (GPT-5 45.14) — **but tool-vs-no-tool = +0.15 pp** |
| **AgentThink** | 2505.15298 (EMNLP'25) | Qwen2.5-VL-7B | Depth Anything V2, YOLO-World, Agent-Driver | SFT → GRPO | DriveLMM-o1 overall **80.51** vs base 46.44 |
| **S-Agent** | 2606.20515 | **Qwen3-VL-8B** | 3-level: GroundingDINO → Depth-Anything-3 → 5 experts incl. **Visual Orientation** | training-free planner + distilled 8B (S-300K) | MMSI **46.4%**; distilled 8B **41.6 vs 31.1 base** |
| **SpatialClaw** | 2606.13673 (NVIDIA) | any (26B–397B) | **persistent Python kernel** + Depth Anything 3 + SAM3 + NumPy/SciPy | **training-free, code-as-action** | **59.9% avg over 20 benchmarks, +11.2 pp over SpaceTools** |
| **Think3D** | 2601.13029 | GPT-4.1 / Qwen3-VL-4B | 3D manipulation tools | zero-shot plugin + Think3D-RL | Qwen3-VL-4B: **+0.7% → +10.7% once RL applied** |
| **Visual Sketchpad** | 2406.09403 | GPT-4o | detectors, segmenters, **depth colormaps** | prompting | V\*Bench 80.3, BLINK spatial 83.9 |

**Two structural lessons:**

- **Small models need RL to use tools at all.** Think3D: zero-shot tool access gives Qwen3-VL-4B **+0.7%**;
  with RL it becomes **+10.7%**. Our 8B is in this regime — a tool without RL will likely do nothing.
- **But a good tool *interface* can beat tool-specific RL entirely.** SpatialClaw is training-free and beats
  RL-trained SpaceTools by 11.2 pp on a capable backbone. The lesson is that tool *plumbing quality* matters
  more than tool *RL* — which cuts against investing heavily in a bespoke RL tool loop at 8B.

### Code / Python REPL as a visual tool — the direct answer to our question

The RL-trained code-tool line is real and reproducible: **Visual-ARFT** (2505.14246), **VTool-R1**
(2505.19255), **Thyme** (2508.11630, GRPO-ATS), **DeepEyesV2** (2511.05271, DAPO), **CodeV** (2511.19661,
CVPR'26 Oral, TAPO), mostly on Qwen2.5-VL-7B. **PyVision** (2507.07998) does it prompting-only. So the
machinery exists and works.

**But on our crux — does a code tool improve *metric/geometric* estimation? — the answer is no, and it is now
directly tested.**

**⭐ The decisive paper: arXiv 2606.02357 (1 Jun 2026), "Do Multimodal Agents Really Benefit from Tool Use?
A Systematic Study of Capability Gains."** Compares **Thyme** and **DeepEyesV2** against **tool-free
counterparts** and a **pure-text reasoner** trained from the same source pool. Verbatim from the abstract:

> "**Tool access yields little consistent aggregate improvement**, does not reliably reduce generated-token
> cost, and leaves only a small tool-only solved set: **93% of DeepEyesV2's tool-solved problems and 96% of
> Thyme's are also solved by at least one non-tool setting.** Mechanism ablations further show that the full
> tool-use loop does not consistently outperform either the tool-call format or the returned execution result
> alone. In the settings we study, the analyzed agents appear to **learn tool-calling patterns more reliably
> than tool-contributed capabilities.**"

Four independent corroborations:

- **Visual Sketchpad's numbers are widely misread.** Its headline "+12.7% on math" is dominated by
  **graph-algorithm tasks where the tool literally computes the answer** (maxflow **+41.3**, isomorphism
  +14.5). **Actual geometry is the *worst* math result: +4.2.** And its "depth" win (+12.1) is BLINK
  ***relative*** depth — an ordinal A-vs-B judgment, not metric depth in meters.
- **GeoMathCode (2605.25384, May 2026)** tests whether code helps geometry directly. **It does not.** Code
  execution accuracy 97% while answer accuracy is 55%; ablating code generation in vs. out gives *"highly
  similar performance"*; reasoning and code-generation occupy *"disentangled latent subspaces."*
- **CodeV's own math columns**: MathVerse-mini **49.2 < GPT-4o 50.2**, MathVision-mini **33.6 < 35.9** — while
  V\* is **+20.4 above** it. The code tool's value is concentrated in visual search, not math.
- **TIGeR, the one strong positive, doesn't survive scrutiny as a code-tool result.** On the only genuinely
  metric benchmark (Q-Spatial++, δ≤2), it gains **+4.2 over its own base and *loses* to tool-free
  Gemini-2.5-Pro (86.01 vs 91.61)**. There is **no tool-on/tool-off ablation** anywhere in the paper — only
  data-mix ablations. And mechanically, TIGeR doesn't compute depth with numpy: it **queries an external
  metric-depth model or RGB-D sensor plus known intrinsics**, then uses code for exact algebra.

**And arithmetic isn't even the bottleneck.** arXiv **2502.11492** isolates it: a linear probe on **frozen
vision features** already reaches 74.4–89.9% on length comparison; fine-tuning **only the text decoder** with
the encoder frozen reaches **95.4–98.9%**. **The bottleneck is the vision→language hand-off, not arithmetic.**
A calculator/REPL addresses the part that isn't broken. (No paper was found testing a calculator for VLM
numeric estimation — plausibly because the answer is obviously negative.)

> **The unifying principle across every positive tool result:** a tool helps precisely when it supplies
> **information absent from the model's forward pass** — a 4× zoom of a 20-px region, a metric depth value, a
> maxflow computation. It does **not** help when the missing thing is a perceptual judgment the model must
> still make itself. Metric depth and yaw are the latter.

**What *does* work for metric quantities** (two independent papers, same recipe): **inject depth + camera
intrinsics and force the projective equation.** TIGeR (2510.07181) and **"Equation-Anchored Tool-Use for
MLLMs"** (2605.19528, May 2026) both do exactly this — the latter writes the pinhole back-projection
`X̂ = (u_c − c_x)·Z̄ / f_x` explicitly into the CoT. The code is the *executor of a known equation*, not the
source of the win.

**Two further warnings if we ever do build a tool loop:**

- **Outcome-only reward produces tool-call theater.** CodeV measured faithfulness (did the crop actually
  contain the queried object?): only **57% for DeepEyes** and **43% for Pixel-Reasoner**. These agents score
  well while frequently calling tools on the wrong region. CodeV's process-level tool reward raises this to
  ~70–85%. This rhymes exactly with our own GRPO aggregation-artifact experience.
- **Naive RL alone fails to induce tool use** — DeepEyesV2 states this flatly; TIGeR, Thyme, and CodeV all do
  SFT→RL. This directly confirms the concern in our zoom design doc that cp896 (never trained to emit
  `<tool_call>`) would need format-warmup SFT. **Option (A) pure-RL is not supported by the literature.**
- Agents also **over-call tools** on queries resolvable from raw visual context (Metis / "Act Wisely",
  2604.08545).

**On yaw with tools: nothing exists.** No tool-augmented or code-execution work reports gains on object
orientation in degrees. A genuine gap — and also a warning that nobody has made it work.

---

## 2. What actually moves metric depth

**DepthLM — arXiv 2509.25413 (ICLR 2026 Oral, top 1.2%, Meta FAIR + Princeton).** Our task, essentially:
monocular metric depth in meters, trained on Argoverse2, Waymo, **nuScenes**, ScanNet++, Taskonomy, HM3D,
Matterport3D. ~16M images, **one labeled pixel each**.

δ₁ (within ±25%):

| Model | δ₁ |
|---|---|
| **DepthLM-7B** | **0.838** (nuScenes: **0.865**) |
| Metric3Dv2 (specialist) | 0.841 |
| UniDepthV2 (specialist) | 0.870 |
| GPT-5 | 0.370 |
| Gemini-2.5-Pro | 0.342 |
| Qwen2.5-VL-7B | 0.118 |

**Four findings that bear directly on us:**

1. **Stop passing coordinates as text.** DepthLM renders a **visual marker (arrow) at the query pixel** and
   asks "how many meters is this point from the camera?" — *"VLMs understand marker-based pixel reference much
   better than text-based one"*, worth **~0.15 δ₁**. This would also structurally eliminate our recurring
   0–1000-vs-pixels frame bug: if the pixel is *drawn on the image*, there is no coordinate frame to get wrong.
2. **SFT ≈ GRPO**, at 8–16× less compute per sample. Their GRPO used a dense negative-L1-depth-error reward.
3. **CoT is useless for metric depth.** Post-GRPO the model emits near-identical traces regardless of input.
4. Plain cross-entropy on the number text suffices — no regression loss needed.

**The representation lineage agrees.** VLM-3R lifted VSI-Bench absolute distance **10.9 → 49.4 with LoRA SFT
only**, by feeding 3D geometry tokens. SpatialRGPT's depth plugin took direct-distance success 29.7 → 41.2 and
direction error 69.9° → 15.4°. **The wins on metric tasks come from representation, not from the optimizer.**

---

## 3. Yaw / orientation — we now have a *mechanism* for the 180° flip

This is the most important section of the survey. **Our mirror-flip is a documented, replicated property of
VLMs with a published explanation — not a pipeline artifact, and not something a tool can fix.**

### 3.1 The mechanism: COMFORT's "reflected egocentric" frame

**COMFORT — arXiv 2410.17385, ICLR 2025 Oral** (SLED, UMich). Evaluates 9 SOTA VLMs on spatial frame-of-
reference (FoR) under ambiguity. Four findings, each of which lands directly on our problem
(εcos = region parsing error, lower = better):

1. **Egocentric ≫ intrinsic.** Egocentric εcos: XComposer2 15.8, MiniCPM-V 26.7, GPT-4o 35.1. Intrinsic
   (object-centred): **54.3 / 51.5 / 50.9**. Models effectively **cannot adopt an object's own frame**.
2. **Reflected (mirror) transformation dominates** — left/right preserved, **front/back reversed**, matching
   English usage. XComposer2 **20.0 reflected vs 73.2 translated**; GPT-4o **27.4 vs 75.7**. This is a
   hard-wired 180° sign convention on the sagittal axis.
3. **Sagittal ≪ lateral.** Verbatim: *"Models generally perform better in the lateral directions (left and
   right) than the sagittal ones (front and behind)."*
4. **Prompting does not fix it.** Explicitly instructed to adopt an intrinsic or addressee-centred frame,
   **all models fall to ~50% (chance)**.

**Read-across to us:** "which way is this car heading" is a **sagittal-axis, intrinsic-frame** question — the
single worst cell in COMFORT's matrix, and exactly where models apply a **reflection**. This explains
(a) why the flip is 180° and not random, (b) **why the 235B teacher shares it** — a shared language-derived
convention, not a shared perception failure, and (c) **why our coordinate primer only fixed ~⅓** — COMFORT
predicts prompting cannot fix an FoR prior. Every piece of our diagnosis is independently corroborated.

Corroborating: **"Your other Left!" (arXiv 2508.00549, MICCAI 2025)** — in radiology, image-left is anatomical
right. GPT-4o and Pixtral are above chance **only when image-left coincides with anatomical left, and at chance
otherwise** — they apply a fixed prior and **never perform the swap**. The cleanest published demonstration
that an annotation-frame mismatch yields a systematic, near-deterministic directional error.

### 3.2 How bad is it — the numbers

| Benchmark | Chance | Best frontier VLM | Fine-tuned small model | Human |
|---|---|---|---|---|
| **DORI** (ECCV'26, 33.6k Q) coarse/granular | 35.7 / 25.5 | Gemini-3-Flash **68.5 / 71.0** | — | ~88 / ~84.8 |
| **3DSRBench** orientation (ICCV'25) | 16.8 | GPT-4o **21.6**; SpatialReasoner 55.2 | — | — |
| **ViewSpatial-Bench** object view orientation | 26.33 | GPT-4o **34.98** | **MVSM 82.09 (+46.24)** | — |
| **EgoOrientBench** Choose (8-way, CVPR'25) | 12.5 | GPT-4o 41.1; LLaVA-1.5 **17.9** | +egocentric tuning **33.7 (+15.8)** | — |
| **Ori-Bench** | — | GPT-4o 32.50 | Orient-Anything+LLM **51.50** | — |
| **Orient Anything** azimuth Acc@22.5° (real) | 12.5 | GPT-4o **19.94** | Orient-Anything **73.94** | — |

Three things to note:
- **EgoOrientBench finds zero-shot predictions are biased toward specific classes, particularly *Front* and
  *Front-Right*** — a systematic directional prior visible in the confusion matrices. Same shape as ours.
- DORI: huge coarse→granular collapse (33.9% → 5.8% on single-axis rotation) ⇒ models use **categorical
  heuristics, not geometry**. And ~25% drop on allocentric tasks.
- **Reasoning prompts can actively hurt**: LRR-Bench (2507.20174) — InternVL2.5-72B camera rotation
  **57 → 0** with reasoning prompts.

### 3.3 The published fixes — and they are all SFT, not tools

**Two independent replications of the same fix: SFT on ONE consistent viewer-anchored convention.**

- **EgoOrientBench (CVPR 2025)** — "egocentric instruction tuning" with a single consistent annotation
  standard: LLaVA-1.5 **+15.8**, mPLUG-Owl2 **+13.5**, InternVL2 **+14.9**, **no loss of general capability**.
  Residual errors become *adjacent-bin* (Left vs Front-Left) — i.e. the flip dissolves. **This is the closest
  published analogue to our coordinate-convention primer — and it worked as training, not prompting.**
- **ViewSpatial / MVSM** — ~43k annotated samples, frozen vision encoder → **82.09% (+46.24 abs)**.

**Orient Anything (ICML 2025) / V2 (2601.05573)** — models orientation as a **probability distribution over
azimuth/polar/rotation**, fitted rather than regressed. V2 adds **0..N valid front faces per object** and a
**symmetry-aware periodic distribution-fitting objective**. Also ships an **orientation-confidence head**
(<0.5 ⇒ object has no defined front) — the established way to handle symmetric/front-less objects instead of
forcing a sign. ⚠️ Reality check: **Orient Anything's KITTI azimuth error is 44.22°** — on driving imagery
even the expert is not a clean oracle.

**APC (ICCV 2025, 2504.17207)** — pipes an orientation expert + explicit perspective change into context.
States plainly that VLMs *"perform well … from the egocentric (camera's) perspective, [but] struggle when the
same questions are posed from an allocentric perspective."* Beats fine-tuned spatial models.

### 3.4 FlipEval — the cheap diagnostic to run first

**3DSRBench ships exactly the test we need.** Flip the image and re-ask: height/location answers must stay,
left/right answers must invert; score only pairs where both hold. It was built to expose exploited priors —
the paper names *"driver often sitting on the left side of the car."*

**Applied to our yaw set: if predicted yaw does not mirror correctly under horizontal flip, the error is a
convention/prior artifact, not perception.** That cleanly separates our 180° convention gap from our 90°
perception gap using data we already have, with no training. **This should be the first yaw experiment.**

Background on why flips are implicated at all: **Visual Chirality (CVPR 2020 Oral, 2006.09512)** shows networks
detect mirroring easily because chiral signals pervade imagery, and draws the explicit implication that
**horizontal-flip augmentation is not label-preserving**. ⚠️ *No paper directly attributes VLM orientation
errors to flip augmentation in the pretraining corpus* — that chain is assembled from separate works and
should be stated as a hypothesis, not a result.

### 3.5 The novel contribution available: an angular RL reward

**The survey found no paper using an angular, cosine-of-angle, or wrap-around-at-360° reward for RL/GRPO
post-training of a VLM.** Searched across angular+GRPO, heading/yaw rewards, circular/periodic rewards, RLVR
angle tolerance — nothing matched.

- SVQA-R1's cosine is over *sentence embeddings*, not angles. SpaceR / Spatial-R1 use rule-based rewards with
  no angular term.
- Nearest prior art in *shape*: distance-decayed continuous rewards for localisation (e.g. 2506.14674), whose
  stated principle — *predictions spatially close to the answer should not be penalised as harshly as entirely
  wrong ones* — transfers directly to angles. **The circular version is unpublished.**
- Closest circular *objective* anywhere is Orient Anything V2's symmetry-aware periodic distribution fit —
  a loss, not an RL reward.
- Useful precedent for the formulation: **nuScenes evaluates orientation error over 360° for all classes
  except barriers, which are evaluated over 180°** — the benchmark itself formalises class-conditional genuine
  180° ambiguity.

**An angular GRPO reward with class-conditional 180°/360° wrapping would be genuinely novel**, and it directly
targets the failure mode we've diagnosed.

### 3.6 Other borrowable mechanisms

- **SVQA-R1 (2506.01371)** — **view-consistency reward**: paired original/mirrored samples with
  mirror-consistent QA, and **the view scoring substantially higher is penalised**, forcing view-invariant
  grounding. **58.42% on Q-Spatial++, +20.8 pp over SFT.**
- **Spatial-SSRL (2510.27606, CVPR'26)** — annotation-free RLVR including a **flipped-patch recognition**
  pretext task. Free and self-supervised. *(Caveat: its depth pretext is purely ordinal — not evidence for
  metric depth.)*

---

## 4. SURDS itself — and a serious comparability problem

**SURDS = arXiv 2411.13112 (NeurIPS 2025 D&B).** Their published GRPO result:

| Model | Yaw | Pixel | Depth | Dist | L/R | F/B | **Score** |
|---|---|---|---|---|---|---|---|
| Random | 5.73 | 1.12 | 34.27 | 8.76 | 11.57 | 11.89 | 12.22 |
| Gemini-2.0-flash | 9.30 | 5.41 | 32.97 | 69.30 | 77.30 | 20.00 | 35.71 |
| SFT (Qwen2.5-VL-3B) | 13.95 | 21.11 | 51.35 | 33.95 | 19.68 | 21.62 | 26.94 |
| **SFT+GRPO (Loc+Logic)** | **20.97** | **44.81** | **69.84** | **49.30** | **51.35** | 8.54 | **40.80** |

**⚠️ Their yaw is 4-way cardinal classification and their depth is binned classification.** We score metric
depth in meters (±4 m) and hard 50 px L2. **Our numbers are not comparable to this table** — we're running a
strictly harder continuous-regression variant. *(Get the centerness formula and depth-bin edges from the PDF;
they weren't extractable from HTML.)*

### The reward ablation we should act on (SURDS Table 3)

| Rewards | Yaw | **Pixel** | Depth | Dist | L/R | F/B | Score |
|---|---|---|---|---|---|---|---|
| SFT only | 13.95 | 21.11 | 51.35 | 33.95 | 19.68 | 21.62 | 26.94 |
| +Format+Accuracy | 19.24 | 15.02 | 62.59 | 39.14 | 32.65 | 9.30 | 29.66 |
| +Location | 17.84 | 22.72 | 64.65 | 41.41 | 30.92 | 11.68 | 31.53 |
| +Logic | 20.54 | 14.81 | 62.49 | 36.54 | 32.65 | 11.35 | 29.73 |
| **+Location+Logic** | 20.97 | **44.81** | 69.84 | 49.30 | 51.35 | 8.54 | **40.80** |

Four takeaways:

1. **Location + Logic are super-additive on pixel localization**: 22.72 / 14.81 alone → **44.81** together.
   Neither alone gets there. The "logic" reward = *the reasoning trace alone, with the question removed, must
   reproduce the final answer.* **If our GRPO uses accuracy+format only, adding a trace-consistency reward is
   the highest-EV change available — validated on this exact benchmark.**
2. **Yaw is the hardest subtask for everyone.** Best-ever RL result is 20.97, barely above random 5.73.
3. **Front/behind *collapses* under RL** (21.62 → 8.54) while five subtasks rise — the same per-category
   regression hidden by an aggregate score that we already diagnosed in the stall analysis. Spatial-R1 shows
   the same (room size −5.6, relative direction −5.0).
4. **SURDS's all-binary reward design is the weakest in the field**, and likely a large part of why its
   depth/yaw numbers are low.

### The SFT-only result that beats it

**arXiv 2603.06985 (Mar 2026, NUS/A\*STAR/MIT), "Perception-Aware Multimodal Spatial Reasoning."** Uses
**Visual Reference Tokens** — objects referenced by the visual tokens in their spatial extent rather than by
text boxes; output format `<loc>VRTs</loc><think>…</think><answer>…</answer>`. **Plain SFT, explicitly no RL.**

**Overall SURDS 68.07 vs SURDS's own GRPO 40.80. Yaw 49.11 vs 20.97.**

This is the same lesson as DepthLM and VLM-3R, arriving on our benchmark: **fix how spatial locations are
represented to the model, and SFT beats RL.** *(⚠️ Only the overall and yaw cells were stable across two
extractions — do not quote per-column Pixel/Depth/Distance without re-reading the PDF.)*

---

## 5. Reward shaping for continuous outputs ⚠️ *(thinnest section — thread did not complete)*

**The field has converged on an MRA-style staircase — not binary, not fully dense:**

```
R_num = (1/N) · Σ_i  1[ |ŷ − y| / y ≤ 1 − θ_i ],   θ_i ∈ {0.50, 0.55, …, 0.95}
```

Ten discrete credit bands from 50% down to 5% relative error (SpaceR, Spatial-MLLM). Shaped enough to avoid
binary gradient collapse, discrete enough to preserve GRPO's group-normalization behavior. **This is the design
to use for our metric depth.**

| Reward design | Paper | Outcome |
|---|---|---|
| `exp(−γ‖a−a*‖)` continuous answer + `exp(−α‖p−p*‖₂)` params | TIGeR | works; each of 5 components worth 0.2–3.3 pts |
| NNDC pointing, mean IoU boxes | SpaceTools | DIRL +12% over SFT |
| **Negative L1 depth error (fully dense)** | **DepthLM** | **no benefit over SFT, 8–16× cost** |
| **MRA staircase** | SpaceR, Spatial-MLLM | SpaceR +11.2 in-domain, and improves OOD where SFT *regresses* |
| Point-L1-within-radius | RoboRefer | RFT beats Gemini-2.5-Pro by +17.4% |
| **h-flip view-consistency** | SVQA-R1 | **+20.8 pp over SFT** |
| Conditional auxiliary (applied **only when task reward = 1**) | SpaceR | clean gating pattern worth copying |
| **Strictly binary ×4** | **SURDS** | weakest in field; F/B collapsed 21.6 → 8.5 |

**Keep the reward simple.** The one systematic study — arXiv **2604.13993** (PNNL, Apr 2026) — compares
format-only / binary-accuracy / 5-term rubric / attention-weighted rewards and finds **simpler accuracy rewards
beat complex rubric supervision in small models**, because multi-objective variance destabilizes GRPO gradient
flow and secondary objectives improve at the primary's expense. (MCQ: Fmt+Acc **0.462** > Rubric 0.440.)
**So: 2–3 reward terms max at 8B. Do not build a 5-term weighted rubric.**

**On "RL sharpens rather than teaches"** — our own framing: DepthLM (SFT ≈ GRPO) and SURDS's F/B collapse both
support it. The counterweight is SpaceR: SFT gained +7.2 in-domain but **regressed −4.3/−5.0 out-of-domain**,
while RL improved everywhere. Reasonable synthesis: **RL doesn't add metric perception, but it buys
generalization and format robustness that SFT actively costs you.**

---

## 5.5 Multi-turn agentic RL methodology — what changed in 2026

Even if we never build a tool loop, several of these are directly usable.

### ⭐ Tool-use collapse, and the entropy fix — arXiv 2606.00096 (ICML 2026)

**The most decision-relevant paper in the survey**, because it is one of the only ones evaluated on
**3DSRBench (has an Orientation category)** and **CV-Bench-3D (depth)** rather than the usual high-res-search
benchmarks. It documents **tool-use collapse**: under vanilla RFT on 3DSRBench, tool-use ratio falls
**~20% → ~2% by step 80, monotonically, while accuracy rises.**

The matched comparison demolishes reward-based tool forcing:

| Method | 3DSRBench | CV-Bench-3D (depth) |
|---|---|---|
| Vanilla RFT | 59.2 | 76.7 *(a regression from Mini-o3's 77.6)* |
| Tool-Banned | 58.1 | — |
| **DeepEyes-style tool bonus** (tool use → 100%) | 59.9 | **74.5** *(worse than vanilla)* |
| **Entropy-Regularized** (tool use still ~3%) | **62.9** | **78.8** |

**A 2%→100% swing in tool calls bought +0.7 pp; entropy regularization bought +3.7 pp — and on depth the tool
bonus actively hurt.** Their framing is the key corrective to the DeepEyes narrative: **tools are training-time
scaffolding, not runtime necessities.** Usable mechanism — proportional entropy controller:

```
J = J_GRPO + λ_t · E[H̄(τ)],   λ_t = K_p · [H_target − H_t]₊,   K_p = 0.03, H_target = 0.9
```

**Context:** this adjudicates the field's central unresolved tension. **Pixel Reasoner** (2505.15966) adds a
curiosity bonus to *force* tool adoption (naming the **"learning trap"** — initial incompetence at a new tool
draws more negative reward than text reasoning, so the policy abandons it; without the term, tool-use rate
decays 0.55 → 0 in 240 steps). **Chain-of-Focus** (2505.15436) does the opposite, penalizing a zoom trajectory
when a sibling rollout in the same GRPO group answered correctly *without* zooming — reaching DeepEyes parity
at **5.4% of its visual tokens**. 2606.00096's answer: **neither — use entropy.**

### Cold-start SFT is now consensus (and this kills option A in our zoom design doc)

**DeepEyesV2 (2511.05271)** — the original group's own successor — states flatly that **direct RL alone fails
to induce robust tool use**. That is the DeepEyes authors retracting the "pure RL is enough" story.
Corroborations: pure GRPO on BFCL-V3 gave **catastrophic collapse (0.0 vs vanilla 4.0)** while staged SFT+RL
gave 21.0 and interleaved gave 26.0; GeoEyes found general-domain cold start worthless (47.86%) vs
domain-matched (52.87%). **Interleaved SFT+RL > staged > pure RL** — with an explicit tax: the
format-stabilizing fix **degrades OOD** (ACEBench OOD: vanilla 26.0, SFT+RL 0.0, GRPO-only 24.75).

**Our own base model's recipe agrees.** The **Qwen3-VL tech report (2511.21631)** does two-stage cold start
(10k grounding SFT → ~120k distilled multi-turn interactions), then RL with **three rewards: answer accuracy,
multi-turn reasoning, and a tool-calling reward comparing actual call count to an expert-estimated target.*

### Other directly usable mechanisms

- **Over-turn masking (Mini-o3, 2509.07969).** In standard GRPO a rollout hitting max-turns gets zero reward →
  negative advantage → the policy is actively taught to answer *earlier*, killing test-time scaling. Mask the
  advantage instead of penalizing (`A'_i = M_i·A_i`); a model trained at ≤6 turns then rolls out to tens of
  turns at inference.
- **Don't reward tool frequency** — OTC-PO (2504.14870, NeurIPS'25) cuts NQ tool calls **−68.3% with +215.4%
  tool productivity at flat accuracy**; HiPRAG drives over-search 27% → 2.3%.
- **Audit tool-call faithfulness** — CodeV measured 43–57% faithful calls in prior agents.
- **Monitoring: entropy is NOT a reliable collapse alarm.** RAGEN-2 (2604.06268, ICML'26 oral) shows template
  collapse (reasoning drifting to input-agnostic templates) runs at **stable entropy and stable surface
  diversity** — use **cross-input mutual information** instead, and filter prompts by reward variance. Watch
  per-token probabilities of tool control tokens; collapse is detectable before reward craters and is
  *recoverable* (it's a format failure, not capability loss).
- **AdaReasoner (2601.18631)** — randomize tool identifiers (`GetWeather` → `Func_X7a2`) + paraphrase docs, so
  the policy learns "when to call something with this signature" and generalizes zero-shot to unseen tools.

### Algorithm-level: what actually moves the ceiling

**ScaleRL (2510.13786, >400k GPU-hours)** finds loss aggregation, advantage normalization, curriculum, and
async all move only **compute-efficiency, not the asymptote**. Only the **loss type (CISPO vs DAPO: A 0.52 →
0.61)** and **FP32 logits at the LM head** move the ceiling. **Lite PPO (2508.08221)** gets a simpler
two-component recipe (group-mean/batch-std hybrid advantage norm + token-level aggregation) beating both DAPO
and GRPO, and notes **clip-higher helps aligned models but is negligible on base models**.

⚠️ **Tool-output token masking has surprisingly thin evidence.** It's universal practice (Search-R1, ReTool,
VerlTool) but the only quantified ablation is Search-R1's **0.431 with vs 0.343 without** — secondary source,
retrieval setting. **Nobody has ablated masking-off in a multi-turn tool-call setting with numbers.**

---

## 5.6 ⭐ Our pass@1/pass@8 yaw gap has a name: pass@k inversion

**"When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion" — arXiv 2607.20543 (Jul 2026).**

Formalizes exactly our yaw situation as an **"absence-of-evidence failure" localized to *boundary prompts*** —
problems where correct solutions exist but are **rare and non-dominant**, so finite-sample training commits to
incorrect modes before the rare correct trajectories ever get reinforced. That is a precise description of
"pass@8 at teacher level, pass@1 36 pp behind."

**Fix — Per-Problem Base Anchoring:** estimate each prompt's correct-mode mass from *frozen base* rollouts;
apply standard RL only where evidence is sufficient (**≥1 success in 8 samples**); **KL-anchor the rest to
base**. Reported **+3.9 pass@1, +4.7 pass@256, and 7.2× fewer boundary-prompt losses** over GRPO.

⚠️ No VLM experiments (math verifiers only), though the authors explicitly extend the claim to vision-language
agents with external verifiers. Converges with three other 2026 results (2510.02230 curate toward
low-likelihood problems; 2606.15455 target unsolved; ScaleRL's No-Positive-Resampling) — **all four
independently say the operative variable is whether the training distribution contains boundary problems.**

**This is arguably the most actionable single idea in the survey for yaw**, because it targets the *exact*
failure signature we measured rather than the subtask.

---

## 6. Metrics — our ±4 m is not comparable to the literature

| Benchmark | Metric | Definition |
|---|---|---|
| VSI-Bench | **MRA** | mean over θ ∈ {0.50…0.95} of `1[|ŷ−y|/y < 1−θ]` |
| Q-Spatial | **δ≤2 / δ≤1.25** | δ = max(d̂/d\*, d\*/d̂) |
| DepthLM, SpatialRGPT | **δ₁ / success@±25%** | \|d̂−d\*\|/d\* ≤ 0.25 |
| STI-Bench | tolerance bands | **outdoor 0.5–5 m** (closest analogue to ours) |

**Our ±4 m is a fixed absolute band; the literature uses relative criteria.** 4 m = 25% relative at **16 m**.
For most nuScenes objects (<16 m) our tolerance is **looser** than δ₁@25% — at 8 m we allow 50% relative error.
Beyond 16 m it's tighter.

**Recommendation: report δ₁(±25%) and δ≤1.25/δ≤2 alongside the 4 m number, stratified by GT depth band.**
Otherwise nothing we publish is comparable, and our 4 m figure is systematically flattering at close range.

**⚠️ Also: never headline VSI-Bench MRA on absolute distance.** A constant-frequency predictor scores **62.1**
— above every model *and* above the human 47.0. Any abs-distance MRA below 62.1 is uninterpretable.

---

## 7. What this means for us — ranked recommendations

**The honest answer to "should we move to a DeepEyes / Python-REPL agent?": no.** This is now *tested*, not
inferred. Four independent lines:

- **Head-to-head against tool-free controls, tools add little** (2606.02357): 93–96% of tool-solved problems
  are solved tool-free; agents learn tool-*calling patterns* more reliably than tool-contributed *capability*.
- **Metric depth responds to representation and data, not reasoning or RL** (DepthLM: SFT ≈ GRPO, CoT useless;
  VLM-3R: +38.5 abs-distance from geometry tokens with SFT only).
- **Yaw is a frame-of-reference prior**, not a perception limit (COMFORT) — zoom and code cannot touch it, and
  convention-consistent SFT demonstrably fixes it (EgoOrientBench +14–16, MVSM +46.24).
- **Arithmetic isn't the bottleneck** (2502.11492): the vision→language hand-off is. A REPL addresses the part
  that isn't broken.

Ranked by expected value per unit of effort:

1. **[Free, do first] Run FlipEval on the yaw set.** Flip the image, re-ask, check whether predicted yaw
   mirrors correctly. Cleanly separates our **180° convention gap** from our **90° perception gap** using data
   we already have, no training. This is the diagnostic 3DSRBench built for exactly this class of prior.
2. **Change the input representation, not the algorithm.** Adopt DepthLM's **visual marker** for pixel
   reference (draw the query point on the image) instead of text coordinates — worth ~0.15 δ₁, and it
   *structurally kills our recurring coordinate-frame bug*. Visual Reference Tokens (2603.06985) is the
   stronger version and beats SURDS's GRPO with plain SFT. **Highest-EV change in the survey.**
3. **For yaw, SFT on ONE consistent viewer-anchored convention.** This is the published fix, replicated twice
   (EgoOrientBench +14–16 pts with no general-capability loss; MVSM +46.24). It is the *training* version of
   the primer we already tried as a prompt — and COMFORT explains why the prompt version could only ever get
   part of the way.
4. **Add a trace-consistency ("logic") reward alongside the spatial reward.** SURDS Table 3: Location+Logic
   together take pixel 21.11 → **44.81** while either alone does nothing. Keep total reward terms to 2–3
   (per 2604.13993 — complex rubrics *hurt* at small scale).
5. **Move metric depth from binary to the MRA staircase reward**; gate any auxiliary geometric term on the
   task reward already being correct (SpaceR's pattern).
6. **Fix our reporting**: δ₁ + δ≤1.25/δ≤2 stratified by depth band; per-subtask breakdowns always (the field
   repeatedly hides category collapse behind aggregate scores — SURDS F/B 21.6→8.5, Spatial-R1 −5.6/−5.0).
7. **Before claiming any RL win, run a random-reward control.** *Spurious Rewards* (2506.10947): on
   Qwen2.5-Math-7B, MATH-500 improves **+21.4% with random rewards** and **+24.1% with deliberately incorrect
   labels**, vs +29.1% with ground truth — and **none of it reproduces on Llama3 or OLMo2**. **We are on a Qwen
   backbone.** A positive RLVR result on Qwen with one reward design is not evidence the reward is doing
   anything. This is cheap and protects every claim we make.
8. **If we ever do build a tool loop**, four non-negotiables from the 2026 evidence:
   - **SFT cold start is mandatory** — DeepEyesV2 states naive RL fails to induce tool use; BFCL-V3 pure GRPO
     collapsed to 0.0 vs 26.0 interleaved. **This kills option (A) in our zoom design doc.** Cold-start data
     must be *domain-matched* (GeoEyes: general-domain cold start was worthless).
   - **Never reward tool frequency — regularize entropy instead** (2606.00096's proportional controller,
     K_p = 0.03, H_target = 0.9). On CV-Bench-3D depth the tool bonus *hurt* (74.5 vs 76.7 vanilla).
   - **Add a process/faithfulness reward** — outcome-only RL yields 43–57% faithful tool calls (CodeV).
   - **The tool must inject information the forward pass lacks** — depth + intrinsics + a forced projective
     equation (TIGeR, 2605.19528) — not a general Python REPL.
   - Add **over-turn masking** (Mini-o3) if we want turn-count to scale at inference.

**A nuance worth noting on tools-for-yaw specifically:** SpaceTools' per-category split shows tools delivering
a **huge win on pose estimation (34.37 IoU vs Claude-Sonnet-4.5's 10.67)** but **exact parity on 3D depth
(70.00 vs 70.16)**. Combined with Orient Anything + LLM beating GPT-4o on Ori-Bench (51.50 vs 32.50), the
orientation-expert-as-tool case is genuinely stronger than the depth case. It is still dominated on published
evidence by the cheaper convention-SFT fix (#3), and Orient Anything's **44.22° KITTI azimuth error** means
we'd be routing to a noisy oracle on driving imagery. But if we ever do add one tool, make it an orientation
expert, not a REPL — and note **no published work reports tool gains on yaw in degrees**, so it is unproven
either way.

### The unoccupied niche (if we want the paper)

Tool-calling driving VLMs exist (AgentThink, DriveAgent-R1) and call exactly the tools we'd want (Depth
Anything V2, DetAny3D, RoI zoom) — **but both target planning/decision, not spatial primitives.** A
tool-augmented agent evaluated on **SURDS-style depth/yaw primitives** appears genuinely unoccupied. Likewise:

- **No one publishes metric depth in meters on nuScenes from a VLM** (SURDS bins it; NuScenes-SpatialQA
  reports <1% on some quantitative tasks but per-model tables weren't extractable).
- **No one publishes continuous yaw error in degrees on driving data.**
- **Our 180°-convention vs 90°-perception yaw decomposition is undocumented in this literature.**

Those three gaps are the contribution — but they are *evaluation and diagnosis* contributions, reachable
without an agentic loop. Temper tool expectations accordingly.

---

## 7.5 Two structural warnings about the literature itself

**The thinking-with-images line is narrower than it looks.** V\*, HR-Bench-4K/8K, and VisualProbe carry almost
every headline result, and all three measure essentially **one skill: find a tiny target among distractors in a
large image** — exactly the regime where cropping supplies pixels the model never saw. **Our subtasks are not
that regime.** The two papers that deliberately stepped outside it (2606.00096 on 3DSRBench/CV-Bench-3D,
2606.02357 on math/OCR/chart) *both* found the tool contribution collapses. And a clean per-category
"where does zoom help vs vanish" ablation **does not exist in the published literature as of July 2026**.

**⚠️ Evaluation-protocol trap: never mix single-pass and Avg@k numbers across papers.** Mini-o3 re-evaluated
DeepEyes at Avg@32 and got **V\* 83.3, not the reported 90.1**. Any cross-paper comparison in this doc that
mixes protocols is unreliable — including our own earlier framing of DeepEyes' V\* result.

**Reality check on the whole agentic framing:** AgentVista (2602.23166) puts **Gemini-3-Pro with tools at 27.3%
overall** on realistic hybrid tool use, some instances needing >25 turns. Whatever V\* saturation suggests, the
agentic setting is nowhere near solved.

---

## 8. ⚠️ Verification flags — re-check before citing externally

- **2603.06985 per-column SURDS numbers** — HTML table extracted with inconsistent column alignment across two
  fetches. **Only overall (68.07 vs 40.80) and yaw (49.11 vs 20.97) are stable.** Read the PDF.
- **DepthLM's exact SFT-vs-GRPO δ₁ pair** — only the verbatim "perform similarly well" and "8–16 times slower"
  are confirmed; the precise gap is not.
- **SURDS centerness formula, depth-bin edges (m), distance threshold** — not extractable from HTML. Needed to
  quantify exactly how much harder our continuous variant is.
- **Cross-paper reproduction is unstable**: SD-VLM reproduces SpatialRGPT's direct-distance at **20.5** where
  SpatialRGPT's own paper reports **41.2**. Don't trust any single cross-paper cell.
- **Naming correction:** *Spatial-R1* and *SpaceR* are the **same arXiv entry, 2504.01805** (v1 vs v2 titles),
  not two papers.
- **SpaceR (v2) per-task VSI-Bench breakdown** unverified — only v1 per-task numbers confirmed.
- **NuScenes-SpatialQA per-model tables** not extracted; the "<1%" figure is secondary-source.
- **No verified VSI-Bench number for Qwen3-VL-8B** (our model) — only 2B (0.422 abs-dist MRA) and 4B (0.435),
  from a community lmms-eval reproduction.
- **DepthVLM (2605.15876)** — unverified whether stage 2 is RL; no head-to-head numbers.
- **Do not cite from this doc:** GeoAlign (2604.12630) — PDF extraction failed; **SpatialLM** — no arXiv paper
  exists; RoboPoint — not retrieved. SpatialSense (1908.02660) is pre-VLM; cite for framing only.
- **Single-extraction only:** SPAR-Bench Table 4, Orient Anything column semantics, SD-VLM deltas,
  OmniSpatial per-subtype, AgentThink's exact reward terms, OmniDrive-R1 / SpaceDrive numbers.
- **§5 incomplete** — never chased: VLM-R1 (2504.07615), Seg-Zero, Perception-R1, and the pass@k analysis
  (2504.13837) that speaks directly to our ceiling-vs-consistency framing.
- **⚠️ A retracted confabulation, recorded deliberately.** An early extraction pass on 2606.02357 produced a
  tidy "helps: localization/counting/OCR; doesn't help: math/geometry/metric" category breakdown. That is
  **not in the paper** — it matched the question too neatly and was self-caught. **The verbatim abstract quoted
  in §1 is solid; the category breakdown is not. Do not cite it.** (Its studied domains are real-world
  understanding, OCR, chart, and math.) Numeric tables from that paper were never retrieved.
- **Not extractable:** PyVision's per-benchmark math table; Thyme's per-benchmark table; numeric tables in
  2605.19528, 2511.22659 (GCA), 2605.23281 (DepthAgent). Visual Sketchpad has an unresolved 12.7% vs 11.2%
  math-average discrepancy across abstract vs table.
- **SOFA** (OpenReview `8sggKfEtSQ`) — bot-check page on both HTML and PDF; no numbers, no arXiv ID.
- **Orient Anything's KITTI azimuth error is 44.22°** — even the dedicated expert is not a clean oracle on
  driving imagery. Do not assume a tool-injected orientation would be ground truth.
- **The flip-augmentation causal chain is a hypothesis, not a result.** Visual Chirality shows flips aren't
  label-preserving; 3DSRBench/SVQA-R1 measure and correct the bias. **No paper attributes VLM orientation
  errors to flip augmentation in the pretraining corpus.** State it as a hypothesis.
- **Two subagents hit their WebSearch quota** (200/200) before chasing: Act Wisely's tables, VESTA
  (2606.00384), TACO tool-call credit assignment (2606.30251), S1-VL (2604.21409).
- **Mini-o3's per-benchmark table and over-turn-masking ablation** — abstract-only fetch.
- **DeepEyes' scalar reward magnitudes** are not in the paper.
- **"SAPO" could not be disambiguated** as a standalone method — multiple unrelated acronyms; it appears in the
  Qwen3-VL report as *Smooth and Adaptive Policy Optimization* via a secondary review, unverified from the
  paper. (An earlier turn in this project described SAPO as "the Qwen-family RL algo" — treat that as
  unverified.)
- **DORI's venue is reported inconsistently** across fetches (one pass said withdrawn, another said ECCV 2026).
- **2607.20543 (pass@k inversion) has no VLM experiments** — math verifiers only; the extension to
  vision-language agents is the authors' claim, not a demonstrated result.
- Several **2026 arXiv IDs show month-prefix/date mismatches**; abs-page dates are reported verbatim.
- **Not covered at all:** context-management techniques for many-turn agentic RL; and the per-category
  zoom-helps-vs-vanishes ablation (**a genuine gap in the literature**, not a gap in the search).

---

## 9. Reading list, if you only read five

1. **COMFORT** — 2410.17385 (ICLR'25 **Oral**). The mechanism for our yaw flip: reflected egocentric frame,
   sagittal ≪ lateral, and prompting cannot fix it. Explains the teacher-shares-the-flip result *and* why our
   primer only got ~⅓.
2. **DepthLM** — 2509.25413 (ICLR'26 **Oral**). Our exact task on nuScenes. Marker-based pixel reference beats
   text coordinates by ~0.15 δ₁; SFT ≈ GRPO; CoT useless.
3. **"Do Multimodal Agents Really Benefit from Tool Use?"** — 2606.02357. The tool-free control study.
   93–96% of tool-solved problems are solved without tools. Read before building any tool loop.
4. **2603.06985** — SFT-only with Visual Reference Tokens, beats SURDS's own GRPO on SURDS
   (68.07 vs 40.80; yaw 49.11 vs 20.97).
5. **SURDS** — 2411.13112. Re-read **Table 3** (Location+Logic super-additivity) and confirm the
   classification-vs-regression mismatch with our harness.

*Runner-up:* **EgoOrientBench** — 2411.16761 (CVPR'25). The convention-consistent SFT fix for orientation,
+14–16 pts with no general-capability loss.

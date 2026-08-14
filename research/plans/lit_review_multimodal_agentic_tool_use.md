# Training Vision-Language Models as Multimodal Agentic Tool-Use Agents: A Literature Review

*Prepared: 25 July 2026. All identifiers verified against arXiv abstract pages at time of writing.*

---

## Abstract

Between early 2025 and mid-2026 the dominant recipe for vision-language models (VLMs) shifted from *single-pass captioning and visual question answering* to *interactive, multi-turn, tool-mediated problem solving*. A VLM is now routinely trained to interleave natural-language reasoning with calls to external tools — image crop/zoom, code interpreters, OCR and detection modules, text and image search engines, browsers, and raw graphical-user-interface (GUI) actions — and to condition subsequent reasoning on the returned observations. This review surveys 32 verified works spanning that shift. We organise them along three axes: (i) the *problem formulation*, which has converged on a partially-observable Markov decision process (POMDP) over interleaved text/image observation tokens; (ii) the *training paradigm*, which is almost universally a two-stage pipeline of supervised trajectory distillation ("cold start") followed by reinforcement learning with verifiable rewards (RLVR), most often a Group Relative Policy Optimization (GRPO) variant; and (iii) the *tool ecosystem*, which has fragmented into four largely non-interoperable clusters — "thinking with images", search/deep-research, code execution, and GUI/computer use. We argue that the field's headline numbers substantially overstate progress: multiple 2025–2026 papers demonstrate that high final-answer accuracy coexists with *unfaithful* tool use (tools invoked on irrelevant regions, or outputs ignored), that trajectory-level GRPO systematically misassigns credit to tool-calling tokens, and that the strongest proprietary models still pass fewer than one in five tasks on the hardest tool-use benchmarks. We close with seven open problems, emphasising faithfulness-aware process rewards, turn-level credit assignment, environment/trajectory synthesis with verifiable rewards, and the near-total absence of standardised, contamination-controlled evaluation for multimodal tool use.

---

## 1 Introduction and Scope

### 1.1 Motivation

Chain-of-thought reasoning lifted large language model (LLM) performance across mathematics, code, and knowledge tasks, but the reasoning itself remained confined to text. For visually intensive problems this is a hard ceiling: a 4K screenshot or a dense infographic contains detail that a fixed visual encoder, operating at a fixed resolution, simply discards before any reasoning begins. [Wang, Haozhe et al., 2025, #20](#ref-20) frame this precisely — chain-of-thought "has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks" — and propose *pixel-space reasoning* as the remedy. [Zheng et al., 2025, #32](#ref-32) frame the same gap cognitively: large VLMs "struggle to deeply integrate visual information into their predominantly text-based reasoning processes."

The proximate trigger for the 2025 explosion of work was OpenAI's o3, which demonstrated a proprietary model natively cropping, zooming, and searching mid-reasoning. Essentially every open-source paper in this review positions itself explicitly as an attempt to reproduce or extend that behaviour: Thyme states that "no open-source work currently offers a feature set as rich as proprietary models (O3)" [Zhang, Yi-Fan et al., 2025, #30](#ref-30); Mini-o3 describes "our recipe for reproducing OpenAI o3-style behaviors" [Lai et al., 2025, #12](#ref-12); TIR-Bench opens on "models like OpenAI o3" [Li et al., 2025, #13](#ref-13). This is worth stating plainly because it shapes the literature's methodology: the target behaviour is defined by a closed system whose training recipe is unknown, so the open literature is reverse-engineering a behavioural signature rather than optimising an independently motivated objective.

### 1.2 Scope

We cover work that **trains** (not merely prompts) a vision-language backbone to emit tool calls and consume tool outputs, plus the algorithmic infrastructure and evaluation resources that such training depends on. Concretely:

- **In scope**: visual tool-augmented reasoning ("thinking with images"); multimodal search / deep-research agents; code-interpreter-augmented VLMs; GUI, mobile, and computer-use agents; RL algorithms for multi-turn tool-calling credit assignment; trajectory and environment synthesis pipelines; benchmarks for multimodal tool use.
- **Out of scope**: training-free prompting frameworks (e.g. Visual Sketchpad-style test-time scaffolds), text-only tool-use agents except where the algorithmic contribution is directly inherited by multimodal systems (we include ToolRL, DAPO, GiGPO, VerlTool, and EnvFactory on those grounds), embodied robotics, and pure visual grounding without an action interface.
- **Recency**: 29 of 32 entries are from 2025 or 2026. Three 2024 works are retained as load-bearing foundations: OSWorld (the de-facto computer-use benchmark), and OS-Genesis (the reverse-task-synthesis idea that most later GUI data pipelines build on).

### 1.3 Method and verifiability caveat

Every entry below was verified by retrieving its arXiv abstract page and reading the title, full author list, submission date, and abstract. Papers that appeared in search results but whose identifiers we could not independently confirm were **dropped**; they are listed in §6.6 so that a reader does not mistake their absence for a judgement of quality. No claim in this review is sourced from a search-engine snippet alone.

---

## 2 Taxonomy and Problem Formulation

### 2.1 The common formalism

Across the corpus, a multimodal tool-use agent is a policy $\pi_\theta$ over a POMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, R \rangle$ in which:

- **Actions** $\mathcal{A}$ are token sequences that either continue free-form reasoning or emit a structured tool call (a function name plus arguments, a Python program, or a GUI primitive such as `click(x, y)`).
- **Observations** $\mathcal{O}$ are *multi-modal*: a tool returns text (search snippets, stdout), an image (a crop, a re-rendered plot, a screenshot), or both, which are appended to the context as new observation tokens. VerlTool makes this explicit, formalising agentic RL with tool use (which it abbreviates ARLT) as "multi-turn trajectories with multi-modal observation tokens (text/image/video)" [Jiang et al., 2025, #11](#ref-11).
- **Rewards** $R$ are predominantly *outcome-verifiable*: exact match, F1, rule-checkable format, or an environment-provided success signal.

The critical structural difference from text-only agentic RL is **partial observability induced by perception**. VAGEN states this as the field's defining challenge: "the shift from textual states to complex visual observations... introduces partial observability and demands robust world modeling" [Wang, Kangrui et al., 2025, #21](#ref-21). This is not a cosmetic difference. In text agents the state is (approximately) fully described by the transcript; in a VLM agent the transcript contains a *rendering* of the state whose informativeness depends on resolution, cropping, and the encoder — which is exactly what tool calls are for.

### 2.2 Four axes of variation

**Axis 1 — Who owns the tool?** Three regimes appear:
1. *Intrinsic tools*: the capability is already latent in the backbone and the tool call merely re-routes it. DeepEyes' image-zoom "emerges natively, leveraging the model's own grounding capability as an intrinsic function rather than relying on external specialized models or APIs" [Zheng et al., 2025, #32](#ref-32).
2. *Fixed external tool library*: a curated set of vision modules (OCR, detector, segmenter, chart parser) behind a standardised interface, as in OpenThinkIMG [Su et al., 2025, #17](#ref-17) and VisTA [Huang et al., 2025, #10](#ref-10).
3. *Generated tools*: the model writes the tool. PyVision has the model "autonomously generate, execute, and refine Python-based tools tailored to the task at hand," an approach the authors summarise as models "not just to use tools, but to invent them" [Zhao et al., 2025, #31](#ref-31). Thyme and CodeV take the same code-as-tool position [Zhang, Yi-Fan et al., 2025, #30](#ref-30); [Hou et al., 2025, #8](#ref-8).

The trend line runs 1 → 3 over 2025–2026, because a fixed library caps the achievable task distribution while code generation does not. The cost is a sandboxing and reproducibility burden that almost no paper quantifies.

**Axis 2 — Interaction depth.** Early works permit 1–3 tool turns; Mini-o3 is the inflection point, training with a cap of six turns yet producing trajectories that "naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases," enabled by an over-turn masking strategy that avoids penalising truncated rollouts [Lai et al., 2025, #12](#ref-12). GUI and computer-use agents operate at the deepest end (tens to hundreds of steps).

**Axis 3 — Tool granularity.** Coarse (one "search" call) versus fine (a Python program composing five image ops). Granularity determines whether credit assignment is tractable: coarse calls admit trajectory-level rewards, fine calls demand step-level rewards (§3.2.3).

**Axis 4 — Environment fidelity.** Static image + offline tool (VTool-R1, VisTA) → live internet (MMSearch-R1, WebWatcher) → full virtual machine (UI-TARS-2, OpenCUA, ScaleCUA). Fidelity correlates inversely with reproducibility: a live-internet reward is not replayable, which is a systemic and under-acknowledged threat to the validity of search-agent comparisons.

### 2.3 A note on terminology drift

"Agentic", "tool-integrated reasoning (TIR)", "thinking with images", "interleaved multimodal chain-of-thought", and "active perception" are used near-interchangeably across this corpus for overlapping but non-identical constructs. We use **multimodal agentic tool use** for the union, and reserve **thinking with images** for the subset in which at least one tool returns a *new image* into the context (§4.1). Readers should be aware that cross-paper baseline comparisons frequently compare systems that do not share a tool interface, a turn budget, or an observation format, and are therefore weaker evidence than their tables suggest.

---

## 3 Training Paradigms

### 3.1 Supervised fine-tuning and trajectory distillation ("cold start")

The near-universal first stage is SFT on synthetic tool-use trajectories. Its function is *format acquisition and exploration priming*, not capability acquisition: a base VLM asked to emit `<tool_call>` tokens it has never seen will produce near-zero-reward rollouts, and GRPO-style algorithms have zero gradient when all rollouts in a group share the same reward.

Representative instantiations:

- **Two-phase with explicit motivation.** Pixel Reasoner runs "instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations", explicitly to overcome "the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations" [Wang, Haozhe et al., 2025, #20](#ref-20). This *reluctance* observation — that a competent text reasoner actively avoids a newly offered visual tool because text reasoning is locally cheaper — recurs throughout the corpus and is the single most-cited reason cold start is required.
- **Scale.** Thyme uses "an initial SFT on a curated dataset of 500K samples to teach code generation, followed by a RL phase to refine decision-making" [Zhang, Yi-Fan et al., 2025, #30](#ref-30). OpenSearch-VL releases `SearchVL-SFT-36k` for SFT and `SearchVL-RL-8k` for RL, a roughly 4.5:1 ratio [Chen et al., 2026, #1](#ref-1).
- **Diversity as the design target.** Mini-o3's iterative cold-start pipeline is engineered to yield "diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance" [Lai et al., 2025, #12](#ref-12) — i.e. the SFT set is chosen to seed *exploration modes*, not to maximise imitation accuracy.
- **Reflective long chain-of-thought over human demonstrations.** OpenCUA's pipeline "transforms demonstrations into state-action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales" [Wang, Xinyuan et al., 2025, #22](#ref-22) — the strongest published evidence that CoT augmentation of *human* trajectories, not just synthetic ones, is what makes SFT scale in the GUI domain.

**The dissenting position.** Two works argue cold start is unnecessary or harmful. DeepEyes is "trained end-to-end with reinforcement learning without requiring pre-collected reasoning data for cold-start supervised fine-tuning" [Zheng et al., 2025, #32](#ref-32), substituting a tailored data-selection and reward strategy. IMAgent goes further, claiming a two-layer motion-trajectory masking strategy plus a tool-use reward gain lets the agent acquire "an effective tool-use paradigm through pure reinforcement learning, eliminating the need for costly supervised fine-tuning data" [Dong et al., 2025, #3](#ref-3). ToolRL, in the text-only setting, provides the mechanistic argument: "SFT struggles to generalize to unfamiliar or complex tool use scenarios" [Qian et al., 2025, #16](#ref-16).

**Critical assessment.** The disagreement is not resolved, and is probably not resolvable as posed, because it is confounded by backbone choice. Models whose pretraining already contains grounding-shaped supervision (DeepEyes leverages the backbone's *own* grounding function) can bootstrap without SFT; models being taught a genuinely novel action space (arbitrary Python image ops, GUI primitives) cannot. Papers on both sides run their ablation on a single backbone family, so "SFT is/isn't needed" should be read as "SFT is/isn't needed *for this backbone and this action space*."

### 3.2 Reinforcement learning with verifiable rewards

#### 3.2.1 The algorithmic substrate

GRPO — critic-free, group-normalised advantages — is the default. Its appeal for this setting is practical: no value network to train over interleaved image observations, and low memory. DAPO [Yu et al., 2025, #28](#ref-28) supplies the open-source system and the now-standard stabilisation tricks (decoupled clipping, dynamic sampling, token-level loss, over-long reward shaping) that most multimodal follow-ups inherit implicitly, often without citation of which specific components they adopted — a real reproducibility problem, since "we use GRPO" now denotes a family, not an algorithm.

Multimodal variants that modify the optimiser:

| Variant | Paper | Modification |
|---|---|---|
| V-ToolRL | OpenThinkIMG [#17](#ref-17) | GRPO over tool-invocation policies with task-success feedback |
| GRPO-ATS | Thyme [#30](#ref-30) | Adaptive temperature sampling: different sampling temperature for reasoning vs. code tokens |
| BN-GSPO | SenseNova-MARS [#2](#ref-2) | Batch-normalised group *sequence* policy optimization for stability under tool interleaving |
| Fatal-aware GRPO | OpenSearch-VL [#1](#ref-1) | Masks post-failure tokens after cascading tool failures; one-sided advantage clamping preserves pre-failure reasoning |
| Bi-Level GAE | VAGEN [#21](#ref-21) | Turn-aware advantage estimation over a POMDP with a dense world-modeling reward |
| TRPO (trajectory-aware) | GUI-Owl [#27](#ref-27) | Trajectory-aware relative policy optimization for asynchronous online GUI RL |
| GiGPO | [#5](#ref-5) | Two-level advantage: episode-level groups plus step-level "anchor state" groups |
| TAPO (process) | CodeV [#8](#ref-8) | Dense process rewards defined on tool *inputs and outputs*, not CoT tokens |
| TAPO (credit transfer) | [#4](#ref-4) | Counterfactual witnesses within a batch; confidence-gated conservative advantage correction |

Note the **name collision**: two distinct 2025–2026 papers both use the acronym TAPO for "Tool-Aware Policy Optimization" with different mechanisms [Hou et al., 2025, #8](#ref-8); [Dong et al., 2026, #4](#ref-4). This is symptomatic of the field's speed and is a live hazard for anyone reading only abstracts.

#### 3.2.2 Reward design

ToolRL is the reference study: "the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm", systematically varying "types, scales, granularity, and temporal dynamics" and reporting 17% over base and 15% over SFT [Qian et al., 2025, #16](#ref-16). Its practical conclusion — decompose the reward into tool-name correctness, parameter-schema correctness, and parameter-value correctness — has been widely adopted in text agents but, notably, *not* systematically transferred to multimodal agents, where most works still use a single outcome reward plus a format term.

Multimodal-specific reward innovations worth isolating:

- **Search penalty / cost-aware rewards.** MMSearch-R1 uses "an outcome-based reward with a search penalty", achieving parity with a larger RAG baseline "while reducing search calls by over 30%" [Wu, Jinming et al., 2025, #23](#ref-23). This is the cleanest demonstration in the corpus that *on-demand* tool use — deciding **not** to call a tool — is itself a learnable and valuable behaviour.
- **Curiosity / exploration bonuses.** Pixel Reasoner's "curiosity-driven reward scheme" exists to "balance exploration between pixel-space reasoning and textual reasoning" [Wang, Haozhe et al., 2025, #20](#ref-20), i.e. to counteract the tool-avoidance failure mode directly at the reward level rather than via SFT.
- **Dense world-modeling rewards.** VAGEN decomposes reasoning into State Estimation and Transition Modeling and rewards accurate state prediction turn-by-turn; it also reports that the optimal belief representation is task-dependent — "Natural Language excels at capturing semantic relationships in general tasks, while Structured formats are indispensable for precise manipulation and control" [Wang, Kangrui et al., 2025, #21](#ref-21).
- **Process rewards grounded in tool outputs.** CodeV's TAPO "augments GRPO with dense rewards defined directly on visual tool inputs and outputs, rather than on chain-of-thought tokens, making supervision easier to verify and less susceptible to reward hacking" [Hou et al., 2025, #8](#ref-8). This is, in our reading, the most important reward-design idea of the past year: it relocates process supervision from the *unverifiable* CoT to the *verifiable* tool interface.

#### 3.2.3 Multi-turn credit assignment

Trajectory-level GRPO broadcasts one scalar advantage to every token, including tool-call tokens. Three papers attack this:

1. **GiGPO** constructs step-level groups by "identifying repeated environment states across trajectories", giving micro relative advantages without a critic; gains exceed 12% on ALFWorld and 9% on WebShop over GRPO at identical GPU memory and rollout cost [Feng et al., 2025, #5](#ref-5). The anchor-state mechanism is elegant but assumes states *recur* — which holds in ALFWorld/WebShop and is far less true for open-web multimodal search or free-form image editing.
2. **VAGEN's Bi-Level GAE** supplies turn-aware credit under partial observability [Wang, Kangrui et al., 2025, #21](#ref-21).
3. **TAPO (credit transfer)** gives the sharpest diagnosis. It "formally characterize[s] credit misassignment as a systematic failure mode of GRPO in tool-augmented multimodal search agents: its uniform broadcast of trajectory-level advantages to all tokens causes valuable tool-use steps in failing trajectories to be penalized no differently from valueless ones", and quantifies it: "Over half of failing trajectories and failing tool-use actions exhibit correctable credit misassignment" [Dong et al., 2026, #4](#ref-4). The fix exploits parameter-determinism of information-acquisition tools (similar call parameters ⇒ equivalent action ⇒ comparable credit), constructing counterfactual witnesses within the batch at negligible overhead, and reports plug-and-play gains over GRPO, GSPO, and SAPO.

**Critical assessment.** That >50% of failing tool actions are mis-credited is a strong indictment of the field's default optimiser. It also implies that a large fraction of published GRPO-vs-SFT multimodal deltas are measured against a self-handicapped RL baseline, and that reported gains from *architectural* or *data* interventions may partly be gains from accidentally mitigating credit misassignment.

### 3.3 Data and environment synthesis

Three families, in increasing order of automation:

**(a) Human demonstration capture with post-hoc reasoning augmentation.** OpenCUA's annotation infrastructure "seamlessly captures human computer-use demonstrations" and yields AgentNet, "the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites" [Wang, Xinyuan et al., 2025, #22](#ref-22). Highest fidelity, lowest scalability.

**(b) Reverse task synthesis.** OS-Genesis "reverses the conventional trajectory collection process": agents first explore the environment with step-wise interactions, then *retrospectively derive* task instructions from the observed state changes, with a trajectory reward model filtering quality [Sun et al., 2024, #18](#ref-18). This solves the chicken-and-egg problem (you need tasks to collect trajectories, but writing tasks requires knowing what the environment affords) and is the conceptual ancestor of most later GUI pipelines.

**(c) Self-evolving and verifiable synthesis loops.** GUI-Owl's Self-Evolving GUI Trajectory Production runs over a cloud virtual environment spanning Android, Ubuntu, macOS, and Windows, with "automated query generation and correctness validation, leveraging GUI-Owl to refine trajectories iteratively, forming a self-improving loop" [Ye et al., 2025, #27](#ref-27). UI-TARS-2 calls its version a "data flywheel" [Wang, Haoming et al., 2025, #19](#ref-19). ScaleCUA sharpens the verification step with VeriGen, "an end-to-end framework for generating verifiable RL tasks through iterative docker interactions and a multi-agent feedback loop", scaled to 100+ concurrent workers to produce 24K+ verifiable tasks and ~3K high-quality RL tasks [Lv et al., 2026, #15](#ref-15).

**Environment synthesis as the new bottleneck.** EnvFactory reframes the problem: the scarce resource is not trajectories but *stateful, executable environments*. It "autonomously explores and verifies stateful, executable tool environments from authentic resources, and synthesizes natural multi-turn trajectories through topology-aware sampling and calibrated refinement, producing grounded queries with implicit intents", and reports that 85 verified environments across 7 domains suffice to beat pipelines using ~5× more environments (+15% BFCLv3, +8.6% MCP-Atlas) [Xu et al., 2026, #26](#ref-26). Its critique of prior synthetic data is pointed and, we think, correct: synthetic trajectories are "frequently over-specified, resembling instruction sequences rather than natural human intents." A query that already names the tool and its arguments teaches formatting, not decision-making.

**Anti-shortcut data design.** OpenSearch-VL is unusually explicit that naive synthesis leaks answers: its Wikipedia path sampling, fuzzy entity rewriting, and source-anchor visual grounding "jointly reduce shortcuts and one-step retrieval collapse" [Chen et al., 2026, #1](#ref-1). *One-step retrieval collapse* — the agent learning that a single search always suffices because the synthesis pipeline made it so — is the multimodal-search analogue of reward hacking and deserves to be checked for in every search-agent paper.

### 3.4 Infrastructure

VerlTool is the reference open framework: upstream-aligned with VeRL, unified tool APIs across code execution, search, SQL, and vision, asynchronous rollout giving "near 2× speedup by eliminating synchronization bottlenecks", evaluated across 6 ARLT domains [Jiang et al., 2025, #11](#ref-11). Systems-level contributions of this kind are load-bearing and under-cited: ScaleCUA's Visual Context Segmentation (a sliding window over recent visual context) yields "a 2.83x training speedup over step-wise decomposition" [Lv et al., 2026, #15](#ref-15) — a larger practical effect than most algorithmic deltas in this review, because rollout throughput, not sample efficiency, is usually the binding constraint for GUI RL.

---

## 4 Tool Ecosystems

### 4.1 Thinking with images

The image-returning tool cluster splits by *who computes the crop*:

- **Backbone-intrinsic grounding.** DeepEyes [#32](#ref-32) — no external module; the model's grounding head selects the region. Reports "distinct evolution of active perception from initial exploration to efficient and accurate exploitation."
- **Fixed operation set.** Pixel Reasoner [#20](#ref-20) — `zoom-in`, `select-frame`; 7B reaching 84% on V*, 74% on TallyQA-Complex, 84% on InfographicsVQA.
- **Structured-data editing tools.** VTool-R1 [#24](#ref-24) — Python visual editing over charts and tables, "the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps", trained purely with outcome rewards and no process supervision.
- **Standardised external vision-tool servers.** OpenThinkIMG [#17](#ref-17) — an RL-trained Qwen2-VL-2B beats its SFT initialisation by +28.83 points on chart reasoning and reported baselines including GPT-4.1 by +8.68.
- **Tool *selection* as the learned skill.** VisTA [#10](#ref-10) — the agent explores and combines tools from a diverse library via GRPO, "without requiring explicit reasoning supervision", with gains concentrated on out-of-distribution examples. This is a distinct and under-explored problem from tool *execution*.
- **Code as universal tool.** PyVision [#31](#ref-31) (dynamic tool generation, +7.8% on V* for GPT-4.1, +31.1% on VLMsAreBlind-mini for Claude-4.0-Sonnet — note these are *inference-time* gains on proprietary models, not a trained open model), Thyme [#30](#ref-30) (500K SFT + GRPO-ATS, ~20 benchmarks), CodeV [#8](#ref-8).
- **Multi-image extension.** IMAgent [#3](#ref-3) observes that "most open-source methods restrict inputs to a single image" and adds visual reflection/verification tools plus an attention-level analysis of *why* tool use helps — namely that VLMs "gradually neglect visual inputs" over long contexts, and tool calls forcibly refocus attention.

**The faithfulness problem.** CodeV supplies the most important negative result in this cluster: "high final-answer accuracy often hides unfaithful visual reasoning: models may invoke tools on irrelevant regions or ignore tool outputs entirely, yet still guess the correct answer", and under an explicit faithfulness protocol (does the crop actually contain the queried evidence?) "recent visual agents achieve high final-answer accuracy but exhibit low rates of faithful tool-use on visual search benchmarks" [Hou et al., 2025, #8](#ref-8). Taken seriously, this invalidates accuracy-only comparisons across the entire §4.1 cluster: two systems with equal accuracy may differ completely in whether the tool did any work. Every paper in this section reports accuracy; one reports faithfulness.

### 4.2 Search and deep research

- **MMSearch-R1** [#23](#ref-23): first end-to-end RL for on-demand multi-turn search in *real* internet environments, image + text search tools, outcome reward with search penalty, search-balanced training data mixing search-required and search-free samples.
- **WebWatcher** [#6](#ref-6): synthetic multimodal trajectories for cold start + tools + RL; introduces BrowseComp-VL.
- **SenseNova-MARS** [#2](#ref-2): interleaves image search, text search, and image crop — i.e. unifies §4.1 and §4.2 tooling — trained with BN-GSPO; introduces HR-MMSearch, "the first search-oriented benchmark composed of high-resolution images."
- **OpenSearch-VL** [#1](#ref-1): the most complete open recipe, motivated by the observation that "top-tier multimodal search agents remain difficult to reproduce, largely due to the absence of open high-quality training data, transparent trajectory synthesis pipelines, or detailed training recipes." Unifies text search, image search, OCR, cropping, sharpening, super-resolution, and perspective correction in one environment.
- **TAPO** [#4](#ref-4): the optimiser-side fix for this cluster (§3.2.3).

**Critical assessment.** Multimodal search agents have a structural evaluation problem: the reward and the benchmark both depend on a live, mutating index. None of the papers above report a frozen-corpus replay protocol. Consequently, cross-paper numbers on BrowseComp-VL, HR-MMSearch, or MMSearch are not strictly comparable, and reported deltas over "RAG baselines" conflate policy quality with index freshness. Note also that three of the five works introduce their *own* benchmark alongside their model — an arrangement that is efficient but structurally favourable to the proposing system.

### 4.3 Code and general-purpose tool interfaces

Code execution is the most expressive tool interface and increasingly the substrate for everything else (image ops, math, data wrangling). ToolRL [#16](#ref-16) and VerlTool [#11](#ref-11) supply the reward-design and infrastructure foundations; EnvFactory [#26](#ref-26) supplies the environments. Notably, the text-agent community has converged on standardised evaluation (BFCLv3, MCP-Atlas, τ²-Bench, all cited by EnvFactory) in a way the multimodal community has not.

### 4.4 GUI, mobile, and computer use

This cluster is methodologically the most mature, because the environment is a real operating system and the reward is genuinely verifiable.

| System | Backbone/scale | Headline result |
|---|---|---|
| OpenCUA [#22](#ref-22) | OpenCUA-72B | 45.0% average success on OSWorld-Verified; SOTA among open-source at publication |
| GUI-Owl / Mobile-Agent-v3 [#27](#ref-27) | GUI-Owl-7B | 66.4 AndroidWorld, 29.4 OSWorld (model); 73.3 / 37.7 with the Mobile-Agent-v3 framework |
| UI-TARS-2 [#19](#ref-19) | native GUI agent | 88.2 Online-Mind2Web, 47.5 OSWorld, 50.6 WindowsAgentArena, 73.3 AndroidWorld; 59.8 mean normalised score on a 15-game suite (~60% of human level) |
| ScaleCUA [#15](#ref-15) | online RLVR | 68.7% OSWorld, 54.0% ScienceBoard; SOTA among open-source CUAs |

The 2024→2026 OSWorld trajectory (OSWorld's own 2024 baselines were in the low single digits to ~12%; OpenCUA 45.0%; ScaleCUA 68.7%) is the steepest verified capability curve in this review. Two caveats: OSWorld-Verified is a revised variant of the original OSWorld [Xie et al., 2024, #25](#ref-25), so cross-year numbers are not strictly identical protocols; and the top systems differ in whether a scaffolding framework is permitted (compare GUI-Owl-7B's 29.4 to Mobile-Agent-v3's 37.7 on the *same* model — the framework, not the policy, contributes 8.3 points).

Two structural insights from the GUI-RL survey [Hu et al., 2026, #9](#ref-9) generalise beyond GUIs: "GUI I/O latency bottlenecks are accelerating the shift toward world-model-based training", and "the spontaneous emergence of System-2-style deliberation suggests that explicit reasoning supervision may not be necessary when sufficiently rich reward signals are available." The latter is a direct challenge to the SFT-cold-start orthodoxy of §3.1 and aligns with DeepEyes and IMAgent.

Cross-domain transfer is claimed but weakly evidenced: UI-TARS-2 reports that "training on diverse environments promotes parameter sharing and capability transfer, giving rise to hybrid skills that integrate graphical interaction with more complex forms of reasoning" [Wang, Haoming et al., 2025, #19](#ref-19). No paper in this review isolates that transfer with a controlled single-environment ablation at matched compute.

---

## 5 Benchmarks and Evaluation

### 5.1 The benchmark inventory

- **OSWorld** [Xie et al., 2024, #25](#ref-25) — 369 real-computer tasks with execution-based validation across operating systems; the field's anchor for computer use.
- **VisualToolBench** [Guo et al., 2025, #7](#ref-7) — 1,204 open-ended vision tasks (603 single-turn, 601 multi-turn) across five domains with per-task rubrics, explicitly built for the think-*with*-images paradigm as opposed to the think-*about*-images paradigm of prior VQA benchmarks.
- **TIR-Bench** [Li et al., 2025, #13](#ref-13) — 13 tasks each requiring novel tool use, 22 MLLMs evaluated including tool-augmented variants; includes a pilot study comparing direct versus agentic fine-tuning.
- **MAT (MAT-Search / MAT-Coding)** — introduced with Visual-ARFT [Liu et al., 2025, #14](#ref-14).
- **BrowseComp-VL** — introduced with WebWatcher [#6](#ref-6). **HR-MMSearch** — introduced with SenseNova-MARS [#2](#ref-2). **Visual Probe Dataset** — introduced with Mini-o3 [#12](#ref-12). **AgentNetBench** — offline evaluator released with OpenCUA [#22](#ref-22).

### 5.2 What the benchmarks say

The headline finding is that *the field is far less capable than its per-paper tables suggest*. On VisualToolBench, "even the strongest model (GPT-5-think) reaches only 18.68% pass rate", and the authors report that model families differ qualitatively — "OpenAI models benefit from diverse image manipulations while Gemini-2.5-pro shows no improvement" [Guo et al., 2025, #7](#ref-7). TIR-Bench concurs: it "is universally challenging, and strong performance requires genuine thinking-with-images capabilities" [Li et al., 2025, #13](#ref-13). TIR-Bench's framing of the standard evaluation target is also a direct criticism of the §4.1 literature: "Even Visual Search, the most common benchmark for current thinking-with-images methods, tests only basic operations such as localization and cropping."

### 5.3 Methodological problems with current evaluation

1. **Saturated proxy tasks.** V*, TallyQA, and visual search generally reduce to crop-and-read. A system can top them with a single well-placed zoom and no genuine multi-step tool policy.
2. **Self-introduced benchmarks.** Six of the resources in §5.1 were released by the same authors as the model they primarily validate. This is understandable (no alternative existed) but means the field currently lacks an independent, adversarially-constructed evaluation for multimodal tool use, with VisualToolBench and TIR-Bench the closest to that role.
3. **Accuracy without faithfulness.** See §4.1; only CodeV [#8](#ref-8) reports a faithfulness metric, and it finds the gap is large.
4. **Non-replayable environments.** Live-internet search and cloud VM environments make exact reproduction impossible; no paper reports variance across environment snapshots.
5. **Unreported inference cost.** Multi-turn agents consume 10–100× the tokens of a single-pass VLM. Almost no comparison in this corpus is compute-matched at inference time, so "tool use beats no tool use" is partly "more test-time compute beats less". MMSearch-R1's explicit search-call accounting [#23](#ref-23) is the honourable exception, and should be the norm.
6. **Framework/model conflation.** As noted in §4.4, scaffolds contribute large deltas that are frequently attributed to the trained policy.

---

## 6 Open Problems and Research Directions

**6.1 Faithful, verifiable process supervision.** The CodeV result — high accuracy with unfaithful tool use — implies that outcome-only RLVR is an under-specified objective for tool agents. The promising direction is rewards computed on the *tool interface* (was the returned crop sufficient to answer the question?) rather than on the CoT, since the former is machine-checkable and the latter is not [Hou et al., 2025, #8](#ref-8). Open: how to define such rewards for open-ended tools (search, code) where "sufficiency" has no closed form.

**6.2 Credit assignment beyond anchor states.** GiGPO's anchor-state grouping requires recurring states; TAPO's counterfactual witnesses require parameter-determinism of information-acquisition tools. Neither assumption holds for image-editing pipelines or long-horizon GUI tasks with irreversible actions. A general, critic-free, turn-level advantage estimator for multimodal POMDPs remains open; VAGEN's Bi-Level GAE is the closest candidate but is validated on a restricted task suite.

**6.3 Tool avoidance and tool overuse — the two-sided pathology.** Pixel Reasoner documents *reluctance* to use new tools; MMSearch-R1 documents *excessive* search. Both are reward-shaping artefacts, and current practice treats them with hand-tuned bonuses/penalties. A principled cost-of-information formulation (call the tool iff expected information gain exceeds its cost) has not been attempted in this corpus.

**6.4 Environment scarcity, not data scarcity.** EnvFactory's result that 85 verified environments beat 5× more environments [#26](#ref-26), and ScaleCUA's verifiable-task generator [#15](#ref-15), together suggest the field's returns now come from *verifiability density* rather than trajectory volume. Extending programmatic, verifiable environment synthesis from API/tool domains into genuinely visual domains (rendered documents, dynamic charts, real applications) is the highest-leverage open engineering problem.

**6.5 Unification of the four tool clusters.** SenseNova-MARS [#2](#ref-2) and OpenSearch-VL [#1](#ref-1) are the only systems that jointly train image manipulation *and* external search; nothing in this corpus jointly trains image manipulation, search, code, and GUI control in one policy. Whether these capabilities interfere or transfer is an open empirical question that UI-TARS-2 asserts optimistically but does not isolate.

**6.6 Reproducibility.** Concrete deficits observed across the corpus: (a) "GRPO" is used to denote a family including DAPO's stabilisers without specifying which; (b) live-environment rewards are not replayable; (c) several strong systems report results with code "to be released"; (d) inference-time compute is rarely matched; (e) SFT-vs-no-SFT conclusions are each drawn on a single backbone. **Papers dropped from this review for unverifiable identifiers**: AgentTrek (referenced only second-hand in OS-Genesis-adjacent text), and a set of titles that appeared in search results but whose arXiv identifiers we did not independently confirm — SimpleSearch-VL, MTA-Agent, VSearcher, VistaHop, Poivre, AdaTooler-V, Diversity Over Frequency, ISE, EnvScaler, and the "Visual Reasoning through Tool-supervised RL" (ToolsRL) entry. Their omission reflects verification discipline, not assessment.

**6.7 Safety and irreversibility.** The GUI-RL survey names "safe exploration in irreversible environments" as a core motivation for RL [Hu et al., 2026, #9](#ref-9), and OpenCUA argues the research community needs open CUA frameworks precisely because these agents "will increasingly mediate digital interactions and execute consequential decisions on our behalf" [#22](#ref-22). Despite this framing, no paper in the corpus reports a safety-constrained RL objective or an irreversibility-aware exploration policy. This is the largest gap between stated motivation and delivered method in the literature reviewed.

---

## 7 Conclusion

Multimodal agentic tool use consolidated remarkably fast. By mid-2026 the field has a shared formalism (multi-turn POMDP with multimodal observation tokens), a shared default recipe (trajectory-distillation cold start followed by GRPO-family RLVR with outcome-verifiable rewards), shared infrastructure (VerlTool-style asynchronous rollout over standardised tool APIs), and, in the computer-use domain, a genuinely steep and verifiable capability curve (OSWorld from single digits in 2024 to 68.7% in 2026).

The consolidation has, however, outrun the field's measurement apparatus. Three findings from 2025–2026 should temper any optimistic reading: agents achieve high accuracy while using tools *unfaithfully* [#8](#ref-8); the default optimiser mis-assigns credit on more than half of failing tool actions [#4](#ref-4); and the strongest available model passes under 19% of a carefully-constructed tool-use benchmark [#7](#ref-7). The near-term research agenda that follows is not "more tools" or "larger backbones" but *verifiable process supervision at the tool interface*, *turn-level credit assignment that does not assume state recurrence*, *programmatically verifiable visual environments*, and *evaluation that is replayable, compute-matched, faithfulness-aware, and not authored by the system's own designers*.

---

## Per-Paper Summary Table

**Legend for column names.** *Paper Name*: short system or benchmark name, with the bibliography anchor number. *Publication Year*: year of first arXiv submission (not the year of the latest revision). *Modality and Tools Used*: input modalities and the concrete tool interface the agent is trained to invoke. *Training Signal Used*: the supervision the work relies on — SFT means supervised fine-tuning on trajectories; RLVR means reinforcement learning with verifiable rewards; GRPO means Group Relative Policy Optimization or a named variant thereof; "none (benchmark)" means the work contributes evaluation rather than training. *Key Contribution*: the single most load-bearing claim of the paper.

| Paper Name | Publication Year | Modality and Tools Used | Training Signal Used | Key Contribution |
|---|---|---|---|---|
| DeepEyes [#32](#ref-32) | 2025 | Image; intrinsic grounding-based zoom/crop | RLVR only, explicitly no SFT cold start | "Thinking with images" emerges end-to-end from RL using the backbone's own grounding, no external vision APIs |
| Pixel Reasoner [#20](#ref-20) | 2025 | Image and video; zoom-in, select-frame | SFT on synthetic traces, then RL with curiosity-driven reward | Formalises pixel-space reasoning; curiosity bonus counteracts the model's reluctance to use new visual operations |
| OpenThinkIMG [#17](#ref-17) | 2025 | Image; standardised external vision tool servers | SFT for policy initialisation, then V-ToolRL (GRPO) | First open end-to-end framework for tool-augmented large vision-language models; +28.83 over its own SFT initialisation on chart reasoning |
| VTool-R1 [#24](#ref-24) | 2025 | Chart and table images; Python visual editing tools | RL fine-tuning with outcome-based rewards, no process supervision | First training of interleaved text-and-image chains of thought rather than text-only reasoning over static images |
| VisTA [#10](#ref-10) | 2025 | Image; diverse external tool library | GRPO on task outcome, no reasoning supervision | Learns tool *selection and composition* as the policy; gains concentrate out of distribution |
| PyVision [#31](#ref-31) | 2025 | Image; dynamically generated Python tools | Inference-time framework, no policy training | Models generate their own tools rather than selecting from a fixed set; taxonomy of emergent tool types |
| Thyme [#30](#ref-30) | 2025 | Image; executable code for image ops and mathematics | 500K-sample SFT, then RL with GRPO-ATS (adaptive temperature sampling) | Richest open image-manipulation-plus-computation action space; separates sampling temperature for code and reasoning |
| Mini-o3 [#12](#ref-12) | 2025 | Image; image-based tools for visual search | Cold-start SFT on diverse exploration patterns, then RL with over-turn masking | Trains at six turns yet scales to tens of turns at test time; accuracy increases with turn count |
| Visual-ARFT [#14](#ref-14) | 2025 | Image; web browsing plus code-based image manipulation | Agentic reinforcement fine-tuning | Joint search-and-code agentic abilities for open large vision-language models; releases the MAT benchmark |
| CodeV / TAPO (process) [#8](#ref-8) | 2025 | Image; visual tools represented as executable Python | Two-stage SFT then process-level RL with dense tool-output rewards | Defines and measures *faithfulness* of visual tool use; shows high accuracy hides unfaithful tool calls |
| IMAgent [#3](#ref-3) | 2025 | Multiple images; visual reflection and verification tools | Pure RL, no SFT, with motion-trajectory masking and tool-use reward gain | Extends thinking-with-images to multi-image inputs; attention-level account of why tool calls help |
| MMSearch-R1 [#23](#ref-23) | 2025 | Image and text; live image search and text search | End-to-end RL with outcome reward plus explicit search penalty | On-demand search: matches a larger retrieval-augmented baseline with over 30% fewer search calls |
| WebWatcher [#6](#ref-6) | 2025 | Image and web pages; browsing and multimodal research tools | Cold start on synthetic multimodal trajectories, then RL | Vision-language deep-research agent; releases the BrowseComp-VL benchmark |
| SenseNova-MARS [#2](#ref-2) | 2025 | High-resolution image and web; image search, text search, image crop | RL with Batch-Normalized Group Sequence Policy Optimization | Interleaves image manipulation with external search in one policy; releases HR-MMSearch |
| OpenSearch-VL [#1](#ref-1) | 2026 | Image and web; text search, image search, OCR, crop, sharpen, super-resolution, perspective correction | SFT on 36k trajectories then RL on 8k with multi-turn fatal-aware GRPO | Fully open reproduction recipe; anti-shortcut data construction against one-step retrieval collapse |
| TAPO (credit transfer) [#4](#ref-4) | 2026 | Image and web; multimodal search tools | Advantage correction layered on GRPO, GSPO, and SAPO | Quantifies credit misassignment in over half of failing tool actions and corrects it with in-batch counterfactual witnesses |
| UI-TARS-2 [#19](#ref-19) | 2025 | Screenshots; GUI actions plus file system and terminal | Data flywheel SFT plus stabilised multi-turn RL | 47.5 on OSWorld and 73.3 on AndroidWorld; hybrid GUI-plus-terminal environment |
| OpenCUA [#22](#ref-22) | 2025 | Screenshots across three operating systems; GUI actions | SFT on human demonstrations augmented with reflective long chain-of-thought | AgentNet dataset and full open computer-use-agent stack; 45.0% on OSWorld-Verified |
| GUI-Owl / Mobile-Agent-v3 [#27](#ref-27) | 2025 | Mobile and desktop screenshots; GUI actions | Self-evolving trajectory production plus asynchronous online RL with trajectory-aware policy optimization | Cloud environment across four operating systems with a self-improving data loop; 73.3 AndroidWorld with the agent framework |
| ScaleCUA [#15](#ref-15) | 2026 | Screenshots; GUI actions in Docker environments | Online RLVR with verifiable synthesised tasks and frontier sampling | 68.7% on OSWorld; verifiable task generation plus a 2.83× training speedup from visual context segmentation |
| OS-Genesis [#18](#ref-18) | 2024 | Mobile and web screenshots; GUI actions | Trajectory data synthesis for SFT, filtered by a trajectory reward model | Reverse task synthesis: explore first, derive tasks retrospectively |
| VAGEN [#21](#ref-21) | 2025 | Visual observations in interactive environments; environment actions | RL with a dense world-modeling reward and Bi-Level Generalized Advantage Estimation | Formalises vision-language-model agents as a POMDP; shows belief representation format is task-dependent |
| GiGPO [#5](#ref-5) | 2025 | Text agent environments (algorithm transfers to multimodal) | Two-level group RL: episode groups plus anchor-state step groups | Critic-free step-level credit assignment; over 12% on ALFWorld and 9% on WebShop above GRPO at equal memory |
| ToolRL [#16](#ref-16) | 2025 | Text; general function calling | GRPO with decomposed tool-name, schema, and parameter-value rewards | First systematic study of reward design for tool use; 17% over base and 15% over supervised fine-tuning |
| VerlTool [#11](#ref-11) | 2025 | Text, image, and video observations; code, search, SQL, vision tools | Infrastructure for agentic RL with tool use | Unified modular framework with asynchronous rollout giving near 2× speedup across six tool-use domains |
| DAPO [#28](#ref-28) | 2025 | Text (algorithmic foundation) | Decoupled clipping, dynamic sampling, token-level loss, over-long reward shaping | Open, reproducible large-scale RL system whose stabilisers are inherited by most multimodal follow-ups |
| EnvFactory [#26](#ref-26) | 2026 | Text and tool APIs; stateful executable environments | Automated environment synthesis for SFT and RL | Environments, not trajectories, are the bottleneck: 85 verified environments beat pipelines with five times more |
| OSWorld [#25](#ref-25) | 2024 | Real desktop screenshots; full computer control | None (benchmark) with execution-based validation | 369 real-computer tasks; the field's anchor benchmark for computer-use agents |
| VisualToolBench [#7](#ref-7) | 2025 | Image plus general-purpose tools; 1,204 tasks | None (benchmark) with per-task rubrics | Think-with-images evaluation; strongest model reaches only 18.68% pass rate |
| TIR-Bench [#13](#ref-13) | 2025 | Image; 13 tasks each needing novel tool use | None (benchmark), plus a direct-versus-agentic fine-tuning pilot | Shows visual search alone is too easy a proxy for agentic thinking-with-images |
| Agentic RL survey [#29](#ref-29) | 2025 | Cross-modal survey | None (survey) | Reframes large language models from sequence generators to decision-making agents; organises the agentic RL landscape |
| GUI agents with RL survey [#9](#ref-9) | 2026 | GUI survey | None (survey) | Offline / online / hybrid taxonomy; identifies latency-driven shift to world-model training and emergent System-2 deliberation |

---

## Bibliography

<a id="ref-1"></a>**[1]** Shuang Chen, Kaituo Feng, Hangting Chen, Wenxuan Huang, Dasen Dai, Quanxin Shou, Yunlong Lin, Xiangyu Yue, Shenghua Gao, Tianyu Pang. *OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents.* arXiv:2605.05185, 2026. https://arxiv.org/abs/2605.05185

<a id="ref-2"></a>**[2]** Yong Xien Chng, Tao Hu, Wenwen Tong, Xueheng Li, Jiandong Chen, Haojia Yu, Jiefan Lu, Hewei Guo, Hanming Deng, Chengjun Xie, Gao Huang, Dahua Lin, Lewei Lu. *SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning.* arXiv:2512.24330, 2025 (v2, Jan 2026). https://arxiv.org/abs/2512.24330

<a id="ref-3"></a>**[3]** Chengqi Dong, Chuhuai Yue, Hang He, Rongge Mao, Fenghe Tang, S. Kevin Zhou, Zekun Xu, Xiaohan Wang, Jiajun Chai, Guojun Yin. *Training Multi-Image Vision Agents via End2End Reinforcement Learning* (IMAgent). arXiv:2512.08980, 2025 (v3, Apr 2026). https://arxiv.org/abs/2512.08980

<a id="ref-4"></a>**[4]** Chengqi Dong, Chuhuai Yue, Hang He, Yandong Liu, Fenghe Tang, S. Kevin Zhou, Xiaohan Wang, Jiajun Chai, Guojun Yin. *TAPO: Tool-Aware Policy Optimization via Credit Transfer for Multimodal Search Agents.* arXiv:2606.05784, 2026. https://arxiv.org/abs/2606.05784

<a id="ref-5"></a>**[5]** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An. *Group-in-Group Policy Optimization for LLM Agent Training.* NeurIPS 2025; arXiv:2505.10978, 2025. https://arxiv.org/abs/2505.10978

<a id="ref-6"></a>**[6]** Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou. *WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent.* arXiv:2508.05748, 2025. https://arxiv.org/abs/2508.05748

<a id="ref-7"></a>**[7]** Xingang Guo, Utkarsh Tyagi, Advait Gosai, Paula Vergara, Jayeon Park, Ernesto Gabriel Hernández Montoya, Chen Bo Calvin Zhang, Bin Hu, Yunzhong He, Bing Liu, Rakshith Sharma Srinivasa. *Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning* (VisualToolBench). arXiv:2510.12712, 2025. https://arxiv.org/abs/2510.12712

<a id="ref-8"></a>**[8]** Xinhai Hou, Shaoyuan Xu, Manan Biyani, Moyan Li, Jia Liu, Todd C. Hollon, Bryan Wang. *CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization.* arXiv:2511.19661, 2025 (v2, Mar 2026). https://arxiv.org/abs/2511.19661

<a id="ref-9"></a>**[9]** Junan Hu, Jian Liu, Jingxiang Lai, Jiarui Hu, Yiwei Sheng, Shuang Chen, Jian Li, Dazhao Du, Song Guo. *GUI Agents with Reinforcement Learning: Toward Digital Inhabitants.* arXiv:2604.27955, 2026. https://arxiv.org/abs/2604.27955

<a id="ref-10"></a>**[10]** Zeyi Huang, Yuyang Ji, Anirudh Sundara Rajan, Zefan Cai, Wen Xiao, Haohan Wang, Junjie Hu, Yong Jae Lee. *VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection.* arXiv:2505.20289, 2025. https://arxiv.org/abs/2505.20289

<a id="ref-11"></a>**[11]** Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, Wenhu Chen. *VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use.* arXiv:2509.01055, 2025. https://arxiv.org/abs/2509.01055

<a id="ref-12"></a>**[12]** Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, Hengshuang Zhao. *Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search.* arXiv:2509.07969, 2025. https://arxiv.org/abs/2509.07969

<a id="ref-13"></a>**[13]** Ming Li, Jike Zhong, Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Yuxiang Lai, Chen Wei, Konstantinos Psounis, Kaipeng Zhang. *TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images Reasoning.* arXiv:2511.01833, 2025. https://arxiv.org/abs/2511.01833

<a id="ref-14"></a>**[14]** Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang. *Visual Agentic Reinforcement Fine-Tuning* (Visual-ARFT). arXiv:2505.14246, 2025. https://arxiv.org/abs/2505.14246

<a id="ref-15"></a>**[15]** Bowen Lv, Xiao Liu, Yanyu Ren, Hanyu Lai, Bohao Jing, Hanchen Zhang, Yanxiao Zhao, Shuntian Yao, Jie Tang, Yuxiao Dong. *ScaleCUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL.* arXiv:2607.11185, 2026. https://arxiv.org/abs/2607.11185

<a id="ref-16"></a>**[16]** Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji. *ToolRL: Reward is All Tool Learning Needs.* arXiv:2504.13958, 2025. https://arxiv.org/abs/2504.13958

<a id="ref-17"></a>**[17]** Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, Yu Cheng. *OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning.* arXiv:2505.08617, 2025. https://arxiv.org/abs/2505.08617

<a id="ref-18"></a>**[18]** Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, Zhiyong Wu. *OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis.* ACL 2025; arXiv:2412.19723, 2024. https://arxiv.org/abs/2412.19723

<a id="ref-19"></a>**[19]** Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, et al. (approximately 105 authors), Guang Shi. *UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning.* arXiv:2509.02544, 2025. https://arxiv.org/abs/2509.02544

<a id="ref-20"></a>**[20]** Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, Wenhu Chen. *Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning.* arXiv:2505.15966, 2025. https://arxiv.org/abs/2505.15966

<a id="ref-21"></a>**[21]** Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Chi Wan, Yiping Lu, Zhengyuan Yang, Lijuan Wang, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Yejin Choi, Manling Li. *VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents.* NeurIPS 2025; arXiv:2510.16907, 2025. https://arxiv.org/abs/2510.16907

<a id="ref-22"></a>**[22]** Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, Tao Yu. *OpenCUA: Open Foundations for Computer-Use Agents.* NeurIPS 2025 Spotlight; arXiv:2508.09123, 2025. https://arxiv.org/abs/2508.09123

<a id="ref-23"></a>**[23]** Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, Ziwei Liu. *MMSearch-R1: Incentivizing LMMs to Search.* arXiv:2506.20670, 2025. https://arxiv.org/abs/2506.20670

<a id="ref-24"></a>**[24]** Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, ChengXiang Zhai, Klara Nahrstedt. *VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use.* ICLR 2026; arXiv:2505.19255, 2025. https://arxiv.org/abs/2505.19255

<a id="ref-25"></a>**[25]** Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, Tao Yu. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS 2024 Datasets and Benchmarks; arXiv:2404.07972, 2024. https://arxiv.org/abs/2404.07972

<a id="ref-26"></a>**[26]** Minrui Xu, Zilin Wang, Mengyi Deng, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, Heyuan Deng, Fei Mi, Lifeng Shang, Xingshan Zeng, Zhijiang Guo. *EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL.* arXiv:2605.18703, 2026. https://arxiv.org/abs/2605.18703

<a id="ref-27"></a>**[27]** Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, Ming Yan. *Mobile-Agent-v3: Fundamental Agents for GUI Automation* (introduces GUI-Owl). arXiv:2508.15144, 2025. https://arxiv.org/abs/2508.15144

<a id="ref-28"></a>**[28]** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476, 2025. https://arxiv.org/abs/2503.14476

<a id="ref-29"></a>**[29]** Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita-Velez, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai. *The Landscape of Agentic Reinforcement Learning for LLMs: A Survey.* Transactions on Machine Learning Research; arXiv:2509.02547, 2025. https://arxiv.org/abs/2509.02547

<a id="ref-30"></a>**[30]** Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, Guorui Zhou. *Thyme: Think Beyond Images.* ICLR 2026; arXiv:2508.11630, 2025. https://arxiv.org/abs/2508.11630

<a id="ref-31"></a>**[31]** Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, Chen Wei. *PyVision: Agentic Vision with Dynamic Tooling.* MTI-LLM Workshop, NeurIPS 2025; arXiv:2507.07998, 2025. https://arxiv.org/abs/2507.07998

<a id="ref-32"></a>**[32]** Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, Xing Yu. *DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning.* ICLR 2026; arXiv:2505.14362, 2025. https://arxiv.org/abs/2505.14362

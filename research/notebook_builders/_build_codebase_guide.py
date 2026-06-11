"""Builds codebase_guide.ipynb — an explanatory notebook for the vl-training-engine repo."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})

md(r"""# vl-training-engine — Codebase Guide

> A map of this repo: what it is, how the pieces connect, and how to drive it for **SFT**, **RL/RLHF**, and the other use cases (inference, deploy, export, eval, Megatron).

This is a **heavily-refactored fork of [ms-swift](https://github.com/modelscope/ms-swift)** (v4.3.0.dev0) — ModelScope's training framework for 600+ LLMs and 400+ multimodal LLMs (VLMs). It covers the full lifecycle: pre-train → SFT → RLHF → infer → eval → quantize → deploy.

**Note on the fork:** upstream ms-swift keeps most logic under `swift/llm/`. This fork has **flattened** that — the modules now live directly under `swift/` (e.g. `swift/model`, `swift/template`, `swift/dataset`, `swift/pipelines`, `swift/trainers`, `swift/rlhf_trainers`). Also note this fork uses the flag **`--tuner_type`** (upstream uses `--train_type`). Keep that in mind when copying commands from the public docs.
""")

md(r"""## 1. The 10-second mental model

Everything is one pattern repeated per task:

```
CLI subcommand   →   *_main() entry   →   Swift* pipeline class   →   Trainer/Engine
  swift sft           sft_main()            SwiftSft                  Seq2SeqTrainer
  swift rlhf          rlhf_main()           SwiftRLHF                 DPO/GRPO/PPO/... Trainer
  swift pt            pretrain_main()       SwiftPretrain             Seq2SeqTrainer
  swift infer         infer_main()          (infer pipeline)          InferEngine (vLLM/sglang/...)
  swift deploy        deploy_main()         (deploy pipeline)         InferEngine + OpenAI server
  swift rollout       rollout_main()        (rollout pipeline)        vLLM sampling server (feeds GRPO)
  swift export        export_main()         (export pipeline)         GPTQ/AWQ/FP8/BNB quant
  swift eval          eval_main()           (eval pipeline)           EvalScope/OpenCompass
```

A pipeline always does the same four things, then trains:

1. **Load model + processor** — `get_model_processor()` (`swift/model`)
2. **Build the chat template** — `get_template()` (`swift/template`) — turns messages into token ids + loss `labels`
3. **Load + encode the dataset** — `load_dataset()` + `EncodePreprocessor` (`swift/dataset`)
4. **Apply the tuner** — LoRA/QLoRA/full/... (`swift/tuners`, selected by `--tuner_type`)
5. **Pick a trainer** via `TrainerFactory` and call `.train()`.

The four building blocks (model / template / dataset / tuner) are **registries** — pluggable maps you can extend without touching the pipeline.
""")

md(r"""## 2. Repo layout (the parts that matter)

```
swift/
├── cli/              # thin entry points, one file per subcommand → calls a *_main()
│   ├── main.py       #   ROUTE_MAPPING: 'sft' → swift.cli.sft, etc. + torchrun wiring
│   └── sft.py rlhf.py infer.py deploy.py rollout.py export.py eval.py ...
├── pipelines/        # the orchestration layer (the "do the 5 steps" logic)
│   ├── train/        #   sft.py (SwiftSft), rlhf.py (SwiftRLHF), pretrain.py
│   ├── infer/        #   infer.py, deploy.py, rollout.py
│   ├── export/       #   export.py, merge_lora.py, quant.py, ollama.py
│   ├── eval/  sampling/  app/
├── arguments/        # ALL config dataclasses (this is your "API reference")
│   ├── base_args.py  #   model/dataset/template common args
│   ├── sft_args.py   #   SftArguments  (+ TunerArguments + Seq2SeqTrainingArguments)
│   ├── rlhf_args.py  #   RLHFArguments (rlhf_type, reward_model, GRPO/PPO knobs)
│   └── infer_args.py deploy_args.py export_args.py eval_args.py ...
├── model/            # MODEL registry: get_model_processor(), register_model()
├── template/         # TEMPLATE registry: get_template(), register_template()
├── dataset/          # DATASET registry: load_dataset(), EncodePreprocessor, packing
├── tuners/           # Swift PEFT methods: LoRA, QLoRA, DoRA, adapter, ReFT, ... (Swift class)
├── trainers/         # base trainers: Seq2SeqTrainer (SFT/PT/seq-cls/embedding/reranker)
├── rlhf_trainers/    # DPO/GRPO/PPO/KTO/CPO/ORPO/GKD/reward trainers + vllm_client
├── rollout/          # GRPO rollout: multi_turn.py, gym_env.py (agentic/env rollouts)
├── rewards/          # reward functions for GRPO: orm.py (outcome), prm.py (process)
├── infer_engine/     # vLLM / SGLang / LMDeploy / Transformers backends + InferClient
├── megatron/         # Megatron-LM integration (TP/PP/CP/EP) — `megatron` CLI
├── sequence_parallel/ ray/  loss/  optimizers/  metrics/  callbacks/  ui/
└── __init__.py       # top-level Python API exports (lazy-loaded)

examples/train/       # copy-paste bash recipes: lora_sft.sh, grpo/, rlhf/{dpo,ppo,gkd,kto}/,
                      # full/, qlora/, multimodal/, multi-node/, packing/, rft/, ...
```
""")

md(r"""## 3. The four registries (the heart of the framework)

These are what make it support hundreds of models with one codebase. Each is a dict you can register into.

| Registry | Lives in | Key functions | What it gives you |
|---|---|---|---|
| **Model** | `swift/model` | `get_model_processor()`, `register_model()`, `ModelMeta` | loads weights + processor from ModelScope/HF, knows the default template, multimodal flags, quant support |
| **Template** | `swift/template` | `get_template()`, `register_template()`, `TemplateMeta` | formats `messages` into the model's chat format; in `train` mode emits `input_ids`/`labels`, in `infer` mode emits a generation prompt. Handles image/video tokens for VLMs. |
| **Dataset** | `swift/dataset` | `load_dataset()`, `register_dataset()`, `EncodePreprocessor` | resolves dataset ids → rows → encoded tensors; supports `#N` sampling, packing, lazy/streaming |
| **Tuner** | `swift/tuners` | `Swift`, `--tuner_type` | wraps the model with LoRA/QLoRA/DoRA/adapter/ReFT/full; integrates with PEFT |

**Why this matters for you:** to support a new model or dataset, you usually *register* one — you don't fork the trainer.
""")

md(r"""## 4. SFT — Supervised Fine-Tuning

- **CLI:** `swift sft`  → `swift/cli/sft.py` → `sft_main()`
- **Pipeline:** `swift/pipelines/train/sft.py` → class `SwiftSft`
- **Args:** `swift/arguments/sft_args.py` → `SftArguments`
- **Trainer:** `swift/trainers` → `Seq2SeqTrainer`

`SftArguments` = `BaseArguments` (model/dataset/template) + `TunerArguments` (lora_rank, target_modules, ...) + HF `Seq2SeqTrainingArguments` (lr, epochs, batch size, deepspeed, ...). So **any HF TrainingArguments flag works** alongside the swift-specific ones.

### Minimal LoRA SFT (see `examples/train/lora_sft.sh`)
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' 'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --max_length 2048 \
    --output_dir output
```

### Full-parameter / efficiency knobs
- `--tuner_type full` for full fine-tuning (use `--deepspeed zero2|zero3`).
- `--packing true` concatenates short samples up to `max_length` (throughput).
- `--padding_free true --attn_impl flash_attn` removes pad tokens (memory/throughput).
- `--lazy_tokenize true` / `--streaming true` for big datasets.

### VLM SFT
Same command, just point `--model` at a VLM (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) and use a dataset whose rows carry `images`/`videos`. The template auto-handles vision tokens. See `examples/train/multimodal/`.
""")

md(r"""## 5. RL / RLHF

- **CLI:** `swift rlhf` → `swift/cli/rlhf.py` → `rlhf_main()`
- **Pipeline:** `swift/pipelines/train/rlhf.py` → class `SwiftRLHF(SwiftSft)` (so it inherits the whole SFT machinery, then adds ref/reward/value/teacher models)
- **Args:** `swift/arguments/rlhf_args.py` → `RLHFArguments(SftArguments)`
- **Trainers:** `swift/rlhf_trainers/` — one file per algorithm.

Pick the algorithm with **`--rlhf_type`**:

| `--rlhf_type` | Trainer file | Needs | Notes |
|---|---|---|---|
| `dpo`  | `dpo_trainer.py`  | preference pairs | no reward model; `--rpo_alpha` adds an SFT term |
| `kto`  | `kto_trainer.py`  | binary good/bad labels | unpaired preference |
| `cpo` / `orpo` | `cpo_trainer.py` / `orpo_trainer.py` | preference pairs | reference-free variants |
| `rm`   | `reward_trainer.py` | preference pairs | trains a reward model |
| `ppo`  | `ppo_trainer.py`  | `--reward_model` + ref + value | classic online RL |
| **`grpo`** | `grpo_trainer.py` | reward funcs/model | group-relative; **vLLM rollout**; the main VLM-RL path |
| `gkd`  | `gkd_trainer.py`  | `--teacher_model` | on-policy distillation |

### DPO (offline, simplest RL) — `examples/train/rlhf/dpo/`
```bash
swift rlhf --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct --tuner_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --rpo_alpha 0.1 --learning_rate 1e-4 --lora_rank 8
```

### GRPO (the online RL workhorse) — `examples/train/grpo/`
GRPO samples `num_generations` completions per prompt, scores them with **reward functions** (`swift/rewards/orm.py`, `prm.py`) and/or a `--reward_model`, and optimizes the group-relative advantage. Generation is done by **vLLM** for speed.

Two vLLM modes:
- **colocate** (`--vllm_mode colocate`): vLLM shares the training GPUs. Simplest; one job.
- **server** (`--vllm_mode server`): a separate `swift rollout` server generates; trainer pulls samples and pushes updated weights. Scales better. See `examples/train/grpo/external/`.

```bash
NPROC_PER_NODE=8 swift rlhf --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct --tuner_type lora \
    --dataset lmms-lab/multimodal-open-r1-8k-verified#1000 \
    --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 4 \
    --reward_funcs external_r1v_acc format --reward_weights 1 0.1 \
    --num_generations 8 --max_completion_length 2048 \
    --temperature 1.1 --log_completions true --deepspeed zero3
```

**Custom rewards** are the lever for your task: drop a function in `swift/rewards/` (or via `--external_plugins`) returning a score per completion, and reference it in `--reward_funcs`. For agentic / multi-turn / environment rollouts see `swift/rollout/multi_turn.py` and `gym_env.py`.

### GKD / on-policy distillation — `examples/train/rlhf/gkd/`, `examples/train/on_policy_distillation.sh`
Student generates, teacher scores/provides logits. `--teacher_model` (local) or a remote teacher server. This is the relevant path for teacher→student transfer.
""")

md(r"""## 6. Everything else (one-liners)

| Task | Command | Pipeline / file |
|---|---|---|
| **Pre-train** | `swift pt --model ... --dataset ...` | `pipelines/train/pretrain.py` |
| **Inference / chat** | `swift infer --model <ckpt> --infer_backend vllm` | `pipelines/infer/infer.py` |
| **Deploy (OpenAI API)** | `swift deploy --model <ckpt> --infer_backend vllm --port 8000` | `pipelines/infer/deploy.py` |
| **Rollout server (GRPO)** | `swift rollout --model <ckpt> --vllm_tensor_parallel_size 4` | `pipelines/infer/rollout.py` |
| **Merge LoRA** | `swift merge-lora --adapters <ckpt> --model <base>` | `pipelines/export/merge_lora.py` |
| **Quantize / export** | `swift export --model <ckpt> --quant_method gptq` | `pipelines/export/quant.py` |
| **Evaluate** | `swift eval --model <ckpt> --eval_dataset mmlu` | `pipelines/eval/eval.py` |
| **Batch sample** | `swift sample --model <ckpt> --dataset prompts.jsonl` | `pipelines/sampling/` |
| **Web UI / app** | `swift web-ui` / `swift app` | `pipelines/app/`, `swift/ui/` |
| **Megatron train** | `megatron sft ...` (separate CLI) | `swift/megatron/` |

**Distributed:** set `NPROC_PER_NODE` (the CLI auto-wraps in `torchrun` for pt/sft/rlhf/infer — see `swift/cli/main.py`). Use `--deepspeed zero2|zero3` for ZeRO; `swift/megatron` for TP/PP/CP/EP on very large / MoE models; `swift/sequence_parallel` for long-context.
""")

md(r"""## 7. Dataset format

A row is a list of `messages` (preferred), or the classic `instruction`/`input`/`output` fields. Multimodal rows add `images` / `videos` / `audios` paths or URLs.

```jsonc
// SFT (messages form)
{"messages": [{"role": "system", "content": "You are helpful."},
              {"role": "user", "content": "What is 2+2?"},
              {"role": "assistant", "content": "4"}]}

// SFT with an image (VLM) — <image> tag marks where the image goes
{"messages": [{"role": "user", "content": "<image>What is this?"},
              {"role": "assistant", "content": "A cat."}],
 "images": ["/path/cat.jpg"]}

// DPO / preference
{"messages": [{"role": "user", "content": "..."}],
 "chosen":   {"role": "assistant", "content": "good answer"},
 "rejected": {"role": "assistant", "content": "bad answer"}}

// GRPO — prompt + ground truth your reward function checks against
{"messages": [{"role": "user", "content": "Solve: ..."}], "solution": "42"}
```

- Pass datasets by **id** (ModelScope/HF, e.g. `AI-ModelScope/alpaca-gpt4-data-en`) or **local path** (`data.jsonl`).
- `#N` after a name **samples N rows**: `'alpaca-en#500'`.
- `--split_dataset_ratio 0.1` carves a val set; or pass `--val_dataset` explicitly.
- Register reusable custom datasets with `register_dataset(DatasetMeta(...))` in `swift/dataset`.
""")

md(r"""## 8. Python API (skip the CLI)

The CLI is a thin wrapper. Anything you can do on the command line you can do in Python — pass an args object (or a list of CLI-style strings) to the `*_main()` functions.

The cells below are runnable references (they will download models, so run deliberately).
""")

code(r"""# What the top-level package exposes (no heavy import side effects thanks to lazy loading)
import swift
print("version:", swift.__version__)
# Most-used entry points:
from swift.pipelines import (
    sft_main, rlhf_main, pretrain_main,      # training
    infer_main, deploy_main, rollout_main,   # inference / serving
    export_main, merge_lora,                  # export
    eval_main, sampling_main,                 # eval / sampling
)
from swift.arguments import SftArguments, RLHFArguments, InferArguments
print("ok")""")

code(r"""# --- SFT entirely in Python ---
from swift.pipelines import sft_main
from swift.arguments import SftArguments

args = SftArguments(
    model='Qwen/Qwen2.5-0.5B-Instruct',         # tiny model for a smoke test
    dataset=['AI-ModelScope/alpaca-gpt4-data-en#200'],
    tuner_type='lora', lora_rank=8, lora_alpha=32, target_modules='all-linear',
    num_train_epochs=1, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=1e-4,
    torch_dtype='bfloat16', max_length=1024,
    output_dir='output/py_sft_demo',
)
# sft_main(args)   # <- uncomment to actually train
print("configured SFT:", args.model, "->", args.output_dir)""")

code(r"""# --- DPO / GRPO in Python: same shape, RLHFArguments + rlhf_main ---
from swift.arguments import RLHFArguments
# from swift.pipelines import rlhf_main

dpo_args = RLHFArguments(
    rlhf_type='dpo',
    model='Qwen/Qwen2.5-0.5B-Instruct', tuner_type='lora', lora_rank=8,
    dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#200'],
    rpo_alpha=0.1, learning_rate=1e-4, output_dir='output/py_dpo_demo',
)
# rlhf_main(dpo_args)
print("rlhf_type:", dpo_args.rlhf_type)""")

code(r"""# --- The building blocks directly: load model, template, dataset (the 'manual' path) ---
from swift.model import get_model_processor
from swift.template import get_template
from swift.dataset import load_dataset, EncodePreprocessor

# model, processor = get_model_processor('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='bfloat16')
# template = get_template(processor, default_system='You are helpful.', max_length=1024)
# template.set_mode('train')
# train_ds, val_ds = load_dataset(['AI-ModelScope/alpaca-gpt4-data-en#100'], split_dataset_ratio=0.1)
# train_ds = EncodePreprocessor(template=template)(train_ds, num_proc=2)
# print(train_ds[0].keys())   # -> input_ids / labels / attention_mask (+ pixel_values for VLMs)
print("see swift/__init__.py for the full export list")""")

code(r"""# --- Inference in Python (vLLM / transformers backends behind one interface) ---
from swift.arguments import InferArguments
# from swift.pipelines import infer_main
infer_args = InferArguments(
    model='output/py_sft_demo',     # or any base model / merged ckpt
    infer_backend='vllm',           # 'vllm' | 'sglang' | 'lmdeploy' | 'pt'
    # stream=True,
)
# infer_main(infer_args)
print("infer backend:", infer_args.infer_backend)""")

md(r"""## 9. How to find things fast

- **"What flag does X?"** → grep `swift/arguments/`. Every option is a dataclass field with a default, often with a comment. Start at `base_args.py`, `sft_args.py`, `rlhf_args.py`.
- **"How does algorithm Y work?"** → read the matching file in `swift/rlhf_trainers/` (e.g. `grpo_trainer.py`). The `compute_loss` / `_prepare_inputs` methods are the meat.
- **"A working command for Z?"** → `examples/train/` is organized by use case (`grpo/`, `rlhf/dpo/`, `full/`, `qlora/`, `multimodal/`, `multi-node/`, `rft/`, `on_policy_distillation.sh`, ...). Each `.sh` is tested and copy-pasteable.
- **"Is model/dataset M supported?"** → search `swift/model/` / `swift/dataset/` registries, or `swift infer --model M` and let auto-detection tell you.
- **End-to-end notebook example** → upstream ships `examples/notebook/`; the manual-trainer pattern there mirrors §8's building-block cells.

### The one connection to remember
`cli/<x>.py` → `<x>_main()` in `pipelines/` → a `Swift*` class that **(1) loads a model, (2) builds a template, (3) encodes a dataset, (4) wraps with a tuner, (5) hands off to a Trainer** — all four of those are swappable registries configured entirely through the `*Arguments` dataclasses. Learn the registries + the args dataclasses and the whole repo opens up.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("/mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/notebooks/codebase_guide.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote codebase_guide.ipynb with", len(cells), "cells")

#!/usr/bin/env python3
"""
compute_surds_acceptance.py — vLLM acceptance-rate generation + scoring for SURDS GRPO data prep.

Reads SURDS train_qa.jsonl (raw QA schema, NOT ms-swift messages/images format).
For each prompt, generates n_samples rollouts via vLLM and scores each against the gold answer
using score_surds.parse_answer + score_surds.score_one.

Output: acceptance_train.parquet with columns:
    prompt_id, template_type, image_path, image_wh (JSON "[W,H]"), gold, question,
    n_samples, n_pass, acceptance, sample_texts (list[str]), sample_correct (list[bool])

Usage (smoke test — validate 8 prompts first per project convention):
    python research/data_scripts/compute_surds_acceptance.py --limit 8

Usage (full 41k run):
    python research/data_scripts/compute_surds_acceptance.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: add research/eval to sys.path so score_surds is importable
# as a sibling of this script's parent package (research/eval/).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # vl-training-engine/
_EVAL_DIR = _REPO_ROOT / "research" / "eval"
sys.path.insert(0, str(_EVAL_DIR))

import score_surds  # noqa: E402  (imported after sys.path bootstrap)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_JSONL = (
    "/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/train_qa.jsonl"
)
DEFAULT_MODEL = (
    "/mnt/data4/shasta/amar.amarjyoti/research_data/sft_runs/"
    "pretrain_model_15_surdsXmulberry_full_1063208/v0-20260609-113540/checkpoint-37119"
)
DEFAULT_OUT = (
    "/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/acceptance_train.parquet"
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="vLLM acceptance-rate generation + scoring for SURDS GRPO data prep."
    )
    p.add_argument(
        "--train_jsonl",
        default=DEFAULT_TRAIN_JSONL,
        help="Path to SURDS train_qa.jsonl (raw QA schema).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to SFT student checkpoint (vLLM model dir).",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output parquet path.",
    )
    p.add_argument("--tp", type=int, default=8, help="Tensor parallel size (number of GPUs).")
    p.add_argument(
        "--gpu_mem_util", type=float, default=0.9, help="vLLM GPU memory utilization."
    )
    p.add_argument(
        "--max_model_len", type=int, default=4096, help="vLLM max_model_len."
    )
    p.add_argument(
        "--n_samples", type=int, default=16, help="Number of sampled rollouts per prompt."
    )
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling.")
    p.add_argument(
        "--max_tokens", type=int, default=2048, help="Max new tokens per generation."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process only the first N prompts (smoke test). "
             "Recommended: --limit 8 before launching the full 41k run.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load train data (raw QA schema)
# ---------------------------------------------------------------------------

def load_train(jsonl_path: str, limit=None):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


# ---------------------------------------------------------------------------
# Build vLLM prompts from raw QA records
#
# train_qa.jsonl `prompt` field already contains the full task text:
#   description + question + options + "think/answer format" instruction.
# T1 notes: "prompt: full task text (description + question + options + think/answer format) inline."
# We therefore use it verbatim — no system prompt is injected (the Thinking model
# emits <think>…</think><answer>…</answer> from its built-in instruction tuning).
#
# Prompt building mirrors gen_val_ablation.build_prompts:
#   1. Open image as PIL RGB.
#   2. Build a user content list: [image_part, text_part].
#   3. Apply processor.apply_chat_template(tokenize=False, add_generation_prompt=True).
#   4. Return (prompt_str, pil_image) pairs for vLLM TextPrompt.
# ---------------------------------------------------------------------------

def build_prompts(records, processor):
    """
    Returns list of (prompt_str, pil_image) tuples.

    Assumption: train_qa.jsonl `prompt` field is self-contained (includes question,
    options, and the think/answer format instruction). No extra system prompt is added.
    """
    prompts = []
    for rec in tqdm(records, desc="Building prompts", file=sys.stderr):
        img_path = rec["image_path"]
        img = Image.open(img_path).convert("RGB")

        user_content_list = [
            {"type": "image", "image": img},
            {"type": "text", "text": rec["prompt"].strip()},
        ]
        chat_msgs = [{"role": "user", "content": user_content_list}]

        prompt_str = processor.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append((prompt_str, img))
    return prompts


# ---------------------------------------------------------------------------
# vLLM generation (batched, inner batch 32 — same pattern as gen_val_ablation)
# ---------------------------------------------------------------------------

def run_vllm_pass(llm, vllm_prompts, sampling_params, desc: str):
    """Run vLLM generate in batches of 32, return list of RequestOutput."""
    outputs = []
    batch_size = 32
    for i in tqdm(range(0, len(vllm_prompts), batch_size), desc=desc, file=sys.stderr):
        batch = vllm_prompts[i: i + batch_size]
        results = llm.generate(batch, sampling_params)
        outputs.extend(results)
    return outputs


# ---------------------------------------------------------------------------
# Score a single record's n rollout texts
# ---------------------------------------------------------------------------

def score_rollouts(rec, sample_texts):
    """
    Given a raw QA record and a list of generated text strings, score each.
    Returns (n_pass, sample_correct).

    xy2d: passes image_wh to score_one for px-space scoring (per T3 spec).
    All other templates: image_wh=None.
    """
    gold = rec["answer"]
    tt = rec["template_type"]

    if tt == "xy2d":
        image_wh = score_surds.get_image_wh(rec["image_path"])
    else:
        image_wh = None

    n_pass = 0
    sample_correct = []
    for text in sample_texts:
        pred = score_surds.parse_answer(text)
        if pred is None:
            # parse failure → incorrect (not a crash)
            sample_correct.append(False)
            continue
        result = score_surds.score_one(
            pred_answer=pred,
            gold_answer=gold,
            template_type=tt,
            image_wh=image_wh,
        )
        correct = bool(result["correct"])
        sample_correct.append(correct)
        if correct:
            n_pass += 1

    return n_pass, sample_correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    # Validate inputs
    if not os.path.isfile(args.train_jsonl):
        sys.exit(f"ERROR: train_jsonl not found: {args.train_jsonl}")
    model_path = Path(args.model)
    if not model_path.is_dir():
        sys.exit(f"ERROR: model dir not found: {model_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[compute_surds_acceptance] train_jsonl={args.train_jsonl}", flush=True)
    print(f"[compute_surds_acceptance] model={model_path}", flush=True)
    print(f"[compute_surds_acceptance] out={out_path}", flush=True)
    if args.limit:
        print(f"[compute_surds_acceptance] SMOKE TEST: processing first {args.limit} prompts.", flush=True)

    # ------------------------------------------------------------------
    # 1. Load train data
    # ------------------------------------------------------------------
    print("[compute_surds_acceptance] Loading train data...", flush=True)
    records = load_train(args.train_jsonl, limit=args.limit)
    print(f"[compute_surds_acceptance] Loaded {len(records)} records.", flush=True)

    # ------------------------------------------------------------------
    # 2. Load processor (tokenizer only, no model weights) for chat template
    # ------------------------------------------------------------------
    print("[compute_surds_acceptance] Loading processor for chat template...", flush=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    # ------------------------------------------------------------------
    # 3. Build prompts from raw QA records
    # ------------------------------------------------------------------
    print("[compute_surds_acceptance] Building prompts...", flush=True)
    prompts = build_prompts(records, processor)

    # ------------------------------------------------------------------
    # 4. Build vLLM TextPrompt list
    # ------------------------------------------------------------------
    from vllm.inputs.data import TextPrompt
    vllm_prompts = [
        TextPrompt(prompt=p, multi_modal_data={"image": img})
        for p, img in prompts
    ]

    # ------------------------------------------------------------------
    # 5. Launch vLLM engine
    # ------------------------------------------------------------------
    print(
        f"[compute_surds_acceptance] Launching vLLM "
        f"(tp={args.tp}, dtype=bfloat16, max_model_len={args.max_model_len})...",
        flush=True,
    )
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        disable_custom_all_reduce=True,
    )

    # ------------------------------------------------------------------
    # 6. Sampled generation: n=args.n_samples rollouts per prompt
    # ------------------------------------------------------------------
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n_samples,
        max_tokens=args.max_tokens,
    )
    print(
        f"[compute_surds_acceptance] Generating {args.n_samples} rollouts per prompt "
        f"(temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens})...",
        flush=True,
    )
    raw_outputs = run_vllm_pass(
        llm, vllm_prompts, sampling_params, desc="Sampled rollouts"
    )

    # ------------------------------------------------------------------
    # 7. Score rollouts and build output rows
    # ------------------------------------------------------------------
    print("[compute_surds_acceptance] Scoring rollouts...", flush=True)
    rows = []
    for rec, out in tqdm(
        zip(records, raw_outputs), total=len(records), desc="Scoring", file=sys.stderr
    ):
        texts = [o.text for o in out.outputs]   # list of n_samples strings
        n_pass, sample_correct = score_rollouts(rec, texts)

        # image_wh: always compute (stored as JSON string "[W,H]" per T3 schema)
        wh = score_surds.get_image_wh(rec["image_path"])
        image_wh_str = json.dumps(list(wh))     # e.g. "[1600, 900]"

        rows.append(
            {
                "prompt_id": rec["sample_id"],
                "template_type": rec["template_type"],
                "image_path": rec["image_path"],
                "image_wh": image_wh_str,
                "gold": rec["answer"],
                "question": rec["prompt"],
                "n_samples": args.n_samples,
                "n_pass": n_pass,
                "acceptance": n_pass / args.n_samples,
                "sample_texts": texts,
                "sample_correct": sample_correct,
            }
        )

    df = pd.DataFrame(rows)

    # Cast to compact dtypes per T3 schema
    df["n_samples"] = df["n_samples"].astype("int8")
    df["n_pass"] = df["n_pass"].astype("int8")
    df["acceptance"] = df["acceptance"].astype("float32")

    # ------------------------------------------------------------------
    # 8. Write parquet atomically (temp → rename)
    # ------------------------------------------------------------------
    tmp_path = str(out_path) + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, str(out_path))
    print(f"[compute_surds_acceptance] Wrote {len(df)} rows to {out_path}", flush=True)

    # ------------------------------------------------------------------
    # 9. Quick sanity print
    # ------------------------------------------------------------------
    overall_acc = df["acceptance"].mean()
    print(
        f"[compute_surds_acceptance] Mean acceptance across {len(df)} prompts: "
        f"{overall_acc:.4f}",
        flush=True,
    )
    per_tt = df.groupby("template_type")["acceptance"].mean()
    print("[compute_surds_acceptance] Per-template mean acceptance:", flush=True)
    for tt, acc in per_tt.items():
        print(f"  {tt:12s}  {acc:.4f}", flush=True)

    wall_time = time.time() - t0
    print(f"[compute_surds_acceptance] Done. Wall time: {wall_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()

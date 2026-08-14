#!/usr/bin/env python3
"""
gen_val_ablation.py — vLLM offline inference for SURDS×Mulberry ablation evaluation.

Runs TWO passes per val example:
  (a) greedy  — temperature=0, n=1
  (b) sampled — temperature=<--temp>, top_p=0.95, n=<--n-sample>

Output: parquet with columns [idx, arm, image_path, prompt_tokens, greedy_text, samples]
        + JSON sidecar with run metadata.

Usage:
    python gen_val_ablation.py \
        --ckpt  <checkpoint-dir> \
        --arm   <arm-name> \
        --val   /path/to/val_1k.jsonl \
        --out   /path/to/output.parquet \
        --base  thinking   # or instruct
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="vLLM offline inference for ablation eval")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint dir")
    p.add_argument("--arm", required=True, help="Arm name (used in output metadata)")
    p.add_argument("--val", default="/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/sft_stageB/val_1k.jsonl",
                   help="Path to val jsonl")
    p.add_argument("--out", required=True, help="Output parquet path")
    p.add_argument("--base", choices=["thinking", "instruct"], default="thinking",
                   help="Model base type: thinking (emits <think>) or instruct")
    p.add_argument("--n-sample", type=int, default=8,
                   help="Number of sampled responses per example (pass@n)")
    p.add_argument("--temp", type=float, default=0.8,
                   help="Sampling temperature for pass@n")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max new tokens per generation")
    p.add_argument("--tp", type=int, default=8,
                   help="Tensor parallel size (number of GPUs)")
    p.add_argument("--gpu-mem-util", type=float, default=0.90,
                   help="vLLM GPU memory utilization")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Number of prompts per vLLM .generate() call")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="vLLM max_model_len. Default 4096 (val prompts fit). Raise for "
                        "long-reasoning teachers if prompt+max_tokens can exceed it.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graphs. REQUIRED for the Qwen3-VL-235B FP8 MoE teacher "
                        "(flashinfer MoE-kernel workaround); harmless (slower) otherwise.")
    p.add_argument("--quantization", default=None,
                   help="vLLM quantization arg (e.g. 'fp8'). Default None = auto-detect from the "
                        "model's config (FP8 checkpoints carry their own quant_config).")
    p.add_argument("--primers", default=None,
                   help="Comma-list of convention primers to inject, by template: "
                        f"{sorted(PRIMERS)}. 'none' disables all. Default: "
                        f"{','.join(DEFAULT_PRIMERS)} (yaw is PERMANENT — A/B-validated "
                        "+4.6pp greedy). Pass --primers none to reproduce pre-2026-08 arms.")
    p.add_argument("--coord-primer", action="store_true",
                   help="DEPRECATED alias: force-include the yaw primer (now on by default).")
    p.add_argument("--only-templates", default=None,
                   help="Comma-list of templates to keep (e.g. 'yaw' or 'fb'), detected by "
                        "prompt signature. Preserves original idx for the meta join.")
    p.add_argument("--yaw-only", action="store_true",
                   help="DEPRECATED alias for --only-templates yaw.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load val data
# ---------------------------------------------------------------------------

def load_val(val_path: str):
    records = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Build prompts using the Qwen3-VL processor's apply_chat_template
#
# vLLM 0.15.0 chat API: llm.chat(messages) accepts OpenAI-style content lists
# with {"type": "image_url", "image_url": {"url": "file://..."}} or
# {"type": "image", "image": <PIL>}.  However, the most reliable path that
# avoids double-counting the <image> placeholder is to:
#   1. Use the HF processor (tokenizer only, no model weights) to apply the
#      chat template and obtain the text prompt string with vision tokens
#      already inserted (e.g. <|vision_start|><|image_pad|>...<|vision_end|>).
#   2. Pass that tokenized-text prompt + the PIL image via vLLM's
#      TextPrompt(prompt=<str>, multi_modal_data={"image": <PIL>}).
#
# This approach:
#   - Avoids the <image> double-counting issue because we strip the plain
#     "<image>" text from the user turn and let apply_chat_template insert
#     the proper vision tokens via the image object in the content list.
#   - Works with vLLM 0.15.0's generate() API which accepts TextPrompt dicts
#     with multi_modal_data.
#   - Handles Thinking vs Instruct: for Thinking, apply_chat_template adds
#     "<think>\n" continuation by default; for Instruct we omit it.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONVENTION PRIMERS
#
# Two SURDS templates define a term in a way that contradicts everyday usage, and
# every model we tested (our 8B student AND the 32B/235B teachers) silently applies
# the everyday meaning instead. These primers state the task's own definition
# explicitly. They fix a CONVENTION gap, not a perception gap.
#
#   yaw : camera-relative compass frame. Unstable image-axis->compass rule =>
#         180deg toward/away flips (+ 90deg axis errors, which are perception and
#         NOT fixed by the primer). A/B job 1067475 (2026-07-15, cp896, 333 ex):
#         greedy .486->.532, sampled .459->.538, 180-flips 57->39 (-32%).
#         => PERMANENT (default on) as of 2026-08.
#
#   fb  : "front/back" is defined by distance from camera — the object FARTHER
#         from the camera is "more forward". So "A in front of B" = A farther,
#         and "A behind B" = A CLOSER. Measured on heldout (rl_init_cp896):
#         'in front of' phrasing acc .832 (perception is fine — cf. distance .811)
#         but 'behind' phrasing acc .230 — FAR BELOW the .50 binary chance floor,
#         i.e. a systematic inversion, not guessing. The 235B teacher shows the
#         same split (.787 / .252). A/B job 1071082 (2026-08-12, cp896, 333 ex):
#         greedy .544->.811, 'behind' .236->.732, 'in front of' .818->.881 (no
#         trade-off), pass@16 .799->.979. fb now matches distance (.811), i.e. the
#         convention was the entire gap. => PERMANENT (default on).
#
# Injected per-template by prompt signature; see --primers.
# ---------------------------------------------------------------------------
COORD_PRIMER = """Coordinate reference (use this to convert the object's visible orientation into a compass direction):

The camera's facing direction (stated above) fixes how image directions map to the compass. Find the row matching the camera's facing direction:

  Camera faces NORTH -> far/top = North, near/bottom (toward camera) = South, image-right = East,  image-left = West
  Camera faces SOUTH -> far/top = South, near/bottom (toward camera) = North, image-right = West,  image-left = East
  Camera faces EAST  -> far/top = East,  near/bottom (toward camera) = West,  image-right = South, image-left = North
  Camera faces WEST  -> far/top = West,  near/bottom (toward camera) = East,  image-right = North, image-left = South

An object's facing direction is the compass direction its FRONT points. Find where the front points in the image, then read the compass value from the row above:
  - You see the object's REAR (taillights, back)         -> front points away into the scene -> use far/top.
  - You see the object's FRONT (grille, headlights, face) -> front points toward the camera    -> use near/bottom.
  - You see the object's RIGHT side                       -> front points toward image-right   -> use image-right.
  - You see the object's LEFT side                        -> front points toward image-left    -> use image-left.
  - For a 3/4 view, combine the two nearest directions (e.g. front + right side is between near/bottom and image-right).

Critical: "facing toward the camera" is the OPPOSITE of the camera's own facing direction; "facing away from the camera" is the SAME as the camera's facing direction. Do not swap these."""

FB_PRIMER = """Front/back convention for this task (it is defined by DISTANCE FROM THE CAMERA, and is the OPPOSITE of everyday usage — read carefully):

  "A is IN FRONT OF B"  means A is FARTHER from the camera than B.
  "A is BEHIND B"       means A is CLOSER to the camera than B.

In everyday speech "behind" suggests farther away. That is NOT the meaning here. Here the object farther from the camera is the one that is more forward.

Procedure:
  1. First decide purely which object is CLOSER to the camera and which is FARTHER (use occlusion, apparent size, and ground contact — lower in the image is usually closer).
  2. Then answer using the definitions above:
       - Asked "Is A in front of B?"  -> answer Yes if A is FARTHER from the camera than B, otherwise No.
       - Asked "Is A behind B?"       -> answer Yes if A is CLOSER to the camera than B, otherwise No.
  3. If the two are at nearly the same distance, choose the "almost the same" option.

Do not skip step 2: decide the distance order first, then apply the definition literally, even when it feels backwards."""


_YAW_SIG = re.compile(r"camera\b.{0,40}?\bis facing\b", re.I)
# fb prompts uniquely say "front-back position(ing)"; distance asks "closer to the
# camera" and never uses that phrase.
_FB_SIG = re.compile(r"front-back position", re.I)


def is_yaw_prompt(user_text: str) -> bool:
    """Yaw prompts uniquely state the camera heading ('The camera in the image
    is facing North'); no other SURDS template mentions it."""
    return bool(user_text) and bool(_YAW_SIG.search(user_text))


def is_fb_prompt(user_text: str) -> bool:
    """fb prompts uniquely describe 'relative front-back positioning'."""
    return bool(user_text) and bool(_FB_SIG.search(user_text))


# template -> (detector, primer text)
PRIMERS = {
    "yaw": (is_yaw_prompt, COORD_PRIMER),
    "fb": (is_fb_prompt, FB_PRIMER),
}
# Default-ON primers. Both are PERMANENT — each A/B-validated as strictly positive
# on the heldout set with its own controlled job (see the per-template notes above).
DEFAULT_PRIMERS = ("yaw", "fb")


def parse_primers(spec: str):
    """'yaw,fb' -> ('yaw','fb'); 'none'/'' -> (). Unknown names are an error."""
    if spec is None:
        return tuple(DEFAULT_PRIMERS)
    spec = spec.strip().lower()
    if spec in ("none", "off", ""):
        return ()
    out = tuple(s.strip() for s in spec.split(",") if s.strip())
    bad = [s for s in out if s not in PRIMERS]
    if bad:
        sys.exit(f"ERROR: unknown --primers value(s): {bad}; known: {sorted(PRIMERS)}")
    return out


def matching_template(user_text: str):
    """Return the template name whose detector matches, else None."""
    for name, (det, _) in PRIMERS.items():
        if det(user_text):
            return name
    return None


def build_prompts(records, processor, base: str, primers=DEFAULT_PRIMERS):
    """
    Returns list of (prompt_str, pil_image) tuples.
    primers: iterable of template names whose convention primer is appended.
    """
    primers = tuple(primers or ())
    prompts = []
    n_primed = Counter()
    for rec in tqdm(records, desc="Building prompts", file=sys.stderr):
        messages = [m for m in rec["messages"] if m["role"] != "assistant"]
        image_path = rec["images"][0]
        img = Image.open(image_path).convert("RGB")

        # Separate system and user messages
        sys_msg = None
        user_text = None
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                user_text = m["content"]

        # Append the convention primer for whichever template this prompt is.
        # Placed AFTER the task/question text so the rule is the freshest context
        # right before <think> (validated placement in the yaw A/B).
        _tpl = matching_template(user_text) if primers else None
        if _tpl in primers:
            user_text = user_text.rstrip() + "\n\n" + PRIMERS[_tpl][1]
            n_primed[_tpl] += 1

        # Build content list for user: strip the plain "<image>" tag from text
        # and pass the image as a separate content part.
        user_content_list = [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text.replace("<image>", "").strip()},
        ]

        chat_msgs = []
        if sys_msg:
            chat_msgs.append({"role": "system", "content": sys_msg})
        chat_msgs.append({"role": "user", "content": user_content_list})

        # apply_chat_template: tokenize=False → text string with vision tokens
        # For Thinking models, add_generation_prompt inserts <think> continuation.
        prompt_str = processor.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append((prompt_str, img))
    print(f"[gen_val_ablation] primers={list(primers) or 'none'} injected={dict(n_primed)}",
          flush=True)
    return prompts


# ---------------------------------------------------------------------------
# Tokenize to get prompt token counts (cheap — just the tokenizer, no model)
# ---------------------------------------------------------------------------

def count_prompt_tokens(prompts, processor):
    counts = []
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for prompt_str, img in tqdm(prompts, desc="Counting tokens", file=sys.stderr):
        # Encode the text portion only for a rough count (images add more)
        ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"]
        counts.append(ids.shape[1])
    return counts


# ---------------------------------------------------------------------------
# vLLM generation helpers
# ---------------------------------------------------------------------------

def run_vllm_pass(llm, vllm_prompts, sampling_params, desc: str):
    """Run vLLM generate in batches, return list of RequestOutput."""
    from vllm import SamplingParams  # noqa: F401 (already imported at top)
    outputs = []
    batch_size = 32  # inner vllm batching; vllm also queues internally
    for i in tqdm(range(0, len(vllm_prompts), batch_size), desc=desc, file=sys.stderr):
        batch = vllm_prompts[i : i + batch_size]
        results = llm.generate(batch, sampling_params)
        outputs.extend(results)
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    t0 = time.time()

    # Validate inputs
    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        sys.exit(f"ERROR: checkpoint dir does not exist: {ckpt}")
    val_path = args.val
    if not os.path.isfile(val_path):
        sys.exit(f"ERROR: val file does not exist: {val_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gen_val_ablation] arm={args.arm}  base={args.base}", flush=True)
    print(f"[gen_val_ablation] ckpt={ckpt}", flush=True)
    print(f"[gen_val_ablation] val={val_path}", flush=True)
    print(f"[gen_val_ablation] out={out_path}", flush=True)

    # ------------------------------------------------------------------
    # 1. Load val data
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Loading val data...", flush=True)
    records = load_val(val_path)
    print(f"[gen_val_ablation] Loaded {len(records)} val examples.", flush=True)

    # Tag every record with its ORIGINAL position so a filtered run still writes
    # meta-aligned idx (scoring joins gen.idx -> heldout_val_meta.idx by position).
    for _i, _rec in enumerate(records):
        _rec["_orig_idx"] = _i

    def _user_text(r):
        for m in r["messages"]:
            if m["role"] == "user":
                return m["content"]
        return ""

    keep = args.only_templates
    if args.yaw_only and not keep:
        keep = "yaw"          # deprecated alias
    if keep:
        want = parse_primers(keep)   # same name validation as --primers
        records = [r for r in records if matching_template(_user_text(r)) in want]
        print(f"[gen_val_ablation] --only-templates {list(want)}: kept {len(records)} records.",
              flush=True)

    # ------------------------------------------------------------------
    # 2. Load processor (tokenizer only, no model weights) for chat template
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Loading processor for chat template...", flush=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)

    # ------------------------------------------------------------------
    # 3. Build prompts
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Building prompts...", flush=True)
    _primers = parse_primers(args.primers)
    if args.coord_primer and "yaw" not in _primers:   # deprecated alias
        _primers = _primers + ("yaw",)
    prompts = build_prompts(records, processor, args.base, primers=_primers)

    # ------------------------------------------------------------------
    # 4. Token counts (text portion only — cheap proxy)
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Counting prompt tokens (text portion)...", flush=True)
    prompt_token_counts = count_prompt_tokens(prompts, processor)

    # ------------------------------------------------------------------
    # 5. Build vLLM TextPrompt list
    #    TextPrompt = {"prompt": <str>, "multi_modal_data": {"image": <PIL>}}
    # ------------------------------------------------------------------
    from vllm.inputs.data import TextPrompt
    vllm_prompts = [
        TextPrompt(prompt=p, multi_modal_data={"image": img})
        for p, img in prompts
    ]

    # ------------------------------------------------------------------
    # 6. Launch vLLM engine
    # ------------------------------------------------------------------
    print(f"[gen_val_ablation] Launching vLLM (tp={args.tp}, dtype=bfloat16)...", flush=True)
    import vllm
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=str(ckpt),
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,   # default 4096; val examples fit well within
        limit_mm_per_prompt={"image": 1},
        # tp=8 custom all-reduce kernel fails on this node's GPU topology
        # (custom_all_reduce.cuh:455 'invalid argument'); fall back to NCCL.
        disable_custom_all_reduce=True,
        enforce_eager=args.enforce_eager,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    # ------------------------------------------------------------------
    # 7a. Pass (a): GREEDY — temperature=0, n=1
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Pass (a): greedy (temp=0, n=1)...", flush=True)
    greedy_params = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=args.max_tokens,
    )
    greedy_outputs = run_vllm_pass(llm, vllm_prompts, greedy_params, desc="Greedy pass")
    greedy_texts = [out.outputs[0].text for out in greedy_outputs]

    # ------------------------------------------------------------------
    # 7b. Pass (b): SAMPLED — temperature=args.temp, n=args.n_sample
    # ------------------------------------------------------------------
    print(f"[gen_val_ablation] Pass (b): sampled (temp={args.temp}, n={args.n_sample})...", flush=True)
    sample_params = SamplingParams(
        temperature=args.temp,
        top_p=0.95,
        n=args.n_sample,
        max_tokens=args.max_tokens,
    )
    sample_outputs = run_vllm_pass(llm, vllm_prompts, sample_params, desc="Sample pass")
    sample_texts = [[o.text for o in out.outputs] for out in sample_outputs]

    # ------------------------------------------------------------------
    # 8. Build output dataframe
    # ------------------------------------------------------------------
    print("[gen_val_ablation] Building output dataframe...", flush=True)
    rows = []
    for idx, (rec, greedy, samples, n_tok) in enumerate(
        zip(records, greedy_texts, sample_texts, prompt_token_counts)
    ):
        rows.append({
            "idx": rec.get("_orig_idx", idx),
            "arm": args.arm,
            "image_path": rec["images"][0],
            "prompt_tokens": n_tok,
            "greedy_text": greedy,
            "samples": samples,  # list of n_sample strings
        })
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 9. Write parquet atomically (temp file → rename)
    # ------------------------------------------------------------------
    tmp_path = str(out_path) + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, str(out_path))
    print(f"[gen_val_ablation] Wrote {len(df)} rows to {out_path}", flush=True)

    # ------------------------------------------------------------------
    # 10. Write JSON sidecar with run metadata
    # ------------------------------------------------------------------
    wall_time = time.time() - t0
    sidecar = {
        "arm": args.arm,
        "ckpt": str(ckpt),
        "base": args.base,
        "val": val_path,
        "out": str(out_path),
        "n_val": len(records),
        "n_sample": args.n_sample,
        "temp": args.temp,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "quantization": args.quantization,
        "tp": args.tp,
        "vllm_version": vllm.__version__,
        "wall_time_sec": round(wall_time, 1),
        # Which convention primers were in the prompt. Recorded because it changes
        # what the arm MEANS — arms generated before 2026-08 have primers=[].
        "primers": list(_primers),
        "only_templates": keep,
    }
    sidecar_path = str(out_path).replace(".parquet", "_meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[gen_val_ablation] Metadata written to {sidecar_path}", flush=True)
    print(f"[gen_val_ablation] Done. Wall time: {wall_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()

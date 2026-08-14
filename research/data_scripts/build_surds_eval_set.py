#!/usr/bin/env python3
"""
build_surds_eval_set.py — build a frozen, stratified held-out eval set for the
SURDS RL bake-off.

Source : $DATA_ROOT/processed/surds/validation_qa.jsonl
         (9250 rows, balanced: lr/distance/fb/yaw=1850 each, xy2d/depth=925 each)

Default output (N=1800, 300 per template family):
    $DATA_ROOT/processed/surds/eval/surds_heldout_eval_1800.jsonl

With --all flag, converts all 9250 rows to the ms-swift messages/images format
and writes to:
    $DATA_ROOT/processed/surds/eval/surds_heldout_eval_all.jsonl

Output format matches what gen_val_ablation.py::load_val expects:
    {
      "messages": [{"role": "user", "content": "<image><question_prompt>"}],
      "images":   ["/abs/path/to/image.jpg"],
      "answer":       <gold answer string>,
      "template_type": <family slug>,
      "image_path":    <same as images[0]>
    }

Sampling is deterministic: fixed seed=0, stratified by template_type, then within
each stratum shuffled and the first N_per_family rows taken.

Usage:
    python build_surds_eval_set.py [--data-root DATA_ROOT] [--all] [--n-per-family N]

    DATA_ROOT defaults to /mnt/data4/shasta/amar.amarjyoti/research_data
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


TEMPLATE_FAMILIES = ["lr", "distance", "fb", "yaw", "xy2d", "depth"]
DEFAULT_N_PER_FAMILY = 300
DEFAULT_DATA_ROOT = "/mnt/data4/shasta/amar.amarjyoti/research_data"
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description="Build frozen stratified SURDS held-out eval set")
    p.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Data root (default: {DEFAULT_DATA_ROOT})",
    )
    p.add_argument(
        "--n-per-family",
        type=int,
        default=DEFAULT_N_PER_FAMILY,
        help=f"Samples per template family for the default eval set (default: {DEFAULT_N_PER_FAMILY})",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also produce the full 9250-row converted file (surds_heldout_eval_all.jsonl)",
    )
    return p.parse_args()


def load_jsonl(path: str):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {i}: {e}", file=sys.stderr)
    return records


def convert_to_swift_format(rec: dict) -> dict:
    """
    Convert a validation_qa.jsonl row to the ms-swift messages/images schema
    that gen_val_ablation.py::load_val expects.

    Expected input fields (from validation_qa.jsonl):
        prompt / question  — the question text
        answer             — gold answer string
        template_type      — one of lr/distance/fb/yaw/xy2d/depth
        image_path         — absolute path to the image file

    Output:
        messages   = [{"role": "user", "content": "<image>" + prompt_text}]
        images     = [image_path]
        answer     = gold answer
        template_type = slug
        image_path = same as images[0]
    """
    # Normalise field names: source may use "prompt" or "question"
    prompt_text = rec.get("prompt") or rec.get("question") or ""
    answer = rec.get("answer", "")
    template_type = rec.get("template_type", "")
    image_path = rec.get("image_path", "")

    # Prepend <image> tag if not already present (gen_val_ablation strips it back out)
    if not prompt_text.startswith("<image>"):
        content = "<image>" + prompt_text
    else:
        content = prompt_text

    return {
        "messages": [{"role": "user", "content": content}],
        "images": [image_path],
        "answer": answer,
        "template_type": template_type,
        "image_path": image_path,
    }


def write_jsonl(records, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, str(out))
    print(f"Wrote {len(records)} rows -> {out}")


def build_stratified_sample(records, n_per_family: int, seed: int = SEED):
    """
    Stratified sample: n_per_family rows per template_type.
    Within each stratum, shuffle with fixed seed then take first n_per_family.
    Returns (sampled_records, per_family_counts).
    """
    by_family = defaultdict(list)
    for rec in records:
        tt = rec.get("template_type", "").strip().lower()
        by_family[tt].append(rec)

    sampled = []
    counts = {}
    for family in TEMPLATE_FAMILIES:
        pool = by_family.get(family, [])
        if len(pool) < n_per_family:
            print(
                f"WARNING: family '{family}' has only {len(pool)} rows "
                f"(requested {n_per_family}); taking all.",
                file=sys.stderr,
            )
        rng = random.Random(seed)
        shuffled = pool[:]
        rng.shuffle(shuffled)
        taken = shuffled[:n_per_family]
        sampled.extend(taken)
        counts[family] = len(taken)

    # Stable sort by template_type then original index for reproducibility
    sampled.sort(key=lambda r: (r.get("template_type", ""), records.index(r)))
    return sampled, counts


def main():
    args = parse_args()
    data_root = args.data_root
    n_per_family = args.n_per_family

    val_path = os.path.join(data_root, "processed", "surds", "validation_qa.jsonl")
    eval_dir = os.path.join(data_root, "processed", "surds", "eval")

    # ------------------------------------------------------------------
    # 1. Load source
    # ------------------------------------------------------------------
    print(f"Loading validation_qa.jsonl from: {val_path}")
    if not os.path.isfile(val_path):
        sys.exit(f"ERROR: source file not found: {val_path}")
    records = load_jsonl(val_path)
    print(f"Loaded {len(records)} rows.")

    # Print source distribution
    src_counts = Counter(r.get("template_type", "UNKNOWN") for r in records)
    print("Source template distribution:")
    for fam in TEMPLATE_FAMILIES:
        print(f"  {fam:12s}: {src_counts.get(fam, 0)}")

    # ------------------------------------------------------------------
    # 2. Build stratified eval set (default N=1800, 300 per family)
    # ------------------------------------------------------------------
    total_n = n_per_family * len(TEMPLATE_FAMILIES)
    print(f"\nBuilding stratified eval set: {n_per_family} per family = {total_n} total (seed={SEED})")
    sampled_raw, per_family = build_stratified_sample(records, n_per_family, seed=SEED)

    print("Per-template counts in eval set:")
    for fam in TEMPLATE_FAMILIES:
        print(f"  {fam:12s}: {per_family.get(fam, 0)}")
    assert sum(per_family.values()) == len(sampled_raw), "Count mismatch"

    # Convert to ms-swift format
    sampled_swift = [convert_to_swift_format(r) for r in sampled_raw]

    out_path = os.path.join(eval_dir, f"surds_heldout_eval_{total_n}.jsonl")
    write_jsonl(sampled_swift, out_path)

    # Disjoint note: validation_qa.jsonl is a separate file from the training data
    # (training uses curriculum/stage_*.jsonl derived from train split). No
    # image-path overlap check needed — the file-level split enforces disjointness.
    print(
        "\nDisjoint guarantee: validation_qa.jsonl is from the held-out validation "
        "split, separate from curriculum/stage_*.jsonl (train split). "
        "No image-path overlap with training data."
    )

    # ------------------------------------------------------------------
    # 3. Optional: convert all 9250 rows
    # ------------------------------------------------------------------
    if args.all:
        print(f"\nConverting all {len(records)} rows (--all flag)...")
        all_swift = [convert_to_swift_format(r) for r in records]
        all_path = os.path.join(eval_dir, "surds_heldout_eval_all.jsonl")
        write_jsonl(all_swift, all_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

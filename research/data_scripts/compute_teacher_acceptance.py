"""
compute_teacher_acceptance.py
-----------------------------
Build a per-prompt acceptance parquet from the EXISTING Stage-B rejection-sampling
generations (Qwen3-VL-235B teacher, N=16 @ T0.8) — no GPU needed. Scores each of the
16 teacher answers per prompt with research/eval/score_surds.score_one and writes a
parquet in the SAME schema build_surds_curriculum_shards.py consumes.

NOTE: this is the TEACHER's pass-rate, used as a difficulty proxy (see plan for the
teacher-vs-student caveat). xy2d is scored in pixel space via get_image_wh (matches the
student-side harness + eval).

Output columns: prompt_id, template_type, image_path, image_wh, gold, question,
n_samples, n_pass, acceptance, sample_correct.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

# import score_surds (sibling of this file's repo at research/eval)
REPO = "/mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine"
sys.path.insert(0, os.path.join(REPO, "research", "eval"))
import score_surds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n16_jsonl",
        default="/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/"
                "cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl",
    )
    ap.add_argument(
        "--out",
        default="/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/"
                "acceptance_train_teacher235b.parquet",
    )
    ap.add_argument("--n_samples", type=int, default=16)
    args = ap.parse_args()

    # group rows by prompt id
    groups = defaultdict(list)
    meta = {}
    n_rows = 0
    print(f"Reading {args.n16_jsonl} ...")
    with open(args.n16_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d["id"]
            groups[pid].append(d)
            n_rows += 1
            if pid not in meta:
                meta[pid] = {
                    "template_type": d.get("template_type"),
                    "image_path": d.get("image_path"),
                    "gold": d.get("gt_answer"),
                    "question": d.get("prompt"),
                }
            if n_rows % 100000 == 0:
                print(f"  read {n_rows:,} rows ...")
    print(f"  total rows {n_rows:,}, unique prompts {len(groups):,}")

    records = []
    wh_cache = {}
    done = 0
    for pid, samples in groups.items():
        m = meta[pid]
        tt = (m["template_type"] or "").strip().lower()
        img = m["image_path"]
        image_wh = None
        if tt == "xy2d":
            if img not in wh_cache:
                try:
                    wh_cache[img] = score_surds.get_image_wh(img)
                except Exception:
                    wh_cache[img] = None
            image_wh = wh_cache[img]
        sample_correct = []
        for s in samples:
            pred = s.get("answer")
            try:
                res = score_surds.score_one(pred, m["gold"], tt, image_wh=image_wh)
                sample_correct.append(bool(res.get("correct", False)))
            except Exception:
                sample_correct.append(False)
        n_pass = int(sum(sample_correct))
        n = len(sample_correct)
        records.append({
            "prompt_id": pid,
            "template_type": tt,
            "image_path": img,
            "image_wh": json.dumps(list(image_wh)) if image_wh else "",
            "gold": str(m["gold"]),
            "question": m["question"] or "",
            "n_samples": n,
            "n_pass": n_pass,
            "acceptance": float(n_pass) / float(n) if n else 0.0,
            "sample_correct": sample_correct,
        })
        done += 1
        if done % 10000 == 0:
            print(f"  scored {done:,}/{len(groups):,} prompts ...")

    df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df):,} rows -> {args.out}")
    # quick distribution print
    acc = df["acceptance"]
    print("\nAcceptance distribution (teacher 235B):")
    print(f"  mean {acc.mean():.3f}  median {acc.median():.3f}")
    print(f"  ==0.00            : {(acc == 0).sum():>6}")
    print(f"  (0.00, 0.20)      : {((acc > 0) & (acc < 0.20)).sum():>6}")
    print(f"  [0.20, 0.25]      : {((acc >= 0.20) & (acc <= 0.25)).sum():>6}")
    print(f"  base [0.20, 0.75] : {((acc >= 0.20) & (acc <= 0.75)).sum():>6}")
    print(f"  hard [0.00, 0.25] : {((acc >= 0.00) & (acc <= 0.25)).sum():>6}")
    print(f"  >0.75 (drop)      : {(acc > 0.75).sum():>6}")
    print("\nper-template mean acceptance:")
    print(df.groupby("template_type")["acceptance"].mean().to_string())


if __name__ == "__main__":
    main()

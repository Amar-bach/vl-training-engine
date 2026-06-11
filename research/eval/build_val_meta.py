"""Build val_meta.parquet — one row per val_1k example.

Columns: idx, image_path, question, gold_answer, template_type, answer_kind.

template_type is recovered from the "Task Description:" line of the user message
using the SURDS family signatures, and cross-checked against the source QA file's
own template_type slugs (lr/distance/fb/yaw/xy2d/depth).
"""
import json
import re
from pathlib import Path

import pandas as pd

from score_surds import CATEGORICAL_TEMPLATES, CONTINUOUS_TEMPLATES

VAL = Path("/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/sft_stageB/val_1k.jsonl")
OUT = Path(__file__).parent / "val_meta.parquet"

# Task-Description signature -> canonical source-QA slug.
# Signatures are distinctive substrings of the val_1k "Task Description" line.
FAMILY_SIG = [
    ("relative left-right positioning", "lr"),
    ("determine which of the two objects is closer", "distance"),
    ("relative front-back positioning", "fb"),
    ("accurately identify and provide the coordinates", "xy2d"),
    ("estimate the vertical distance", "depth"),
    ("identify the direction that the specified object is facing", "yaw"),
]

DESC_RE = re.compile(r"Task Description:\s*(.*?)\n\s*\nQuestion:\s*(.*?)(?:\n\s*\n|\Z)", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def template_of(user_text):
    for sig, slug in FAMILY_SIG:
        if sig in user_text:
            return slug
    return None


def answer_kind(slug):
    if slug in CONTINUOUS_TEMPLATES:
        return "continuous"
    if slug in CATEGORICAL_TEMPLATES:
        return "categorical"
    return "unknown"


def main():
    rows = []
    unmatched = []
    with open(VAL) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r["messages"]
            user = msgs[1]["content"]
            asst = msgs[2]["content"]
            img = r["images"][0]

            slug = template_of(user)
            if slug is None:
                unmatched.append(idx)

            m = DESC_RE.search(user)
            question = m.group(2).strip() if m else ""
            # also strip leading <image> if question capture failed
            if not question:
                question = user.replace("<image>", "").strip()

            ga = ANSWER_RE.search(asst)
            gold = ga.group(1).strip() if ga else ""

            rows.append({
                "idx": idx,
                "image_path": img,
                "question": question,
                "gold_answer": gold,
                "template_type": slug,
                "answer_kind": answer_kind(slug),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, engine="pyarrow", index=False)

    print(f"rows: {len(df)}   ->  {OUT}")
    print(f"unmatched template_type: {len(unmatched)} {unmatched[:10]}")
    print("\ntemplate_type value_counts:")
    print(df["template_type"].value_counts(dropna=False).to_string())
    print("\nanswer_kind value_counts:")
    print(df["answer_kind"].value_counts(dropna=False).to_string())
    print("\nper-template gold example:")
    for slug in ["lr", "distance", "fb", "yaw", "xy2d", "depth"]:
        ex = df[df.template_type == slug]["gold_answer"].iloc[0] if (df.template_type == slug).any() else "N/A"
        print(f"  {slug:9s} ({answer_kind(slug):11s}): {ex!r}")


if __name__ == "__main__":
    main()

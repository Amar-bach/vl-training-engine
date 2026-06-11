#!/usr/bin/env python
"""Synthesize a tiny fake eval_runs dir for self-testing score_and_aggregate.py.

Builds 2 fake arms x ~20 questions matching the generation schema, drawing idx from
the real val_meta so template_type/gold_answer are real. Some samples are made
correct (copy gold) and some wrong, with realistic <grounding>/<think>/<answer>.
"""
import json
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_selftest", "eval_runs")
os.makedirs(OUT, exist_ok=True)

meta = pd.read_parquet(os.path.join(HERE, "val_meta.parquet"))

# pick ~20 idx covering a mix of templates (both categorical + continuous)
rng = random.Random(0)
picks = []
for tt in ["lr", "distance", "fb", "yaw", "xy2d", "depth"]:
    sub = meta[meta.template_type == tt]
    picks += sub.sample(min(12, len(sub)), random_state=0).idx.tolist()
picks = sorted(set(picks))
sel = meta[meta.idx.isin(picks)].reset_index(drop=True)


def wrong_answer(tt, gold):
    if tt == "fb":
        return "No" if str(gold).strip().lower().startswith("y") else "Yes"
    if tt == "yaw":
        return "North"
    if tt == "xy2d":
        return "[10, 10]"
    if tt == "depth":
        return "Between 40 meters and 46 meters"
    return "The nonexistent purple object"


def make_gen(tt, ans, with_grounding=True, n_obj=2, truncated=False):
    g = ""
    if with_grounding:
        objs = " ".join(f"<obj{i}>thing{i} [{100+i*30}, {200+i*20}]</obj{i}>" for i in range(1, n_obj + 1))
        g = f"<grounding>{objs}</grounding>\n"
    think = ("Let me look at the scene. " * rng.randint(3, 12)).strip()
    if truncated:
        # truncated generation: no closing answer tag
        return f"{g}<think>{think}"
    return f"{g}<think>{think}</think>\n<answer>{ans}</answer>"


def build_arm(arm, correctness_bias):
    """correctness_bias in [0,1]: higher -> more correct greedy + samples."""
    rows = []
    r = random.Random(hash(arm) & 0xFFFF)
    for _, m in sel.iterrows():
        tt, gold = m.template_type, m.gold_answer
        # greedy
        greedy_correct = r.random() < correctness_bias
        if r.random() < 0.05:  # occasional truncated/unparseable greedy
            greedy_text = make_gen(tt, "", with_grounding=tt in ("lr", "distance"), truncated=True)
        else:
            ans = gold if greedy_correct else wrong_answer(tt, gold)
            greedy_text = make_gen(tt, ans, with_grounding=(r.random() < 0.85), n_obj=r.randint(1, 3))
        # 8 samples
        samples = []
        for _ in range(8):
            if r.random() < 0.03:
                samples.append(make_gen(tt, "", truncated=True))
                continue
            sc = r.random() < correctness_bias
            ans = gold if sc else wrong_answer(tt, gold)
            samples.append(make_gen(tt, ans, with_grounding=(r.random() < 0.8), n_obj=r.randint(1, 3)))
        rows.append(dict(
            idx=int(m.idx), arm=arm, image_path=m.image_path,
            prompt_tokens=int(r.randint(200, 600)),
            greedy_text=greedy_text, samples=samples,
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, f"{arm}.parquet"), index=False)
    with open(os.path.join(OUT, f"{arm}_meta.json"), "w") as f:
        json.dump({"arm": arm, "n": len(df), "synthetic": True}, f)
    return len(df)


# All 8 arms, with varied correctness bias so the figures (heatmaps, deltas,
# pass@k spread, reasoning panel) are meaningfully exercised on the self-test.
ARM_BIAS = {
    "baseline_instruct": 0.40,
    "baseline_thinking": 0.45,   # reference arm for Δ
    "geometry_math":     0.62,
    "chart_plot":        0.47,
    "science_diagram":   0.52,
    "doc_text":          0.43,   # slightly below baseline -> a regression arm
    "general_vqa":       0.50,
    "full":              0.58,
}
counts = {arm: build_arm(arm, bias) for arm, bias in ARM_BIAS.items()}
print(f"wrote {len(counts)} fake arms to {OUT}")
for arm, n in counts.items():
    print(f"  - {arm:18s} {n} questions")

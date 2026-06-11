#!/usr/bin/env python3
"""Build the Stage-C (DeepSeek enrichment) SFT dataset, matched 1:1 to the Stage-B1 winner set.

For each of the exact Stage-B1 points (same ids / order / images / questions as
sft_stageB/{train,val_1k}.jsonl), the assistant target becomes:

    <grounding>{Stage-B winner grounding}</grounding>
    <think>{cleaned Stage-C c2_think}</think>
    <answer>{cleaned Stage-C c2_answer}</answer>

whenever a USABLE Stage-C trace exists for that id (parse_ok_c2 AND c2_lands_on_gold,
and the cleaned text contains no structural-tag leak). Otherwise the original Stage-B
assistant content is kept UNCHANGED (fallback). The <grounding> block is always the
Stage-B winner's (Stage-C is text-only and has no points); only the reasoning/answer
prose is swapped to the deformity-cleaned Stage-C version.

Alignment contract (verified by main agent, bit-for-bit reproduction):
    train_ids.json[i]  <->  sft_stageB/train.jsonl  line i
    val_ids.json[i]    <->  sft_stageB/val_1k.jsonl line i

Pure stdlib + local stage_c_clean. Deterministic, answer-preserving.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage_c_clean as C  # noqa: E402

DATA_ROOT = "/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill"
SCRATCH = "/mnt/sandbox/amar.amarjyoti/research_code/vl-training-engine/subagent_research/stagec-trace-cleanup"

_GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.S | re.I)
_STRUCT_TAGS = ("<grounding>", "</grounding>", "<think>", "</think>", "<answer>", "</answer>")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--stageb-dir", default=os.path.join(DATA_ROOT, "sft_stageB"))
    p.add_argument("--phase-c", default=os.path.join(DATA_ROOT, "phase_c_1060281_deepseekv4_c2_v3_thinking.jsonl"))
    p.add_argument("--train-ids", default=os.path.join(SCRATCH, "train_ids.json"))
    p.add_argument("--val-ids", default=os.path.join(SCRATCH, "val_ids.json"))
    p.add_argument("--out-dir", default=os.path.join(DATA_ROOT, "sft_stageC"))
    return p.parse_args()


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_usable_phase_c(path):
    """id -> {c2_think, c2_answer} for usable traces (parse_ok_c2 AND c2_lands_on_gold)."""
    usable = {}
    n_total = n_usable = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if r.get("parse_ok_c2") and r.get("c2_lands_on_gold"):
                usable[r["id"]] = {"think": r.get("c2_think") or "", "answer": r.get("c2_answer") or ""}
                n_usable += 1
    print(f"  phase_c: {n_total} records, {n_usable} usable (parse_ok & lands_on_gold)")
    return usable


def has_struct_tag(*fields):
    for fld in fields:
        for t in _STRUCT_TAGS:
            if t in fld:
                return True
    return False


def build_split(stageb_recs, ids, usable, stats):
    assert len(stageb_recs) == len(ids), \
        f"alignment mismatch: {len(stageb_recs)} stageB recs vs {len(ids)} ids"
    out = []
    for rec, rid in zip(stageb_recs, ids):
        b_assistant = rec["messages"][-1]["content"]
        new_rec = {
            "messages": [rec["messages"][0], rec["messages"][1], {"role": "assistant", "content": b_assistant}],
            "images": rec["images"],
        }
        src = usable.get(rid)
        if src is not None:
            # c2_think is sometimes the raw model output: <reasoning>...</think><final summary><answer>...
            # Keep only the reasoning before the first </think> separator, and drop any stray
            # opening <think> tag, so the target's <think> block holds reasoning only (parallels Stage-B).
            raw_think = src["think"]
            if "</think>" in raw_think:
                raw_think = raw_think.split("</think>", 1)[0]
            raw_think = raw_think.replace("<think>", "")
            think = C.clean_think(raw_think).strip()
            # c2_answer is occasionally a malformed tail: "...</think>... <answer>REAL ANSWER</answer>".
            # The gold-match (c2_lands_on_gold) was computed on the post-<answer> span, so recover it:
            # take the text after the last <answer>, drop a closing </answer>.
            raw_answer = src["answer"]
            if "<answer>" in raw_answer:
                raw_answer = raw_answer.rsplit("<answer>", 1)[1]
            raw_answer = raw_answer.replace("</answer>", "")
            answer = C.clean_answer(raw_answer).strip()
            # Guard: cleaned Stage-C prose must not carry structural tags (would break tag balance),
            # and must be non-empty. Otherwise fall back to the Stage-B trace.
            if think and answer and not has_struct_tag(think, answer):
                m = _GROUNDING_RE.search(b_assistant)
                grounding_inner = m.group(1) if m else ""
                target = ("<grounding>{g}</grounding>\n"
                          "<think>{t}</think>\n"
                          "<answer>{a}</answer>").format(g=grounding_inner, t=think, a=answer)
                new_rec["messages"][-1]["content"] = target
                stats["substituted"] += 1
            else:
                stats["fallback_bad_clean"] += 1
        else:
            stats["fallback_no_usable_c"] += 1
        out.append(new_rec)
    return out


def main():
    args = parse_args()
    print("[1/4] Loading Stage-B SFT splits + id order")
    tr_b = load_jsonl(os.path.join(args.stageb_dir, "train.jsonl"))
    va_b = load_jsonl(os.path.join(args.stageb_dir, "val_1k.jsonl"))
    train_ids = json.load(open(args.train_ids))
    val_ids = json.load(open(args.val_ids))
    print(f"  stageB train={len(tr_b)} (ids={len(train_ids)})  val={len(va_b)} (ids={len(val_ids)})")

    print("[2/4] Loading usable Stage-C traces")
    usable = load_usable_phase_c(args.phase_c)

    print("[3/4] Building matched Stage-C splits")
    tr_stats = {"substituted": 0, "fallback_no_usable_c": 0, "fallback_bad_clean": 0}
    va_stats = {"substituted": 0, "fallback_no_usable_c": 0, "fallback_bad_clean": 0}
    tr_out = build_split(tr_b, train_ids, usable, tr_stats)
    va_out = build_split(va_b, val_ids, usable, va_stats)

    def report(name, recs, st):
        sub = st["substituted"]
        fb = st["fallback_no_usable_c"] + st["fallback_bad_clean"]
        print(f"  {name}: n={len(recs)}  stage_c-substituted={sub} ({100*sub/len(recs):.1f}%)  "
              f"fallback={fb} (no_usable_c={st['fallback_no_usable_c']}, bad_clean={st['fallback_bad_clean']})")
    report("train", tr_out, tr_stats)
    report("val",   va_out, va_stats)

    print("[4/4] Writing")
    os.makedirs(args.out_dir, exist_ok=True)
    tp = os.path.join(args.out_dir, "train.jsonl")
    vp = os.path.join(args.out_dir, "val_1k.jsonl")
    with open(tp, "w") as f:
        for r in tr_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(vp, "w") as f:
        for r in va_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  train -> {tp}")
    print(f"  val   -> {vp}")
    print("DONE.")


if __name__ == "__main__":
    main()

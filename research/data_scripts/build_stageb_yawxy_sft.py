"""Build the Stage-B yaw+xy2d consolidation SFT set.

Harvests 235B-teacher-CORRECT yaw and xy2d traces from the teacher N16 grounding pool,
reformats them into the exact sft_stageB swift schema (system/user/assistant with
<grounding><think><answer>), and concatenates with the existing 20k Stage-B SFT train set.

Correctness scoring uses research/eval/score_surds.score_one with the RIGHT coordinate
frame: xy2d gold (gt_answer) is SURDS PIXELS, teacher answer is 0-1000 normalised, so we
pass image_wh=(W,H) -> pred rescaled to px, compared to pixel gold. (See repo CLAUDE.md
"SURDS xy2d coordinate frames".) Targets are kept in the teacher's 0-1000 output format
(model output language stays normalised, per user decision 2026-06-18).

yaw is class-balanced (per-compass cap) so cardinals don't dominate the diagonals we are
trying to fix. One correct trace per prompt.

Out:
  <DATA>/vlm_cot_distill/sft_stageB_yawxy/harvest_teacher_yawxy.jsonl   (new traces only)
  <DATA>/vlm_cot_distill/sft_stageB_yawxy/train_stageb_yawxy.jsonl      (20k + harvest, shuffled)

Run:  python research/data_scripts/build_stageb_yawxy_sft.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from score_surds import score_one, get_image_wh  # noqa: E402

DATA = Path("/mnt/data4/shasta/amar.amarjyoti/research_data")
POOL = DATA / "vlm_cot_distill/cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl"
STAGEB = DATA / "vlm_cot_distill/sft_stageB/train.jsonl"
OUTDIR = DATA / "vlm_cot_distill/sft_stageB_yawxy"
OUTDIR.mkdir(parents=True, exist_ok=True)

SYSTEM = ("You are a spatial visual-reasoning assistant. First, ground every object "
          "referenced in the question by listing it inside a <grounding> block, each as "
          "<objN>label [x, y]</objN> with a normalized 0-1000 point. Then reason step by "
          "step inside a <think> block. Finally, give a concise, definitive response "
          "inside an <answer> block.")

COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
YAW_PER_CLASS_CAP = 700      # balance: cardinals capped so they don't swamp the diagonals
XY2D_PER_PROMPT = 1          # one correct trace per prompt


def to_compass(t):
    if not t:
        return None
    t = re.sub(r"[^a-z]", "", t.lower())
    for c in sorted(COMPASS, key=len, reverse=True):
        if t == c or t.endswith(c) or t.startswith(c):
            return c
    return None


def build_row(rec):
    """Teacher-pool record -> sft_stageB swift row (system/user/assistant + images)."""
    user = "<image>" + rec["prompt"] if not rec["prompt"].lstrip().startswith("<image>") else rec["prompt"]
    grounding = (rec.get("grounding") or "").strip()
    think = (rec.get("thinking") or "").strip()
    answer = (rec.get("answer") or "").strip()
    assistant = f"<grounding>{grounding}</grounding>\n<think>{think}</think>\n<answer>{answer}</answer>"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [rec["image_path"]],
    }


# Drop traces that would TRUNCATE under max_length=4096 (image ~1024 tok leaves ~3000
# text tok ≈ 12000 chars). Truncation would cut the trailing <answer> -> corrupt target.
MAX_TRACE_CHARS = 12000


def quality_ok(rec):
    if not (rec.get("finish_reason") == "stop"
            and (rec.get("grounding") or "").strip()
            and (rec.get("thinking") or "").strip()
            and (rec.get("answer") or "").strip()):
        return False
    approx = len(rec.get("prompt") or "") + len(rec.get("thinking") or "") + \
        len(rec.get("grounding") or "") + len(rec.get("answer") or "")
    return approx <= MAX_TRACE_CHARS


# ---------------------------------------------------------------------------
# Harvest: one correct trace per prompt, scored in the right frame
# ---------------------------------------------------------------------------
seen_prompt = set()
yaw_rows = defaultdict(list)   # compass -> [row, ...]
xy_rows = []
wh_cache = {}
stats = defaultdict(int)

with open(POOL) as f:
    for line in f:
        rec = json.loads(line)
        tt = (rec.get("template_type") or "").lower()
        if tt not in ("yaw", "xy2d"):
            continue
        pid = rec.get("id")
        if pid in seen_prompt:        # already harvested a correct trace for this prompt
            continue
        if not quality_ok(rec):
            continue
        gt, pred, img = rec.get("gt_answer"), rec.get("answer"), rec.get("image_path")

        if tt == "xy2d":
            if img not in wh_cache:
                wh_cache[img] = get_image_wh(img)
            res = score_one(pred, gt, "xy2d", image_wh=wh_cache[img])   # pixel-frame scoring
            if res["correct"]:
                xy_rows.append(build_row(rec))
                seen_prompt.add(pid)
                stats["xy2d_kept"] += 1
        else:  # yaw
            res = score_one(pred, gt, "yaw")
            if res["correct"]:
                cls = to_compass(gt)
                if cls and len(yaw_rows[cls]) < YAW_PER_CLASS_CAP:
                    yaw_rows[cls].append(build_row(rec))
                    seen_prompt.add(pid)
                    stats[f"yaw_{cls}"] += 1

harvest = list(xy_rows)
for cls in COMPASS:
    harvest.extend(yaw_rows[cls])

print("xy2d kept:", len(xy_rows))
print("yaw kept by class:", {c: len(yaw_rows[c]) for c in COMPASS}, "total yaw:", sum(len(v) for v in yaw_rows.values()))
print("harvest total:", len(harvest))

# ---------------------------------------------------------------------------
# Write harvest + combined (20k Stage-B + harvest), deterministically shuffled
# ---------------------------------------------------------------------------
harvest_path = OUTDIR / "harvest_teacher_yawxy.jsonl"
with open(harvest_path, "w") as f:
    for r in harvest:
        f.write(json.dumps(r) + "\n")

base = [json.loads(l) for l in open(STAGEB)]
combined = base + harvest
# deterministic shuffle (index-based, no RNG-state dependence)
combined.sort(key=lambda r: (hash(json.dumps(r["messages"][1]["content"])) & 0xffffffff))

combined_path = OUTDIR / "train_stageb_yawxy.jsonl"
with open(combined_path, "w") as f:
    for r in combined:
        f.write(json.dumps(r) + "\n")

print(f"\nwrote {harvest_path}  ({len(harvest)} rows)")
print(f"wrote {combined_path}  ({len(combined)} rows = {len(base)} base + {len(harvest)} harvest)")

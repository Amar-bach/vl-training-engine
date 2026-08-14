"""Consolidate yaw/direction-subtask data for notebooks/yaw_direction_diagnosis.ipynb.

Pulls from the L2-direct GRPO run rollouts (completions.jsonl) and joins each yaw
prompt-group back to its source image via the shared task text in the L2 curriculum.

Produces:
  * 8x8 WORLD-frame confusion matrix (gold compass vs predicted compass)
  * camera-RELATIVE confusion (object heading minus camera heading) -> tests the
    "model defaults to facing-the-viewer" hypothesis independent of camera frame
  * per-class accuracy + pred marginals
  * 10 random worked examples: image path, camera-facing, object, gold, and up to
    3 real rollouts (CoT + predicted answer + correctness)

Output: notebooks/data/yaw_direction_data.json
Run:    python research/notebook_builders/_prep_yaw_direction_data.py
"""
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/data4/shasta/amar.amarjyoti/research_data")
RUN = DATA_ROOT / "rl_runs/bakeoff/A_grpo_v3/L2_direct/v1-20260617-011220"
L2 = DATA_ROOT / "processed/surds/curriculum/L2.jsonl"
OUT = REPO / "notebooks" / "data" / "yaw_direction_data.json"

COMPASS = ["north", "northeast", "east", "southeast",
           "south", "southwest", "west", "northwest"]
DEG = {c: i * 45 for i, c in enumerate(COMPASS)}      # N=0, NE=45, ... NW=315
ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]   # display order
ABBR = {c: ORDER[i] for i, c in enumerate(COMPASS)}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
CAM_RE = re.compile(r"camera in the image is facing (\w+)", re.I)
OBJ_RE = re.compile(r"direction is the (.+?) facing", re.I)


def parse_answer(text):
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


def to_compass(text):
    """Normalise free text to one of the 8 compass words, else None."""
    if not text:
        return None
    t = re.sub(r"[^a-z]", "", text.lower())
    for c in COMPASS:                      # longest-first so 'northeast' beats 'north'
        pass
    for c in sorted(COMPASS, key=len, reverse=True):
        if t == c:
            return c
    # answer may be a short phrase ending in the word
    for c in sorted(COMPASS, key=len, reverse=True):
        if t.endswith(c) or t.startswith(c):
            return c
    return None


def task_key(text):
    i = text.find("Task Description")
    j = text.find("Reason carefully")
    return re.sub(r"\s+", " ", text[i:j]).strip() if i >= 0 else None


def think_snippet(text, n=600):
    t = re.sub(r"</?think>", "", text or "", flags=re.I)
    t = re.sub(r"<answer>.*", "", t, flags=re.S | re.I).strip()
    return (t[:n] + " …") if len(t) > n else t


def groups(sols):
    i, n = 0, len(sols)
    while i < n:
        j = i
        while j < n and sols[j] == sols[i]:
            j += 1
        yield i, j
        i = j


# --------------------------------------------------------------------------
# 1. L2 yaw map: task_key -> {image, camera, object}
# --------------------------------------------------------------------------
L2map = {}
for line in open(L2):
    r = json.loads(line)
    if r.get("template_type") != "yaw":
        continue
    content = r["messages"][0]["content"]
    k = task_key(content)
    if not k:
        continue
    cam = CAM_RE.search(content)
    obj = OBJ_RE.search(content)
    L2map[k] = {
        "image": r["images"][0],
        "camera": cam.group(1).lower() if cam else None,
        "object": obj.group(1).strip() if obj else None,
    }
print(f"L2 yaw keys: {len(L2map)}")

# --------------------------------------------------------------------------
# 2. Scan rollouts: confusion matrices + collect per-group records
# --------------------------------------------------------------------------
# world-frame confusion: counts[gold_idx][pred_idx]; pred None -> unparsed bucket
world = [[0] * 8 for _ in range(8)]
unparsed = [0] * 8                       # per gold class, count of unparseable preds
rel_pred = [0] * 8                       # camera-relative predicted-heading histogram
rel_gold = [0] * 8                       # camera-relative gold histogram
per_class_correct = [0] * 8
per_class_total = [0] * 8

records = {}                             # task_key -> example record (first time seen)
with open(RUN / "completions.jsonl") as f:
    for line in f:
        row = json.loads(line)
        sols, comps, accs, prompts = (row["solution"], row["completion"],
                                      row["SurdsAccuracy"], row["prompt"])
        for s, e in groups(sols):
            gold = to_compass(str(sols[s]))
            if gold is None:
                continue
            gi = COMPASS.index(gold)
            k = task_key(prompts[s])
            meta = L2map.get(k)
            cam = meta["camera"] if meta else None
            for c, a in zip(comps[s:e], accs[s:e]):
                pred = to_compass(parse_answer(c))
                per_class_total[gi] += 1
                if a == 1:
                    per_class_correct[gi] += 1
                if pred is None:
                    unparsed[gi] += 1
                    continue
                pj = COMPASS.index(pred)
                world[gi][pj] += 1
                if cam in DEG:
                    rel_pred[((DEG[pred] - DEG[cam]) // 45) % 8] += 1
                    rel_gold[((DEG[gold] - DEG[cam]) // 45) % 8] += 1
            # stash a worked-example record (first occurrence of this prompt)
            if meta and k not in records:
                rolls = []
                # prefer 1 correct + 2 wrong for contrast
                idx = list(range(s, e))
                cor = [i for i in idx if accs[i] == 1]
                wro = [i for i in idx if accs[i] == 0]
                pick = (cor[:1] + wro[:2]) if cor else wro[:3]
                for i in sorted(pick):
                    rolls.append({
                        "pred": parse_answer(comps[i]),
                        "correct": bool(accs[i] == 1),
                        "think": think_snippet(comps[i]),
                    })
                records[k] = {
                    "image": meta["image"], "camera": cam,
                    "object": meta["object"], "gold": str(sols[s]),
                    "rollouts": rolls,
                }

n_groups = len(records)
print(f"yaw prompt-groups: {n_groups}")

# --------------------------------------------------------------------------
# 3. Pick 10 random worked examples (deterministic)
# --------------------------------------------------------------------------
rng = random.Random(20260618)
keys = sorted(records.keys())
chosen = rng.sample(keys, min(10, len(keys)))
examples = [records[k] for k in chosen]

# camera-facing distribution (sanity: is the frame really varying?)
cam_dist = {}
for v in records.values():
    cam_dist[v["camera"]] = cam_dist.get(v["camera"], 0) + 1

# --------------------------------------------------------------------------
# 4. Write
# --------------------------------------------------------------------------
payload = {
    "meta": {
        "run": "A_grpo_v3 L2-direct (job 1064116)",
        "source": str(RUN / "completions.jsonl"),
        "order": ORDER,
        "compass": COMPASS,
        "rel_labels": ["+0° same as camera (faces away/into scene)", "+45°", "+90° (camera-right)",
                       "+135°", "+180° toward camera (faces viewer)", "+225°",
                       "+270° (camera-left)", "+315°"],
        "n_prompt_groups": n_groups,
        "n_rollout_samples": sum(per_class_total),
        "camera_facing_distribution": cam_dist,
    },
    "world_confusion": world,
    "world_unparsed_per_gold": unparsed,
    "per_class_acc": [
        {"cls": ORDER[i], "correct": per_class_correct[i], "total": per_class_total[i],
         "acc": round(100 * per_class_correct[i] / per_class_total[i], 1) if per_class_total[i] else None}
        for i in range(8)],
    "rel_pred_hist": rel_pred,
    "rel_gold_hist": rel_gold,
    "examples": examples,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
print(f"camera-facing dist: {cam_dist}")
print(f"examples: {len(examples)}  (golds: {[e['gold'] for e in examples]})")

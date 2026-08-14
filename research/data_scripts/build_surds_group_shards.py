"""
build_surds_group_shards.py
---------------------------
Split the existing cumulative SURDS curriculum shards (L12 = Stage-B band, L123 =
Stage-C band) by TEMPLATE GROUP for the per-template GRPO loops (user spec 2026-06-20):

  group 'categorical' = {lr, distance, fb}     -> phase2 (B) + phase3 (C), binary reward
  group 'yaw'         = {yaw}                   -> phase3 (C) only,          binary reward
  group 'continuous'  = {xy2d, depth}           -> phase2 (B) + phase3 (C), dense+binary reward

Reads the committed L12.jsonl / L123.jsonl (each row carries template_type) and writes
filtered cumulative shards under curriculum/by_group/. Pure filter — no re-derivation of
acceptance bands, so it stays consistent with curriculum_manifest.json.
"""
import json
from collections import Counter
from pathlib import Path

CUR = Path("/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/curriculum")
OUT = CUR / "by_group"
OUT.mkdir(parents=True, exist_ok=True)

# Split by REWARD TYPE (user spec 2026-06-20): the binary-reward subtasks train
# together (lr/distance/fb/yaw), the dense+binary subtasks train together (xy2d/depth).
GROUPS = {
    "binary": {"lr", "distance", "fb", "yaw"},
    "dense": {"xy2d", "depth"},
}
# which cumulative band each group trains on: B=L12 (Stage B / phase2), C=L123 (Stage C / phase3)
BANDS = {"B": "L12.jsonl", "C": "L123.jsonl"}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    for band, fname in BANDS.items():
        src = load(CUR / fname)
        for gname, tts in GROUPS.items():
            recs = [r for r in src if (r.get("template_type") or "").lower() in tts]
            out = OUT / f"{gname}_{band}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            by_tt = Counter((r.get("template_type") or "").lower() for r in recs)
            print(f"  {out.name:24s} {len(recs):>6,} rows  {dict(by_tt)}")
    print(f"\nWrote group shards -> {OUT}")


if __name__ == "__main__":
    main()

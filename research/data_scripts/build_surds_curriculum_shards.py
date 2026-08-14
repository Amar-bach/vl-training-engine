"""
build_surds_curriculum_shards.py
---------------------------------
Turns the SURDS acceptance parquet into a 3-level difficulty curriculum of ms-swift
GRPO dataset shards (user spec 2026-06-15, revised after xy2d-scoring fix).

Levels (on `acceptance` = n_pass/16), chosen to maximise GRPO within-group variance:
  L1_learnable : 0.35 <= acc <= 0.75   (highest-variance — the bake-off substrate)
  L2_hard      : 0.10 <= acc <  0.35
  L3_frontier  : 0.00 <  acc <  0.10   (sparse; pair with dynamic sampling / dense reward)
  dropped_solved : acc > 0.75          (near-solved, weak GRPO signal)
  dropped_zero   : acc == 0.0          (zero GRPO advantage under binary reward)

Cumulative curriculum mixes (for L1->L2->L3 annealed staging):
  L1.jsonl, L12.jsonl (L1+L2), L123.jsonl (L1+L2+L3)

Output JSONL schema per row:
  messages      : [{"role":"user","content":"<image>"+question}]
  images        : [image_path]
  solution      : gold answer string
  template_type : passthrough for reward plugin
  image_path    : passthrough for reward plugin (xy2d get_image_wh)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def build_user_content(row, train_jsonl_map):
    q = row.get("question", None)
    if q and isinstance(q, str) and q.strip():
        return q.strip()
    pid = row.get("prompt_id", "")
    if pid in train_jsonl_map:
        p = train_jsonl_map[pid].get("prompt", "")
        if p and p.strip():
            return p.strip()
    return ""


def row_to_jsonl(row, train_jsonl_map):
    user_text = build_user_content(row, train_jsonl_map)
    image_path = str(row["image_path"])
    return {
        "messages": [{"role": "user", "content": "<image>" + user_text}],
        "images": [image_path],
        "solution": str(row["gold"]),
        "template_type": str(row["template_type"]),
        "image_path": image_path,
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>6,} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Build SURDS L1/L2/L3 GRPO curriculum shards.")
    ap.add_argument("--acceptance_parquet",
                    default="/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/acceptance_train_teacher235b.parquet")
    ap.add_argument("--train_jsonl",
                    default="/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/train_qa.jsonl")
    ap.add_argument("--out_dir",
                    default="/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/curriculum")
    # Level thresholds
    ap.add_argument("--l1_lo", type=float, default=0.35)
    ap.add_argument("--l1_hi", type=float, default=0.75)
    ap.add_argument("--l2_lo", type=float, default=0.10)   # L2: [l2_lo, l1_lo)
    ap.add_argument("--l3_lo", type=float, default=0.0)    # L3: (l3_lo, l2_lo)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading acceptance parquet: {args.acceptance_parquet}")
    df = pd.read_parquet(args.acceptance_parquet)
    df["acceptance"] = df["acceptance"].astype(float)
    print(f"  Rows: {len(df):,}   Columns: {list(df.columns)}")

    train_jsonl_map = {}
    tp = Path(args.train_jsonl)
    if tp.exists():
        for line in open(tp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id", rec.get("prompt_id", ""))
            if sid:
                train_jsonl_map[sid] = rec
        print(f"  Loaded {len(train_jsonl_map):,} train_qa records (prompt fallback).")

    acc = df["acceptance"]
    mask_solved = acc > args.l1_hi
    mask_L1 = (acc >= args.l1_lo) & (acc <= args.l1_hi)
    mask_L2 = (acc >= args.l2_lo) & (acc < args.l1_lo)
    mask_L3 = (acc > args.l3_lo) & (acc < args.l2_lo)
    mask_zero = acc == 0.0

    df_L1, df_L2, df_L3 = df[mask_L1], df[mask_L2], df[mask_L3]
    df_solved, df_zero = df[mask_solved], df[mask_zero]

    print("\n" + "=" * 64)
    print("CURRICULUM LEVEL SUMMARY")
    print("=" * 64)
    print(f"{'Level':<18}  {'Count':>7}  acceptance range")
    print("-" * 64)
    print(f"{'L1_learnable':<18}  {len(df_L1):>7}  [{args.l1_lo:.2f}, {args.l1_hi:.2f}]")
    print(f"{'L2_hard':<18}  {len(df_L2):>7}  [{args.l2_lo:.2f}, {args.l1_lo:.2f})")
    print(f"{'L3_frontier':<18}  {len(df_L3):>7}  ({args.l3_lo:.2f}, {args.l2_lo:.2f})")
    print(f"{'dropped_solved':<18}  {len(df_solved):>7}  > {args.l1_hi:.2f}")
    print(f"{'dropped_zero':<18}  {len(df_zero):>7}  == 0.0")
    print(f"{'TOTAL':<18}  {len(df):>7}")

    print("\n" + "=" * 64)
    print("LEVEL x TEMPLATE_TYPE CROSSTAB")
    print("=" * 64)
    tagged = []
    for name, bdf in [("L1", df_L1), ("L2", df_L2), ("L3", df_L3),
                      ("solved", df_solved), ("zero", df_zero)]:
        if len(bdf):
            t = bdf[["template_type"]].copy()
            t["level"] = name
            tagged.append(t)
    if tagged:
        comb = pd.concat(tagged, ignore_index=True)
        ct = pd.crosstab(comb["level"], comb["template_type"], margins=True)
        order = [v for v in ["L1", "L2", "L3", "solved", "zero", "All"] if v in ct.index]
        print(ct.reindex(order).to_string())
    print("=" * 64)

    def to_recs(bdf):
        return [row_to_jsonl(r, train_jsonl_map) for _, r in bdf.iterrows()]

    rL1, rL2, rL3 = to_recs(df_L1), to_recs(df_L2), to_recs(df_L3)
    rSolved, rZero = to_recs(df_solved), to_recs(df_zero)

    print("\nWriting shard files ...")
    write_jsonl(rL1, out_dir / "L1.jsonl")
    write_jsonl(rL2, out_dir / "L2.jsonl")
    write_jsonl(rL3, out_dir / "L3.jsonl")
    write_jsonl(rSolved, out_dir / "dropped_solved.jsonl")
    write_jsonl(rZero, out_dir / "dropped_zero.jsonl")
    # cumulative curriculum mixes
    write_jsonl(rL1 + rL2, out_dir / "L12.jsonl")
    write_jsonl(rL1 + rL2 + rL3, out_dir / "L123.jsonl")

    manifest = {
        "thresholds": {"l1_lo": args.l1_lo, "l1_hi": args.l1_hi,
                       "l2_lo": args.l2_lo, "l3_lo": args.l3_lo},
        "level_definitions": {
            "L1_learnable": f"[{args.l1_lo}, {args.l1_hi}]",
            "L2_hard": f"[{args.l2_lo}, {args.l1_lo})",
            "L3_frontier": f"({args.l3_lo}, {args.l2_lo})",
            "dropped_solved": f"> {args.l1_hi}",
            "dropped_zero": "== 0.0",
        },
        "counts": {"L1": len(rL1), "L2": len(rL2), "L3": len(rL3),
                   "dropped_solved": len(rSolved), "dropped_zero": len(rZero),
                   "L12": len(rL1) + len(rL2), "L123": len(rL1) + len(rL2) + len(rL3),
                   "total_parquet": len(df)},
        "files": {k: str(out_dir / f"{k}.jsonl")
                  for k in ["L1", "L2", "L3", "L12", "L123", "dropped_solved", "dropped_zero"]},
        "source": str(args.acceptance_parquet),
        "note": "acceptance = 235B-teacher pass-rate over 16 rejection samples; xy2d scored "
                "with pred-only 0-1000->px un-normalization (gold is pixel bbox-centre).",
    }
    with open(out_dir / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest -> {out_dir / 'curriculum_manifest.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

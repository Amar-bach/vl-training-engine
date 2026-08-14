# Repo guidance — vl-training-engine

## wandb logging (set by user 2026-06-16)

Runs from this repo log to the user's **PERSONAL wandb account `samarjyo`**, NOT the company
`blueriver` team that the machine's `~/.netrc` (login `amar-brt`) points at.

- The personal API key lives in **`./.env`** as `WANDB_API_KEY` (gitignored — never commit it,
  never echo it into a log).
- In job scripts (which run `set -ex`), load the key with **xtrace suppressed** so the secret never
  lands in the SLURM log, then restore the prior xtrace state. Pattern used in
  `slurm_scripts/grpo_bakeoff_*.sh`:
  ```bash
  { __xt=$(set +o | grep xtrace); set +x; } 2>/dev/null
  export WANDB_API_KEY="$(grep -E '^[[:space:]]*WANDB_API_KEY' "$REPO/.env" | head -1 \
      | sed -E 's/^[^=]*=[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/')"
  export WANDB_ENTITY=samarjyo
  eval "$__xt" 2>/dev/null
  ```
- Always set `WANDB_ENTITY=samarjyo` explicitly and pass `--run_name "$WANDB_NAME"` (ms-swift
  otherwise sets `run_name=output_dir`, overriding `WANDB_NAME`).
- **Confirm the destination with the user before launching anything that logs to wandb.** See the
  global `~/.claude/CLAUDE.md` wandb policy for the full rule.

## SURDS xy2d coordinate frames — READ BEFORE SCORING ANY xy2d (set by user 2026-06-18)

This bug recurs every session. The model output and the gold label live in **different coordinate
frames**, and the frame of the gold **varies by dataset**. Get this wrong and xy2d metrics are
garbage (silently — everything reads as wrong, or everything as right).

**The two frames**
- **Model prediction (Qwen3-VL student AND the 235B teacher): always `0–1000` NORMALISED relative
  coords.** Qwen emits points on a 0–1000 grid regardless of image size. This is non-negotiable Qwen
  behaviour (see global memory `reference-qwen-relative-coordinates`).
- **SURDS images are the native nuScenes frame `1600×900`** (verified: every `.webp` is exactly
  1600×900, no downscaling). To convert a prediction to pixels: `x_px = x*1600/1000`,
  `y_px = y*900/1000`.

**Gold (`solution` / `<answer>`) frame — DIFFERS BY DATASET (the actual trap):**

| dataset | xy2d gold frame | how to tell |
|---|---|---|
| GRPO curriculum (`processed/surds/curriculum/L{1,2,3}.jsonl`) | **absolute PIXELS** (1600×900) | ~28% of gold x values exceed 1000 (max ~1598) |
| source QA (`vlm_cot_distill/_train_qa_for_cot.jsonl`), teacher pool (`cot_*_grounding.jsonl` `gt_answer`) | **absolute PIXELS** | same |
| SFT data (`sft_stageB/train.jsonl`), val_1k, heldout eval (`<answer>` = teacher distilled trace) | **`0–1000` NORMALISED** | NO gold x exceeds 1000 |

**Scoring rule (`research/eval/score_surds.score_one`) — as of 2026-06-19 it has an explicit
`gold_space=('pixels'|'norm')` param (default `'pixels'`).** It reconciles BOTH pred and gold into
the SURDS **pixel** frame before the 50 px L2 compare: pred is always rescaled `×(W,H)/1000`; gold is
rescaled too **only** when `gold_space='norm'`. Pass it per the gold's dataset frame:
- **Pixel-gold data (curriculum / source-QA / teacher pool): `score_one(..., image_wh=(W,H))`**
  (default `gold_space='pixels'`). Correct. The GRPO reward plugin does this (passes `image_path`) —
  687/688 agreement with the logged reward, so **GRPO RL training scored xy2d correctly**.
- **Normalised-gold data (SFT / val_1k / heldout / val_meta): `score_one(..., image_wh=(W,H),
  gold_space='norm')`** so the gold is rescaled into px too. (Equivalently `image_wh=None` compares
  both in 0–1000 with `NORM_XY_TOL≈38.5`, ~1–3 pp stricter than the canonical 50 px px-frame tol.)
- **Safety rails now built in:** (a) any gold coord > 1000 is auto-treated as pixels even under
  `gold_space='norm'`; (b) pixel gold with `image_wh=None` returns `detail.space=='frame_error'`
  (correct=False) instead of silently comparing 0–1000 vs ~1600. Previously that silent path made a
  teacher audit read 0.8% (true **76%**), and passing `image_wh` *without* `gold_space='norm'` on
  normalised gold scored everything 0.000 — a LATENT footgun in `score_and_aggregate.py` (the committed
  Jun-13 ablation figures were computed in norm space and are correct; the bug would only have bitten on
  the next re-run, now fixed).

**When in doubt, detect the frame:** if any gold coord > 1000 it is pixels; if a whole dataset's
xy2d gold maxes ≤1000 it is normalised. Always reconcile pred-frame and gold-frame into ONE space
before the L2 compare. Tolerance: 50 px on the 1600×900 frame, or `NORM_XY_TOL≈38.5` in 0–1000 space.

# heldout_metrics_primed/ — CURRENT numbers (convention primers ON)

Heldout SURDS val (1998 ex, 333/template), scored with primed prompts, i.e. what
`gen_val_ablation.py` produces by default as of **2026-08-12**
(`DEFAULT_PRIMERS = ("yaw", "fb")`).

**This is the directory to read for current performance.** The older
`research/eval/heldout_metrics/` holds the pre-primer ablation arms and its `yaw`, `fb`, and
`ALL` rows are stale — see the README there.

## Arms present

| arm | ckpt | primers | n_sample |
|---|---|---|---|
| `rl_init_cp896_primed`   | `sft_runs/stageb_yawxy_1064290/v0-20260619-025339/checkpoint-896` | `yaw,fb` | 16 |
| `rl_init_cp896_unprimed` | same checkpoint                                                    | none     | 16 |

Same model, so the pair isolates the prompt effect exactly.

## Results — pass@1

| template | unprimed | **primed** | Δ |
|---|---|---|---|
| **ALL** | 0.652 | **0.706** | **+0.054** |
| fb | 0.535 | **0.811** | **+0.276** |
| yaw | 0.483 | **0.532** | +0.048 |
| depth | 0.529 | 0.529 | 0.000 |
| distance | 0.811 | 0.811 | 0.000 |
| lr | 0.820 | 0.820 | 0.000 |
| xy2d | 0.733 | 0.733 | 0.000 |

The four untouched templates being **exactly** 0.000 is the validity check on how the primed
arm was built (below). Note `fb` (0.811) now equals `distance` (0.811) — the convention was
the entire gap; perception was never the problem.

## How `rl_init_cp896_primed` was built (no new GPU time)

The primers only alter yaw/fb *prompts*, so the arm was **spliced** from existing generations:

- non-yaw/fb rows ← `eval_runs/dapo_vs_grpo_gen/rl_init_cp896.parquet` (unprimed, but those
  templates are unaffected by definition)
- yaw rows ← `eval_runs/yaw_coord_primer_ab/yaw_primer.parquet` (job 1067475)
- fb rows  ← `eval_runs/fb_convention_primer_ab/fb_primer.parquet` (job 1071082)

All three are the same checkpoint at 16 samples. Row counts and idx sets were asserted to match
the meta exactly (333 yaw, 333 fb, 1998 total, unique idx). Source parquets:
`$DATA_ROOT/eval_runs/heldout_primed/`.

## Not yet updated

The 14 SFT-ablation arms (`full`, `stage_c_mulberry_full`, the mulberry-domain arms, both
teachers) still need a primed regeneration — that requires new GPU jobs. Until then, keep
primed and unprimed numbers in separate tables.

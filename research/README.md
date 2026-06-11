# research/ — VLM spatial-reasoning research code

User research code for the SURDS VLM SFT/RLVR project, kept **out** of the vendored
ms-swift framework tree and out of `notebooks/` (which now holds only `.ipynb` files
and their output dirs).

```
research/
  cot_lib/            Importable shared modules (no side effects on import)
    cot_metrics.py      CoT intrinsic-quality feature defs (trace_features, split_trace, ...)
  data_scripts/       Runnable data-prep / metrics CLI scripts
    generate_mulberry_subsets.py   build per-domain Mulberry subsets
    format_mulberry_swift.py       Mulberry/VisionR1 -> ms-swift schema
    run_cot_metrics.py             dump per-record CoT-metric parquet (-> notebooks/visionr1_out/)
  eval/               SURDS x Mulberry ablation evaluation on val_1k
    gen_val_ablation.py   vLLM generation: pass@1 greedy + pass@8 sampled per checkpoint
    eval_ablation.sh      SLURM sbatch wrapper (one arm per submission)
    score_surds.py        answer scoring (categorical exact-match; xy2d/depth abs-diff+tol)
    build_val_meta.py     recover template_type + gold for val_1k
    val_meta.parquet      cached val_1k metadata (1001 rows)
  notebook_builders/  Scripts that GENERATE the .ipynb in ../notebooks/
    _build_codebase_guide.py
    _build_visionr1_mulberry_cot.py
```

## Conventions
- `cot_lib/` is on `sys.path` via a small bootstrap at the top of each script/notebook
  (`sys.path.insert(0, .../research/cot_lib)`); it is not a pip package.
- Notebooks live in `../notebooks/` and run with that as the cwd; their output dirs
  (`audit_out/`, `visionr1_out/`, ...) are siblings there.
- Generated artifacts > 100 MB go under `/mnt/data4/shasta/amar.amarjyoti/research_data/`,
  never committed here.
- conda env for everything: `rlvr_conda`.

## Ablation eval — submit (run from anywhere; paths are absolute)
See the header of `eval/eval_ablation.sh` for the per-arm `CKPT=/ARM=/BASE= sbatch` lines.

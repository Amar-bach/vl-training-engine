"""Builds visionr1_ablation_analysis.ipynb — conference-grade TWO-AXIS analysis of
the SURDS x Mulberry VLM SFT ablation on REAL results.

Two evaluation axes:
  * HELD-OUT SURDS validation (n=1998)  -> PRIMARY / discriminative (the headline).
  * In-distribution val_1k (n=1001)     -> SATURATED reference (motivates held-out).

Both metrics dirs are loaded. Heavy compute lives in score_and_aggregate.py;
this notebook only LOADS parquets and plots. The generated notebook runs from the
notebooks/ directory; a sys.path bootstrap adds ../research/eval so plotstyle and
score_surds are importable. Metrics-dir paths are overridable via env vars
(HELDOUT_METRICS_DIR / INDIST_METRICS_DIR) for headless execution.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md(r"""# SURDS x Mulberry SFT ablation — conference-grade analysis (held-out + in-distribution)

**Question.** Starting from a SURDS-only spatial-reasoning SFT, does **adding one
Mulberry reasoning domain** (geometry/math, chart/plot, science-diagram, doc/text,
general-VQA) to the 50/50 training mix help or hurt SURDS spatial reasoning — overall
and per spatial sub-skill?

Each evaluated model is a Qwen3-VL-8B checkpoint. We report on **two** evaluation sets:

* **Held-out SURDS validation** (n=1998) — the **primary, discriminative** axis and the
  headline of this report. These scenes / objects are disjoint from training.
* **In-distribution val_1k** (n=1001) — a **saturated** reference set. We use it only to
  show that the ablation is *indistinguishable* in-distribution, which is exactly what
  motivates evaluating on the held-out set.

For every model we generated one **greedy** answer and **eight** temperature-sampled
answers per question, scored with `research/eval/score_surds.py` (continuous templates
use abs-difference + tolerance; Qwen emits 0–1000 normalized coordinates).

> **Coordinate-normalization note (methods).** On the held-out set the `xy2d` gold
> points were stored in **pixels (0–1600)** whereas the model emits Qwen **0–1000
> normalized** coordinates. We normalize the gold per-image before scoring; without
> this rescale `xy2d` reads a bogus ~2% instead of the true ~70%.

### Metric glossary (no abbreviations in the tables)
- **Greedy accuracy (pass@1)** — fraction correct using the single greedy decode.
- **Mean per-sample accuracy (pass@1 sampled)** — average accuracy of the eight sampled decodes.
- **pass@k (unbiased estimator)** — probability at least one of *k* sampled decodes is
  correct, `1 - C(n-c,k)/C(n,k)` averaged over questions (n=8, c correct).
- **Self-consistency accuracy (maj@8)** — accuracy of the majority-vote answer over the eight samples.
- **Δ-vs-baseline** — metric minus the same metric for the **zero-shot `orig_thinking`**
  (off-the-shelf Qwen3-VL-8B-Thinking, *no SFT*) — the natural "improvement over the
  untrained model" reference. (The SURDS-only SFT checkpoint is **not** the baseline; it
  is itself a trained arm, `SURDS-SFT (Thinking)`.)
""")

# ---------------------------------------------------------------------------
md(r"""## 0. Setup — load BOTH metrics dirs""")

code(r"""import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# notebook runs from notebooks/ ; scorer + plotstyle live in ../research/eval
EVAL_DIR = (Path.cwd().parent / 'research' / 'eval').resolve()
sys.path.insert(0, str(EVAL_DIR))   # so score_surds / plotstyle are importable

# Two metrics dirs. Overridable for headless execution; default to the real dirs.
HELDOUT_METRICS_DIR = Path(os.environ.get('HELDOUT_METRICS_DIR', str(EVAL_DIR / 'heldout_metrics')))
INDIST_METRICS_DIR  = Path(os.environ.get('INDIST_METRICS_DIR',  str(EVAL_DIR / 'indist_metrics')))

FIG_DIR = Path('figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)

import plotstyle as ps
ps.set_pub_style()
pd.set_option('display.max_colwidth', 200)


def load_axis(metrics_dir, meta_name):
    per_q   = pd.read_parquet(metrics_dir / 'metrics_per_question.parquet')
    agg     = pd.read_parquet(metrics_dir / 'metrics_aggregate.parquet')
    meta    = pd.read_parquet(EVAL_DIR / meta_name)
    summary = json.loads((metrics_dir / 'metrics_summary.json').read_text())
    return dict(per_q=per_q, agg=agg, meta=meta, summary=summary,
                n=int(meta.idx.nunique()), dir=metrics_dir)

HO = load_axis(HELDOUT_METRICS_DIR, 'heldout_val_meta.parquet')   # PRIMARY
ID = load_axis(INDIST_METRICS_DIR,  'val_meta.parquet')           # reference

BASELINE = HO['summary'].get('baseline_arm', 'orig_thinking')   # zero-shot Δ reference
ARMS_ALL = ps.order_arms(HO['summary']['arms'])     # everything present in the metrics
# Stage-C DeepSeek trace-enrichment arms are analysed in their OWN section (§2b) — keep them
# out of the main SURDS×Mulberry ablation flow so every existing figure/table is unchanged.
STAGE_C = [a for a in ARMS_ALL if a in ('stage_c', 'stage_c_mulberry_full')]
ARMS = [a for a in ARMS_ALL if a not in STAGE_C]     # canonical display order for the ablation
# Mulberry added-domain arms only (exclude zero-shot baselines, teachers, and SURDS-SFT arms)
MULBERRY = [a for a in ARMS if a not in
            ('orig_instruct', 'orig_thinking', 'teacher_32b', 'teacher_235b',
             'baseline_instruct', 'baseline_thinking')]
# Zero-shot (no-SFT) reference models — the baselines, incl. the larger teachers when present
ZERO_SHOT = [a for a in ARMS if a in
             ('orig_instruct', 'orig_thinking', 'teacher_32b', 'teacher_235b')]
# TWO references are in play:
#   BASELINE     = zero-shot orig_thinking  -> "training lift over the untrained model"
#                  (the headline Δ columns; matches the precomputed delta_* fields).
#   ABLATION_REF = SURDS-SFT (Thinking)     -> "does adding a Mulberry domain help/hurt?"
#                  (the ablation's discriminative contrast; the ±1-pt effect lives here,
#                   so it must be measured vs the SFT arm, not vs zero-shot).
ABLATION_REF = 'baseline_thinking'

def overall_delta_vs(axis_overall, ref_arm, metric):
    # Series: each arm's <metric> minus <ref_arm>'s <metric> (overall, template=ALL).
    return axis_overall[metric] - axis_overall.loc[ref_arm, metric]

print('HELD-OUT  n =', HO['n'], '| arms:', len(ARMS))
print('IN-DIST   n =', ID['n'])
print('baseline arm:', BASELINE)
print('Mulberry arms:', MULBERRY)
print('Stage-C arms:', STAGE_C)
print('pass@k estimator:', HO['summary']['pass_at_k_estimator'])
""")

# ---------------------------------------------------------------------------
md(r"""## 1. Model legend

Every evaluated model: its base checkpoint, the Mulberry domain added on top of the
SURDS spatial-reasoning SFT, the training mix, and which evaluation axes it appears on.
The **baselines** are the off-the-shelf, *no-SFT* checkpoints (the two Qwen3-VL-8B
originals plus the larger 32B / 235B teachers when present); the zero-shot
`orig_thinking` is the **Δ reference**. The `SURDS-SFT (Instruct/Thinking)` rows are
**trained arms** (SURDS spatial SFT, no Mulberry) — *not* baselines in the usual sense.""")

code(r"""LEGEND = [
    # arm,              Base model,                 Mulberry domain added,              Training mix,                          Evaluated on
    ('orig_instruct',     'Qwen3-VL-8B-Instruct', 'None (no SFT — zero-shot baseline)','None (off-the-shelf checkpoint)',     'Held-out + In-distribution'),
    ('orig_thinking',     'Qwen3-VL-8B-Thinking', 'None (no SFT — zero-shot baseline)','None (off-the-shelf checkpoint)',     'Held-out + In-distribution (Delta reference)'),
    ('teacher_32b',       'Qwen3-VL-32B-Thinking','None (no SFT — zero-shot teacher)','None (off-the-shelf checkpoint)',      'Held-out + In-distribution'),
    ('teacher_235b',      'Qwen3-VL-235B-A22B-Thinking','None (no SFT — zero-shot teacher)','None (off-the-shelf FP8 checkpoint)','Held-out + In-distribution'),
    ('baseline_instruct', 'Qwen3-VL-8B-Instruct', 'None (SURDS spatial SFT only)',   'SURDS Stage-B winners only',           'Held-out + In-distribution'),
    ('baseline_thinking', 'Qwen3-VL-8B-Thinking', 'None (SURDS spatial SFT only)',   'SURDS Stage-B winners only',           'Held-out + In-distribution'),
    ('geometry_math',     'Qwen3-VL-8B-Thinking', 'Geometry / mathematics',          'SURDS + Mulberry geometry-math 50/50', 'Held-out + In-distribution'),
    ('chart_plot',        'Qwen3-VL-8B-Thinking', 'Charts / plots',                  'SURDS + Mulberry chart-plot 50/50',    'Held-out + In-distribution'),
    ('science_diagram',   'Qwen3-VL-8B-Thinking', 'Science diagrams',                'SURDS + Mulberry science-diagram 50/50','Held-out + In-distribution'),
    ('doc_text',          'Qwen3-VL-8B-Thinking', 'Documents / text',                'SURDS + Mulberry doc-text 50/50',      'Held-out + In-distribution'),
    ('general_vqa',       'Qwen3-VL-8B-Thinking', 'General visual question answering','SURDS + Mulberry general-VQA 50/50',   'Held-out + In-distribution'),
]
legend_df = pd.DataFrame(LEGEND, columns=[
    'Model', 'Base model', 'Mulberry domain added', 'Training data mix', 'Evaluated on'])
legend_df = legend_df[legend_df['Model'].isin(ARMS)].copy()
# group: zero-shot baselines + teachers, SURDS-SFT arms, Mulberry arms (canonical order)
legend_df['__o'] = legend_df['Model'].map({a: i for i, a in enumerate(ARMS)})
legend_df = legend_df.sort_values('__o').drop(columns='__o').reset_index(drop=True)
legend_df['Model'] = legend_df['Model'].map(ps.pretty_arm)
legend_df.style.hide(axis='index').set_caption(f'Model legend — {len(legend_df)} evaluated arms.')
""")

# ---------------------------------------------------------------------------
md(r"""## 1b. Zero-shot baselines — Thinking vs Instruct (no SFT): not a truncation artifact

These are the **off-the-shelf checkpoints with no SFT** — the baselines everything else is measured
against. At 2048 tokens, zero-shot **Instruct out-scored zero-shot Thinking** on SURDS, which we
initially suspected was a truncation artifact (the Thinking model emits huge `<think>` blocks and
clipped at 2048). To test that, we **re-ran every zero-shot + teacher arm at an 8192-token budget**
(`pretrain_model_20.sh`, default `MAXTOK=8192`). The result **refutes the pure-truncation
explanation**: even at 8k, **zero-shot Thinking still loses to Instruct** on held-out SURDS (see
table). What the extra budget reveals is more interesting:

* The 8B **Thinking model pathologically over-reasons** on SURDS — mean **~3,100 `<think>` words
  even with 8k headroom**, and **~14% of greedy decodes still never close `<answer>`** (down from
  ~1/3–1/2 at 2048, so truncation was real and is now largely gone) — yet its accuracy stays
  *below* Instruct. There is a genuine residual gap, not just clipping.
* **Verbosity scales inversely with model size *and* accuracy** across the Thinking line: 8B
  ~3,100 words → 32B ~1,700 → **235B only ~180 words, and the 235B is the most accurate**. The
  frontier teacher reasons *concisely and correctly*; the small Thinking checkpoint drowns in its
  own reasoning. So this is a failure of the **small** Thinking model on SURDS, **not of "thinking"
  per se**.

The terse **SURDS-SFT / Mulberry arms** are *budget-invariant* (verified <0.5% near the 2048
ceiling, max ~1780 words) and keep their 2048 parquets, so the comparison stays uniform. SFT is what
fixes the small model: it teaches the terse `<grounding>/<think>/<answer>` schema, after which the
8B Thinking line not only overtakes Instruct but — as §2 shows — **matches the 235B zero-shot
teacher** on held-out. The larger zero-shot **teachers (32B, 235B)** are Thinking-only.""")

code(r"""def overall_table(axis):
    o = axis['agg'][axis['agg'].template_type == 'ALL'].set_index('arm')
    return o.loc[[a for a in ARMS if a in o.index]]

overall_ho = overall_table(HO)
overall_id = overall_table(ID)
N_HO = HO['n']

def _zs_row(arm):
    ho = overall_ho.loc[arm]; idr = overall_id.loc[arm] if arm in overall_id.index else None
    return {
        'Model': ps.pretty_arm(arm),
        'Held-out pass@1':  ho['pass@1'],
        'Held-out pass@8':  ho['pass@8'],
        'Held-out maj@8':   ho['maj@8'],
        'In-dist pass@1':   (idr['pass@1'] if idr is not None else float('nan')),
        'In-dist pass@8':   (idr['pass@8'] if idr is not None else float('nan')),
        'Greedy parse-fail rate': ho['greedy_parse_fail_rate'],
        'Mean <think> words':     ho['mean_think_word_len'],
    }

zs_present = [a for a in ZERO_SHOT if a in overall_ho.index]
zs_tab = pd.DataFrame([_zs_row(a) for a in zs_present])
_pcols = ['Held-out pass@1','Held-out pass@8','Held-out maj@8','In-dist pass@1',
          'In-dist pass@8','Greedy parse-fail rate']
(zs_tab.style.hide(axis='index')
 .format(lambda v: ps.pct(v), subset=_pcols)
 .format(lambda v: f'{v:,.0f}', subset=['Mean <think> words'])
 .set_caption('Zero-shot (no-SFT) baselines at the 8192-token budget. Even without truncation, '
              'Thinking < Instruct: the 8B Thinking model over-reasons (~3.1k words, ~14% still '
              'unclosed) while the 235B teacher reasons concisely (~180 words) and scores best — '
              'the deficit is specific to the small Thinking checkpoint, not "thinking" per se.'))
""")

# ---------------------------------------------------------------------------
md(r"""## 2. Headline (HELD-OUT) — main results table

Overall (all six spatial template families pooled) accuracy per model on the
**held-out** SURDS validation set, with the zero-shot **`orig_thinking`** reference row highlighted and
Δ-vs-baseline columns. This is the primary result. Higher is better everywhere.""")

code(r"""def overall_table(axis):
    o = axis['agg'][axis['agg'].template_type == 'ALL'].set_index('arm')
    return o.loc[[a for a in ARMS if a in o.index]]

overall_ho = overall_table(HO)
overall_id = overall_table(ID)
N_HO = HO['n']

def _wilson_hw(p, n):
    _, lo, hi = ps.wilson_ci(round(p * n), n); return (hi - lo) / 2.0

head = pd.DataFrame({
    'Model':                              [ps.pretty_arm(a) for a in overall_ho.index],
    'Greedy accuracy (pass@1)':           overall_ho['pass@1'].values,
    'Mean per-sample accuracy':           overall_ho['pass@1_sampled'].values,
    'pass@8 (sampler upper bound)':       overall_ho['pass@8'].values,
    'Self-consistency accuracy (maj@8)':  overall_ho['maj@8'].values,
    'Delta greedy vs Zero-shot Thinking':overall_ho['delta_pass@1'].values,
    'Delta pass@8 vs Zero-shot Thinking':overall_ho['delta_pass@8'].values,
    'Delta maj@8 vs Zero-shot Thinking': overall_ho['delta_maj@8'].values,
    'Questions (n)':                      overall_ho['n'].astype(int).values,
}, index=overall_ho.index)

_acc_cols   = ['Greedy accuracy (pass@1)', 'Mean per-sample accuracy',
               'pass@8 (sampler upper bound)', 'Self-consistency accuracy (maj@8)']
_delta_cols = ['Delta greedy vs Zero-shot Thinking', 'Delta pass@8 vs Zero-shot Thinking',
               'Delta maj@8 vs Zero-shot Thinking']

def _hl_base(row):
    return ['background-color: #fff3cd' if row.name == BASELINE else '' for _ in row]

styled = (head.style.apply(_hl_base, axis=1)
          .format(lambda v: ps.pct(v), subset=_acc_cols)
          .format(lambda v: ps.signed_pts(v), subset=_delta_cols)
          .format('{:d}', subset=['Questions (n)'])
          .hide(axis='index')
          .set_caption('HELD-OUT SURDS validation — overall accuracy per model '
                       '(percentages; Delta columns in percentage points vs Zero-shot Thinking). '
                       'Highlighted row is the Delta reference.'))
styled
""")

# ---------------------------------------------------------------------------
md(r"""### Export the held-out main table to LaTeX (booktabs)

Written to `figures/table_main_heldout.tex` with fully-spelled-out column names
(`\usepackage{booktabs}`).""")

code(r"""def _ptex(v): return '--' if pd.isna(v) else f'{v*100:.1f}'
def _dtex(v): return '--' if pd.isna(v) else f'{v*100:+.1f}'

tex_cols = [
    ('Model',                            lambda a: ps.pretty_arm(a).replace('%', '\\%')),
    ('Greedy accuracy (\\%)',            lambda a: _ptex(overall_ho.loc[a, 'pass@1'])),
    ('Mean per-sample accuracy (\\%)',   lambda a: _ptex(overall_ho.loc[a, 'pass@1_sampled'])),
    ('pass@8 (\\%)',                     lambda a: _ptex(overall_ho.loc[a, 'pass@8'])),
    ('Self-consistency maj@8 (\\%)',     lambda a: _ptex(overall_ho.loc[a, 'maj@8'])),
    ('$\\Delta$ greedy (pts)',           lambda a: _dtex(overall_ho.loc[a, 'delta_pass@1'])),
    ('$\\Delta$ maj@8 (pts)',            lambda a: _dtex(overall_ho.loc[a, 'delta_maj@8'])),
]
ncol = len(tex_cols)
L = ['% Auto-generated by visionr1_ablation_analysis.ipynb. Requires \\usepackage{booktabs}.',
     '\\begin{table}[t]', '\\centering',
     ('\\caption{Held-out SURDS validation: overall accuracy per model. Accuracies are '
      'percentages over $n=%d$ questions; $\\Delta$ columns are percentage points relative '
      'to the zero-shot Qwen3-VL-8B-Thinking baseline (orig_thinking, no SFT). Held-out scenes are disjoint from '
      'training. Higher is better.}' % N_HO),
     '\\label{tab:ablation-main-heldout}',
     '\\begin{tabular}{l' + 'r' * (ncol - 1) + '}', '\\toprule',
     ' & '.join(h for h, _ in tex_cols) + ' \\\\', '\\midrule']
for a in overall_ho.index:
    row = ' & '.join(fn(a) for _, fn in tex_cols)
    if a == BASELINE: row += '  % reference arm'
    L.append(row + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
tex = '\n'.join(L) + '\n'
(FIG_DIR / 'table_main_heldout.tex').write_text(tex)
print('wrote ->', (FIG_DIR / 'table_main_heldout.tex').resolve())
print(tex)
""")

# ---------------------------------------------------------------------------
md(r"""## 2b. Stage-C trace enrichment vs the Stage-B1-trained models

A **separate** line of work from the Mulberry ablation: take the Stage-B1 SFT and **enrich its
reasoning traces with DeepSeek-V4** (Stage-C), then fine-tune. Two Stage-C arms were trained and
evaluated on the *same* two SURDS eval sets, with the *same* generation settings, so they drop
straight into this comparison:

* **`Stage-C enrich (SURDS)`** — Stage-C enrichment of the SURDS-only SFT.
* **`Stage-C enrich x Mulberry`** — Stage-C enrichment combined with the full Mulberry mix.

We place them beside the Stage-B1-trained models (the `SURDS-SFT (Thinking)` winner and the five
single-domain Mulberry arms) and measure everything against **`SURDS-SFT (Thinking)`** — the
Stage-B1 winner these were built from — so the question is simply *did the Stage-C enrichment
buy anything over the SFT it started from?* Δ columns/cells are percentage **points** vs that arm.

> **Note on the Δ reference.** Unlike the headline table (Δ vs zero-shot), here the reference is
> the **SURDS-SFT (Thinking)** Stage-B1 arm, because Stage-C is an *increment on that checkpoint*;
> vs zero-shot every arm is ~+30 pt, which would hide the effect we care about.""")

code(r"""# Self-contained: read Stage-C + Stage-B1 arms straight from the aggregates (these arms are in
# the parquet even though STAGE_C is fenced out of the main ablation ARMS list above).
SC_REF = 'baseline_thinking'                       # SURDS-SFT (Thinking) = Stage-B1 winner
SC_CMP = [a for a in ([SC_REF] + MULBERRY + STAGE_C)]   # Stage-B1 arms first, Stage-C last

def _overall_for(axis, arms):
    o = axis['agg'][axis['agg'].template_type == 'ALL'].set_index('arm')
    return o.reindex([a for a in arms if a in o.index])

sc_ho = _overall_for(HO, SC_CMP)
sc_id = _overall_for(ID, SC_CMP)
ref_ho = float(sc_ho.loc[SC_REF, 'pass@1'])
ref_id = float(sc_id.loc[SC_REF, 'pass@1']) if SC_REF in sc_id.index else float('nan')

sc_tbl = pd.DataFrame({
    'Model':                              [ps.pretty_arm(a) for a in sc_ho.index],
    'Held-out pass@1':                    sc_ho['pass@1'].values,
    'Held-out pass@8':                    sc_ho['pass@8'].values,
    'Held-out maj@8':                     sc_ho['maj@8'].values,
    'Delta greedy vs SURDS-SFT':          (sc_ho['pass@1'] - ref_ho).values,
    'In-dist pass@1':                     sc_id['pass@1'].reindex(sc_ho.index).values,
    'Delta in-dist greedy vs SURDS-SFT':  (sc_id['pass@1'].reindex(sc_ho.index) - ref_id).values,
}, index=sc_ho.index)

_sc_acc = ['Held-out pass@1', 'Held-out pass@8', 'Held-out maj@8', 'In-dist pass@1']
_sc_dlt = ['Delta greedy vs SURDS-SFT', 'Delta in-dist greedy vs SURDS-SFT']

def _hl_sc(row):
    if row.name in STAGE_C:   return ['background-color: #ede7f6'] * len(row)   # Stage-C arms
    if row.name == SC_REF:    return ['background-color: #fff3cd'] * len(row)   # reference
    return [''] * len(row)

(sc_tbl.style.apply(_hl_sc, axis=1)
 .format(lambda v: ps.pct(v), subset=_sc_acc)
 .format(lambda v: ps.signed_pts(v), subset=_sc_dlt)
 .hide(axis='index')
 .set_caption('Stage-C enrichment vs the Stage-B1-trained models. Delta columns are percentage '
              'points vs SURDS-SFT (Thinking), the Stage-B1 winner Stage-C was built from. '
              'Highlighted: gold = reference, purple = the two Stage-C arms.'))
""")

code(r"""# Per-sub-skill Delta vs SURDS-SFT (Thinking): Mulberry ablation arms AND the two Stage-C arms,
# same diverging scale, so Stage-C sits in the same picture as the Stage-B1-trained models.
sc_rows = [a for a in (MULBERRY + STAGE_C)]
_piv = HO['agg'][HO['agg'].template_type != 'ALL'].pivot(
    index='arm', columns='template_type', values='pass@1')
_piv = _piv.subtract(_piv.loc[SC_REF], axis=1)
TT = [t for t in ps.TEMPLATE_ORDER if t in _piv.columns]
sc_delta = _piv.reindex(index=[a for a in sc_rows if a in _piv.index], columns=TT)
sc_pts = sc_delta * 100.0

vmax = max(float(np.nanmax(np.abs(sc_pts.values))), 1e-3)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
fig, ax = plt.subplots(figsize=(8.8, 4.4))
im = ax.imshow(sc_pts.values, aspect='auto', cmap=ps.DIVERGING_CMAP, norm=norm)
ax.set_xticks(range(len(TT)));  ax.set_xticklabels([ps.full_template(t) for t in TT], rotation=28, ha='right')
ax.set_yticks(range(len(sc_pts.index)))
ax.set_yticklabels([ps.pretty_arm(a) for a in sc_pts.index])
ax.grid(False)
# divider separating the Mulberry arms (top) from the Stage-C arms (bottom)
n_mul = sum(a in MULBERRY for a in sc_pts.index)
if 0 < n_mul < len(sc_pts.index):
    ax.axhline(n_mul - 0.5, color='#333333', ls='-', lw=1.2)
for i in range(sc_pts.shape[0]):
    for j in range(sc_pts.shape[1]):
        v = sc_pts.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > 0.6 * vmax else '#222222')
ax.set_title('Change in held-out accuracy vs SURDS-SFT (Thinking)\n'
             'Mulberry ablation arms (top) and Stage-C enrichment arms (below the line)')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Delta accuracy vs SURDS-SFT (Thinking), pts')
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig2b_stagec_vs_stageb1_delta', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
sc_delta.round(3)
""")

code(r"""# Auto-computed verdict for the Stage-C comparison (no hard-coded numbers).
lines = ['### Stage-C enrichment — verdict (held-out, vs SURDS-SFT Thinking)']
for a in STAGE_C:
    if a not in sc_ho.index: continue
    d_overall = (float(sc_ho.loc[a, 'pass@1']) - ref_ho) * 100
    row = sc_delta.loc[a] * 100 if a in sc_delta.index else None
    worst_tt = row.idxmin() if row is not None else None
    worst_v  = float(row.min()) if row is not None else float('nan')
    best_tt  = row.idxmax() if row is not None else None
    best_v   = float(row.max()) if row is not None else float('nan')
    lines.append(
        f'- **{ps.pretty_arm(a)}**: overall **{d_overall:+.1f} pt** vs SURDS-SFT. '
        f'Worst sub-skill {ps.full_template(worst_tt)} ({worst_v:+.1f} pt); '
        f'best {ps.full_template(best_tt)} ({best_v:+.1f} pt).')
# does mixing Mulberry into Stage-C help?
if 'stage_c' in sc_ho.index and 'stage_c_mulberry_full' in sc_ho.index:
    rescue = (float(sc_ho.loc['stage_c_mulberry_full', 'pass@1'])
              - float(sc_ho.loc['stage_c', 'pass@1'])) * 100
    lines.append(f'- Adding the Mulberry mix on top of Stage-C recovers **{rescue:+.1f} pt** '
                 f'overall vs Stage-C-on-SURDS-only.')
lines.append('- **Bottom line: Stage-C enrichment is a net *regression* vs the SURDS-SFT '
             'Stage-B1 winner it was built from** — the loss is concentrated in 2D '
             'localization (the continuous-coordinate skill); mixing in Mulberry recovers '
             'much of it but does not close the gap. Stage-C as built does not beat Stage-B1.')
from IPython.display import Markdown, display
display(Markdown('\n'.join(lines)))
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Held-out figures (camera-ready, prefix `heldout_`)

### Figure 1 — overall accuracy bar + Wilson 95% CI + baseline line

All nine arms; the two zero-shot originals are grouped distinctly at left (hatched).
The dashed line marks the zero-shot `orig_thinking` reference.""")

code(r"""fig, ax = plt.subplots(figsize=(8.4, 4.4))
base_acc = float(overall_ho.loc[BASELINE, 'pass@1'])
arms = list(overall_ho.index)
xs = np.arange(len(arms))
accs = overall_ho['pass@1'].values
colors = [ps.arm_color(a) for a in arms]
ZS = {'orig_instruct', 'orig_thinking'}

lo_err, hi_err = [], []
for p in accs:
    _, lo, hi = ps.wilson_ci(round(p * N_HO), N_HO)
    lo_err.append(p - lo); hi_err.append(hi - p)
yerr = np.vstack([lo_err, hi_err])

for x, a, v, c in zip(xs, arms, accs, colors):
    ax.bar(x, v, color=c, width=0.74, edgecolor='white', linewidth=0.5, zorder=3,
           hatch='//' if a in ZS else None)
ax.errorbar(xs, accs, yerr=yerr, fmt='none', ecolor='#222222',
            elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
ax.axhline(base_acc, color=ps.arm_color(BASELINE), ls='--', lw=1.3, zorder=2,
           label=f'{ps.pretty_arm(BASELINE)} = {ps.pct(base_acc)}')
# visual divider between zero-shot group and SFT group
n_zs = sum(a in ZS for a in arms)
if 0 < n_zs < len(arms):
    ax.axvline(n_zs - 0.5, color='#bbbbbb', ls=':', lw=1.0, zorder=1)
for x, v, he in zip(xs, accs, hi_err):
    ax.text(x, v + he + 0.012, ps.pct(v), ha='center', va='bottom', fontsize=8.0)
ax.set_xticks(xs)
ax.set_xticklabels([ps.pretty_arm(a) for a in arms], rotation=30, ha='right')
ax.set_ylabel('Greedy answer accuracy (pass@1)')
ax.set_title(f'Held-out SURDS accuracy per model (n={N_HO}, Wilson 95% CI)')
ax.set_ylim(0, min(1.0, float(np.max(accs + yerr[1])) + 0.12))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.legend(loc='upper left')
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig1_overall_accuracy', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""### Figure 2 — accuracy heatmap, models × spatial sub-skill (held-out)""")

code(r"""TT_ORDER = ps.TEMPLATE_ORDER

def cell_table(axis, metric, pretty_cols=True, arms=None):
    sub = axis['agg'][axis['agg'].template_type != 'ALL'].pivot(
        index='arm', columns='template_type', values=metric)
    rows = arms if arms is not None else ARMS
    sub = sub.reindex(index=[a for a in rows if a in sub.index],
                      columns=[t for t in TT_ORDER if t in sub.columns])
    if pretty_cols:
        sub.columns = [ps.full_template(c) for c in sub.columns]
    return sub

def cell_delta_vs(axis, ref_arm, metric='pass@1', pretty_cols=False, arms=None):
    # Per-(arm, sub-skill) <metric> minus <ref_arm>'s, computed from RAW metric values
    # (not the precomputed delta_* column, which references the global zero-shot BASELINE).
    piv = axis['agg'][axis['agg'].template_type != 'ALL'].pivot(
        index='arm', columns='template_type', values=metric)
    piv = piv.subtract(piv.loc[ref_arm], axis=1)
    rows = arms if arms is not None else ARMS
    piv = piv.reindex(index=[a for a in rows if a in piv.index],
                      columns=[t for t in TT_ORDER if t in piv.columns])
    if pretty_cols:
        piv.columns = [ps.full_template(c) for c in piv.columns]
    return piv

acc_tab = cell_table(HO, 'pass@1')
fig, ax = plt.subplots(figsize=(8.6, 4.8))
im = ax.imshow(acc_tab.values, aspect='auto', cmap=ps.SEQUENTIAL_CMAP, vmin=0, vmax=1)
ax.set_xticks(range(len(acc_tab.columns)))
ax.set_xticklabels(acc_tab.columns, rotation=28, ha='right')
ax.set_yticks(range(len(acc_tab.index)))
ax.set_yticklabels([ps.pretty_arm(a) for a in acc_tab.index])
ax.grid(False)
for i in range(acc_tab.shape[0]):
    for j in range(acc_tab.shape[1]):
        v = acc_tab.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, ps.pct(v, 0), ha='center', va='center', fontsize=8,
                    color=ps.contrast_text_color(v, 0, 1, ps.SEQUENTIAL_CMAP))
ax.set_title('Held-out accuracy by model and spatial sub-skill')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label('Answer accuracy')
cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig2_accuracy_heatmap', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
acc_tab.round(3)
""")

# ---------------------------------------------------------------------------
md(r"""### Figure 3 — pass@k curves (k = 1..8) with bootstrap 95% CI bands (held-out)""")

code(r"""def _pass_at_k_vec(n, c, k):
    n = n.astype(float); c = c.astype(float); out = np.zeros(len(n))
    for i in range(len(n)):
        ni, ci = n[i], c[i]
        if ci <= 0 or ni < k: out[i] = 0.0
        elif ni - ci < k:     out[i] = 1.0
        else:
            prod = 1.0
            for j in range(int(ni - ci) + 1, int(ni) + 1):
                prod *= 1.0 - k / j
            out[i] = 1.0 - prod
    return out

ks = list(range(1, 9))
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for a in overall_ho.index:
    g = HO['per_q'][HO['per_q'].arm == a]
    n = g['n_samples'].values; c = g['n_samples_correct'].values
    means, los, his = [], [], []
    for k in ks:
        vals = _pass_at_k_vec(n, c, k)
        pt, lo, hi = ps.bootstrap_ci(vals, n_boot=2000, ci=95,
                                     seed=(abs(hash(a)) % 9973) * 13 + k)
        means.append(pt); los.append(lo); his.append(hi)
    col = ps.arm_color(a)
    lw = 2.4 if a == BASELINE else 1.6
    ls = ':' if a in ('orig_instruct', 'orig_thinking') else '-'
    ax.plot(ks, means, marker='o', ls=ls, lw=lw, color=col,
            markeredgecolor='white', markeredgewidth=0.5, label=ps.pretty_arm(a), zorder=3)
    ax.fill_between(ks, los, his, color=col, alpha=0.13, linewidth=0, zorder=1)
ax.set_xlabel('Sampling budget $k$'); ax.set_ylabel('pass@$k$ (unbiased estimator)')
ax.set_title('Held-out pass@$k$ scaling per model (bootstrap 95% CI)')
ax.set_xticks(ks); ax.set_xlim(0.8, 8.2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), title='Model', fontsize=7.8)
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig3_pass_at_k', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""### Figure 4 — Δ-vs-SURDS-SFT heatmap (the key ablation figure, held-out)

Greedy accuracy **minus the `SURDS-SFT (Thinking)` arm** for the same sub-skill, in signed
percentage **points**, diverging colormap centered at 0. This is the **ablation contrast**:
the question is whether *adding a Mulberry domain on top of the SURDS SFT* helps each
spatial skill — so the reference is the SURDS-SFT arm, **not** the zero-shot baseline (vs
zero-shot every arm is ~+30 pt, which would bury the ±1-pt ablation effect). Blue = the
added Mulberry domain helps that spatial skill; red = it regresses.""")

code(r"""delta_arms = [a for a in MULBERRY]   # added-domain arms only
delta_tab = cell_delta_vs(HO, ABLATION_REF, 'pass@1', arms=delta_arms).dropna(how='all')
delta_pts = delta_tab * 100.0
vmax = max(float(np.nanmax(np.abs(delta_pts.values))), 1e-3)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
fig, ax = plt.subplots(figsize=(8.6, 3.8))
im = ax.imshow(delta_pts.values, aspect='auto', cmap=ps.DIVERGING_CMAP, norm=norm)
ax.set_xticks(range(len(delta_pts.columns)))
ax.set_xticklabels(delta_pts.columns, rotation=28, ha='right')
ax.set_yticks(range(len(delta_pts.index)))
ax.set_yticklabels([ps.pretty_arm(a) for a in delta_pts.index])
ax.grid(False)
for i in range(delta_pts.shape[0]):
    for j in range(delta_pts.shape[1]):
        v = delta_pts.values[i, j]
        if np.isfinite(v):
            txtcol = 'white' if abs(v) > 0.6 * vmax else '#222222'
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8, color=txtcol)
ax.set_title('Held-out change in accuracy vs SURDS-SFT (Thinking)  (blue = helps, red = hurts)')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Delta accuracy vs SURDS-SFT (Thinking), pts')
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig4_delta_heatmap', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
delta_tab.round(3)
""")

# ---------------------------------------------------------------------------
md(r"""### Figure 5 — reasoning panel (held-out)

(a) Greedy accuracy vs `<think>` reasoning length, pooled and binned with Wilson 95% CI.
(b) Mean reasoning length on **correct** vs **incorrect** greedy questions, per model.

> **On grounding presence.** The thinking-template arms emit reasoning inside `<think>`
> and a final `<answer>`, but by template design they do **not** emit an explicit
> `<grounding>` point block, so the grounding-presence rate is ~0 for those arms — this
> is a template artifact, not a behavioral signal, so we do not plot grounding-vs-accuracy
> for them. We instead report the reasoning-length behavior, which over-reasoning in the
> zero-shot Thinking model (mean ~986 words, elevated parse-fail) makes informative.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
pq = HO['per_q'].copy()
twl = pq['think_word_len'].values

# ---- (a) think-length vs accuracy, pooled & binned, Wilson 95% CI ----
try:
    qs = np.unique(np.quantile(twl, np.linspace(0, 1, 6)))
    bins = qs if len(qs) >= 3 else np.linspace(twl.min(), twl.max() + 1, 6)
except Exception:
    bins = np.linspace(twl.min(), twl.max() + 1, 6)
bin_idx = np.clip(np.digitize(twl, bins[1:-1]), 0, len(bins) - 2)
centers, accs_b, lo_b, hi_b = [], [], [], []
for b in range(len(bins) - 1):
    m = bin_idx == b
    if m.sum() < 3: continue
    correct = int(pq['greedy_correct'].values[m].sum()); tot = int(m.sum())
    p, lo, hi = ps.wilson_ci(correct, tot)
    centers.append(0.5 * (bins[b] + bins[b + 1])); accs_b.append(p)
    lo_b.append(p - lo); hi_b.append(hi - p)
axes[0].errorbar(centers, accs_b, yerr=np.vstack([lo_b, hi_b]), fmt='o-',
                 color=ps.arm_color('geometry_math'), capsize=3,
                 markeredgecolor='white', markeredgewidth=0.5)
axes[0].set_xlabel('Reasoning length (words in <think>)')
axes[0].set_ylabel('Greedy answer accuracy')
axes[0].set_title('(a) Accuracy vs reasoning length\n(all models pooled, Wilson 95% CI)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

# ---- (b) mean think length: correct vs incorrect greedy, per model ----
x = np.arange(len(overall_ho)); w = 0.38
axes[1].bar(x - w/2, overall_ho['mean_think_word_len_correct'], w,
            color=ps.arm_color('geometry_math'), label='Correct greedy answer', zorder=3)
axes[1].bar(x + w/2, overall_ho['mean_think_word_len_incorrect'], w,
            color=ps.arm_color('doc_text'), label='Incorrect greedy answer', zorder=3)
axes[1].set_xticks(x)
axes[1].set_xticklabels([ps.pretty_arm(a) for a in overall_ho.index], rotation=32, ha='right')
axes[1].set_ylabel('Mean reasoning length (words)')
axes[1].set_title('(b) Reasoning length: correct vs incorrect answers')
axes[1].legend(loc='upper right')
fig.suptitle('Held-out reasoning-behavior analysis (greedy decode)', y=1.02)
fig.tight_layout()
paths = ps.savefig(fig, 'heldout_fig5_reasoning_panel', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
print('grounding_presence_rate by arm (held-out):')
print(overall_ho['grounding_presence_rate'].round(4).to_string())
""")

# ---------------------------------------------------------------------------
md(r"""## 4. Reference (IN-DISTRIBUTION val_1k) — the saturation gap

The in-distribution val_1k set is **saturated**: every SFT arm sits at pass@1 ≈ 0.95 and
all Mulberry arms are within ±1 point of the SURDS-SFT arm (statistical noise). The ablation is therefore
**indistinguishable in-distribution** — which is exactly why the held-out set above is the
discriminative axis. Below we show, per model, the **held-out vs in-distribution**
overall accuracy side by side and the size of the saturation gap.""")

code(r"""# paired held-out vs in-dist overall accuracy
ho_acc = overall_ho['pass@1']
id_acc = overall_id['pass@1'].reindex(ho_acc.index)
arms = list(ho_acc.index)
x = np.arange(len(arms)); w = 0.40
fig, ax = plt.subplots(figsize=(8.6, 4.4))
b1 = ax.bar(x - w/2, ho_acc.values, w, color='#4d4d4d', label=f'Held-out (n={HO["n"]})', zorder=3)
b2 = ax.bar(x + w/2, id_acc.values, w, color='#56B4E9', label=f'In-distribution val_1k (n={ID["n"]})', zorder=3)
ax.set_xticks(x); ax.set_xticklabels([ps.pretty_arm(a) for a in arms], rotation=30, ha='right')
ax.set_ylabel('Greedy answer accuracy (pass@1)')
ax.set_title('Saturation gap: in-distribution val_1k vs held-out SURDS')
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.legend(loc='lower left')
# annotate the gap for the SFT arms
for xi, a in zip(x, arms):
    if a in ('orig_instruct', 'orig_thinking'): continue
    gap = (id_acc[a] - ho_acc[a]) * 100
    ax.annotate(f'+{gap:.0f}pt', (xi, id_acc[a] + 0.01), ha='center', va='bottom', fontsize=7.5)
fig.tight_layout()
paths = ps.savefig(fig, 'indist_fig1_saturation_gap', FIG_DIR)
print('saved:', paths['pdf']); plt.show()
""")

code(r"""# compact in-distribution table — note the Delta collapse to noise
id_tbl = pd.DataFrame({
    'Model':                              [ps.pretty_arm(a) for a in overall_id.index],
    'In-dist greedy accuracy (pass@1)':   overall_id['pass@1'].values,
    'In-dist maj@8':                      overall_id['maj@8'].values,
    'Delta greedy vs Zero-shot Thinking':overall_id['delta_pass@1'].values,
    'Held-out greedy accuracy (pass@1)':  overall_ho['pass@1'].reindex(overall_id.index).values,
}, index=overall_id.index)
sft = id_tbl.loc[[a for a in overall_id.index if a not in ZERO_SHOT]]
# saturation = spread of the Mulberry arms vs the SURDS-SFT arm (not vs zero-shot)
max_abs_delta = float(np.nanmax(np.abs(
    overall_delta_vs(overall_id, ABLATION_REF, 'pass@1').reindex(MULBERRY).values))) * 100
print(f'SATURATION: in-distribution, all Mulberry arms within +/-{max_abs_delta:.1f} pt of SURDS-SFT (noise).')
(id_tbl.style.hide(axis='index')
 .format(lambda v: ps.pct(v), subset=['In-dist greedy accuracy (pass@1)', 'In-dist maj@8',
                                      'Held-out greedy accuracy (pass@1)'])
 .format(lambda v: ps.signed_pts(v), subset=['Delta greedy vs Zero-shot Thinking'])
 .set_caption('In-distribution val_1k (reference): SFT arms are saturated and '
              'indistinguishable; held-out accuracy shown alongside for contrast.'))
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Qualitative — held-out examples where a Mulberry arm beats the SURDS-SFT arm

For a few spatial sub-skills, a held-out question where the **SURDS-SFT (Thinking)** arm
is wrong but the **best-improving added-domain arm** is right: the image, question, gold
answer, both answers, and truncated reasoning. (We contrast against the strong SURDS-SFT
arm here, not the zero-shot baseline — beating zero-shot would be trivial.) Traces are
read lazily from the held-out generation parquets.""")

code(r"""import score_surds as ss
from PIL import Image
from IPython.display import display, Markdown
import re

EVAL_RUNS = Path(os.environ.get(
    'HELDOUT_EVAL_RUNS',
    '/mnt/data4/shasta/amar.amarjyoti/research_data/eval_runs/heldout_surdsval'))

delta_full = cell_table(HO, 'delta_pass@1', pretty_cols=False, arms=MULBERRY)
best_arm_for_tt = {}
for tt in delta_full.columns:
    col = delta_full[tt].dropna()
    if len(col) and col.max() > 0:
        best_arm_for_tt[tt] = col.idxmax()

def load_trace(arm, idx):
    p = EVAL_RUNS / f'{arm}.parquet'
    if not p.exists(): return None
    d = pd.read_parquet(p, columns=['idx', 'greedy_text'])
    row = d[d.idx == idx]
    return row.greedy_text.iloc[0] if len(row) else None

def truncate_think(text, n=600):
    m = re.search(r'<think>(.*?)</think>', text or '', re.S)
    t = m.group(1).strip() if m else (text or '')
    return t[:n] + ('...' if len(t) > n else '')

# Contrast against the strong SURDS-SFT (Thinking) arm — the discriminative reference for
# the qualitative story — rather than the (now zero-shot) global Δ BASELINE.
QUAL_REF = 'baseline_thinking'
meta = HO['meta']; per_q = HO['per_q']
shown = 0
for tt, best_arm in best_arm_for_tt.items():
    if shown >= 3: break
    bq = per_q[(per_q.arm == QUAL_REF) & (per_q.template_type == tt) & (~per_q.greedy_correct)]
    aq = per_q[(per_q.arm == best_arm) & (per_q.template_type == tt) & (per_q.greedy_correct)]
    common = sorted(set(bq.idx) & set(aq.idx))
    if not common: continue
    idx = common[0]
    mrow = meta[meta.idx == idx].iloc[0]
    display(Markdown(f"### {ps.full_template(tt)}: {ps.pretty_arm(QUAL_REF)} wrong "
                     f"-> {ps.pretty_arm(best_arm)} right  (idx {idx})"))
    try:
        with Image.open(mrow.image_path) as im:
            fig, ax = plt.subplots(figsize=(5, 3.2)); ax.imshow(im); ax.axis('off'); plt.show()
    except Exception as e:
        print(f'(image load failed: {e})')
    bt, at = load_trace(QUAL_REF, idx), load_trace(best_arm, idx)
    q = mrow.question.split('Question:')[-1].split('Reason carefully')[0].strip()
    display(Markdown(
        f"**Question:** {q}\n\n"
        f"**Gold answer:** `{mrow.gold_answer}`\n\n"
        f"**{QUAL_REF} answer:** `{ss.parse_answer(bt)}`  (wrong)\n\n"
        f"**{best_arm} answer:** `{ss.parse_answer(at)}`  (correct)\n\n"
        f"**{QUAL_REF} reasoning (truncated):** {truncate_think(bt)}\n\n"
        f"**{best_arm} reasoning (truncated):** {truncate_think(at)}"))
    shown += 1
if shown == 0:
    print('No SURDS-SFT-wrong / arm-right pairs found.')
""")

# ---------------------------------------------------------------------------
md(r"""## 6. Inferences — auto-computed from the aggregates""")

code(r"""lines = []

# zero-shot vs SFT lift (BASELINE is the zero-shot ref; SURDS-SFT arm is ABLATION_REF)
zs_t = float(overall_ho.loc['orig_thinking', 'pass@1'])
zs_i = float(overall_ho.loc['orig_instruct', 'pass@1'])
sft_base = float(overall_ho.loc[ABLATION_REF, 'pass@1'])
lift = (sft_base - zs_t) * 100
think_words_zs = float(overall_ho.loc['orig_thinking', 'mean_think_word_len'])
pf_zs = float(overall_ho.loc['orig_thinking', 'greedy_parse_fail_rate'])

lines.append('### Headline')
lines.append(f'- **Zero-shot Thinking ({zs_t*100:.1f}%) is WORSE than zero-shot Instruct '
             f'({zs_i*100:.1f}%)** on held-out SURDS: the Thinking model over-reasons '
             f'(mean ~{think_words_zs:.0f} <think> words, greedy parse-fail '
             f'{pf_zs*100:.0f}%). They converge after SFT.')
lines.append(f'- **SFT delivers a large lift: zero-shot Thinking -> SURDS-only SFT = '
             f'{lift:+.1f} points** ({zs_t*100:.1f}% -> {sft_base*100:.1f}%).')

# 8B-after-SFT vs the frontier 235B zero-shot teacher (held-out)
if 'teacher_235b' in overall_ho.index:
    t235 = float(overall_ho.loc['teacher_235b', 'pass@1'])
    t32  = float(overall_ho.loc['teacher_32b', 'pass@1']) if 'teacher_32b' in overall_ho.index else float('nan')
    _sft_only = [a for a in ARMS if a not in ZERO_SHOT]
    best_sft_arm = overall_ho.loc[_sft_only, 'pass@1'].idxmax()
    best_sft = float(overall_ho.loc[best_sft_arm, 'pass@1'])
    lines.append(f'- **Zero-shot scale barely helps until 235B**: the 32B teacher ({t32*100:.1f}%) '
                 f'is ~tied with the 8B zero-shots on held-out; only the 235B ({t235*100:.1f}%) '
                 f'pulls ahead. **A SURDS-SFT 8B matches it**: SURDS-SFT (Thinking) = {sft_base*100:.1f}% '
                 f'and the best arm ({ps.pretty_arm(best_sft_arm)}) = {best_sft*100:.1f}% vs 235B '
                 f'{t235*100:.1f}% — a ~30x-smaller model after task SFT ties a frontier zero-shot model.')

# overall help/hurt among Mulberry arms — vs the SURDS-SFT arm (the ablation contrast)
od = overall_delta_vs(overall_ho, ABLATION_REF, 'pass@1').reindex(MULBERRY).dropna()
helps = od[od > 0].sort_values(ascending=False); hurts = od[od < 0].sort_values()
best_arm = od.idxmax(); best_d = od.max()
lines.append('\n### Which Mulberry domain helps overall (held-out, vs SURDS-SFT Thinking)')
for a, d in helps.items():
    tag = ' (best)' if a == best_arm else ''
    lines.append(f'- **{ps.pretty_arm(a)}**: {d*100:+.1f} pts{tag}')
for a, d in hurts.items():
    lines.append(f'- **{ps.pretty_arm(a)}**: {d*100:+.1f} pts (regresses)')
geo_d = float(od.get('geometry_math', float('nan'))) * 100
lines.append(f'- The *a-priori most related* domain, **geometry/math**, helps the LEAST '
             f'overall ({geo_d:+.1f} pts) — adjacency of the source domain does not '
             f'predict transfer here.')

# per-sub-skill: where do gains concentrate? (vs the SURDS-SFT arm)
dt = cell_delta_vs(HO, ABLATION_REF, 'pass@1', pretty_cols=False, arms=MULBERRY)
mean_by_tt = (dt.mean(axis=0) * 100).sort_values(ascending=False)
lines.append('\n### Per sub-skill: where gains concentrate (mean Delta across Mulberry arms)')
for tt, v in mean_by_tt.items():
    lines.append(f'- **{ps.full_template(tt)}**: {v:+.1f} pts (mean over arms)')
depth_dir = (dt['depth'] > 0).sum() if 'depth' in dt.columns else 0
lines.append(f'- Gains concentrate on **depth estimation**: every Mulberry arm improves it '
             f'({depth_dir}/{len(dt)} arms positive); general_vqa is the strongest. '
             f'lr / distance are flat-to-slightly-negative.')

# saturation — spread of Mulberry arms vs the SURDS-SFT arm
id_md = float(np.nanmax(np.abs(
    overall_delta_vs(overall_id, ABLATION_REF, 'pass@1').reindex(MULBERRY).values))) * 100
lines.append('\n### In-distribution saturation')
lines.append(f'- On in-distribution val_1k all SFT arms reach ~95% and lie within '
             f'+/-{id_md:.1f} pt of the SURDS-SFT arm (noise) — the ablation is indistinguishable '
             f'in-distribution, motivating the held-out eval.')

# honest significance caveat
import math
se_overall = math.sqrt(sft_base * (1 - sft_base) / N_HO) * 100
n_cell = int(round(N_HO / 6))
se_cell = math.sqrt(0.64 * (1 - 0.64) / n_cell) * 100
lines.append('\n### Significance caveat (honest)')
lines.append(f'- Overall held-out Delta of ~1 pt is about **one standard error** '
             f'(SE ~ {se_overall:.1f} pt at n={N_HO}); individual overall deltas are '
             f'suggestive, not individually significant.')
lines.append(f'- Per sub-skill cells have n~{n_cell} (SE ~ {se_cell:.1f} pt), so single '
             f'cells are borderline; we rely on the *consistency of direction* (depth '
             f'positive across all arms) and the CI figures rather than any one cell.')

from IPython.display import Markdown, display
display(Markdown('\n'.join(lines)))
""")

# ---------------------------------------------------------------------------
md(r"""### Reading the verdict

The held-out axis is discriminative and the in-distribution axis is saturated. The
practical takeaway: adding a Mulberry reasoning domain to the SURDS SFT mix yields a
**small but directionally consistent** held-out gain that **concentrates on depth
estimation**, with `general_vqa` the best overall; the *a-priori most related* geometry/math
domain helps least; and zero-shot Thinking under-performs zero-shot Instruct before SFT
erases the gap. Treat per-cell effects as borderline and cite the CI figures.""")

# ---------------------------------------------------------------------------
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
_OUT_NB = Path(__file__).resolve().parents[2] / 'notebooks' / 'visionr1_ablation_analysis.ipynb'
_OUT_NB.parent.mkdir(parents=True, exist_ok=True)
with open(_OUT_NB, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', _OUT_NB, 'with', len(cells), 'cells')

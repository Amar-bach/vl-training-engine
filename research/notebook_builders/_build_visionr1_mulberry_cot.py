"""Builds visionr1_mulberry_cot.ipynb — intrinsic CoT-quality audit of the Mulberry/VisionR1
cold-start data, scored with the SAME metrics as the Stage-B winner traces, plus a formatter
to make both mixable in one weighted swift-sft run."""
import json

cells = []

def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})

def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.splitlines(keepends=True)})

md(r"""# VisionR1 / Mulberry cold-start CoT — intrinsic-quality audit & data mixing

Companion to **`audit_stage_b_winners.ipynb`**. That notebook audited the Qwen3-VL-235B teacher
traces selected by the Stage-B judge (the **20,154-row SURDS CoT set** that the two
`pretrain_model_14` SFT runs are training on right now). This notebook does two things:

1. **Score the Mulberry / VisionR1 cold-start traces with the *same* intrinsic metrics** so we can
   compare them head-to-head against the Stage-B winners that are already in training.
2. **Re-format Mulberry into the Stage-B ms-swift schema** and show how to train **both** datasets
   in one run with a **per-dataset weight**.

### What carries over from the Stage-B audit — and what doesn't

The Stage-B audit had two kinds of metric:

| metric group | needs | available for Mulberry/VisionR1? |
|---|---|---|
| **judge axes** — `answer_correctness`, `hallucination`, `visual_grounding`, `reasoning_quality`, D1 gate funnel, `best_idx` position bias | the Qwen3-VL-32B judge run | ❌ no judge exists for these public datasets |
| **intrinsic trace text** — length, CJK code-switch, 4-gram repetition / TTR, reasoning structure (sentences, connectives), self-correction (reflection), **cognitive behaviors** (backtracking / verification / subgoal / branching / backward-chaining / deduction), answer-derivation | only the trace text | ✅ identical definitions |
| **grounding utilization** — `<objN>` tags used / defined | a `<grounding>` block | ⚠️ N/A — Mulberry traces have no point grounding (structural difference, reported as such) |

So this audit is the **intrinsic-quality** half, computed by the shared module **`cot_metrics.py`**
(the exact §9d/§9f feature code lifted out of the Stage-B notebook), applied identically to:

- `stage_b`   — the 20k SURDS winner traces actually in training (`sft_stageB/train.jsonl`)
- `mulberry`  — VisionR1 cold-start **Mulberry** subset (`vision_r1_mulberry_sft_full.json`, 197,975)
- `llava_cot` — VisionR1 cold-start **LLaVA-CoT** subset (`vision_r1_llava_cot_full.json`, 63,019)

> ⚠️ **LLaVA-CoT is analysis-only.** Its *text* metrics are computed and shown for comparison, but
> its **images are not downloaded** (`llava_cot_images/` is empty), so it cannot be formatted into a
> trainable file yet. The trainable mix in §10 is **Stage-B + Mulberry**. Download the LLaVA-CoT
> images and re-run the formatter to add it.

Per-record metrics were precomputed by `run_cot_metrics.py` (run in parallel) into
`visionr1_out/<name>_metrics.parquet`; this notebook loads those and visualizes them.
""")

md(r"""## 0. Setup — load precomputed per-record metrics

If a parquet is missing, regenerate it with:
```bash
R=/mnt/data4/shasta/amar.amarjyoti/research_data
S=research/data_scripts/run_cot_metrics.py   # run from repo root
python $S --name stage_b   --path $R/vlm_cot_distill/sft_stageB/train.jsonl              --schema messages
python $S --name mulberry  --path $R/raw/vision_r1_cold/vision_r1_mulberry_sft_full.json --schema conversations
python $S --name llava_cot --path $R/raw/vision_r1_cold/vision_r1_llava_cot_full.json    --schema conversations
```
""")

code(r"""import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# shared lib now lives in research/cot_lib (notebook runs from notebooks/)
sys.path.insert(0, str(Path.cwd().parent / 'research' / 'cot_lib'))
import cot_metrics as cm   # shared feature definitions (same as Stage-B audit)

pd.set_option('display.max_colwidth', 200)
plt.rcParams['figure.dpi'] = 110

OUT_DIR = Path('visionr1_out'); OUT_DIR.mkdir(exist_ok=True)
DATASETS = ['stage_b', 'mulberry', 'llava_cot']
COLORS   = {'stage_b': 'tab:green', 'mulberry': 'tab:blue', 'llava_cot': 'tab:orange'}
LABELS   = {'stage_b': 'Stage-B winners (in training)',
            'mulberry': 'VisionR1 Mulberry', 'llava_cot': 'VisionR1 LLaVA-CoT'}

dfs, summ = {}, {}
for n in DATASETS:
    pq = OUT_DIR / f'{n}_metrics.parquet'
    sj = OUT_DIR / f'{n}_summary.json'
    if pq.exists():
        dfs[n] = pd.read_parquet(pq)
        summ[n] = json.loads(sj.read_text()) if sj.exists() else {}
        print(f'{n:10s} {len(dfs[n]):>8,} traces  ({pq.name})')
    else:
        print(f'{n:10s} MISSING -> run run_cot_metrics.py (see cell above)')
""")

md(r"""## 1. Dataset overview

`stage_b` is the reference (already in training). The two VisionR1 splits are candidates to mix in.
Note the structural difference: only `stage_b` carries a `<grounding>` point block.""")

code(r"""rows = []
for n in DATASETS:
    if n not in dfs: continue
    d = dfs[n]
    rows.append(dict(
        dataset=LABELS[n], n_traces=len(d),
        has_grounding_block=f"{(d.obj_defined>0).mean()*100:.0f}%",
        median_think_words=int(d.thinking_words.median()),
        median_answer_words=int(d.answer_words.median()),
    ))
pd.DataFrame(rows).set_index('dataset')
""")

md(r"""## 2. Reasoning length — the `max_length` budget question

Stage-B winners are long, grounded spatial traces. Mulberry/LLaVA-CoT reasoning length decides
(a) how much they'll dominate a token-weighted mix and (b) whether the current `--max_length 4096`
truncates them. Right panel is the per-dataset thinking-word quantile table for setting the cap.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
allw = pd.concat([dfs[n].thinking_words for n in dfs])
hi = float(np.nanpercentile(allw, 99))
bins = np.linspace(0, hi, 60)
for n in DATASETS:
    if n not in dfs: continue
    axes[0].hist(dfs[n].thinking_words.clip(upper=hi), bins=bins, alpha=0.5,
                 label=f'{LABELS[n]} (med {dfs[n].thinking_words.median():.0f})',
                 color=COLORS[n], density=True)
axes[0].set_xlabel('thinking words'); axes[0].set_ylabel('density')
axes[0].set_title('Reasoning length (density)'); axes[0].legend(fontsize=8)

for n in DATASETS:
    if n not in dfs: continue
    aw = dfs[n].answer_words.clip(upper=dfs[n].answer_words.quantile(0.99))
    axes[1].hist(aw, bins=40, alpha=0.5, label=LABELS[n], color=COLORS[n], density=True)
axes[1].set_xlabel('answer words'); axes[1].set_title('Answer length (density)'); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR / 'lengths_compare.png'); plt.show()

qs = [0.5, 0.9, 0.95, 0.99, 1.0]
qtab = pd.DataFrame({LABELS[n]: dfs[n].thinking_words.quantile(qs) for n in dfs if n in dfs}).round(0)
qtab.index = [f'p{int(q*100)}' for q in qs]
print('thinking-word quantiles (set --max_length above the mix you intend to train):')
qtab
""")

md(r"""## 3. Reasoning structure & self-correction

`sent_count` / `connective_n` = how multi-step the reasoning is; `has_reflection` = self-checking
("wait… let me re-check"). Linear answer-writing scores low on all three.""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for n in DATASETS:
    if n not in dfs: continue
    s = dfs[n].sent_count.clip(upper=dfs[n].sent_count.quantile(0.98))
    axes[0].hist(s, bins=40, alpha=0.5, label=LABELS[n], color=COLORS[n], density=True)
axes[0].set_xlabel('# sentences'); axes[0].set_title('Reasoning step count'); axes[0].legend(fontsize=8)

for n in DATASETS:
    if n not in dfs: continue
    c = dfs[n].connective_n.clip(upper=12)
    axes[1].hist(c, bins=13, alpha=0.5, label=LABELS[n], color=COLORS[n], density=True)
axes[1].set_xlabel('# logical connectives'); axes[1].set_title('Connective density'); axes[1].legend(fontsize=8)

refl = [dfs[n].has_reflection.mean()*100 for n in DATASETS if n in dfs]
axes[2].bar([LABELS[n] for n in DATASETS if n in dfs], refl,
            color=[COLORS[n] for n in DATASETS if n in dfs])
for i, v in enumerate(refl): axes[2].text(i, v, f'{v:.0f}%', ha='center', va='bottom', fontsize=9)
axes[2].set_ylabel('% with self-correction'); axes[2].set_title('Reflection / self-correction rate')
axes[2].tick_params(axis='x', rotation=15)
plt.tight_layout(); plt.savefig(OUT_DIR / 'structure_compare.png'); plt.show()
""")

md(r"""## 4. Language purity & degeneration

Hard SFT defects the answer protocol does not catch: **CJK code-switch** (Qwen leaking Chinese) and
**looping** (high max 4-gram repetition / low type-token ratio).""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
cjk = [(dfs[n].think_cjk_frac > 0).mean()*100 for n in DATASETS if n in dfs]
axes[0].bar([LABELS[n] for n in DATASETS if n in dfs], cjk,
            color=[COLORS[n] for n in DATASETS if n in dfs])
for i, v in enumerate(cjk): axes[0].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
axes[0].set_ylabel('% traces with ANY CJK'); axes[0].set_title('Code-switch (CJK leakage)')
axes[0].tick_params(axis='x', rotation=15)

for n in DATASETS:
    if n not in dfs: continue
    axes[1].hist(dfs[n].rep_4gram.clip(upper=0.4), bins=50, alpha=0.5,
                 label=LABELS[n], color=COLORS[n], density=True)
axes[1].axvline(0.10, color='r', ls='--', label='flag thr')
axes[1].set_xlabel('max 4-gram repetition'); axes[1].set_title('Repetition (loop detector)'); axes[1].legend(fontsize=8)

for n in DATASETS:
    if n not in dfs: continue
    axes[2].hist(dfs[n].ttr.dropna(), bins=40, alpha=0.5, label=LABELS[n], color=COLORS[n], density=True)
axes[2].set_xlabel('type-token ratio'); axes[2].set_title('Lexical diversity'); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR / 'purity_degeneration_compare.png'); plt.show()
""")

md(r"""## 5. Cognitive reasoning behaviors (Stage-B audit §9f, across datasets)

The canonical self-improving-reasoner moves. Heuristic lexical detectors — read as a **relative**
cross-dataset signal, not absolute ground truth. Presence rate = % of traces exhibiting the
behavior. This is the headline "are these traces teaching the reasoning behaviors we want" view.""")

code(r"""beh = cm.COG_NAMES
present = pd.DataFrame({
    LABELS[n]: [(dfs[n][f'cog_{b}'] > 0).mean()*100 for b in beh]
    for n in DATASETS if n in dfs
}, index=beh)
display(present.round(1))

y = np.arange(len(beh)); ncol = len([n for n in DATASETS if n in dfs]); h = 0.8/ncol
fig, ax = plt.subplots(figsize=(11, 5))
for k, n in enumerate([n for n in DATASETS if n in dfs]):
    ax.barh(y + (k-(ncol-1)/2)*h, present[LABELS[n]].values, h, label=LABELS[n], color=COLORS[n])
ax.set_yticks(y); ax.set_yticklabels(beh)
ax.set_xlabel('% of traces with behavior'); ax.legend(fontsize=8)
ax.set_title('Cognitive-behavior presence rate by dataset')
plt.tight_layout(); plt.savefig(OUT_DIR / 'cognitive_behaviors_compare.png'); plt.show()

rich = pd.DataFrame({
    'mean distinct behaviors/trace': [dfs[n].cog_n_distinct.mean() for n in DATASETS if n in dfs],
    '% traces >=3 behaviors':        [(dfs[n].cog_n_distinct>=3).mean()*100 for n in DATASETS if n in dfs],
    '% traces 0 behaviors':          [(dfs[n].cog_n_distinct==0).mean()*100 for n in DATASETS if n in dfs],
}, index=[LABELS[n] for n in DATASETS if n in dfs]).round(2)
rich
""")

md(r"""## 6. Red-flag summary & headline comparison

`flag_no_ground` (grounding ignored) only applies to `stage_b` (the others have no grounding block,
so it is structurally 0). The closing table is the means side-by-side with the ratio vs Stage-B —
the one view for deciding the mix.

> **Caveat — `answer not derived in trace` is weak for multiple-choice.** It checks whether the
> answer *body* string appears in the reasoning. For Stage-B (free-form spatial answers) it is
> meaningful (~13%). For Mulberry/LLaVA-CoT the answers are mostly a single option letter / short
> number, so this flag is near-uninformative there — read it with that in mind, not as a defect rate.""")

code(r"""flags = list(cm.FLAG_LABELS)
ftab = pd.DataFrame({
    LABELS[n]: [dfs[n][f].mean()*100 for f in flags] for n in DATASETS if n in dfs
}, index=[cm.FLAG_LABELS[f] for f in flags]).round(2)
print('Red-flag rates (%):'); display(ftab)

metrics = ['thinking_words', 'answer_words', 'sent_count', 'connective_n',
           'reflect_n', 'rep_4gram', 'ttr', 'cog_n_distinct']
head = pd.DataFrame({LABELS[n]: [dfs[n][m].mean() for m in metrics]
                     for n in DATASETS if n in dfs}, index=metrics)
if 'stage_b' in dfs:
    for n in DATASETS:
        if n in dfs and n != 'stage_b':
            head[f'{n}/stage_b'] = (head[LABELS[n]] / head[LABELS['stage_b']].replace(0, np.nan))
print('\nHeadline means (and ratio vs Stage-B):'); head.round(3)
""")

md(r"""## 7. Re-format Mulberry/VisionR1 into the Stage-B swift schema

`format_mulberry_swift.py` converts each record to the exact ms-swift target the Stage-B file uses:

```json
{"messages": [
   {"role": "system",    "content": "<think>/<answer> protocol (no grounding)"},
   {"role": "user",      "content": "<image>...question..."},
   {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}],
 "images": ["/abs/path"]}
```

It preserves the reasoning verbatim, **drops VisionR1's `Final Answer:` prefix** so the `<answer>`
body matches Stage-B, makes image paths absolute, and drops records whose think/answer/image can't
be recovered. The full files are written under the data root (kept off the repo):

- `…/research_data/sft_mulberry_visionr1/mulberry_visionr1_train.jsonl`
- `…/research_data/sft_mulberry_visionr1/llava_cot_visionr1_train.jsonl`

(Generated by the parallel run below — the cell just reports the result and shows one record.)""")

code(r"""R = Path('/mnt/data4/shasta/amar.amarjyoti/research_data')
MIX_DIR = R / 'sft_mulberry_visionr1'
files = {
    'stage_b'  : R / 'vlm_cot_distill/sft_stageB/train.jsonl',
    'mulberry' : MIX_DIR / 'mulberry_visionr1_train.jsonl',
    'llava_cot': MIX_DIR / 'llava_cot_visionr1_train.jsonl',
}
counts = {}
for n, p in files.items():
    if p.exists():
        c = sum(1 for _ in open(p)); counts[n] = c
        print(f'{n:10s} {c:>8,} rows  {p}')
    else:
        print(f'{n:10s} MISSING -> run format_mulberry_swift.py')

# show one formatted Mulberry record
if files['mulberry'].exists():
    r = json.loads(open(files['mulberry']).readline())
    print('\n--- sample formatted Mulberry record ---')
    print('system   :', r['messages'][0]['content'][:90])
    print('user     :', r['messages'][1]['content'][:90])
    print('assistant:', r['messages'][2]['content'][:160], '...')
    print('images   :', r['images'])
""")

md(r"""## 8. Qualitative inspector — 20 random Mulberry examples

The §1–6 numbers describe the traces in aggregate; this is the eyeball check. 20 random formatted
**Mulberry** records, each showing the **image · question · `<think>` · `<answer>`** exactly as the
student will see them (post-formatting: `Final Answer:` stripped, `<image>` tag removed from the
shown question). Re-run with a different `SEED` to draw a fresh sample.""")

code(r"""import random
from PIL import Image
from IPython.display import display, Markdown

SEED = 0
N_SHOW = 20
src = files['mulberry']
assert src.exists() and counts.get('mulberry', 0) > 0, 'formatted Mulberry file missing/empty'

# reservoir-sample N_SHOW lines without loading the whole 500MB file
rng = random.Random(SEED)
sample = []
with open(src) as f:
    for i, line in enumerate(f):
        if len(sample) < N_SHOW:
            sample.append(line)
        elif (j := rng.randint(0, i)) < N_SHOW:
            sample[j] = line
recs = [json.loads(s) for s in sample]

for k, r in enumerate(recs, 1):
    msgs = {m['role']: m['content'] for m in r['messages']}
    _, think, answer = cm.split_trace(msgs.get('assistant', ''))
    question = msgs.get('user', '').replace('<image>', '').strip()
    img_path = r['images'][0]
    display(Markdown(f"### {k}. `{Path(img_path).name}`"))
    try:
        with Image.open(img_path) as im:
            fig, ax = plt.subplots(figsize=(4.5, 3)); ax.imshow(im); ax.axis('off'); plt.show()
    except Exception as e:
        print(f'(image load failed: {e})')
    body = (f"**Question:** {question[:700]}\n\n"
            f"**Think ({cm._wc(think)} words):** {think[:1200]}{'…' if len(think) > 1200 else ''}\n\n"
            f"**Answer:** `{answer.strip()[:200]}`")
    display(Markdown(body))
""")

md(r"""## 9. Mulberry subtask composition — what you'd be adding to SURDS

Mulberry is a **mixture of source datasets**, encoded in the image path as
`…/mulberry_images/<SOURCE>/images/<file>` (e.g. `geoqa_plus`, `chartqa`, `docvqa`). That `<SOURCE>`
is the natural "subtask". This section counts them so you can decide *which* subtasks to add to the
SURDS SFT — your stated goal is to measure how adding different subtasks affects SURDS performance,
and not all of Mulberry is equally relevant.

Two views:
1. **Per-subtask** (case-variants like `DocVQA`/`docvqa` merged to one canonical name).
2. **By domain group** — a heuristic bucketing for the transfer question. SURDS is *spatial-relation
   geometry on driving scenes*, so **`geometry_math`** and the spatial-CLEVR items are the closest
   relatives; chart/doc/text/general VQA are progressively farther. Use this to design ablations
   (e.g. SURDS + geometry-only vs SURDS + all-Mulberry).""")

code(r"""import re
from collections import Counter

def subtask_of(img_path):
    m = re.search(r'mulberry_images/([^/]+)/', img_path)
    return m.group(1) if m else '(unknown)'

def canon(name):
    # merge case-variants / cosmetic suffixes: DocVQA==docvqa, ai2d_images==AI2D, A-OKVQA==aokvqa
    return name.lower().replace('-', '').replace('.', '').replace('_images', '').replace('_', '')

# heuristic domain buckets for the SURDS-transfer question (closest relative first)
DOMAIN = {
    'geometry_math':    {'geoqaplus', 'geo3k', 'unigeo', 'geos', 'mathvision',
                         'clevrmath', 'superclevr', 'tabmwp'},
    'chart_plot':       {'figureqa', 'plotqa', 'chartqa', 'dvqa', 'lrvchart', 'mapqa'},
    'science_diagram':  {'ai2d', 'scienceqa', 'sqa', 'tqa', 'iconqa', 'pmcvqa'},
    'doc_text':         {'docvqa', 'infovqa', 'textvqa', 'vqaas'},
    'general_vqa':      {'cauldron', 'aokvqa', 'vqa20', 'vizwiz', 'vqarad'},
}
_canon2domain = {c: d for d, cs in DOMAIN.items() for c in cs}

# fast scan: pull the subtask straight from each line (no full json parse)
src = files['mulberry']
raw_ct, canon_ct, dom_ct = Counter(), Counter(), Counter()
with open(src) as f:
    for line in f:
        m = re.search(r'mulberry_images/([^/]+)/', line)
        s = m.group(1) if m else '(unknown)'
        c = canon(s)
        raw_ct[s] += 1; canon_ct[c] += 1
        dom_ct[_canon2domain.get(c, 'other')] += 1
N = sum(canon_ct.values())
print(f'Mulberry: {N:,} records | {len(raw_ct)} raw source folders | {len(canon_ct)} canonical subtasks')

sub_df = (pd.DataFrame({'records': canon_ct})
          .assign(pct=lambda d: (100*d.records/N).round(2),
                  domain=lambda d: [_canon2domain.get(i, 'other') for i in d.index])
          .sort_values('records', ascending=False))
sub_df.index.name = 'subtask'
display(sub_df)
sub_df.to_csv(OUT_DIR / 'mulberry_subtask_counts.csv')

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
s = sub_df.records.sort_values()
dom_color = {'geometry_math': 'tab:red', 'chart_plot': 'tab:blue', 'science_diagram': 'tab:green',
             'doc_text': 'tab:orange', 'general_vqa': 'gray', 'other': 'black'}
axes[0].barh(s.index, s.values, color=[dom_color[_canon2domain.get(i, 'other')] for i in s.index])
for i, v in enumerate(s.values):
    axes[0].text(v, i, f' {v:,}', va='center', fontsize=7)
axes[0].set_xlabel('# records'); axes[0].set_title('Mulberry subtasks (color = domain group)')

dom_s = pd.Series(dom_ct).sort_values()
axes[1].barh(dom_s.index, dom_s.values, color=[dom_color[d] for d in dom_s.index])
for i, (d, v) in enumerate(zip(dom_s.index, dom_s.values)):
    axes[1].text(v, i, f' {v:,} ({100*v/N:.0f}%)', va='center', fontsize=9)
axes[1].set_xlabel('# records'); axes[1].set_title('By domain group (SURDS-transfer view)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'mulberry_subtasks.png'); plt.show()

print('\nDomain-group totals (closest relative to SURDS first):')
for d in ['geometry_math', 'chart_plot', 'science_diagram', 'doc_text', 'general_vqa', 'other']:
    if d in dom_ct:
        print(f'  {d:16s} {dom_ct[d]:8,}  ({100*dom_ct[d]/N:4.1f}%)')
""")

md(r"""### Building a subtask-filtered subset for the ablation

To test "SURDS + <some subtasks>", filter the formatted Mulberry file to the subtasks/domains you
want and point the §10 mix at the subset. The helper below writes a filtered JSONL (kept in the data
root, off the repo); pass either canonical subtask names or domain-group names.""")

code(r"""def write_subtask_subset(selection, out_name, limit_per_subtask=0):
    # selection: set of canonical subtask names and/or DOMAIN group names.
    # Writes matching records to MIX_DIR/out_name; returns (path, kept_count).
    wanted = set()
    for s in selection:
        wanted |= DOMAIN[s] if s in DOMAIN else {canon(s)}
    out = MIX_DIR / out_name
    per = Counter(); kept = 0
    with open(files['mulberry']) as fi, open(out, 'w') as fo:
        for line in fi:
            m = re.search(r'mulberry_images/([^/]+)/', line)
            c = canon(m.group(1)) if m else '(unknown)'
            if c in wanted and (not limit_per_subtask or per[c] < limit_per_subtask):
                fo.write(line); per[c] += 1; kept += 1
    print(f'wrote {kept:,} records -> {out}')
    return out, kept

# Example (uncomment to materialize a geometry/math-only subset for a SURDS + geometry ablation):
# write_subtask_subset({'geometry_math'}, 'mulberry_geometry_math_train.jsonl')
print('helper ready: write_subtask_subset({"geometry_math"}, "mulberry_geometry_math_train.jsonl")')
""")

md(r"""## 10. Train both, with per-dataset weights

ms-swift mixes multiple `--dataset` entries and the **`#N` suffix sets how many rows to sample from
each** — that is the weighting lever (sampling-without-replacement up to the file size; above it,
it upsamples). So "weights" = the relative `#N` counts.

Set a target epoch size and a weight per dataset; the cell computes the `#N` for each and prints a
ready `swift sft --dataset …` block you can drop into a copy of `pretrain_model_14.sh`.""")

code(r"""# --- choose your mix here ---
WEIGHTS = {        # relative emphasis; need not sum to 1
    'stage_b'  : 1.0,   # the SURDS spatial winners (grounded, in training now)
    'mulberry' : 1.0,   # general chart/table/science VQA CoT
    'llava_cot': 0.0,   # NOTE: LLaVA-CoT images are not downloaded -> not trainable yet (keep 0)
}
TARGET_TOTAL = 60_000   # total sampled rows per epoch across the mix

# only datasets with a non-empty formatted file on disk can be sampled
active = {n: w for n, w in WEIGHTS.items() if counts.get(n, 0) > 0 and w > 0}
wsum = sum(active.values())
plan = []
for n, w in active.items():
    want = int(round(TARGET_TOTAL * w / wsum))
    avail = counts[n]
    note = '' if want <= avail else f'(upsamples x{want/avail:.2f})'
    plan.append((n, w, avail, want, note))
plan_df = pd.DataFrame(plan, columns=['dataset', 'weight', 'available', 'sample_N', 'note'])
display(plan_df)

ds_args = '  '.join(f'{files[n]}#{want}' for n, _, _, want, _ in plan)
# recommend max_length from the p95 of the chosen mix
p95 = max(int(dfs[n].thinking_words.quantile(0.95)) for n, *_ in plan if n in dfs)
rec_maxlen = int(min(8192, max(4096, (p95 + 200) // 256 * 256 + 256)))

print('\n# ---- paste into a copy of pretrain_model_14.sh (replace the --dataset line) ----')
print(f'#   weighted mix: {", ".join(f"{n}={w}" for n,w,*_ in plan)}  | target {TARGET_TOTAL:,} rows/epoch')
print('swift sft \\')
print(f'    --dataset {ds_args} \\')
print(f'    --val_dataset {files["stage_b"].parent / "val_1k.jsonl"} \\')
print(f'    --max_length {rec_maxlen} \\')
print('    --dataset_shuffle true \\')
print('    # ... (keep all other pretrain_model_14.sh flags: --tuner_type full, zero2, etc.)')

cfg = {'weights': WEIGHTS, 'target_total': TARGET_TOTAL,
       'plan': {n: {'weight': w, 'available': a, 'sample_N': s} for n, w, a, s, _ in plan},
       'recommended_max_length': rec_maxlen,
       'dataset_arg': ds_args}
(OUT_DIR / 'mix_config.json').write_text(json.dumps(cfg, indent=2))
print('\nwrote', OUT_DIR / 'mix_config.json')
""")

md(r"""### Notes on the mix

- **`#N` is the weight.** Equal `#N` = equal row counts regardless of file size. Because Mulberry
  (197k) and LLaVA-CoT (63k) dwarf the 20k Stage-B set, *without* `#N` caps they would swamp the
  spatial signal — always cap explicitly.
- **Token weighting vs row weighting.** `#N` weights by *rows*. If you care about *tokens*, note from
  §2 that the datasets have different median reasoning lengths — multiply your intended token share by
  `median_think_words` to back out the row counts.
- **`--max_length`** is recommended from the chosen mix's p95 reasoning length so long traces aren't
  silently truncated; raise it only if the p99 tail matters to you (costs memory).
- **System prompt mismatch is intentional.** Stage-B asks for a `<grounding>` block; Mulberry's
  formatted prompt does not (it has no point annotations). Both still emit `<think>/<answer>`, so the
  answer protocol the student learns is consistent across the mix.
- Validation stays on `sft_stageB/val_1k.jsonl` (the SURDS eval) so the metric is comparable to the
  runs already in flight; add a Mulberry val slice if you want to track general-VQA loss too.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
_OUT_NB = Path(__file__).resolve().parents[2] / 'notebooks' / 'visionr1_mulberry_cot.ipynb'
with open(_OUT_NB, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', _OUT_NB, 'with', len(cells), 'cells')

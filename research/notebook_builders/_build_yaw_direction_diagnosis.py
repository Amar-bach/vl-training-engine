"""Builds notebooks/yaw_direction_diagnosis.ipynb — a focused look at the SURDS
`yaw` / facing-direction subtask: what the task actually asks, 10 worked examples
(image + ground truth + real model rollouts), and the full 8x8 confusion matrix.

All numbers/paths come from a single committed data file produced by
  research/notebook_builders/_prep_yaw_direction_data.py
  -> notebooks/data/yaw_direction_data.json
(the prep joins the GRPO L2-direct rollouts back to their source images). The
notebook only LOADS that JSON, opens the referenced images, and plots.

Run:  python research/notebook_builders/_build_yaw_direction_diagnosis.py
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.splitlines(keepends=True)})


# ===========================================================================
md(r"""# The SURDS `yaw` subtask — what it asks, and where the model fails

This notebook isolates the **facing-direction (`yaw`)** subtask, which dominates the flat
GRPO accuracy curve (see `grpo_accuracy_stall_diagnosis.ipynb`). It answers three things:

1. **What is the task actually asking?** (the problem statement, in plain terms)
2. **10 random worked examples** — the real image, the ground-truth answer, and the model's
   own step-by-step rollouts.
3. **The full 8×8 confusion matrix** (gold compass direction vs. predicted), plus a
   camera-relative view that tests *why* the model is wrong.

All evidence is from the run's own per-rollout `completions.jsonl` (`A_grpo_v3` L2-direct,
SLURM 1064116), joined back to source images via `_prep_yaw_direction_data.py`.

---

## 0 · What the task asks (read this first)

Every `yaw` prompt has the same shape:

> *Task: identify the direction the **specified object** is facing. **The camera in the image
> is facing `<North|South>`**, analyse the object's orientation relative to that reference.*
> *Question: "Which direction is the `<the white car>` facing?"*  Options: 8 compass points.

Two things are easy to misread, and they're the crux of your question:

- **The answer is the *object's* orientation, not the camera's.** The camera-facing line is
  only a **reference frame** so the model can convert "which way the car points *in the image*"
  into a **world compass bearing**. Asking "which way is the white car facing" with "camera
  faces South" given is: *given the camera looks South, the car's heading in absolute compass
  terms is …?*
- **The camera-facing direction is not fixed** — in this band it is ~50 % North, ~50 % South
  (printed below). So a raw count of "the model said South a lot" mixes two different reference
  frames and can be misleading; the confusion matrix and the **camera-relative** view in §3
  disentangle it.""")

# ---------------------------------------------------------------------------
code(r"""import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Markdown, display

EVAL_DIR = (Path.cwd().parent / 'research' / 'eval').resolve()
sys.path.insert(0, str(EVAL_DIR))
import plotstyle as ps
ps.set_pub_style()

FIG_DIR = Path('figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = json.loads((Path('data') / 'yaw_direction_data.json').read_text())
META = DATA['meta']
ORDER = META['order']                     # [N, NE, E, SE, S, SW, W, NW]

print('run                  :', META['run'])
print('yaw prompt-groups    :', META['n_prompt_groups'])
print('rollout samples       :', META['n_rollout_samples'])
print('camera-facing split   :', META['camera_facing_distribution'])
print('(answer = OBJECT world-compass heading; camera-facing is only the reference frame)')""")

# ===========================================================================
md(r"""## 1 · Ten random worked examples

Each panel below is a real held-in `yaw` prompt: the **image**, the stated **camera-facing**,
the **object** being asked about, the **ground-truth** compass answer, and up to **3 of the
model's own 16 rollouts** (a correct one where it exists, plus wrong ones for contrast). Read
the rollouts — they show the model narrating a reference-frame conversion and then committing
to a *mirror* of the right answer.

> Note: the panels render the actual `.webp` frames from `research_data/raw/surds`. The
> object asked about is named in the title; the model is *not* told a bounding box.""")

code(r"""def show_example(ex, idx):
    correct_rolls = [r for r in ex['rollouts'] if r['correct']]
    hit = '✓ has a correct rollout' if correct_rolls else '✗ all 16 rollouts wrong'
    # image
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    try:
        img = Image.open(ex['image']).convert('RGB')
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f'[image unavailable]\n{e}', ha='center', va='center')
    ax.axis('off')
    ax.set_title(f"Example {idx} — object: «{ex['object']}»\n"
                 f"camera faces {ex['camera'].upper()}  |  "
                 f"GROUND TRUTH: {ex['gold']}  |  {hit}",
                 fontsize=10.5)
    plt.show(); plt.close(fig)
    # rollouts as markdown
    lines = []
    for j, r in enumerate(ex['rollouts'], 1):
        mark = '✅ CORRECT' if r['correct'] else '❌ wrong'
        lines.append(f"**Rollout {j} — predicted `{r['pred']}` ({mark})**  ")
        lines.append('> ' + r['think'].replace(chr(10), ' ').strip())
        lines.append('')
    display(Markdown('\n'.join(lines)))

for i, ex in enumerate(DATA['examples'], 1):
    show_example(ex, i)""")

# ===========================================================================
md(r"""## 2 · The 8×8 confusion matrix (world compass frame)

Row = ground-truth compass direction, column = the model's predicted direction, **row-
normalised to %** (each row sums to 100 across the 8 cells + an "unparsed" sliver). This is
the headline figure: it shows the errors are **structured mirror-flips**, not random.""")

code(r"""W = np.array(DATA['world_confusion'], dtype=float)        # [gold][pred] counts
U = np.array(DATA['world_unparsed_per_gold'], dtype=float)
row_tot = W.sum(1) + U
Wn = 100 * W / row_tot[:, None]                            # row-normalised %

fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(Wn, cmap='magma', vmin=0, vmax=70, aspect='equal')
ax.set_xticks(range(8)); ax.set_xticklabels(ORDER)
ax.set_yticks(range(8)); ax.set_yticklabels(ORDER)
ax.set_xlabel('PREDICTED direction'); ax.set_ylabel('GROUND-TRUTH direction')
ax.set_title('Yaw confusion (row-normalised %)  —  diagonal = correct')
for i in range(8):
    for j in range(8):
        v = Wn[i, j]
        if v >= 1:
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    color=ps.contrast_text_color(v, 0, 70, 'magma'), fontsize=9)
# outline the correct diagonal
for i in range(8):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor='#39FF14', lw=1.8))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='% of gold-class rollouts')
paths = ps.savefig(fig, 'yaw_fig1_confusion_matrix')
plt.show(); plt.close(fig)
print('saved:', paths['png'])""")

md(r"""**Read the off-diagonal mass — it is not spread randomly, it sits on the *mirror* cells:**

- **East ↔ West (horizontal flip) is the single largest error.** Gold `E` → predicted `W`
  **50 %**; gold `W` → predicted `E` **37 %**. The model gets the *axis* (left–right) but the
  *sign* wrong.
- **North-diagonals collapse onto South-diagonals (vertical flip).** Gold `NE` → `SE` **65 %**;
  gold `NW` → `SW` **46 %**. This is why `NE` (7.7 % acc) and `NW` (10.7 %) are the worst classes.
- **Cardinals N/S also leak into each other** (`N`→`S` 23 %, `S`→`N` 11 %).

The pattern is a **front/back + left/right mirror ambiguity**: the model recovers the line the
object is oriented along, but not *which of the two opposite ways* along it. That is the classic
hard part of monocular orientation — and it is a **perception** failure, not something a reward
tweak fixes.""")

# ===========================================================================
md(r"""## 3 · Per-class accuracy, and the camera-relative test

**Left:** accuracy per gold class — cardinals N/S survive (~54 %), but E/W and all four
diagonals are at or near chance. **Right:** the *camera-relative* histogram — every prediction
re-expressed as **(object heading − camera heading)**, which cancels the rotating reference
frame. If the model simply "always faced the viewer" we'd see a single spike at **+180°
(toward camera)**; instead +180° is *under*-predicted vs. ground truth and the model leaks mass
into the camera-left quadrant (+270°/+315°). So the bias is a **mirror ambiguity**, not a fixed
"point south / face the camera" default.""")

code(r"""pc = DATA['per_class_acc']
accs = [r['acc'] for r in pc]
ns = [r['total'] for r in pc]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

# left: per-class accuracy
colors = ['#2a9d8f' if c in ('N', 'S') else '#e76f51' for c in ORDER]
a1.bar(range(8), accs, color=colors)
a1.axhline(12.5, ls='--', color='0.4', lw=1.2, label='8-way chance (12.5%)')
a1.set_xticks(range(8)); a1.set_xticklabels(ORDER)
a1.set_ylabel('binary accuracy (%)'); a1.set_title('Per-class accuracy')
for i, (v, n) in enumerate(zip(accs, ns)):
    a1.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=9)
a1.legend(fontsize=9)

# right: camera-relative gold vs pred
gp = np.array(DATA['rel_gold_hist'], float); pp = np.array(DATA['rel_pred_hist'], float)
gp = 100 * gp / gp.sum(); pp = 100 * pp / pp.sum()
xr = np.arange(8); w = 0.4
rel_tick = ['+0°\n(faces away)', '+45°', '+90°\n(cam-right)', '+135°',
            '+180°\n(faces viewer)', '+225°', '+270°\n(cam-left)', '+315°']
a2.bar(xr - w/2, gp, w, label='ground truth', color='#264653')
a2.bar(xr + w/2, pp, w, label='model prediction', color='#e9c46a')
a2.set_xticks(xr); a2.set_xticklabels(rel_tick, fontsize=7.5)
a2.set_ylabel('% of rollouts'); a2.set_title('Camera-relative heading (object − camera)')
a2.legend(fontsize=9)
fig.tight_layout()
paths = ps.savefig(fig, 'yaw_fig2_perclass_and_relative')
plt.show(); plt.close(fig)
print('saved:', paths['png'])""")

# ===========================================================================
md(r"""## 4 · So, to answer the original question

> *"The model consistently predicts the angle as South while the GT assumes North?"*

**Close in spirit, but the confusion matrix sharpens it — and corrects two points:**

1. **The GT is not "North."** The camera-facing reference is ~50/50 North/South, and the gold
   answers are spread across all 8 compass classes. The question asks for the **object's** world
   heading; the camera-facing is only the conversion reference.
2. **The model is not "always South."** The errors are **axis mirror-flips**: East↔West (the
   biggest, ~50 %), and North-diagonals→South-diagonals (NE→SE 65 %, NW→SW 46 %). There *is* an
   asymmetric downward pull (north flips to south more than south flips to north), which is what
   the earlier marginal-only view read as a "south bias" — but the mechanism is a
   **front/back + left/right ambiguity**, not a fixed south default.

**Why it matters for training.** A mirror ambiguity means the SFT model genuinely cannot resolve
the object's *sense* of orientation from a single frame — it lacks the perception, not the
reward signal. GRPO can only sharpen capability that exists, so it cannot fix yaw; worse, the
"correct" yaw rollouts it would reinforce are often the lucky side of a coin-flip, so RFT on them
risks **entrenching the flip**. The fix lives in the *perception* track (grounding-SFT after
auditing whether the 235B teacher shares the flip; crop-zoom tool-use; possibly the Instruct
base), not in any reward/KL/lr knob. See the ranked plan in `grpo_accuracy_stall_diagnosis.ipynb`.""")

# ===========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT = Path(__file__).resolve().parents[2] / "notebooks" / "yaw_direction_diagnosis.ipynb"
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

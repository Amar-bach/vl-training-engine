"""plotstyle.py — publication-quality matplotlib styling for the SURDS×Mulberry
VLM-ablation analysis (NeurIPS/CVPR camera-ready figures).

Importable from any notebook/script:

    import plotstyle as ps
    ps.set_pub_style()
    color = ps.ARM_COLORS['geometry_math']
    paths = ps.savefig(fig, 'fig1_overall_accuracy', outdir='figures')

Design goals
------------
* Offline-safe fonts only (DejaVu Sans / STIXGeneral ship with matplotlib — no
  network download, no system-font dependency).
* Editable vector text (pdf.fonttype = ps.fonttype = 42 / TrueType) so most
  venues accept the PDF/SVG directly.
* A FIXED, colorblind-safe arm -> color palette so a given model is the SAME
  color in every figure across the whole notebook.
* Seeded, reproducible statistics helpers (Wilson CI for proportions, seeded
  bootstrap CI for arbitrary statistics) that provide the error bars.

Nothing here touches the global numpy random state: bootstrap takes an explicit
seed and uses a local ``numpy.random.Generator``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. rcParams — camera-ready
# ---------------------------------------------------------------------------
def set_pub_style(base_fontsize: float = 10.5):
    """Set matplotlib rcParams for conference/journal camera-ready output.

    Uses only offline-available fonts (DejaVu Sans for text, STIX for math),
    thin despined spines, a light grid, 300-DPI tight vector saves, and
    TrueType (type-42) embedded fonts so text stays editable in PDF/SVG/EPS.
    """
    # Resolve an available sans family without requiring any network font.
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    preferred = ["DejaVu Sans", "Helvetica", "Arial", "STIXGeneral", "sans-serif"]
    sans = [f for f in preferred if f in available] or ["sans-serif"]

    rc = {
        # --- fonts (offline-safe) ---
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "font.size": base_fontsize,
        "mathtext.fontset": "stix",          # STIX ships with matplotlib (offline)
        "axes.unicode_minus": True,
        # --- element sizes tuned around the base size ---
        "axes.titlesize": base_fontsize + 1.0,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1.0,
        "ytick.labelsize": base_fontsize - 1.0,
        "legend.fontsize": base_fontsize - 1.0,
        "legend.title_fontsize": base_fontsize - 0.5,
        "figure.titlesize": base_fontsize + 2.0,
        # --- spines: thin, top/right despined ---
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # --- light grid only on the y axis where it aids reading ---
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        # --- lines / markers ---
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        # --- figure / saving ---
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "savefig.transparent": False,        # white bg by default; per-call override below
        # --- EDITABLE vector text (required by most venues) ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",              # keep SVG text as real <text>, not paths
    }
    mpl.rcParams.update(rc)
    return rc


# ---------------------------------------------------------------------------
# 2. Fixed colorblind-safe palette (Okabe-Ito) + colormaps
# ---------------------------------------------------------------------------
# Okabe-Ito qualitative palette — colorblind-safe (deuteranopia/protanopia).
_OKABE_ITO = {
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Stable arm -> color. The zero-shot (no-SFT) baselines in neutral grays — with the
# Δ-reference ``orig_thinking`` in the darkest gray; SURDS-SFT arms in lighter grays;
# Mulberry arms in distinct colorblind-safe hues; ``full`` in a strong dark purple.
ARM_COLORS = {
    "orig_instruct":     "#9e9e9e",        # medium gray (zero-shot 8B baseline)
    "orig_thinking":     "#4d4d4d",        # darkest gray (zero-shot 8B baseline — Δ reference)
    "teacher_32b":       "#8c6d31",        # bronze (zero-shot 32B teacher, larger ref)
    "teacher_235b":      "#000000",        # black (zero-shot 235B teacher — ceiling)
    "baseline_instruct": "#d0d0d0",        # very light gray (SURDS-SFT, instruct)
    "baseline_thinking": "#7a7a7a",        # light/medium gray (SURDS-SFT, thinking)
    "geometry_math":     _OKABE_ITO["blue"],          # #0072B2
    "chart_plot":        _OKABE_ITO["orange"],        # #E69F00
    "science_diagram":   _OKABE_ITO["bluish_green"],  # #009E73
    "doc_text":          _OKABE_ITO["vermillion"],    # #D55E00
    "general_vqa":       _OKABE_ITO["sky_blue"],      # #56B4E9
    "full":              _OKABE_ITO["reddish_purple"],# #CC79A7
}

# Consistent display order: baselines first, then Mulberry arms, then full.
ARM_ORDER = [
    "orig_instruct",
    "orig_thinking",
    "teacher_32b",
    "teacher_235b",
    "baseline_instruct",
    "baseline_thinking",
    "geometry_math",
    "chart_plot",
    "science_diagram",
    "doc_text",
    "general_vqa",
    "full",
]

# Sequential colormap for accuracy heatmaps (cividis is colorblind-safe).
SEQUENTIAL_CMAP = "cividis"
# Diverging colormap for Δ-vs-baseline heatmaps (centered at 0 via TwoSlopeNorm).
# RdBu (not _r): high/positive Δ -> blue (helps), low/negative Δ -> red (hurts),
# matching the figure legend "blue = helps, red = hurts".
DIVERGING_CMAP = "RdBu"


def arm_color(arm: str) -> str:
    """Stable color for an arm; falls back to a neutral gray for unknown arms."""
    return ARM_COLORS.get(arm, "#777777")


def order_arms(arms) -> list:
    """Return ``arms`` in the canonical display order, with any extras appended."""
    arms = list(arms)
    ordered = [a for a in ARM_ORDER if a in arms]
    ordered += [a for a in arms if a not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# 3. savefig — PDF + SVG + PNG (300 DPI) into figures/
# ---------------------------------------------------------------------------
def savefig(fig, name: str, outdir="figures", transparent: bool = False,
            dpi: int = 300, formats=("pdf", "svg", "png")) -> dict:
    """Save ``fig`` as vector (PDF, SVG) AND raster (PNG, 300 DPI) into ``outdir``.

    Returns a dict {format: absolute_path}. ``outdir`` is created if missing.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=dpi, bbox_inches="tight",
                    transparent=transparent)
        paths[fmt] = str(p.resolve())
    return paths


# ---------------------------------------------------------------------------
# 4. Statistics helpers (error bars)
# ---------------------------------------------------------------------------
def wilson_ci(n_success, n_total, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Returns ``(p_hat, lo, hi)``. Robust for small n and proportions near 0/1,
    which is why it is preferred over the normal-approximation interval for
    accuracy bars. ``z=1.96`` -> 95% CI.
    """
    n = float(n_total)
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = float(n_success) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (p, lo, hi)


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 95,
                 statistic=np.mean, seed: int = 0):
    """Seeded bootstrap confidence interval for ``statistic`` of ``values``.

    Uses a LOCAL ``numpy.random.Generator(seed)`` — never the global RNG — so
    results are fully reproducible. Returns ``(point, lo, hi)`` where ``point``
    is the statistic on the observed data and ``lo/hi`` are the percentile-CI
    bounds. NaNs in ``values`` are dropped.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(v))
    if v.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    n = v.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(v[idx], axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(boot, alpha))
    hi = float(np.percentile(boot, 100.0 - alpha))
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# 5. Pretty labels + ordering
# ---------------------------------------------------------------------------
_ARM_PRETTY = {
    "orig_instruct":     "Qwen3-VL-8B-Instruct (zero-shot baseline)",
    "orig_thinking":     "Qwen3-VL-8B-Thinking (zero-shot baseline)",
    "teacher_32b":       "Qwen3-VL-32B-Thinking (zero-shot teacher)",
    "teacher_235b":      "Qwen3-VL-235B-A22B-Thinking (zero-shot teacher)",
    "baseline_instruct": "SURDS-SFT (Instruct)",
    "baseline_thinking": "SURDS-SFT (Thinking)",
    "geometry_math":     "+ Geometry/Math",
    "chart_plot":        "+ Chart/Plot",
    "science_diagram":   "+ Science Diagram",
    "doc_text":          "+ Doc/Text",
    "general_vqa":       "+ General VQA",
    "full":              "+ Full Mulberry",
}

_TEMPLATE_PRETTY = {
    "lr":       "Left–Right",          # en-dash
    "fb":       "Front–Back",
    "distance": "Distance Comp.",
    "yaw":      "Yaw Direction",
    "xy2d":     "2D Localization",
    "depth":    "Depth Estimation",
}

# Spelled-out names for tables / LaTeX (HARD user rule: no cryptic abbreviations).
_TEMPLATE_FULL = {
    "lr":       "Left-vs-Right discrimination",
    "fb":       "Front-vs-Back discrimination",
    "distance": "Distance comparison",
    "yaw":      "Yaw (heading) direction",
    "xy2d":     "2D image localization",
    "depth":    "Depth estimation",
}

TEMPLATE_ORDER = ["lr", "distance", "fb", "yaw", "xy2d", "depth"]


def pretty_arm(arm: str) -> str:
    """Human-readable model label, e.g. 'baseline_thinking' -> 'SURDS-SFT (Thinking)'."""
    return _ARM_PRETTY.get(arm, arm.replace("_", " ").title())


def pretty_template(t: str) -> str:
    """Human-readable sub-skill label, e.g. 'lr' -> 'Left–Right'."""
    return _TEMPLATE_PRETTY.get(t, t)


def full_template(t: str) -> str:
    """Fully spelled-out sub-skill name for tables/LaTeX."""
    return _TEMPLATE_FULL.get(t, t)


def order_templates(templates) -> list:
    """Return template slugs in canonical order, extras appended."""
    templates = list(templates)
    ordered = [t for t in TEMPLATE_ORDER if t in templates]
    ordered += [t for t in templates if t not in ordered]
    return ordered


def pct(x, decimals: int = 1) -> str:
    """Format a 0–1 fraction as a percentage string, e.g. 0.62 -> '62.0%'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def signed_pts(x, decimals: int = 1) -> str:
    """Format a 0–1 delta as signed percentage points, e.g. 0.031 -> '+3.1'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.{decimals}f}"


def contrast_text_color(value, vmin, vmax, cmap_name=SEQUENTIAL_CMAP) -> str:
    """Pick black/white annotation text for a heatmap cell, by luminance of the
    cell's mapped color. Auto-contrast so values are readable on any cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#222222"
    norm = (float(value) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    norm = min(1.0, max(0.0, norm))
    r, g, b, _ = plt.get_cmap(cmap_name)(norm)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"

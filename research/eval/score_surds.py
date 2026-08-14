"""
score_surds.py — scoring API for the SURDS spatial-reasoning eval (val_1k).

SURDS template families (canonical slugs, matching the source QA file
`_train_qa_for_cot.jsonl` `template_type` field):

    slug        kind          gold-answer shape                       example
    ----------- ------------- ----------------------------------------------------------
    lr          categorical   object label / "Almost the same"        "The black truck"
    distance    categorical   object label / "Almost the same"        "The silver car"
    fb          categorical   "Yes" / "No"                            "No"
    yaw         categorical   one of 8 compass directions             "Southeast"
    xy2d        continuous    a 2-D point in Qwen 0-1000 norm coords  "[946, 574]"
    depth       continuous    a metric range "Between X .. and Y .."  "Between 5 meters and 11 meters"

Scoring API
-----------
    parse_answer(text)                          -> str | None
    get_image_wh(path)                          -> (W, H)
    score_one(pred, gold, template_type, image_wh=None) -> dict

CONTINUOUS-EVAL PROVENANCE
--------------------------
The xy2d / depth abs-diff + tolerance + (W,H)/1000 rescale logic is ported
FAITHFULLY from the vetted, corrected eval in
    notebooks/audit_stage_b_winners.ipynb  (cell 11, "Recompute answer_correctness
    for CONTINUOUS templates")
which is the same correction tracked in MEMORY (continuous-answer-eval: xy2d/depth
need abs-diff+tolerance, not exact match; xy2d coords are 0-1000 normalised).

IMPORTANT coord-system note — the recurring SURDS xy2d frame trap (read this):
  * The PREDICTION (Qwen3-VL student AND the 235B teacher) is ALWAYS 0-1000 NORMALISED.
  * The GOLD frame DIFFERS BY DATASET — this is the trap:
      - PIXELS (1600x900): GRPO curriculum L{1,2,3}.jsonl, source QA
        `_train_qa_for_cot.jsonl`, teacher pool `cot_*_grounding.jsonl` `gt_answer`.
      - 0-1000 NORMALISED: SFT `sft_stageB/train.jsonl`, val_1k, heldout eval
        (`<answer>` = teacher distilled trace), val_meta.parquet (extracted from those).
  Callers MUST declare the gold frame via `gold_space=` ('pixels' default, or 'norm').
  score_one then reconciles BOTH operands into ONE pixel space (the native SURDS
  1600x900 frame) before the L2 compare, so 50 px tolerance is always meaningful:
      pred_px = pred * (W,H)/1000                          (pred is always 0-1000)
      gold_px = gold              if gold_space=='pixels'
      gold_px = gold * (W,H)/1000 if gold_space=='norm'
  Defensive auto-detect: if gold_space=='norm' but a gold coord exceeds 1000 it is
  actually pixels (and vice-versa is impossible), so the obviously-pixel value is used
  as-is rather than rescaled again — this stops the silent norm-vs-pixel corruption
  that previously made teacher xy2d read 0.8% (true 76%) and broke the val_1k ablation.
  Full rule: repo CLAUDE.md "SURDS xy2d coordinate frames".

TOLERANCES (ported from CONT_TOL in the audit notebook; see module docstring at
bottom for the rationale):
    xy2d : 50 px  (on the native 1600x900 nuScenes frame -> ~2.7% of the image
                   diagonal of 1835 px). If image_wh is None we score in normalised
                   space with the equivalent normalised tolerance NORM_XY_TOL.
    depth: 4 m absolute on the range MIDPOINT. We additionally accept a prediction
           whose midpoint falls inside the gold range (the gold IS a ~7 m-wide
           range), which is a strict superset-friendly relaxation; either condition
           marks it correct.
"""

import math
import re

# ----------------------------------------------------------------------------
# Tolerances (ported from audit_stage_b_winners.ipynb cell 11: CONT_TOL)
# ----------------------------------------------------------------------------
XY2D_TOL_PX = 50.0      # pixels, on the native frame
DEPTH_TOL_M = 4.0       # metres, abs-diff on range midpoints
# Equivalent normalised xy2d tolerance when image_wh is unknown.
# 50 px / hypot(1600,900) * 1000(diag-norm) is awkward; instead we express the
# px tol as a fraction of the nuScenes diagonal and apply the SAME fraction to the
# normalised 0-1000 diagonal (hypot(1000,1000)=1414.2). 50/1835.3 * 1414.2 ~= 38.5
NORM_XY_TOL = 50.0 / math.hypot(1600.0, 900.0) * math.hypot(1000.0, 1000.0)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

# Categorical template families.
CATEGORICAL_TEMPLATES = {"lr", "distance", "fb", "yaw"}
CONTINUOUS_TEMPLATES = {"xy2d", "depth"}


# ----------------------------------------------------------------------------
# Answer extraction
# ----------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def parse_answer(text):
    """Extract the <answer>...</answer> content.

    Robust to:
      * missing/maformed tags  -> falls back to the last non-empty line
      * a lone closing/opening tag
      * leading/trailing whitespace
    Returns a stripped string, or None if nothing usable.
    """
    if text is None:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: a bare opening tag with no close
    if "<answer>" in text.lower():
        tail = re.split(r"<answer>", text, flags=re.I)[-1]
        tail = re.sub(r"</?answer>", "", tail, flags=re.I).strip()
        if tail:
            return tail.splitlines()[-1].strip()
    # final fallback: last non-empty line of the stripped text
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


# ----------------------------------------------------------------------------
# Image size (lazy, header-only) — ported helper from audit notebook (_img_size)
# ----------------------------------------------------------------------------
_WH_CACHE = {}


def get_image_wh(path):
    """Return (W, H) for an image, reading only the header. Cached.

    Falls back to the nuScenes default (1600, 900) if the file is unreadable,
    matching the audit-notebook behaviour.
    """
    if path in _WH_CACHE:
        return _WH_CACHE[path]
    wh = (1600, 900)
    try:
        from PIL import Image

        with Image.open(path) as im:
            wh = im.size
    except Exception:
        wh = (1600, 900)
    _WH_CACHE[path] = wh
    return wh


# ----------------------------------------------------------------------------
# Continuous parsers (ported from audit_stage_b_winners.ipynb cell 11)
# ----------------------------------------------------------------------------
def _parse_point(s):
    """First two numbers in s -> (x, y) float, else None.  (ported: _parse_point)"""
    n = _NUM.findall(s or "")
    return (float(n[0]), float(n[1])) if len(n) >= 2 else None


def _parse_depth_nums(s):
    """All numbers in s as floats (ported from _parse_depth_mid, extended)."""
    return [float(x) for x in _NUM.findall(s or "")]


def _depth_mid(s):
    """Range/scalar midpoint in metres, or None.  (ported: _parse_depth_mid)"""
    n = _parse_depth_nums(s)
    if not n:
        return None
    return (n[0] + n[1]) / 2.0 if len(n) >= 2 else n[0]


def _depth_range(s):
    """(lo, hi) of a 'Between X and Y' range; (v, v) for a scalar; None if empty."""
    n = _parse_depth_nums(s)
    if not n:
        return None
    if len(n) >= 2:
        return (min(n[0], n[1]), max(n[0], n[1]))
    return (n[0], n[0])


# ----------------------------------------------------------------------------
# Categorical normalisation
# ----------------------------------------------------------------------------
# yes/no synonyms
_YESNO = {
    "yes": "yes", "y": "yes", "true": "yes", "correct": "yes", "yeah": "yes",
    "no": "no", "n": "no", "false": "no", "incorrect": "no", "nope": "no",
}
# left/right (lr family answers are usually object labels, but bare directionals occur)
_LR_DIR = {
    "left": "left", "to the left": "left", "on the left": "left", "the left": "left",
    "right": "right", "to the right": "right", "on the right": "right", "the right": "right",
}
# "almost the same" variants (shared by lr / distance)
_SAME = {"almost the same", "about the same", "roughly the same", "the same", "same",
         "equal", "equally", "almost equal", "neither"}
# yaw compass synonyms / abbreviations
_COMPASS = {
    "n": "north", "north": "north",
    "s": "south", "south": "south",
    "e": "east", "east": "east",
    "w": "west", "west": "west",
    "ne": "northeast", "northeast": "northeast", "north-east": "northeast", "north east": "northeast",
    "nw": "northwest", "northwest": "northwest", "north-west": "northwest", "north west": "northwest",
    "se": "southeast", "southeast": "southeast", "south-east": "southeast", "south east": "southeast",
    "sw": "southwest", "southwest": "southwest", "south-west": "southwest", "south west": "southwest",
}


def _norm_text(s):
    """lowercase, strip, drop articles/option-letter prefixes, collapse punct/ws."""
    if s is None:
        return ""
    t = s.strip().lower()
    # strip a leading "option a)", "a.", "(b)" style option marker
    t = re.sub(r"^\(?[a-d]\)[\.\):\s-]+", "", t)
    t = re.sub(r"^[a-d][\.\)]\s+", "", t)
    # drop punctuation -> spaces, collapse whitespace
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _canon_categorical(s):
    """Map a normalised categorical answer onto a canonical token for comparison."""
    t = _norm_text(s)
    if t in _YESNO:
        return _YESNO[t]
    if t in _COMPASS:
        return _COMPASS[t]
    if t in _SAME:
        return "__same__"
    if t in _LR_DIR:
        return _LR_DIR[t]
    # object-label answers: drop a leading article for robustness
    t2 = re.sub(r"^(the|a|an) ", "", t)
    return t2


def _categorical_match(pred, gold):
    cp, cg = _canon_categorical(pred), _canon_categorical(gold)
    if not cp or not cg:
        return False
    if cp == cg:
        return True
    # containment fallback: e.g. pred "the worker is further left" vs gold "left"
    if cg in _LR_DIR.values() or cg in _COMPASS.values() or cg in _YESNO.values():
        return False  # for these short labels require exact canonical match
    # object labels: allow either to be a substring of the other (robust to extra words)
    return cp in cg or cg in cp


# ----------------------------------------------------------------------------
# score_one
# ----------------------------------------------------------------------------
def score_one(pred_answer, gold_answer, template_type, image_wh=None, gold_space="pixels"):
    """Score a single prediction against gold for one SURDS template.

    Parameters
    ----------
    pred_answer : str   raw predicted <answer> text (or already-extracted string)
    gold_answer : str   gold <answer> text
    template_type : str one of lr/distance/fb/yaw/xy2d/depth
    image_wh : (W,H) | None   pixel size for the xy2d rescale (default get_image_wh
                              gives (1600,900)). If None, xy2d falls back to scoring
                              in normalised 0-1000 space with NORM_XY_TOL — valid only
                              when the gold is ALSO normalised (gold_space='norm').
    gold_space : 'pixels' | 'norm'   the coordinate frame the xy2d GOLD is stored in.
                              The PREDICTION is always 0-1000 normalised. 'pixels' is
                              the default (curriculum / source-QA / teacher-pool gold);
                              'norm' is for SFT/val_1k/heldout/val_meta gold. score_one
                              reconciles BOTH into pixel space before comparing. See the
                              module docstring "frame trap" note. Ignored for non-xy2d.

    Returns
    -------
    dict with keys: correct (bool), kind ('categorical'|'continuous'|'unknown'),
                    template_type, detail (dict with the numeric error / canon tokens),
                    parse_ok (bool).
    """
    tt = (template_type or "").strip().lower()

    # ----- continuous: xy2d -----------------------------------------------
    if tt == "xy2d":
        p, g = _parse_point(pred_answer), _parse_point(gold_answer)
        if not p or not g:
            return {"correct": False, "kind": "continuous", "template_type": tt,
                    "parse_ok": False, "detail": {"error": "unparseable point",
                                                  "pred": p, "gold": g}}
        gs = (gold_space or "pixels").strip().lower()
        # Defensive auto-detect: a gold coord > 1000 can only be pixels (the 0-1000
        # grid can't exceed 1000); a whole-point <= 1000 declared 'pixels' on a
        # >1000-wide frame is suspicious but legal (top-left objects), so we only
        # OVERRIDE the unambiguous direction: declared-norm-but-actually-pixels.
        gold_is_pixels = (gs == "pixels") or (max(g[0], g[1]) > 1000.0)
        frame_corrected = gold_is_pixels and gs == "norm"
        if image_wh is not None:
            W, H = image_wh
            # Qwen prediction is 0-1000 RELATIVE coords -> un-normalize to pixels.
            pp = (p[0] * W / 1000.0, p[1] * H / 1000.0)
            # Reconcile gold into the SAME pixel frame.
            if gold_is_pixels:
                gg = (g[0], g[1])                              # already pixels
            else:
                gg = (g[0] * W / 1000.0, g[1] * H / 1000.0)    # norm -> pixels
            dist = math.hypot(pp[0] - gg[0], pp[1] - gg[1])
            tol = XY2D_TOL_PX
            space = "px"
        else:
            # No image size. Only valid when BOTH operands are already normalised
            # (pred is always 0-1000; gold must be 'norm'). If the gold is pixels we
            # CANNOT reconcile without (W,H) -> flag the frame mismatch loudly instead
            # of silently comparing 0-1000 vs ~1600 (the old corruption path).
            if gold_is_pixels:
                return {"correct": False, "kind": "continuous", "template_type": tt,
                        "parse_ok": True,
                        "detail": {"error": "frame mismatch: pixel gold needs image_wh "
                                            "to reconcile against 0-1000 pred",
                                   "space": "frame_error", "pred": p, "gold": g}}
            dist = math.hypot(p[0] - g[0], p[1] - g[1])
            tol = NORM_XY_TOL
            space = "norm"
        return {"correct": dist <= tol, "kind": "continuous", "template_type": tt,
                "parse_ok": True,
                "detail": {"l2": dist, "tol": tol, "space": space,
                           "gold_space": gs, "frame_corrected": frame_corrected,
                           "pred": p, "gold": g}}

    # ----- continuous: depth ----------------------------------------------
    if tt == "depth":
        pm, gm = _depth_mid(pred_answer), _depth_mid(gold_answer)
        grange = _depth_range(gold_answer)
        if pm is None or gm is None:
            return {"correct": False, "kind": "continuous", "template_type": tt,
                    "parse_ok": False, "detail": {"error": "unparseable depth",
                                                  "pred_mid": pm, "gold_mid": gm}}
        mid_err = abs(pm - gm)
        in_range = grange is not None and (grange[0] <= pm <= grange[1])
        correct = (mid_err <= DEPTH_TOL_M) or in_range
        return {"correct": bool(correct), "kind": "continuous", "template_type": tt,
                "parse_ok": True,
                "detail": {"mid_err": mid_err, "tol": DEPTH_TOL_M,
                           "pred_mid": pm, "gold_mid": gm,
                           "gold_range": grange, "in_range": in_range}}

    # ----- categorical: lr / distance / fb / yaw --------------------------
    if tt in CATEGORICAL_TEMPLATES:
        ok = _categorical_match(pred_answer, gold_answer)
        return {"correct": bool(ok), "kind": "categorical", "template_type": tt,
                "parse_ok": pred_answer is not None and str(pred_answer).strip() != "",
                "detail": {"pred_canon": _canon_categorical(pred_answer),
                           "gold_canon": _canon_categorical(gold_answer)}}

    # ----- unknown template -----------------------------------------------
    ok = _norm_text(pred_answer) == _norm_text(gold_answer) and _norm_text(gold_answer) != ""
    return {"correct": bool(ok), "kind": "unknown", "template_type": tt,
            "parse_ok": pred_answer is not None,
            "detail": {"note": "unknown template_type; fell back to normalised exact match"}}


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _selftest():
    cases = []

    def chk(name, got, want):
        ok = got == want
        cases.append((name, ok, got, want))

    # parse_answer
    chk("parse_answer tagged",
        parse_answer("<think>x</think><answer>The black truck</answer>"),
        "The black truck")
    chk("parse_answer no-tag-fallback",
        parse_answer("some reasoning\nThe silver car"),
        "The silver car")
    chk("parse_answer none", parse_answer(None), None)

    # categorical: lr object label (substring robust)
    chk("lr exact", score_one("The black truck", "The black truck", "lr")["correct"], True)
    chk("lr article-robust", score_one("black truck", "The black truck", "lr")["correct"], True)
    chk("lr wrong", score_one("The silver car", "The black truck", "lr")["correct"], False)
    chk("lr same-synonym",
        score_one("about the same", "Almost the same", "lr")["correct"], True)

    # fb yes/no synonyms
    chk("fb yes", score_one("Yes.", "Yes", "fb")["correct"], True)
    chk("fb y/n-synonym", score_one("no", "No", "fb")["correct"], True)
    chk("fb wrong", score_one("Yes", "No", "fb")["correct"], False)

    # yaw compass synonyms
    chk("yaw exact", score_one("West", "West", "yaw")["correct"], True)
    chk("yaw abbrev", score_one("SE", "Southeast", "yaw")["correct"], True)
    chk("yaw hyphen", score_one("north-east", "Northeast", "yaw")["correct"], True)
    chk("yaw wrong", score_one("East", "West", "yaw")["correct"], False)

    # xy2d PIXEL gold (default gold_space='pixels'): pred 0-1000 -> px, gold as-is.
    # pred [824,578] on 1600x900 -> (1318,520) px vs pixel gold [1321,531] -> err ~12 px
    chk("xy2d close px (pixel gold)",
        score_one("[824, 578]", "[1321, 531]", "xy2d", image_wh=(1600, 900))["correct"], True)
    chk("xy2d far px (pixel gold)",
        score_one("[100, 100]", "[1321, 531]", "xy2d", image_wh=(1600, 900))["correct"], False)
    # xy2d NORMALISED gold (gold_space='norm') + image_wh: BOTH rescaled to px.
    # pred [950,570] vs norm gold [946,574] -> tiny px error < 50
    chk("xy2d close px (norm gold)",
        score_one("[950, 570]", "[946, 574]", "xy2d", image_wh=(1600, 900),
                  gold_space="norm")["correct"], True)
    chk("xy2d far px (norm gold)",
        score_one("[100, 100]", "[946, 574]", "xy2d", image_wh=(1600, 900),
                  gold_space="norm")["correct"], False)
    # the OLD BUG: norm gold scored as pixels (default) -> pred(px ~1518) vs gold(946)
    # -> huge error -> wrongly incorrect. Confirms why callers must pass gold_space='norm'.
    chk("xy2d norm-gold-as-pixels is wrong",
        score_one("[946, 574]", "[946, 574]", "xy2d", image_wh=(1600, 900))["correct"], False)
    # defensive auto-detect: declared 'norm' but gold is clearly pixels (x>1000) -> used as px
    chk("xy2d auto-detect pixel gold under norm flag",
        score_one("[824, 578]", "[1321, 531]", "xy2d", image_wh=(1600, 900),
                  gold_space="norm")["correct"], True)
    # normalised-space fallback (no image_wh, both norm)
    chk("xy2d close norm (no wh)",
        score_one("[946, 576]", "[946, 574]", "xy2d", gold_space="norm")["correct"], True)
    # frame error: pixel gold with no image_wh -> cannot reconcile -> flagged, not silent
    chk("xy2d frame-error pixel gold no wh",
        score_one("[824, 578]", "[1321, 531]", "xy2d")["detail"].get("space"), "frame_error")
    chk("xy2d unparseable",
        score_one("dunno", "[946, 574]", "xy2d")["parse_ok"], False)

    # depth: midpoint abs-diff and in-range
    chk("depth in-range",
        score_one("Between 6 meters and 12 meters", "Between 5 meters and 11 meters",
                  "depth")["correct"], True)
    chk("depth midpoint-close",
        score_one("about 8 meters", "Between 5 meters and 11 meters", "depth")["correct"], True)
    chk("depth far",
        score_one("Between 30 meters and 36 meters", "Between 5 meters and 11 meters",
                  "depth")["correct"], False)
    chk("depth unparseable",
        score_one("far away", "Between 5 meters and 11 meters", "depth")["parse_ok"], False)

    npass = sum(1 for _, ok, _, _ in cases if ok)
    print(f"score_surds self-test: {npass}/{len(cases)} passed\n")
    for name, ok, got, want in cases:
        flag = "PASS" if ok else "FAIL"
        extra = "" if ok else f"   got={got!r} want={want!r}"
        print(f"  [{flag}] {name}{extra}")
    return npass == len(cases)


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)

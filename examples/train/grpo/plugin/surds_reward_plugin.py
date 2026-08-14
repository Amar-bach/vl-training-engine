"""
SURDS Spatial-Reasoning GRPO Reward Plugin
===========================================

Two reward keys are registered:

  surds_accuracy   -- Binary 0/1 reward for ALL template types.
                      Correct answer → 1.0, wrong/unparseable → 0.0.
                      Use this for early/easy curriculum bands or as a simple baseline.

  surds_dense      -- Shaped/dense reward for continuous templates (xy2d, depth);
                      binary 0/1 for categorical templates (lr, distance, fb, yaw).
                      Designed for hard curriculum bands where the binary signal is
                      near-zero (pass rate < 25%), making policy gradients otherwise
                      very noisy.  Dense distance reward provides a learning signal
                      even when the answer is wrong.

Required dataset columns (must be present in the GRPO dataset):
  solution       (str)   -- Gold answer string, e.g. "[946, 574]" or "The black truck"
  template_type  (str)   -- SURDS template slug: lr / distance / fb / yaw / xy2d / depth
  image_path     (str)   -- Absolute path to the scene image (used for xy2d WH rescale).
                            Optional — falls back to normalised-space scoring if absent.

swift rlhf wiring (binary reward):
  swift rlhf --rlhf_type grpo \\
    --external_plugins examples/train/grpo/plugin/surds_reward_plugin.py \\
    --reward_funcs surds_accuracy format \\
    --reward_weights 1.0 0.2

swift rlhf wiring (dense reward):
  swift rlhf --rlhf_type grpo \\
    --external_plugins examples/train/grpo/plugin/surds_reward_plugin.py \\
    --reward_funcs surds_dense format \\
    --reward_weights 1.0 0.2

score_surds import path
-----------------------
This plugin resolves the path to research/eval/ relative to its own location:
  <repo_root>/examples/train/grpo/plugin/surds_reward_plugin.py
  → <repo_root>/research/eval/score_surds.py

You can override the search directory by setting env var SURDS_EVAL_DIR:
  export SURDS_EVAL_DIR=/path/to/research/eval

Dense reward formulas
---------------------
  xy2d:
    D = 2 * tol  (where tol = 50 px when image_wh known, else NORM_XY_TOL ≈ 38.5)
    reward = max(0, 1 - l2 / D)
    This gives 1.0 at l2=0, 0.5 at l2=tol (the pass threshold), 0.0 at l2>=2*tol.
    If correct (l2 <= tol) → hard clamp to 1.0.

  depth:
    k = 2  (scale constant; reward hits 0.5 at mid_err = tol = 4 m)
    reward = max(0, 1 - mid_err / (k * tol))
    Gives 1.0 at mid_err=0, 0.5 at mid_err=4 m, 0.0 at mid_err>=8 m.
    If correct (mid_err<=4 or pred_mid in gold_range) → hard clamp to 1.0.

  categorical (lr / distance / fb / yaw): binary 1.0 / 0.0 same as SurdsAccuracy.
"""

import math
import os
import sys
from typing import List

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Bootstrap: add research/eval to sys.path so score_surds can be imported.
# This file lives at:  <repo>/examples/train/grpo/plugin/surds_reward_plugin.py
# repo root is 4 levels up.
# ---------------------------------------------------------------------------
_SURDS_EVAL_DIR = os.environ.get("SURDS_EVAL_DIR", None)
if _SURDS_EVAL_DIR is None:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    # examples/train/grpo/plugin  →  up 4 levels  →  repo root
    _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
    _SURDS_EVAL_DIR = os.path.join(_REPO_ROOT, "research", "eval")

if _SURDS_EVAL_DIR not in sys.path:
    sys.path.insert(0, _SURDS_EVAL_DIR)

try:
    from score_surds import parse_answer, score_one, get_image_wh  # noqa: E402
    _SCORE_SURDS_OK = True
except ImportError as _e:
    logger.warning(
        f"[surds_reward_plugin] Could not import score_surds from {_SURDS_EVAL_DIR}: {_e}. "
        "All rewards will be 0.0 until the import is fixed."
    )
    _SCORE_SURDS_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse_answer(completion: str):
    """Call parse_answer; return None on any exception."""
    try:
        return parse_answer(completion)
    except Exception:
        return None


def _get_wh(image_path):
    """Return (W, H) or None if path is empty/unavailable."""
    if not image_path:
        return None
    try:
        return get_image_wh(image_path)
    except Exception:
        return None


def _score_sample(pred, gold, template_type, image_path):
    """
    Score one sample.  Returns the full score_one dict, or a failure sentinel.
    """
    if not _SCORE_SURDS_OK:
        return {"correct": False, "kind": "unknown", "template_type": template_type or "",
                "parse_ok": False, "detail": {}}
    try:
        tt = (template_type or "").strip().lower()
        wh = _get_wh(image_path) if tt == "xy2d" else None
        return score_one(pred, gold, template_type, image_wh=wh)
    except Exception as exc:
        logger.warning(f"[surds_reward_plugin] score_one raised: {exc}")
        return {"correct": False, "kind": "unknown", "template_type": template_type or "",
                "parse_ok": False, "detail": {}}


# ---------------------------------------------------------------------------
# SurdsAccuracy  —  binary 0/1 for all template types
# ---------------------------------------------------------------------------

class SurdsAccuracy(ORM):
    """
    Binary accuracy reward for SURDS spatial-reasoning.

    Returns 1.0 if the completion's parsed answer matches the gold answer
    according to score_one, else 0.0.

    Required kwargs (batched lists):
      completions    -- generated text completions
      solution       -- gold answer strings
      template_type  -- SURDS template slug per sample
    Optional kwargs:
      image_path     -- path to scene image (used for xy2d pixel-space scoring)
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions = kwargs.get("solution", [None] * len(completions))
        template_types = kwargs.get("template_type", [None] * len(completions))
        image_paths = kwargs.get("image_path", [None] * len(completions))

        # Graceful fallback if a column is missing entirely (scalar None → replicated list)
        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions] * len(completions)
        if not isinstance(template_types, (list, tuple)):
            template_types = [template_types] * len(completions)
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths] * len(completions)

        rewards = []
        for completion, gold, tt, img_path in zip(completions, solutions, template_types, image_paths):
            try:
                pred = _safe_parse_answer(completion)
                result = _score_sample(pred, gold, tt, img_path)
                rewards.append(1.0 if result["correct"] else 0.0)
            except Exception as exc:
                logger.warning(f"[SurdsAccuracy] Unexpected error: {exc}")
                rewards.append(0.0)
        return rewards


orms["surds_accuracy"] = SurdsAccuracy


# ---------------------------------------------------------------------------
# SurdsDense  —  shaped reward for continuous templates, binary for categorical
# ---------------------------------------------------------------------------

# Scale constants for dense rewards (documented in module docstring above).
_XY2D_SCALE_K = 2.0   # D = k * tol → reward hits 0.5 at l2 == tol
_DEPTH_SCALE_K = 2.0  # reward hits 0.5 at mid_err == tol (4 m)


class SurdsDense(ORM):
    """
    Dense/shaped reward for SURDS spatial-reasoning.

    Categorical templates (lr, distance, fb, yaw): binary 1.0 / 0.0.

    xy2d:
        D = 2 * tol  (tol = 50 px in pixel-space, NORM_XY_TOL ≈ 38.5 in norm-space)
        reward = clamp(1 - l2 / D, 0, 1)
        Hard 1.0 when correct (l2 <= tol).

    depth:
        reward = clamp(1 - mid_err / (2 * 4.0), 0, 1)  i.e. 1 - mid_err/8
        Hard 1.0 when correct (mid_err<=4 m or pred_mid in gold_range).

    Rationale: In hard curriculum bands (<25% pass rate) binary reward is nearly
    all-zero, making gradient variance huge.  The continuous distance signal
    provides a non-zero learning gradient even when the answer is wrong,
    making these bands trainable.

    Required / optional kwargs: same as SurdsAccuracy.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions = kwargs.get("solution", [None] * len(completions))
        template_types = kwargs.get("template_type", [None] * len(completions))
        image_paths = kwargs.get("image_path", [None] * len(completions))

        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions] * len(completions)
        if not isinstance(template_types, (list, tuple)):
            template_types = [template_types] * len(completions)
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths] * len(completions)

        rewards = []
        for completion, gold, tt, img_path in zip(completions, solutions, template_types, image_paths):
            try:
                pred = _safe_parse_answer(completion)
                result = _score_sample(pred, gold, tt, img_path)
                reward = self._dense_reward(result)
                rewards.append(reward)
            except Exception as exc:
                logger.warning(f"[SurdsDense] Unexpected error: {exc}")
                rewards.append(0.0)
        return rewards

    @staticmethod
    def _dense_reward(result: dict) -> float:
        """Compute shaped reward from a score_one result dict."""
        correct = result.get("correct", False)
        tt = result.get("template_type", "").strip().lower()
        detail = result.get("detail", {})

        # Hard 1.0 on correct for all templates (ceiling is always 1.0).
        if correct:
            return 1.0

        # Continuous: xy2d — distance-based shaped reward.
        if tt == "xy2d":
            l2 = detail.get("l2")
            tol = detail.get("tol")
            if l2 is None or tol is None or tol <= 0:
                return 0.0
            D = _XY2D_SCALE_K * tol  # e.g. 100 px in pixel space
            return float(max(0.0, 1.0 - l2 / D))

        # Continuous: depth — midpoint-error shaped reward.
        if tt == "depth":
            mid_err = detail.get("mid_err")
            tol = detail.get("tol")
            if mid_err is None or tol is None or tol <= 0:
                return 0.0
            D = _DEPTH_SCALE_K * tol  # 2 * 4.0 = 8.0 m
            return float(max(0.0, 1.0 - mid_err / D))

        # Categorical (lr, distance, fb, yaw) and unknown: binary.
        return 0.0


orms["surds_dense"] = SurdsDense


# ---------------------------------------------------------------------------
# SurdsDenseBinary  —  binary correctness bonus + Gaussian directional dense term
# ---------------------------------------------------------------------------
#
#   reward = W_BINARY * 1{correct}  +  W_DENSE * closeness
#
#   closeness in (0, 1]:
#       xy2d  : exp(-(l2      / xy_sigma)^2)
#       depth : exp(-(mid_err / depth_sigma)^2)
#       categorical (lr/distance/fb/yaw): == 1{correct}   (no graded error signal)
#       unparseable: 0.0  (no directional credit)
#
#   Defaults: W_BINARY=1.0, W_DENSE=0.20, sigma s.t. closeness ~= 0.5 at the pass
#   tolerance (50 px / 4 m).  Surface:
#       correct + ~0 error       -> 1.0 + 0.20*1.00 = 1.20   (global max)
#       correct at tolerance edge-> 1.0 + 0.20*0.50 = 1.10
#       wrong, just past tol     -> 0.0 + 0.20*~0.50 = ~0.10  (a little directional)
#       wrong, far off           -> 0.0 + 0.20*~0.00 = ~0.00
#       categorical correct      -> 1.0 + 0.20*1.00 = 1.20
#   Any correct (>=1.10) strictly dominates any wrong (<=0.10): a sharp "very high
#   for correct" step, plus a small Gaussian ramp that pulls wrong-but-close
#   continuous answers toward the target.  All knobs are env-overridable:
#       SURDS_W_BINARY, SURDS_W_DENSE, SURDS_XY_SIGMA (px), SURDS_DEPTH_SIGMA (m).

_W_BINARY = float(os.environ.get("SURDS_W_BINARY", "1.0"))
_W_DENSE = float(os.environ.get("SURDS_W_DENSE", "0.20"))
# Gaussian sigmas, in the SAME units score_one reports the error in (px / m).
# 60 px and 5 m put closeness ~= 0.5 at the 50 px / 4 m pass tolerance.
_XY_SIGMA = float(os.environ.get("SURDS_XY_SIGMA", "60.0"))      # px
_DEPTH_SIGMA = float(os.environ.get("SURDS_DEPTH_SIGMA", "5.0"))  # m
# Reference tolerances the sigmas are calibrated against; used to rescale sigma
# into normalised xy space when image_wh is absent (tol != XY2D_TOL_PX).
_XY_REF_TOL = 50.0    # XY2D_TOL_PX
_DEPTH_REF_TOL = 4.0  # DEPTH_TOL_M


class SurdsDenseBinary(ORM):
    """Combined binary-correctness + Gaussian directional dense reward.

    reward = W_BINARY * 1{correct} + W_DENSE * closeness, where closeness is a
    Gaussian on the continuous error for xy2d/depth and == 1{correct} for the
    categorical families.  See the module comment above for the full surface.

    Required / optional kwargs: same as SurdsAccuracy / SurdsDense.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions = kwargs.get("solution", [None] * len(completions))
        template_types = kwargs.get("template_type", [None] * len(completions))
        image_paths = kwargs.get("image_path", [None] * len(completions))

        if not isinstance(solutions, (list, tuple)):
            solutions = [solutions] * len(completions)
        if not isinstance(template_types, (list, tuple)):
            template_types = [template_types] * len(completions)
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths] * len(completions)

        rewards = []
        for completion, gold, tt, img_path in zip(completions, solutions, template_types, image_paths):
            try:
                pred = _safe_parse_answer(completion)
                result = _score_sample(pred, gold, tt, img_path)
                rewards.append(self._reward(result))
            except Exception as exc:
                logger.warning(f"[SurdsDenseBinary] Unexpected error: {exc}")
                rewards.append(0.0)
        return rewards

    @staticmethod
    def _gauss(err, sigma) -> float:
        """exp(-(err/sigma)^2); 0.0 if inputs unusable."""
        if err is None or sigma is None or sigma <= 0:
            return 0.0
        return math.exp(-((err / sigma) ** 2))

    @staticmethod
    def _reward(result: dict) -> float:
        correct = bool(result.get("correct", False))
        tt = result.get("template_type", "").strip().lower()
        detail = result.get("detail", {})
        binary = 1.0 if correct else 0.0

        if tt == "xy2d":
            l2 = detail.get("l2")
            tol = detail.get("tol")
            # rescale sigma if scored in normalised space (tol != 50 px)
            sigma = _XY_SIGMA * (tol / _XY_REF_TOL) if tol else _XY_SIGMA
            closeness = SurdsDenseBinary._gauss(l2, sigma) if l2 is not None else binary
        elif tt == "depth":
            mid_err = detail.get("mid_err")
            tol = detail.get("tol")
            sigma = _DEPTH_SIGMA * (tol / _DEPTH_REF_TOL) if tol else _DEPTH_SIGMA
            closeness = SurdsDenseBinary._gauss(mid_err, sigma) if mid_err is not None else binary
        else:
            # categorical (lr/distance/fb/yaw) and unknown: no graded error.
            closeness = binary

        return _W_BINARY * binary + _W_DENSE * closeness


orms["surds_dense_binary"] = SurdsDenseBinary

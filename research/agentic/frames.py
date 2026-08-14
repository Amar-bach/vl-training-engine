"""Coordinate frames for the SURDS agentic harness.

THIS MODULE EXISTS BECAUSE FRAME CONFUSION IS THE #1 SILENT FAILURE IN THIS PROJECT.
Read `<repo>/CLAUDE.md` (SURDS xy2d coordinate frames) before changing anything here.

There are FOUR frames in the agentic setting. Three predate this work; the fourth is new.

  1. NORM     0-1000, isotropic-normalised. Everything Qwen3-VL *emits* lives here
              (points certainly; bboxes assumed but see `BBOX_FRAME_VERIFIED` below).
  2. FETCHED  the pixel grid of the image after `qwen_vl_utils.fetch_image` has
              smart_resize'd it to fit `max_pixels`. This is what `PIL.Image.crop()`
              operates on inside an ms-swift rollout scheduler.
  3. NATIVE   the on-disk SURDS frame, 1600x900. Curriculum / source-QA gold xy2d
              lives here, and the 50 px tolerance is defined here.
  4. SANDBOX  whatever we hand the python_repl tool. WE CHOOSE: the tool is seeded
              with the NATIVE image, so sandbox coords == NATIVE coords. Documented
              here so it never becomes an open question.

Pinned measurements (2026-08-13, this dataset, this env):

    every SURDS frame is 1600x900   (verified on a 60-file sample of 27,152 images)
    MAX_PIXELS=1003520  ->  fetched 1316x728   == 47 x 26 patches of 28 px
    sx = 1316/1600 = 0.8225      sy = 728/900 = 0.8089

Note sx != sy. smart_resize rounds each dimension independently to a multiple of 28,
so the scaling is ANISOTROPIC and a single scalar rescale is WRONG.

Why this matters concretely: `VisualToolBoxScheduler.step` crops in FETCHED space.
A bbox emitted in NORM space and passed straight to `crop()` is read as pixels on a
1316x728 canvas -- it takes roughly the top-left 76% of the width and clips at 728 in
height. Every crop is wrong, nothing raises, training proceeds, and the tool
contributes nothing. Always convert explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Tuple

# --------------------------------------------------------------------------------------
# Pinned constants
# --------------------------------------------------------------------------------------

SURDS_NATIVE_W = 1600
SURDS_NATIVE_H = 900

NORM_SCALE = 1000.0          # Qwen's normalised coordinate range is [0, 1000]
PATCH = 28                   # Qwen-VL spatial patch factor; smart_resize rounds to this
MIN_SIDE_PX = PATCH          # a crop smaller than one patch carries no visual information

# Production value used by every SURDS job (`MAX_PIXELS=1003520` in slurm_scripts/*.sh).
# Read from the environment so this module and ms-swift's template -- which resolves it
# via `get_env_args('max_pixels', ...)` at template init (swift/template/templates/qwen.py
# :1209) -- cannot disagree. A divergence here silently changes the image token count
# between rollout and training.
#
# NB `qwen_vl_utils.fetch_image` DEFAULTS to 12845056, which on a 1600x900 frame yields
# 1596x896, not 1316x728. Every call site must pass max_pixels explicitly; forgetting
# was caught by a smoke test and would have made every crop land in the wrong place.
DEFAULT_MAX_PIXELS = int(os.environ.get('MAX_PIXELS') or 1003520)

# OPEN PHASE-0 ITEM. Qwen emits *points* in NORM space by documented convention. Whether
# it emits *bboxes* in NORM space has NOT been verified against a live model on SURDS.
# Until the 20-sample dump is done, `bbox_to_fetched` refuses to guess: pass the frame
# explicitly. Set this to True only after the dump, and record the finding in CLAUDE.md.
BBOX_FRAME_VERIFIED = False


@dataclass(frozen=True)
class FrameSpec:
    """The three image-space frames for one sample, with their sizes."""

    native_wh: Tuple[int, int]
    fetched_wh: Tuple[int, int]
    max_pixels: int = DEFAULT_MAX_PIXELS

    @property
    def sx(self) -> float:
        """NATIVE -> FETCHED scale on x. Not equal to `sy` in general."""
        return self.fetched_wh[0] / self.native_wh[0]

    @property
    def sy(self) -> float:
        return self.fetched_wh[1] / self.native_wh[1]


def fetched_wh(native_wh: Tuple[int, int] = (SURDS_NATIVE_W, SURDS_NATIVE_H),
               max_pixels: int = DEFAULT_MAX_PIXELS) -> Tuple[int, int]:
    """Size `qwen_vl_utils.fetch_image` will produce, as (W, H).

    Computed rather than hardcoded so a change to MAX_PIXELS cannot silently
    invalidate the transforms. For the pinned SURDS case this returns (1316, 728).
    """
    from qwen_vl_utils.vision_process import smart_resize

    w, h = native_wh
    new_h, new_w = smart_resize(h, w, factor=PATCH, min_pixels=None, max_pixels=max_pixels)
    return int(new_w), int(new_h)


def frame_spec(native_wh: Tuple[int, int] = (SURDS_NATIVE_W, SURDS_NATIVE_H),
               max_pixels: int = DEFAULT_MAX_PIXELS) -> FrameSpec:
    return FrameSpec(native_wh=tuple(native_wh),
                     fetched_wh=fetched_wh(native_wh, max_pixels),
                     max_pixels=max_pixels)


# --------------------------------------------------------------------------------------
# Point transforms. Named <source>_to_<target>; there is deliberately no generic helper,
# so every call site states both frames.
# --------------------------------------------------------------------------------------

def norm_to_native(xy: Sequence[float],
                   spec: FrameSpec | None = None) -> Tuple[float, float]:
    spec = spec or frame_spec()
    w, h = spec.native_wh
    return xy[0] * w / NORM_SCALE, xy[1] * h / NORM_SCALE


def norm_to_fetched(xy: Sequence[float],
                    spec: FrameSpec | None = None) -> Tuple[float, float]:
    spec = spec or frame_spec()
    w, h = spec.fetched_wh
    return xy[0] * w / NORM_SCALE, xy[1] * h / NORM_SCALE


def native_to_fetched(xy: Sequence[float],
                      spec: FrameSpec | None = None) -> Tuple[float, float]:
    spec = spec or frame_spec()
    return xy[0] * spec.sx, xy[1] * spec.sy


def fetched_to_native(xy: Sequence[float],
                      spec: FrameSpec | None = None) -> Tuple[float, float]:
    spec = spec or frame_spec()
    return xy[0] / spec.sx, xy[1] / spec.sy


# --------------------------------------------------------------------------------------
# Bbox handling
# --------------------------------------------------------------------------------------

class BBoxError(ValueError):
    """Raised for a structurally invalid bbox. Callers must convert this to an
    observation string -- a malformed tool call is a training signal, never a crash."""


def validate_and_clamp(bbox: Sequence[float], wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Clamp a bbox to image bounds and enforce minimum size / sane aspect ratio.

    Ported from `VisualToolBoxScheduler.maybe_resize_bbox`, whose logic is sound: clamp,
    reject degenerate or absurd boxes, then expand tiny boxes about their centre so the
    crop is at least one patch on each side.

    `bbox` and `wh` must already be in the SAME frame. This function does not convert.
    """
    if len(bbox) != 4:
        raise BBoxError(f'bbox must have 4 elements, got {len(bbox)}: {bbox!r}')
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError) as e:
        raise BBoxError(f'bbox values must be numeric: {bbox!r}') from e

    w, h = wh
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(w), right), min(float(h), bottom)

    if not (left < right and top < bottom):
        raise BBoxError(f'degenerate bbox after clamping: {(left, top, right, bottom)}')

    bw, bh = right - left, bottom - top
    if max(bw, bh) / min(bw, bh) > 100:
        raise BBoxError(f'aspect ratio out of range: {bw:.1f}x{bh:.1f}')

    if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
        if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
            raise BBoxError(f'image {w}x{h} is smaller than the minimum crop side {MIN_SIDE_PX}')
        # Grow about the centre, preserving aspect ratio, then SHIFT (not clamp) back
        # inside the image. Clamping here would silently return a sub-minimum crop for
        # boxes near an edge -- Qwen cannot encode an image thinner than one patch.
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        ratio = MIN_SIDE_PX / min(bw, bh)
        want_w = min(max(bw * ratio, MIN_SIDE_PX), float(w))
        want_h = min(max(bh * ratio, MIN_SIDE_PX), float(h))
        left = min(max(0.0, cx - want_w / 2.0), w - want_w)
        top = min(max(0.0, cy - want_h / 2.0), h - want_h)
        right, bottom = left + want_w, top + want_h

    return int(left), int(top), int(right), int(bottom)


def bbox_to_fetched(bbox: Sequence[float],
                    source_frame: str,
                    spec: FrameSpec | None = None) -> Tuple[int, int, int, int]:
    """Convert a model-emitted bbox into FETCHED space and validate it.

    `source_frame` must be given explicitly -- 'norm', 'native' or 'fetched'. There is no
    default, because guessing is exactly the bug this module exists to prevent. See
    `BBOX_FRAME_VERIFIED`: until the Phase-0 dump resolves which frame the model uses,
    the caller is responsible for the choice and for recording the evidence.
    """
    spec = spec or frame_spec()
    if source_frame == 'norm':
        x1, y1 = norm_to_fetched(bbox[0:2], spec)
        x2, y2 = norm_to_fetched(bbox[2:4], spec)
    elif source_frame == 'native':
        x1, y1 = native_to_fetched(bbox[0:2], spec)
        x2, y2 = native_to_fetched(bbox[2:4], spec)
    elif source_frame == 'fetched':
        x1, y1, x2, y2 = (float(v) for v in bbox)
    else:
        raise BBoxError(f"source_frame must be 'norm'|'native'|'fetched', got {source_frame!r}")

    return validate_and_clamp((x1, y1, x2, y2), spec.fetched_wh)


def contains_point(bbox: Sequence[float], xy: Sequence[float]) -> bool:
    """Faithfulness primitive: does this crop contain the gold point?

    Both arguments must be in the SAME frame. Used by the eval harness to compute
    P(correct | crop contains gold) vs P(correct | crop misses gold) -- if those are
    equal, the crop is decorative and the tool is not doing any work.
    """
    x1, y1, x2, y2 = bbox
    return x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2

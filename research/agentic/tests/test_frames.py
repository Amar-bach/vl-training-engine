"""Unit tests for the coordinate-frame layer.

Run:  python -m pytest research/agentic/tests/test_frames.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentic import frames as F  # noqa: E402


# --------------------------------------------------------------------------------------
# The pinned measurements. If any of these fail, the harness is operating on a different
# image pipeline than the one the architecture doc was written against -- stop and
# re-derive rather than adjusting the numbers.
# --------------------------------------------------------------------------------------

def test_pinned_fetched_size():
    assert F.fetched_wh() == (1316, 728)


def test_fetched_size_is_patch_aligned():
    w, h = F.fetched_wh()
    assert w % F.PATCH == 0 and h % F.PATCH == 0
    assert (w // F.PATCH, h // F.PATCH) == (47, 26)


def test_scaling_is_anisotropic():
    """The whole point of this module: sx != sy, so no single scalar rescale is valid."""
    spec = F.frame_spec()
    assert spec.sx == pytest.approx(0.8225, abs=1e-4)
    assert spec.sy == pytest.approx(0.8089, abs=1e-4)
    assert spec.sx != pytest.approx(spec.sy, abs=1e-3)


def test_fetched_respects_max_pixels():
    w, h = F.fetched_wh()
    assert w * h <= F.DEFAULT_MAX_PIXELS


# --------------------------------------------------------------------------------------
# Point transforms
# --------------------------------------------------------------------------------------

def test_norm_to_native_corners():
    assert F.norm_to_native((0, 0)) == pytest.approx((0.0, 0.0))
    assert F.norm_to_native((1000, 1000)) == pytest.approx((1600.0, 900.0))


def test_norm_to_fetched_corners():
    assert F.norm_to_fetched((1000, 1000)) == pytest.approx((1316.0, 728.0))


def test_native_fetched_round_trip():
    for xy in [(0, 0), (800, 450), (1599, 899)]:
        back = F.fetched_to_native(F.native_to_fetched(xy))
        assert back == pytest.approx(xy, abs=1e-6)


def test_norm_route_consistency():
    """norm->native->fetched must equal norm->fetched."""
    for xy in [(1, 1), (500, 500), (999, 999)]:
        assert F.native_to_fetched(F.norm_to_native(xy)) == pytest.approx(
            F.norm_to_fetched(xy), abs=1e-6)


def test_the_bug_this_module_prevents():
    """A NORM bbox read as FETCHED pixels covers only the top-left of the image.

    This is the silent failure described in the architecture doc: it does not raise,
    it just crops the wrong region forever.
    """
    spec = F.frame_spec()
    naive_full_image = (0, 0, 1000, 1000)          # model says "the whole image"
    correct = F.bbox_to_fetched(naive_full_image, 'norm', spec)
    wrong = F.validate_and_clamp(naive_full_image, spec.fetched_wh)
    assert correct == (0, 0, 1316, 728)
    assert wrong == (0, 0, 1000, 728)              # 76% of the width, silently
    assert correct != wrong


# --------------------------------------------------------------------------------------
# BBox validation
# --------------------------------------------------------------------------------------

def test_clamp_to_bounds():
    assert F.validate_and_clamp((-50, -50, 5000, 5000), (1316, 728)) == (0, 0, 1316, 728)


def test_tiny_box_expands_to_min_side():
    x1, y1, x2, y2 = F.validate_and_clamp((600, 400, 604, 404), (1316, 728))
    assert x2 - x1 >= F.MIN_SIDE_PX and y2 - y1 >= F.MIN_SIDE_PX
    # expansion is about the centre
    assert (x1 + x2) / 2 == pytest.approx(602, abs=1)
    assert (y1 + y2) / 2 == pytest.approx(402, abs=1)


def test_tiny_box_in_corner_still_valid():
    x1, y1, x2, y2 = F.validate_and_clamp((0, 0, 3, 3), (1316, 728))
    assert x2 - x1 >= F.MIN_SIDE_PX and y2 - y1 >= F.MIN_SIDE_PX


@pytest.mark.parametrize('bad', [
    (100, 100, 100, 200),        # zero width
    (100, 100, 200, 100),        # zero height
    (300, 100, 100, 200),        # inverted x
    (100, 300, 200, 100),        # inverted y
    (2000, 2000, 3000, 3000),    # entirely outside
])
def test_degenerate_boxes_rejected(bad):
    with pytest.raises(F.BBoxError):
        F.validate_and_clamp(bad, (1316, 728))


def test_absurd_aspect_ratio_rejected():
    with pytest.raises(F.BBoxError):
        F.validate_and_clamp((0, 300, 1316, 302), (1316, 728))


@pytest.mark.parametrize('bad', [(1, 2, 3), (1, 2, 3, 4, 5), ('a', 'b', 'c', 'd')])
def test_malformed_input_rejected(bad):
    with pytest.raises(F.BBoxError):
        F.validate_and_clamp(bad, (1316, 728))


def test_bbox_to_fetched_requires_explicit_frame():
    with pytest.raises(F.BBoxError):
        F.bbox_to_fetched((0, 0, 500, 500), 'whatever')


def test_bbox_to_fetched_frames_differ():
    """Same numbers, three frames, three different crops -- hence no default."""
    b = (100, 100, 500, 400)
    assert len({F.bbox_to_fetched(b, f) for f in ('norm', 'native', 'fetched')}) == 3


def test_bbox_frame_not_yet_verified():
    """Guard: flip to True only after the 20-sample live-model dump, and record it in
    CLAUDE.md at the same time. This test is the reminder."""
    assert F.BBOX_FRAME_VERIFIED is False


# --------------------------------------------------------------------------------------
# Faithfulness primitive
# --------------------------------------------------------------------------------------

def test_contains_point():
    assert F.contains_point((100, 100, 300, 300), (200, 200))
    assert F.contains_point((100, 100, 300, 300), (100, 100))     # boundary counts
    assert not F.contains_point((100, 100, 300, 300), (301, 200))

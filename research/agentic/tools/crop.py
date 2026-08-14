"""`image_crop` -- the v1-style look-closer tool.

Pure and in-process: no sandbox, no network. Targets the xy2d near-miss failure mode
(~30% of predictions are the right object just outside the 50 px tolerance) and `fb`.

THE ONE THING THAT MATTERS HERE IS THE COORDINATE FRAME. The crop must be taken in
FETCHED space (1316x728 at MAX_PIXELS=1003520), because that is the pixel grid the model
actually saw. A bbox emitted in NORM space and cropped directly yields the top-left ~76%
of the image, forever, without raising. See `frames.py` for the full argument.
"""

from __future__ import annotations

from typing import Any, Dict

from .. import frames as F
from .base import Observation, ToolContext

# Which frame the model emits its bbox in. Until the Phase-0 20-sample dump against a
# live model resolves this, it stays configurable and the default is stated loudly
# rather than assumed. Qwen emits *points* in 0-1000 by documented convention; bboxes
# are believed to follow but are NOT verified on SURDS.
DEFAULT_BBOX_FRAME = 'norm'


class ImageCropTool:
    name = 'image_crop'

    schema = {
        'type': 'function',
        'function': {
            'name': 'image_crop',
            'description': (
                'Crop a rectangular region of the image and return it as a new, magnified '
                'image. Use this to inspect a small or distant object more closely before '
                'committing to an answer. Coordinates are relative, on a 0-1000 grid over '
                'the full image, with (0,0) at the top-left.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'bbox_2d': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'description': '[x1, y1, x2, y2] on the 0-1000 grid, x1<x2 and y1<y2.',
                    },
                },
                'required': ['bbox_2d'],
            },
        },
    }

    def __init__(self, bbox_frame: str = DEFAULT_BBOX_FRAME, max_upscale: float = 4.0):
        self.bbox_frame = bbox_frame
        self.max_upscale = max_upscale

    def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> Observation:
        args = arguments or {}
        bbox = args.get('bbox_2d', args.get('bbox'))
        if bbox is None:
            return Observation.error("missing required argument 'bbox_2d'.")

        try:
            box = F.bbox_to_fetched(bbox, self.bbox_frame, ctx.spec)
        except F.BBoxError as e:
            return Observation.error(f'invalid bbox_2d {bbox!r}: {e}')
        except Exception as e:  # never raise into the rollout loop
            return Observation.error(f'could not interpret bbox_2d {bbox!r} ({type(e).__name__}: {e})')

        try:
            fetched = self._fetched_image(ctx)
            crop = fetched.crop(box)
        except Exception as e:
            return Observation.error(f'crop failed ({type(e).__name__}: {e})')

        crop = self._upscale(crop)
        x1, y1, x2, y2 = box
        text = (f'Cropped region [{x1}, {y1}, {x2}, {y2}] '
                f'({x2 - x1}x{y2 - y1} px of the {fetched.width}x{fetched.height} view), '
                f'shown below at {crop.width}x{crop.height}.')
        return Observation(
            text=text,
            images=[crop],
            ok=True,
            meta={'bbox_fetched': box, 'bbox_raw': list(bbox), 'bbox_frame': self.bbox_frame},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fetched_image(ctx: ToolContext):
        """The image in FETCHED space -- what the model actually saw.

        `qwen_vl_utils.fetch_image` is the single source of truth; re-implementing
        smart_resize here would be a second place for the frame to drift.
        """
        from qwen_vl_utils import fetch_image
        # max_pixels MUST be explicit: fetch_image defaults to 12845056, which yields
        # 1596x896 on a SURDS frame instead of the 1316x728 the spec was computed for.
        img = fetch_image({'image': ctx.image, 'max_pixels': ctx.spec.max_pixels})
        if img.size != ctx.spec.fetched_wh:
            raise RuntimeError(
                f'fetched size {img.size} != FrameSpec {ctx.spec.fetched_wh}; '
                'MAX_PIXELS disagreement between the harness and qwen_vl_utils')
        return img

    def _upscale(self, crop):
        """Magnify small crops so the encoder has patches to work with.

        A 40x30 crop re-encoded at native scale is ~2 patches and carries almost no more
        information than the original view did -- the tool would be a no-op. Bounded at
        `max_upscale` so we do not spend thousands of visual tokens on one crop.
        """
        from PIL import Image
        target = F.PATCH * 4  # aim for >=4 patches on the short side
        short = min(crop.width, crop.height)
        if short <= 0:
            return crop
        factor = min(self.max_upscale, max(1.0, target / short))
        if factor <= 1.0:
            return crop
        return crop.resize((int(crop.width * factor), int(crop.height * factor)), Image.BICUBIC)


TOOL = ImageCropTool

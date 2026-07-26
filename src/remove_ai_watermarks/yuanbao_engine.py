"""Tencent Yuanbao visible watermark detector and localizer.

Yuanbao stamps a compact italic two-line mark, ``元宝`` over ``AI生成``, in the
bottom-right corner. The same silhouette is rendered light on dark scenes and
dark on pale scenes, so a one-polarity white top-hat cannot detect it reliably.
This engine uses the shared text-mark pipeline with the ``contrast`` front-end:
normalized absolute local-luma residual followed by silhouette NCC.

The bundled silhouette is synthetic and font-rendered by
``scripts/render_vendor_silhouettes.py``. Removal follows the shared
localize-then-fill path and uses the detector's own match box.

Calibration (2026-07-25) used the metadata-harvested Tencent cohort after byte
deduplication and visual adjudication. The standard two-line variant was detected
on 26 of 28 unique marked carriers (92.9%) at gate 0.38, with 0 fires on 286
hand-labeled clean frames. The separate photographer-overlay variant is not
covered by this silhouette.
"""

# The module-level helpers are imported by tests.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

WM_WIDTH_FRAC = 0.20
WM_HEIGHT_FRAC = 0.15
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.38

_ALPHA_WIDTH_FRAC = 0.08
_ALPHA_HEIGHT_FRAC = 0.0446
_LADDER = (0.95, 1.0, 1.05)

_CONFIG = TextMarkConfig(
    name="Tencent Yuanbao",
    asset_name="yuanbao_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="contrast",
    scale_basis="short",
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=32,
    provenance_ncc_factor=1.0,
)

YuanbaoDetection = TextMarkDetection


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Yuanbao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary two-line Yuanbao silhouette (255 = glyph), or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Yuanbao silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class YuanbaoEngine(TextMarkEngine):
    """Detect and localize the bottom-right Yuanbao mark."""

    _ANCHOR_MAX_RIGHT = 0.04
    _ANCHOR_MAX_BOTTOM = 0.04

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def detect(self, image: NDArray[Any] | None, *, provenance: bool = False) -> TextMarkDetection:
        if image is None or not image.size:
            return TextMarkDetection()
        detection = super().detect(image, provenance=provenance)
        if not detection.detected:
            return detection
        location = self.locate(image)
        _, box = self._contrast_best(image, location)
        if box is None:
            detection.detected = False
            return detection
        h, w = image.shape[:2]
        base = min(h, w)
        right = (w - (location.x + box[2] + 1)) / base
        bottom = (h - (location.y + box[3] + 1)) / base
        if not (0 <= right <= self._ANCHOR_MAX_RIGHT and 0 <= bottom <= self._ANCHOR_MAX_BOTTOM):
            logger.debug(
                "Yuanbao detect: score %.3f but match off-anchor (right=%.3f bottom=%.3f); demoting.",
                detection.confidence,
                right,
                bottom,
            )
            detection.detected = False
        return detection


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as a BGR ndarray."""
    from remove_ai_watermarks import image_io

    image = image_io.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image

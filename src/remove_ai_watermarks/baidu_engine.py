"""Baidu visible watermark detector/localizer.

Baidu stamps its generations with a white bold "百度" text run plus a separate
white rounded tag carrying dark "AI生成", bottom-right -- the China TC260
explicit AIGC label. Detection keys on the **百度 text run only**: a
two-component template (text + pill tag) was measured and REJECTED -- the solid
white pill is a bright-blob magnet and both front-ends scored the clean arm at
cohort levels (tophat clean p95 0.445 / gray clean p95 0.487 vs cohort ~0.5,
2026-07-22). The text-only silhouette separates cleanly (below). The white tag
is still removed with the mark: the fill blob covers both bright components in
the corner box.

Removal is the shared **localize -> fill** (:meth:`footprint_mask` ->
``region_eraser``). This module supplies only Baidu's tuned
:class:`TextMarkConfig` (``assets/baidu_alpha.png`` -- a font-rendered
synthetic silhouette from ``scripts/render_vendor_silhouettes.py``, never cut
from an upload).

The detector uses a synthetic silhouette, short-side geometry, a strict
confidence gate, and a Qwen rival margin. The footprint covers both the text
run and its adjacent pill tag.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# Locate geometry as a fraction of the image SHORT side (measured basis). The
# box covers the text run AND the pill tag to its right (tag right edge ~0.002
# off the frame edge, text run left edge ~0.19 off).
WM_WIDTH_FRAC = 0.25
WM_HEIGHT_FRAC = 0.07
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

# Glyph appearance: white bold text on a usually-darker background (white
# top-hat), same overlay class as Doubao -- inherited, harmless because the
# tophat front-end turns these gates into weights.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
# Calibrated against vendor, rival-mark, and clean compatibility examples.
# The Qwen rival margin handles visually similar marks; the threshold rejects
# remaining unrelated bottom-right text.
DETECT_NCC_THRESHOLD = 0.48

# Detection-silhouette geometry (fraction of the short side): the 百度 text run
# only, measured 0.090 wide with aspect 0.51.
_ALPHA_WIDTH_FRAC = 0.090
_ALPHA_HEIGHT_FRAC = 0.046

# Tight ladder: the NCC comb is sharp in size (see runninghub_engine), so the
# nominal sits exactly on the measured 0.090 with +-5% rungs.
_LADDER = (0.95, 1.0, 1.05)

_CONFIG = TextMarkConfig(
    name="Baidu",
    asset_name="baidu_alpha.png",
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
    detect_frontend="tophat",
    scale_basis="short",
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # Load-bearing rival margins (crossfire measured 2026-07-22): the 百度 and
    # 豆包 silhouettes share their second glyph and a similar first, and 百度 vs
    # 千问 are near-identical after binarization -- at the 0.37 gate this
    # template fires on 45.8% of 400 Doubao-marked frames AND on Qwen-marked
    # frames at 0.38-0.43. Doubao's template beats it by ~0.56 on Doubao marks,
    # Qwen's by 0.17-0.35 on Qwen marks, so the 0.10 margin suppresses all of
    # that crossfire at zero genuine-Baidu cost (cohort fire+m == fire).
    rivals=("doubao_alpha.png", "qwen_alpha.png"),
    # STRICT ONLY: small cohort, the relaxed band is unmeasured.
    provenance_ncc_factor=1.0,
)

BaiduDetection = TextMarkDetection


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Baidu alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "百度" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Baidu glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class BaiduEngine(TextMarkEngine):
    """Detect/localize the visible Baidu "百度 AI生成" mark (bottom-right; localize -> fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def footprint_mask(
        self, image: NDArray[Any] | None, *, force: bool = False, dilate: int | None = None
    ) -> NDArray[Any] | None:
        """Full-frame mask of the WHOLE mark (text run + the pill tag to its right).

        The base class's blob-bbox footprint UNDERCOVERS this mark: the white tag's
        flat interior gives no top-hat response (a top-hat answers edges, not flats),
        so the blob ends at the text run and the fill leaves the tag's right half as
        a ghost (measured 2026-07-22 on the 768x1024 cohort frame: blob bbox x
        632..746 vs the tag ending ~758). The layout is measured and fixed -- the
        text run is at the left of the locate box, the tag runs to the corner -- so
        the footprint is the detector's match box extended RIGHT to the corner.
        """
        if image is None or image.size == 0:
            return None

        from remove_ai_watermarks import image_io, region_eraser

        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return None
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        if force:
            rx1, ry1, rx2, ry2 = bx, by, min(w, bx + bw), min(h, by + bh)
        else:
            if not self.detect(image).detected:
                return None
            _, box = self._tophat_best(image, loc)
            if box is None:
                return None
            gx0, gy0, _gx1, gy1 = box
            pad = max(4, int(0.15 * bh))
            rx1 = max(0, bx + gx0 - pad)
            ry1 = max(0, by + gy0 - pad)
            rx2 = min(w, bx + bw)  # the tag runs to the corner end of the box
            ry2 = min(h, by + gy1 + 1 + pad)
        if rx1 >= rx2 or ry1 >= ry2:
            return None
        d = dilate if dilate is not None else max(3, int(0.02 * bw))
        return region_eraser.boxes_to_mask((h, w), [(rx1, ry1, rx2 - rx1, ry2 - ry1)], dilate=d)


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

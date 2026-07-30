"""Visible AI-watermark localization and removal for video.

Supported marks use fully synthetic silhouettes made from geometric primitives,
OpenCV's built-in font, and Pillow's bundled font. Sora detection searches the
full frame because the wordmark moves. Veo detection covers both the current
four-point diamond and legacy ``Veo`` text. Seedance detects the boxed ``AI``
label, while Dola detects its compact text label. A single frame is never enough
to authorize removal: the temporal arbiter requires the candidate to recur at
the same location across adjacent frames. This keeps isolated lookalikes in
clean videos from becoming removal masks.

Video pixels are decoded with OpenCV and encoded with the system ``ffmpeg``.
Audio is stream-copied from the source. The video stream must be transcoded
because visible-mark removal changes pixels.
"""

# cv2/numpy boundary: these packages do not expose usable types for many array
# operations. Public signatures remain annotated while unknown third-party types
# are relaxed only in this module.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import Backend


log = logging.getLogger(__name__)

Region = tuple[int, int, int, int]

_NORMALIZED_SHORT_SIDE = 480
_SORA_TEMPLATE_SIZE = (180, 64)
_SORA_RELATIVE_HEIGHTS = (0.065, 0.075, 0.085, 0.095, 0.105)
_SORA_PROVENANCE_WEAK_CONFIDENCE = 0.58
_SORA_STRICT_WEAK_CONFIDENCE = 0.60
_SORA_STRONG_CONFIDENCE = 0.65
_VEO_PROVENANCE_WEAK_CONFIDENCE = 0.45
_VEO_STRICT_WEAK_CONFIDENCE = 0.50
_VEO_STRONG_CONFIDENCE = 0.55
_SEEDANCE_WEAK_CONFIDENCE = 0.38
_SEEDANCE_STRONG_CONFIDENCE = 0.43
_DOLA_PROVENANCE_WEAK_CONFIDENCE = 0.48
_DOLA_STRICT_WEAK_CONFIDENCE = 0.50
_DOLA_STRONG_CONFIDENCE = 0.52
_MIN_STABLE_FRAMES = 5
_MIN_VEO_STABLE_FRAMES = 12
_MIN_FIXED_MARK_STABLE_FRAMES = 12
_MAX_STABLE_GAP = 2
_STABLE_IOU = 0.55
_VEO_REFERENCE_SHORT_SIDE = 720
_VEO_DIAMOND_PROFILES = (
    (56, 92, 92),
    (48, 72, 72),
    (44, 29, 40),
)
_DOLA_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(22, 41))


@dataclass(frozen=True)
class FrameLocalization:
    """Best untrusted visible-mark candidate found in one decoded frame."""

    frame_index: int
    confidence: float
    region: Region | None


@dataclass(frozen=True)
class VideoScan:
    """Decoded video geometry plus one localization candidate per frame."""

    width: int
    height: int
    fps: float
    detections: tuple[FrameLocalization, ...]


def _scalable_default_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load Pillow's bundled scalable font, with a Pillow 10.0 fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # ``size=`` was added after Pillow 10.0, which is still within the
        # package's supported dependency range. Resize that bundled bitmap font
        # before compositing it into the synthetic template.
        return ImageFont.load_default()


@lru_cache(maxsize=1)
def _sora_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return synthetic full-wordmark and mascot-only silhouettes.

    No source frame or provider logo asset contributes pixels to these templates.
    The cloud-like mascot is assembled from primitive shapes and the word is
    rendered with Pillow's bundled font.
    """
    width, height = _SORA_TEMPLATE_SIZE
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((4, 8, 58, 57), radius=22, fill=255)
    draw.ellipse((0, 18, 26, 50), fill=255)
    draw.ellipse((38, 16, 64, 51), fill=255)
    draw.ellipse((16, 18, 29, 43), fill=0)
    draw.ellipse((35, 18, 48, 43), fill=0)

    font = _scalable_default_font(50)
    if isinstance(font, ImageFont.FreeTypeFont):
        draw.text((67, 0), "Sora", font=font, fill=255, stroke_width=1, stroke_fill=255)
    else:
        text_box = font.getbbox("Sora")
        text = Image.new("L", (max(1, text_box[2]), max(1, text_box[3])), 0)
        ImageDraw.Draw(text).text((0, 0), "Sora", font=font, fill=255)
        text = text.resize((102, 50), Image.Resampling.NEAREST)
        canvas.paste(text, (67, 0), text)

    full = np.asarray(canvas, dtype=np.uint8)
    return full, full[:, :64]


@lru_cache(maxsize=1)
def _veo_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return synthetic current-diamond and legacy-text Veo silhouettes."""
    size = 256
    diamond_canvas = Image.new("L", (size, size), 0)
    diamond_points = (
        (0.50, 0.02),
        (0.60, 0.39),
        (0.98, 0.50),
        (0.60, 0.61),
        (0.50, 0.98),
        (0.40, 0.61),
        (0.02, 0.50),
        (0.40, 0.39),
    )
    ImageDraw.Draw(diamond_canvas).polygon(
        [(round(x * size), round(y * size)) for x, y in diamond_points],
        fill=255,
    )

    text_canvas = Image.new("L", (140, 60), 0)
    text_draw = ImageDraw.Draw(text_canvas)
    text_draw.text((2, 0), "Veo", font=_scalable_default_font(48), fill=255)
    text = np.asarray(text_canvas, dtype=np.uint8)
    ys, xs = np.where(text > 0)
    text = text[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return np.asarray(diamond_canvas, dtype=np.uint8), text


@lru_cache(maxsize=1)
def _seedance_template() -> NDArray[Any]:
    """Return a synthetic boxed-AI silhouette for Seedance exports."""
    canvas = Image.new("L", (160, 120), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((8, 8, 142, 105), radius=28, outline=255, width=7)
    draw.text(
        (35, 17),
        "AI",
        font=_scalable_default_font(70),
        fill=255,
        stroke_width=1,
        stroke_fill=255,
    )
    draw.rounded_rectangle((142, 94, 157, 109), radius=3, outline=255, width=2)
    draw.text((145, 94), "AI", font=_scalable_default_font(8), fill=255)
    return np.asarray(canvas, dtype=np.uint8)


@lru_cache(maxsize=1)
def _dola_template() -> NDArray[Any]:
    """Return a synthetic Dola AI text silhouette using OpenCV's font."""
    canvas = np.zeros((100, 400), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Dola AI",
        (2, 72),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        255,
        3,
        cv2.LINE_AA,
    )
    ys, xs = np.where(canvas > 0)
    return canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _top_hat(gray: NDArray[Any]) -> NDArray[Any]:
    kernel = np.ones((7, 7), dtype=np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def _normalized_gray(image_bgr: NDArray[Any]) -> tuple[NDArray[Any], float]:
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    short_side = min(height, width)
    if short_side <= 0:
        return gray, 1.0
    scale = min(1.0, _NORMALIZED_SHORT_SIDE / short_side)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _expanded_region(
    location: tuple[int, int],
    template_width: int,
    template_height: int,
    *,
    icon_only: bool,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region:
    x = round(location[0] / scale)
    y = round(location[1] / scale)
    height = max(1, round(template_height / scale))
    width = max(1, round(template_width / scale))
    if icon_only:
        width = round(height * _SORA_TEMPLATE_SIZE[0] / _SORA_TEMPLATE_SIZE[1])
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    return x, y, max(1, width), max(1, height)


def detect_sora_frame(image_bgr: NDArray[Any], *, frame_index: int = 0) -> FrameLocalization:
    """Locate the strongest synthetic Sora-wordmark match in one frame.

    The returned candidate is intentionally untrusted. Call
    :func:`stabilize_sora_localizations` across the full sequence before building
    any removal mask.
    """
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray, scale = _normalized_gray(image_bgr)
    normalized_height, normalized_width = gray.shape[:2]
    feature = _top_hat(gray)
    best_confidence = 0.0
    best_region: Region | None = None

    for template_index, base_template in enumerate(_sora_templates()):
        icon_only = template_index == 1
        for relative_height in _SORA_RELATIVE_HEIGHTS:
            template_height = max(16, round(min(normalized_height, normalized_width) * relative_height))
            template_width = max(1, round(base_template.shape[1] * template_height / base_template.shape[0]))
            if template_height >= normalized_height or template_width >= normalized_width:
                continue
            template = cv2.resize(
                base_template,
                (template_width, template_height),
                interpolation=cv2.INTER_AREA,
            )
            scores = cv2.matchTemplate(feature, _top_hat(template), cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(scores)
            if confidence <= best_confidence:
                continue
            best_confidence = float(confidence)
            best_region = _expanded_region(
                location,
                template_width,
                template_height,
                icon_only=icon_only,
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )

    return FrameLocalization(frame_index, best_confidence, best_region)


def _match_template(
    gray: NDArray[Any],
    template: NDArray[Any],
    *,
    region: Region,
    kernel_size: int,
) -> tuple[float, Region | None]:
    """Match one synthetic silhouette inside a bounded frame region."""
    x, y, width, height = region
    roi = gray[y : y + height, x : x + width]
    template_height, template_width = template.shape[:2]
    if roi.size == 0 or template_height >= roi.shape[0] or template_width >= roi.shape[1]:
        return 0.0, None
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    feature = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    template_feature = cv2.morphologyEx(template, cv2.MORPH_TOPHAT, kernel)
    scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(scores)
    return float(confidence), (
        x + location[0],
        y + location[1],
        template_width,
        template_height,
    )


def _restore_region(
    region: Region | None,
    *,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region | None:
    """Map a localization from normalized pixels back to the source frame."""
    if region is None:
        return None
    x, y, width, height = region
    source_x = round(x / scale)
    source_y = round(y / scale)
    source_width = min(frame_width - source_x, max(1, round(width / scale)))
    source_height = min(frame_height - source_y, max(1, round(height / scale)))
    return source_x, source_y, source_width, source_height


def _detect_fixed_bottom_right_mark(
    image_bgr: NDArray[Any],
    template: NDArray[Any],
    *,
    relative_heights: tuple[float, ...],
    search_origin: tuple[float, float],
    kernel_fraction: float,
    frame_index: int,
) -> FrameLocalization:
    """Match one fixed bottom-right synthetic mark on a normalized frame."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray, scale = _normalized_gray(image_bgr)
    normalized_height, normalized_width = gray.shape[:2]
    short_side = min(normalized_height, normalized_width)
    search_x = round(normalized_width * search_origin[0])
    search_y = round(normalized_height * search_origin[1])
    search_region = (
        search_x,
        search_y,
        normalized_width - search_x,
        normalized_height - search_y,
    )
    best_confidence = 0.0
    best_region: Region | None = None
    for relative_height in relative_heights:
        template_height = max(6, round(short_side * relative_height))
        template_width = max(1, round(template.shape[1] * template_height / template.shape[0]))
        resized = cv2.resize(
            template,
            (template_width, template_height),
            interpolation=cv2.INTER_AREA,
        )
        confidence, candidate = _match_template(
            gray,
            resized,
            region=search_region,
            kernel_size=max(3, round(template_height * kernel_fraction) | 1),
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    return FrameLocalization(
        frame_index,
        best_confidence,
        _restore_region(
            best_region,
            scale=scale,
            frame_width=frame_width,
            frame_height=frame_height,
        ),
    )


def detect_seedance_frame(image_bgr: NDArray[Any], *, frame_index: int = 0) -> FrameLocalization:
    """Locate the strongest fixed Seedance boxed-AI candidate."""
    return _detect_fixed_bottom_right_mark(
        image_bgr,
        _seedance_template(),
        relative_heights=(0.065, 0.075, 0.085, 0.095, 0.105),
        search_origin=(0.68, 0.72),
        kernel_fraction=0.12,
        frame_index=frame_index,
    )


def detect_dola_frame(image_bgr: NDArray[Any], *, frame_index: int = 0) -> FrameLocalization:
    """Locate the strongest fixed Dola AI text candidate."""
    return _detect_fixed_bottom_right_mark(
        image_bgr,
        _dola_template(),
        relative_heights=_DOLA_RELATIVE_HEIGHTS,
        search_origin=(0.65, 0.85),
        kernel_fraction=0.50,
        frame_index=frame_index,
    )


def detect_veo_frame(image_bgr: NDArray[Any], *, frame_index: int = 0) -> FrameLocalization:
    """Locate the strongest current-diamond or legacy-text Veo candidate."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    short_scale = min(frame_height, frame_width) / _VEO_REFERENCE_SHORT_SIDE
    diamond_base, text_base = _veo_templates()
    best_confidence = 0.0
    best_region: Region | None = None

    diamond_sizes: set[int] = set()
    for base_size, right_margin, bottom_margin in _VEO_DIAMOND_PROFILES:
        diamond_size = max(16, round(base_size * short_scale))
        diamond_sizes.add(diamond_size)
        template = cv2.resize(
            diamond_base,
            (diamond_size, diamond_size),
            interpolation=cv2.INTER_AREA,
        )
        expected_x = round(frame_width - (right_margin + base_size) * short_scale)
        expected_y = round(frame_height - (bottom_margin + base_size) * short_scale)
        search_padding = max(6, round(diamond_size * 0.25))
        search_x = max(0, expected_x - search_padding)
        search_y = max(0, expected_y - search_padding)
        search_width = min(frame_width - search_x, diamond_size + search_padding * 2)
        search_height = min(frame_height - search_y, diamond_size + search_padding * 2)
        confidence, candidate = _match_template(
            gray,
            template,
            region=(search_x, search_y, search_width, search_height),
            kernel_size=max(3, round(7 * short_scale) | 1),
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    # Provider layouts have moved before. A bounded corner search is a safety
    # net for a relocated diamond, but it is admitted only at a much stronger
    # per-frame score than the known-profile path. Without this gate, recurring
    # bright scene details in clean API exports can become stable false matches.
    corner_x = round(frame_width * 0.65)
    corner_y = round(frame_height * 0.65)
    corner_region = (corner_x, corner_y, frame_width - corner_x, frame_height - corner_y)
    for diamond_size in diamond_sizes:
        template = cv2.resize(
            diamond_base,
            (diamond_size, diamond_size),
            interpolation=cv2.INTER_AREA,
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=corner_region,
            kernel_size=max(3, round(7 * short_scale) | 1),
        )
        if confidence >= 0.70 and confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    text_region_width = min(frame_width, max(32, round(180 * short_scale)))
    text_region_height = min(frame_height, max(24, round(120 * short_scale)))
    text_region = (
        frame_width - text_region_width,
        frame_height - text_region_height,
        text_region_width,
        text_region_height,
    )
    text_heights = sorted({max(5, round(height * short_scale)) for height in range(8, 22)})
    for text_height in text_heights:
        text_width = max(1, round(text_base.shape[1] * text_height / text_base.shape[0]))
        template = cv2.resize(
            text_base,
            (text_width, text_height),
            interpolation=cv2.INTER_AREA,
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=text_region,
            kernel_size=max(3, round(3 * short_scale) | 1),
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    return FrameLocalization(frame_index, best_confidence, best_region)


def _region_iou(left: Region, right: Region) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0 = max(lx, rx)
    y0 = max(ly, ry)
    x1 = min(lx + lw, rx + rw)
    y1 = min(ly + lh, ry + rh)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def stabilize_sora_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
) -> list[Region | None]:
    """Accept only spatially recurring Sora candidates and bridge short dropouts.

    Metadata never creates a detection. It only allows a stable visual run whose
    scores remain below the strict confidence floor, which covers low-contrast
    Sora marks while clean metadata-bearing exports stay untouched.
    """
    weak_floor = _SORA_PROVENANCE_WEAK_CONFIDENCE if provenance else _SORA_STRICT_WEAK_CONFIDENCE
    return _stabilize_localizations(
        detections,
        provenance=provenance,
        weak_floor=weak_floor,
        strong_floor=_SORA_STRONG_CONFIDENCE,
        transition_floor=0.45,
        min_stable_frames=_MIN_STABLE_FRAMES,
        cover_after_confirmation=False,
    )


def stabilize_veo_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
) -> list[Region | None]:
    """Accept temporally recurring current or legacy Veo candidates."""
    weak_floor = _VEO_PROVENANCE_WEAK_CONFIDENCE if provenance else _VEO_STRICT_WEAK_CONFIDENCE
    return _stabilize_localizations(
        detections,
        provenance=provenance,
        weak_floor=weak_floor,
        strong_floor=_VEO_STRONG_CONFIDENCE,
        transition_floor=0.35,
        min_stable_frames=_MIN_VEO_STABLE_FRAMES,
        cover_after_confirmation=True,
    )


def stabilize_seedance_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
) -> list[Region | None]:
    """Accept a recurring Seedance boxed-AI mark at a fixed position."""
    return _stabilize_localizations(
        detections,
        provenance=provenance,
        weak_floor=_SEEDANCE_WEAK_CONFIDENCE,
        strong_floor=_SEEDANCE_STRONG_CONFIDENCE,
        transition_floor=0.30,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
    )


def stabilize_dola_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
) -> list[Region | None]:
    """Accept a recurring Dola AI text mark at a fixed position."""
    weak_floor = _DOLA_PROVENANCE_WEAK_CONFIDENCE if provenance else _DOLA_STRICT_WEAK_CONFIDENCE
    return _stabilize_localizations(
        detections,
        provenance=provenance,
        weak_floor=weak_floor,
        strong_floor=_DOLA_STRONG_CONFIDENCE,
        transition_floor=0.40,
        min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation=True,
        anchor_iou=0.80,
    )


def _stabilize_localizations(
    detections: tuple[FrameLocalization, ...] | list[FrameLocalization],
    *,
    provenance: bool,
    weak_floor: float,
    strong_floor: float,
    transition_floor: float,
    min_stable_frames: int,
    cover_after_confirmation: bool,
    anchor_iou: float | None = None,
) -> list[Region | None]:
    """Apply the shared recurrence policy to provider-specific candidates."""
    accepted: list[Region | None] = [None] * len(detections)
    runs: list[list[int]] = []
    current: list[int] = []

    for position, detection in enumerate(detections):
        if detection.region is None or detection.confidence < weak_floor:
            continue
        if current:
            previous = detections[current[-1]]
            anchor = detections[current[0]]
            frame_gap = detection.frame_index - previous.frame_index
            if (
                previous.region is None
                or anchor.region is None
                or frame_gap > _MAX_STABLE_GAP + 1
                or _region_iou(previous.region, detection.region) < _STABLE_IOU
                or (anchor_iou is not None and _region_iou(anchor.region, detection.region) < anchor_iou)
            ):
                runs.append(current)
                current = []
        current.append(position)
    if current:
        runs.append(current)

    for run in runs:
        strong = max(detections[position].confidence for position in run) >= strong_floor
        if len(run) < min_stable_frames or (not provenance and not strong):
            continue
        for position in run:
            accepted[position] = detections[position].region
        for left_position, right_position in pairwise(run):
            if right_position - left_position <= 1:
                continue
            left = detections[left_position]
            right = detections[right_position]
            if left.region is None or right.region is None or _region_iou(left.region, right.region) < _STABLE_IOU:
                continue
            for missing_position in range(left_position + 1, right_position):
                distance_left = missing_position - left_position
                distance_right = right_position - missing_position
                accepted[missing_position] = left.region if distance_left <= distance_right else right.region

    # Provider provenance plus a confirmed run establishes a continuously
    # watermarked app export rather than a clean API export that merely shares
    # the generator name. Veo may also cover the sequence without provenance
    # after its longer, strong fixed-position run. Cover low-contrast transition
    # frames with the nearest confirmed position.
    confirmed_positions = [position for position, region in enumerate(accepted) if region is not None]
    if (provenance or cover_after_confirmation) and confirmed_positions:
        confirmed_regions = [accepted[position] for position in confirmed_positions]
        carry_position = confirmed_positions[0]
        for position, region in enumerate(accepted):
            if region is not None:
                carry_position = position
                continue
            raw = detections[position]
            if (
                raw.region is not None
                and raw.confidence >= transition_floor
                and any(
                    confirmed_region is not None and _region_iou(raw.region, confirmed_region) >= _STABLE_IOU
                    for confirmed_region in confirmed_regions
                )
            ):
                accepted[position] = raw.region
                continue
            accepted[position] = accepted[carry_position]

    return accepted


def _scan_video(
    source: Path,
    detector: Any,
) -> VideoScan:
    """Decode a video once and collect one untrusted candidate per frame."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid stream geometry or frame rate: {source}")

    detections: list[FrameLocalization] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            capture.release()
            raise RuntimeError(f"Video changes frame dimensions at frame {frame_index}: {source}")
        detections.append(detector(frame, frame_index=frame_index))
        frame_index += 1
    capture.release()
    if not detections:
        raise RuntimeError(f"Video contains no decodable frames: {source}")
    return VideoScan(width, height, fps, tuple(detections))


def scan_sora_video(source: Path) -> VideoScan:
    """Decode a video once and collect one untrusted Sora candidate per frame."""
    return _scan_video(source, detect_sora_frame)


def scan_veo_video(source: Path) -> VideoScan:
    """Decode a video once and collect one untrusted Veo candidate per frame."""
    return _scan_video(source, detect_veo_frame)


def scan_seedance_video(source: Path) -> VideoScan:
    """Decode a video once and collect one untrusted Seedance candidate per frame."""
    return _scan_video(source, detect_seedance_frame)


def scan_dola_video(source: Path) -> VideoScan:
    """Decode a video once and collect one untrusted Dola candidate per frame."""
    return _scan_video(source, detect_dola_frame)


def _ffmpeg_video_args(suffix: str) -> list[str]:
    if suffix == ".webm":
        return ["-c:v", "libvpx-vp9", "-crf", "18", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "14"]


def _mask_for_region(
    frame_bgr: NDArray[Any],
    region: Region,
    *,
    padding_fraction: float,
    mask_style: Literal["box", "veo"],
) -> NDArray[Any]:
    height, width = frame_bgr.shape[:2]
    x, y, region_width, region_height = region
    mask = np.zeros((height, width), dtype=np.uint8)
    if mask_style == "veo" and 0.80 <= region_width / region_height <= 1.25:
        diamond_base, _ = _veo_templates()
        diamond = cv2.resize(
            diamond_base,
            (region_width, region_height),
            interpolation=cv2.INTER_AREA,
        )
        diamond = np.where(diamond >= 24, 255, 0).astype(np.uint8)
        dilation = max(2, round(region_height * 0.08))
        kernel_size = dilation * 2 + 1
        diamond = cv2.dilate(
            diamond,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        x1 = min(width, x + region_width)
        y1 = min(height, y + region_height)
        mask[y:y1, x:x1] = diamond[: y1 - y, : x1 - x]
        return mask

    # A glyph-shaped mask leaves a thin translucent rim outside the approximate
    # synthetic silhouette. Classical inpainting then pulls that white rim back
    # into the hole, recreating the mascot as a bright blob. The measured clean
    # floor on real Sora frames is a full box with roughly 0.28 mark-heights of
    # context on every side.
    padding = max(4, round(region_height * padding_fraction))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + region_width + padding)
    y1 = min(height, y + region_height + padding)
    mask[y0:y1, x0:x1] = 255
    return mask


def encode_clean_video(
    source: Path,
    output: Path,
    scan: VideoScan,
    regions: list[Region | None],
    *,
    backend: Backend,
    strip_metadata: bool,
    padding_fraction: float = 0.28,
    mask_style: Literal["box", "veo"] = "box",
) -> int:
    """Decode again, fill accepted regions, and encode video while copying audio."""
    from remove_ai_watermarks.watermark_registry import fill, resolve_backend

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Visible video removal requires ffmpeg on PATH")
    if len(regions) != len(scan.detections):
        raise ValueError("Temporal localization count does not match the scanned frame count")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{scan.width}x{scan.height}",
        "-r",
        f"{scan.fps:.12g}",
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        *_ffmpeg_video_args(output.suffix.lower()),
        "-c:a",
        "copy",
        "-map_metadata",
        "-1" if strip_metadata else "1",
        "-map_chapters",
        "-1" if strip_metadata else "1",
        "-shortest",
    ]
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output))
    log.info("Encoding visible-watermark removal with ffmpeg: command=%s", command)

    process = subprocess.Popen(  # noqa: S603
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Could not open ffmpeg pipes")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        process.kill()
        raise RuntimeError(f"OpenCV could not reopen video for removal: {source}")

    removed_frames = 0
    resolved_backend: Literal["cv2", "migan", "lama"] = resolve_backend(backend)
    try:
        for frame_index, region in enumerate(regions):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended while reading frame {frame_index}: {source}")
            if region is not None:
                frame = fill(
                    frame,
                    _mask_for_region(
                        frame,
                        region,
                        padding_fraction=padding_fraction,
                        mask_style=mask_style,
                    ),
                    backend=resolved_backend,
                )
                removed_frames += 1
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        capture.release()

    log.info("ffmpeg visible-watermark encode finished: status=%s stderr=%s", return_code, stderr)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed to encode {output}: {stderr.strip()[:500]}")
    return removed_frames


def has_sora_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance specifically names the Sora generator."""
    return "sora" in markers.get("claim_generator", "").lower()


def has_veo_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance names Google as the AI-video generator."""
    identity = " ".join(
        (
            markers.get("claim_generator", ""),
            markers.get("issuer", ""),
        )
    ).lower()
    return "google" in identity and "trainedalgorithmicmedia" in markers.get("source_type", "").lower()


def has_bytedance_video_provenance(markers: dict[str, str]) -> bool:
    """Whether container provenance names ByteDance or BytePlus AI video."""
    identity = " ".join(
        (
            markers.get("claim_generator", ""),
            markers.get("issuer", ""),
        )
    ).lower()
    source_type = markers.get("source_type", "").lower()
    return ("bytedance" in identity or "byteplus" in identity) and "trainedalgorithmicmedia" in source_type

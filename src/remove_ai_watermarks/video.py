"""High-level video processing API.

Supported experimental stages are container-level AI metadata inspection and
removal, temporally stabilized visible Sora, Veo, Seedance, Dola, Hailuo, and
Kling removal, and VAE regeneration that produces an externally verifiable
SynthID candidate. The visible pixel path reuses the image package's shared
fill backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    DEFAULT_VIDEO_SYNTHID_VAE,
)

if TYPE_CHECKING:
    from remove_ai_watermarks.video_invisible import RegenerationMetrics

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv"})
_ISOBMFF_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v"})
_EBML_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".webm", ".mkv"})
_RIFF_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".avi"})
_FLV_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".flv"})
_REGENERATED_VIDEO_EXTENSIONS: frozenset[str] = _ISOBMFF_VIDEO_EXTENSIONS
_EBML_MAGIC = b"\x1aE\xdf\xa3"


@dataclass(frozen=True)
class VideoMetadataReport:
    """AI metadata found in one supported video container."""

    source: Path
    has_ai_metadata: bool
    markers: dict[str, str]


@dataclass(frozen=True)
class VideoMetadataResult:
    """Result of a verified video metadata-removal operation."""

    source: Path
    output: Path
    detected: dict[str, str]
    remaining: dict[str, str]


@dataclass(frozen=True)
class VideoVisibleResult:
    """Result of visible AI-watermark removal from a video."""

    source: Path
    output: Path | None
    mark: str
    total_frames: int
    detected_frames: int
    removed_frames: int
    remaining_metadata: dict[str, str]


@dataclass(frozen=True)
class VideoInvisibleResult:
    """Result of generating an externally verifiable SynthID candidate."""

    source: Path
    output: Path
    noise_std: float
    metrics: RegenerationMetrics
    remaining_metadata: dict[str, str]
    requires_external_verification: ClassVar[Literal[True]] = True

    @property
    def total_frames(self) -> int:
        return self.metrics.frames

    @property
    def fps(self) -> float:
        return self.metrics.fps

    @property
    def width(self) -> int:
        return self.metrics.width

    @property
    def height(self) -> int:
        return self.metrics.height

    @property
    def psnr_db(self) -> float:
        return self.metrics.psnr_db

    @property
    def temporal_residual_ratio(self) -> float:
        return self.metrics.temporal_residual_ratio


def _video_source(source: str | Path) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video source must be a file: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format {path.suffix or '<none>'}; expected one of: {supported}")
    with path.open("rb") as stream:
        head = stream.read(12)
    suffix = path.suffix.lower()
    matches_container = (
        (suffix in _ISOBMFF_VIDEO_EXTENSIONS and len(head) >= 8 and head[4:8] == b"ftyp")
        or (suffix in _EBML_VIDEO_EXTENSIONS and head.startswith(_EBML_MAGIC))
        or (suffix in _RIFF_VIDEO_EXTENSIONS and len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI ")
        or (suffix in _FLV_VIDEO_EXTENSIONS and head.startswith(b"FLV"))
    )
    if not matches_container:
        raise ValueError(f"Video content does not match its {suffix} extension: {path}")
    return path


def _video_output(
    source: Path,
    output: str | Path | None,
    *,
    operation: str = "metadata removal",
) -> Path:
    path = Path(output) if output is not None else source.with_stem(source.stem + "_clean")
    if path.suffix.lower() != source.suffix.lower():
        raise ValueError(
            f"Video output container must match the source ({source.suffix}); "
            f"{operation} does not change containers to {path.suffix or '<none>'}"
        )
    if path.resolve() == source.resolve():
        raise ValueError(f"Video {operation} requires a distinct output path")
    return path


def inspect_video_metadata(source: str | Path) -> VideoMetadataReport:
    """Inspect supported AI-provenance metadata in a video container."""
    from remove_ai_watermarks.metadata import get_ai_metadata, has_ai_metadata

    source_path = _video_source(source)
    return VideoMetadataReport(
        source=source_path,
        has_ai_metadata=has_ai_metadata(source_path),
        markers=get_ai_metadata(source_path),
    )


def remove_video_metadata(
    source: str | Path,
    output: str | Path | None = None,
    *,
    keep_standard: bool = True,
) -> VideoMetadataResult:
    """Remove AI metadata without transcoding video or audio streams.

    The default output is ``<source_stem>_clean<source_suffix>``. A separate
    output is required so an experimental video operation never overwrites the
    original file.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata, strip_and_verify

    source_path = _video_source(source)
    output_path = _video_output(source_path, output)
    detected = get_ai_metadata(source_path)
    written, remaining = strip_and_verify(source_path, output_path, keep_standard=keep_standard)
    return VideoMetadataResult(
        source=source_path,
        output=written,
        detected=detected,
        remaining=remaining,
    )


def remove_video_visible(
    source: str | Path,
    output: str | Path | None = None,
    *,
    mark: str = "sora",
    backend: str = "cv2",
    strip_metadata: bool = True,
) -> VideoVisibleResult:
    """Remove a supported visible AI wordmark from a video.

    Supported marks are ``sora``, ``veo``, ``seedance``, ``dola``, ``hailuo``,
    and ``kling``. The full sequence is scanned before pixels change, and only
    recurring candidates are accepted. Audio is copied without re-encoding;
    video is transcoded because the pixels change. When no stable mark is found,
    no output is written and ``output`` in the result is ``None``.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_visible import (
        encode_clean_video,
        has_bytedance_video_provenance,
        has_sora_provenance,
        has_veo_provenance,
        scan_dola_video,
        scan_hailuo_video,
        scan_kling_video,
        scan_seedance_video,
        scan_sora_video,
        scan_veo_video,
        stabilize_dola_localizations,
        stabilize_hailuo_localizations,
        stabilize_kling_localizations,
        stabilize_seedance_localizations,
        stabilize_sora_localizations,
        stabilize_veo_localizations,
    )
    from remove_ai_watermarks.watermark_registry import resolve_backend

    if mark not in {"sora", "veo", "seedance", "dola", "hailuo", "kling"}:
        raise ValueError("Unsupported visible video mark; expected sora, veo, seedance, dola, hailuo, or kling")
    if backend not in {"auto", "cv2", "migan", "lama"}:
        raise ValueError("Unsupported fill backend; expected auto, cv2, migan, or lama")

    source_path = _video_source(source)
    output_path = _video_output(source_path, output, operation="visible watermark removal")
    markers = get_ai_metadata(source_path)
    if mark == "sora":
        scan = scan_sora_video(source_path)
        regions = stabilize_sora_localizations(
            scan.detections,
            provenance=has_sora_provenance(markers),
        )
        padding_fraction = 0.28
        mask_style = "box"
    elif mark == "veo":
        scan = scan_veo_video(source_path)
        regions = stabilize_veo_localizations(
            scan.detections,
            provenance=has_veo_provenance(markers),
        )
        padding_fraction = 0.18
        mask_style = "veo"
    elif mark == "seedance":
        scan = scan_seedance_video(source_path)
        regions = stabilize_seedance_localizations(
            scan.detections,
            provenance=has_bytedance_video_provenance(markers),
        )
        padding_fraction = 0.0
        mask_style = "box"
    elif mark == "dola":
        scan = scan_dola_video(source_path)
        regions = stabilize_dola_localizations(
            scan.detections,
            provenance=has_bytedance_video_provenance(markers),
        )
        padding_fraction = 0.20
        mask_style = "box"
    elif mark == "hailuo":
        scan = scan_hailuo_video(source_path)
        regions = stabilize_hailuo_localizations(scan.detections)
        padding_fraction = 0.12
        mask_style = "box"
    else:
        scan = scan_kling_video(source_path)
        regions = stabilize_kling_localizations(scan.detections)
        padding_fraction = 0.12
        mask_style = "box"
    detected_frames = sum(region is not None for region in regions)
    if detected_frames == 0:
        return VideoVisibleResult(
            source=source_path,
            output=None,
            mark=mark,
            total_frames=len(scan.detections),
            detected_frames=0,
            removed_frames=0,
            remaining_metadata=markers if strip_metadata else {},
        )

    # Validate optional model availability before ffmpeg creates or overwrites
    # the requested output.
    resolve_backend(backend)  # type: ignore[arg-type]
    removed_frames = encode_clean_video(
        source_path,
        output_path,
        scan,
        regions,
        backend=backend,  # type: ignore[arg-type]
        strip_metadata=strip_metadata,
        padding_fraction=padding_fraction,
        mask_style=mask_style,
    )
    remaining_metadata = get_ai_metadata(output_path) if strip_metadata else {}
    return VideoVisibleResult(
        source=source_path,
        output=output_path,
        mark=mark,
        total_frames=len(scan.detections),
        detected_frames=detected_frames,
        removed_frames=removed_frames,
        remaining_metadata=remaining_metadata,
    )


def remove_video_invisible(
    source: str | Path,
    output: str | Path | None = None,
    *,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
) -> VideoInvisibleResult:
    """Generate a video SynthID-removal candidate through VAE regeneration.

    The function strips source metadata during the transcode, but cannot verify
    the proprietary pixel watermark locally. ``requires_external_verification``
    therefore remains true for every result. Verify important output with
    Google's matching content-verification flow.
    """
    from remove_ai_watermarks.metadata import get_ai_metadata
    from remove_ai_watermarks.video_invisible import regenerate_video_candidate

    source_path = _video_source(source)
    if source_path.suffix.lower() not in _REGENERATED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(_REGENERATED_VIDEO_EXTENSIONS))
        raise ValueError(f"Video SynthID regeneration requires one of: {supported}")
    candidate_output = (
        Path(output) if output is not None else source_path.with_stem(source_path.stem + "_synthid_candidate")
    )
    output_path = _video_output(
        source_path,
        candidate_output,
        operation="SynthID candidate generation",
    )
    metrics = regenerate_video_candidate(
        source_path,
        output_path,
        noise_std=noise_std,
        long_side=long_side,
        fps=fps,
        batch_size=batch_size,
        seed=seed,
        model=model,
        device=device,
    )
    return VideoInvisibleResult(
        source=source_path,
        output=output_path,
        noise_std=noise_std,
        metrics=metrics,
        remaining_metadata=get_ai_metadata(output_path),
    )

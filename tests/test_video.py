"""Tests for the video processing API and CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw, ImageFont

from remove_ai_watermarks.cli import main
from remove_ai_watermarks.metadata import C2PA_UUID

if TYPE_CHECKING:
    from pathlib import Path


_MP4_FTYP = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
_VIDEO_PAYLOAD = b"synthetic-video-payload"
_TC260_AIGC = (
    b'{"Label":"1","ContentProducer":"00119144030008867405X210002",'
    b'"ProduceID":"sample-001","ReservedCode1":"","ContentPropagator":"",'
    b'"PropagateID":"","ReservedCode2":""}'
)


def _box(box_type: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def _video_with_c2pa(path: Path) -> Path:
    manifest = C2PA_UUID + b"OpenAI trainedAlgorithmicMedia"
    path.write_bytes(_MP4_FTYP + _box(b"uuid", manifest) + _box(b"mdat", _VIDEO_PAYLOAD))
    return path


def _metadata_key(name: bytes) -> bytes:
    return (8 + len(name)).to_bytes(4, "big") + b"mdta" + name


def _metadata_value(index: int, value: bytes) -> bytes:
    data = _box(b"data", b"\x00\x00\x00\x01\x00\x00\x00\x00" + value)
    return _box(index.to_bytes(4, "big"), data)


def _video_with_tc260(path: Path, *, media_payload: bytes = _VIDEO_PAYLOAD) -> Path:
    keys = _box(
        b"keys",
        b"\x00\x00\x00\x00" + (2).to_bytes(4, "big") + _metadata_key(b"AIGC") + _metadata_key(b"title"),
    )
    ilst = _box(
        b"ilst",
        _metadata_value(1, _TC260_AIGC) + _metadata_value(2, b"standard title"),
    )
    meta = _box(b"meta", b"\x00\x00\x00\x00" + keys + ilst)
    path.write_bytes(_MP4_FTYP + _box(b"mdat", media_payload) + _box(b"moov", _box(b"udta", meta)))
    return path


def _ebml_size(value: int) -> bytes:
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    raise ValueError("EBML test value is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def _video_with_tc260_ebml(path: Path, *, value: bytes = _TC260_AIGC) -> Path:
    simple_tag = _ebml_element(
        b"\x67\xc8",
        _ebml_element(b"\x45\xa3", b"AIGC") + _ebml_element(b"\x44\x87", value),
    )
    tags = _ebml_element(b"\x12\x54\xc3\x67", _ebml_element(b"\x73\x73", simple_tag))
    segment = _ebml_element(b"\x18\x53\x80\x67", tags)
    path.write_bytes(_ebml_element(b"\x1a\x45\xdf\xa3", b"") + segment)
    return path


def _regeneration_metrics(
    *,
    frames: int = 24,
    fps: float = 12.0,
    width: int = 512,
    height: int = 288,
    psnr_db: float = 22.0,
    temporal_residual_ratio: float = 1.2,
):
    from remove_ai_watermarks.video_invisible import RegenerationMetrics

    return RegenerationMetrics(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        psnr_db=psnr_db,
        temporal_residual_ratio=temporal_residual_ratio,
    )


class TestVideoMetadataApi:
    def test_top_level_api_is_lazy_exported(self):
        import remove_ai_watermarks as raiw

        assert raiw.inspect_video_metadata is not None
        assert raiw.remove_video_invisible is not None
        assert raiw.remove_video_metadata is not None
        assert raiw.remove_video_visible is not None

    def test_inspects_video_metadata(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")

        report = inspect_video_metadata(source)

        assert report.source == source
        assert report.has_ai_metadata is True
        assert report.markers

    def test_removes_metadata_without_touching_video_payload(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = remove_video_metadata(source, output)

        assert result.output == output
        assert result.detected
        assert result.remaining == {}
        assert _VIDEO_PAYLOAD in output.read_bytes()
        assert C2PA_UUID not in output.read_bytes()

    def test_default_output_preserves_source(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")
        original = source.read_bytes()

        result = remove_video_metadata(source)

        assert result.output == tmp_path / "source_clean.mp4"
        assert result.output.exists()
        assert source.read_bytes() == original

    @pytest.mark.parametrize("suffix", [".mp4", ".mov"])
    def test_inspects_native_tc260_metadata(self, tmp_path: Path, suffix: str):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    def test_inspects_native_tc260_metadata_after_large_media_payload(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(
            tmp_path / "source.mp4",
            media_payload=b"x" * (1024 * 1024),
        )

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert "aigc_label" in report.markers

    def test_removes_native_tc260_metadata_without_touching_media_or_standard_tag(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_tc260(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = remove_video_metadata(source, output)
        cleaned = output.read_bytes()

        assert result.detected["aigc_label"].startswith("China AIGC label")
        assert result.remaining == {}
        assert len(cleaned) == source.stat().st_size
        assert _VIDEO_PAYLOAD in cleaned
        assert b"standard title" in cleaned
        assert b"AIGC" not in cleaned
        assert _TC260_AIGC not in cleaned

    def test_ignores_generic_mp4_aigc_tag_without_tc260_fields(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260(tmp_path / "source.mp4")
        source.write_bytes(source.read_bytes().replace(_TC260_AIGC, b'{"description":"' + b"x" * 146 + b'"}'))

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is False
        assert report.markers == {}

    @pytest.mark.parametrize("suffix", [".mkv", ".webm"])
    def test_inspects_native_tc260_ebml_metadata(self, tmp_path: Path, suffix: str):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_ebml(tmp_path / f"source{suffix}")

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is True
        assert report.markers["aigc_label"].endswith("producer 00119144030008867405X210002")

    def test_ignores_generic_ebml_aigc_tag_without_tc260_fields(self, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        source = _video_with_tc260_ebml(
            tmp_path / "source.mkv",
            value=b'{"description":"ordinary application metadata"}',
        )

        report = inspect_video_metadata(source)

        assert report.has_ai_metadata is False
        assert report.markers == {}

    def test_rejects_image_input(self, tmp_clean_png: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        with pytest.raises(ValueError, match="Unsupported video format"):
            inspect_video_metadata(tmp_clean_png)

    def test_rejects_image_with_video_extension(self, tmp_clean_png: Path, tmp_path: Path):
        from remove_ai_watermarks.video import inspect_video_metadata

        disguised = tmp_path / "image.mp4"
        disguised.write_bytes(tmp_clean_png.read_bytes())

        with pytest.raises(ValueError, match="does not match"):
            inspect_video_metadata(disguised)

    def test_rejects_output_container_change(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_metadata

        source = _video_with_c2pa(tmp_path / "source.mp4")

        with pytest.raises(ValueError, match="must match"):
            remove_video_metadata(source, tmp_path / "clean.mov")


class TestVideoMetadataCli:
    def test_help(self):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "metadata", "--help"])

        assert result.exit_code == 0, result.output
        assert "AI metadata" in result.output

    def test_check_reports_metadata(self, tmp_path: Path):
        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")

        result = runner.invoke(main, ["video", "metadata", str(source), "--check"])

        assert result.exit_code == 0, result.output
        assert "AI metadata detected" in result.output

    def test_remove_reports_output(self, tmp_path: Path):
        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"

        result = runner.invoke(main, ["video", "metadata", str(source), "--remove", "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "AI metadata stripped" in result.output
        assert C2PA_UUID not in output.read_bytes()

    def test_rejects_image_input(self, tmp_clean_png: Path):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "metadata", str(tmp_clean_png), "--check"])

        assert result.exit_code != 0
        assert "Unsupported video format" in result.output


class TestVideoInvisibleApi:
    def test_generates_unverified_candidate_and_strips_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "candidate.mp4"

        def fake_regenerate(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return _regeneration_metrics()

        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regenerate)

        result = remove_video_invisible(source, output)

        assert result.output == output
        assert result.requires_external_verification is True
        assert result.total_frames == 24
        assert result.remaining_metadata == {}

    def test_default_output_is_named_as_candidate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_invisible
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_c2pa(tmp_path / "source.mp4")

        def fake_regenerate(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return _regeneration_metrics(
                frames=2,
                fps=2.0,
                width=16,
                height=16,
                psnr_db=20.0,
                temporal_residual_ratio=1.0,
            )

        monkeypatch.setattr(video_invisible, "regenerate_video_candidate", fake_regenerate)

        result = remove_video_invisible(source)

        assert result.output == tmp_path / "source_synthid_candidate.mp4"

    def test_rejects_webm_regeneration(self, tmp_path: Path):
        from remove_ai_watermarks.video import remove_video_invisible

        source = _video_with_tc260_ebml(tmp_path / "source.webm")

        with pytest.raises(ValueError, match="requires one of"):
            remove_video_invisible(source)


class TestVideoInvisibleCli:
    def test_help_describes_external_verification(self):
        runner = CliRunner()

        result = runner.invoke(main, ["video", "invisible", "--help"])

        assert result.exit_code == 0, result.output
        assert "externally verifiable" in result.output

    def test_reports_unverified_candidate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video

        runner = CliRunner()
        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "candidate.mp4"

        def fake_remove(_source: Path, target: Path, **_kwargs: object):
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return video.VideoInvisibleResult(
                source=_source,
                output=target,
                noise_std=0.1,
                metrics=_regeneration_metrics(),
                remaining_metadata={},
            )

        monkeypatch.setattr(video, "remove_video_invisible", fake_remove)

        result = runner.invoke(main, ["video", "invisible", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "Candidate generated" in result.output
        assert "UNVERIFIED" in result.output
        assert "Gemini Flash" in result.output


class TestSoraFrameLocalization:
    @staticmethod
    def _sora_like_frame() -> tuple[np.ndarray, tuple[int, int, int, int]]:
        frame = np.full((480, 840, 3), 36, dtype=np.uint8)
        mark = Image.new("L", (180, 64), 0)
        draw = ImageDraw.Draw(mark)
        draw.ellipse((1, 14, 32, 54), fill=255)
        draw.ellipse((25, 8, 62, 58), fill=255)
        draw.ellipse((15, 20, 28, 44), fill=0)
        draw.ellipse((37, 18, 50, 43), fill=0)
        try:
            font = ImageFont.load_default(size=49)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((68, 1), "Sora", font=font, fill=255, stroke_width=1)
        mark_array = cv2.resize(np.asarray(mark), (124, 44), interpolation=cv2.INTER_AREA)
        x, y = 620, 398
        alpha = mark_array.astype(np.float32)[:, :, None] / 255 * 0.78
        crop = frame[y : y + 44, x : x + 124].astype(np.float32)
        frame[y : y + 44, x : x + 124] = np.clip(crop * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)
        return frame, (x, y, 124, 44)

    def test_localizes_independently_rendered_sora_like_mark(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_sora_frame

        frame, expected = self._sora_like_frame()

        detection = detect_sora_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.58
        assert _region_iou(detection.region, expected) >= 0.45

    def test_empty_frame_is_not_localized(self):
        from remove_ai_watermarks.video_visible import detect_sora_frame

        detection = detect_sora_frame(np.empty((0, 0, 3), dtype=np.uint8))

        assert detection.confidence == 0.0
        assert detection.region is None


class TestVeoFrameLocalization:
    def test_localizes_independently_rendered_diamond_at_relocated_position(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_veo_frame

        frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
        size = 48
        x, y = 1080, 570
        mark = Image.new("L", (size, size), 0)
        points = (
            (size // 2, 1),
            (round(size * 0.61), round(size * 0.38)),
            (size - 2, size // 2),
            (round(size * 0.61), round(size * 0.62)),
            (size // 2, size - 2),
            (round(size * 0.39), round(size * 0.62)),
            (1, size // 2),
            (round(size * 0.39), round(size * 0.38)),
        )
        ImageDraw.Draw(mark).polygon(points, fill=255)
        alpha = np.asarray(mark, dtype=np.float32)[:, :, None] / 255 * 0.72
        crop = frame[y : y + size, x : x + size].astype(np.float32)
        frame[y : y + size, x : x + size] = np.clip(
            crop * (1 - alpha) + 255 * alpha,
            0,
            255,
        ).astype(np.uint8)

        detection = detect_veo_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.70
        assert _region_iou(detection.region, (x, y, size, size)) >= 0.70

    def test_localizes_independently_rendered_legacy_text(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_veo_frame

        frame = np.full((720, 1280, 3), 42, dtype=np.uint8)
        mark = Image.new("L", (60, 24), 0)
        try:
            font = ImageFont.load_default(size=19)
        except TypeError:
            font = ImageFont.load_default()
        ImageDraw.Draw(mark).text((1, 0), "Veo", font=font, fill=255)
        mark_array = np.asarray(mark)
        ys, xs = np.where(mark_array > 0)
        mark_array = mark_array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mark_height, mark_width = mark_array.shape
        x = frame.shape[1] - mark_width - 20
        y = frame.shape[0] - mark_height - 18
        alpha = mark_array.astype(np.float32)[:, :, None] / 255 * 0.66
        crop = frame[y : y + mark_height, x : x + mark_width].astype(np.float32)
        frame[y : y + mark_height, x : x + mark_width] = np.clip(
            crop * (1 - alpha) + 255 * alpha,
            0,
            255,
        ).astype(np.uint8)

        detection = detect_veo_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.55
        assert _region_iou(detection.region, (x, y, mark_width, mark_height)) >= 0.65

    def test_empty_frame_is_not_localized(self):
        from remove_ai_watermarks.video_visible import detect_veo_frame

        detection = detect_veo_frame(np.empty((0, 0, 3), dtype=np.uint8))

        assert detection.confidence == 0.0
        assert detection.region is None

    def test_diamond_mask_preserves_transparent_box_corners(self):
        from remove_ai_watermarks.video_visible import _mask_for_region

        mask = _mask_for_region(
            np.zeros((100, 100, 3), dtype=np.uint8),
            (20, 20, 48, 48),
            padding_fraction=0.18,
            mask_style="veo",
        )

        assert mask[44, 44] == 255
        assert mask[20, 20] == 0
        assert mask[67, 67] == 0


class TestByteDanceFrameLocalization:
    def test_localizes_independently_rendered_seedance_box(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_seedance_frame

        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        mark = Image.new("L", (80, 60), 0)
        draw = ImageDraw.Draw(mark)
        draw.rounded_rectangle((2, 2, 70, 53), radius=14, outline=255, width=4)
        try:
            font = ImageFont.load_default(size=35)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((18, 8), "AI", font=font, fill=255)
        mark_array = np.asarray(mark, dtype=np.float32)
        x, y = 1130, 620
        alpha = mark_array[:, :, None] / 255 * 0.65
        crop = frame[y : y + 60, x : x + 80].astype(np.float32)
        frame[y : y + 60, x : x + 80] = np.clip(
            crop * (1 - alpha) + 255 * alpha,
            0,
            255,
        ).astype(np.uint8)

        detection = detect_seedance_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.43
        assert _region_iou(detection.region, (x, y, 80, 60)) >= 0.75

    def test_localizes_independently_rendered_dola_text(self):
        from remove_ai_watermarks.video_visible import _region_iou, detect_dola_frame

        frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
        mark = np.zeros((40, 150), dtype=np.uint8)
        cv2.putText(
            mark,
            "Dola AI",
            (2, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            255,
            2,
            cv2.LINE_AA,
        )
        ys, xs = np.where(mark > 0)
        mark = mark[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mark_height, mark_width = mark.shape
        x = frame.shape[1] - mark_width - 18
        y = frame.shape[0] - mark_height - 14
        alpha = mark.astype(np.float32)[:, :, None] / 255 * 0.75
        crop = frame[y : y + mark_height, x : x + mark_width].astype(np.float32)
        frame[y : y + mark_height, x : x + mark_width] = np.clip(
            crop * (1 - alpha) + 255 * alpha,
            0,
            255,
        ).astype(np.uint8)

        detection = detect_dola_frame(frame)

        assert detection.region is not None
        assert detection.confidence >= 0.52
        assert _region_iou(detection.region, (x, y, mark_width, mark_height)) >= 0.75

    def test_seedance_box_mask_covers_the_full_localized_mark(self):
        from remove_ai_watermarks.video_visible import _mask_for_region

        mask = _mask_for_region(
            np.zeros((120, 160, 3), dtype=np.uint8),
            (20, 20, 80, 60),
            padding_fraction=0.0,
            mask_style="box",
        )

        assert mask[15, 15] == 0
        assert mask[16, 16] == 255
        assert mask[83, 103] == 255
        assert mask[84, 104] == 0


class TestSoraTemporalArbiter:
    _BOX = (40, 60, 150, 54)

    def test_four_frame_lookalike_run_is_too_short(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [FrameLocalization(index, 0.70, self._BOX) for index in range(4)]

        assert stabilize_sora_localizations(detections, provenance=False) == [None] * 4

    def test_provenance_accepts_recurring_low_contrast_visual_match(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.59, self._BOX),
            FrameLocalization(1, 0.61, self._BOX),
            FrameLocalization(2, 0.62, self._BOX),
            FrameLocalization(3, 0.60, self._BOX),
            FrameLocalization(4, 0.61, self._BOX),
        ]

        assert stabilize_sora_localizations(detections, provenance=True) == [self._BOX] * 5

    def test_confirmed_provenance_run_covers_transition_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.30, (500, 300, 54, 54)),
            FrameLocalization(1, 0.59, self._BOX),
            FrameLocalization(2, 0.61, self._BOX),
            FrameLocalization(3, 0.62, self._BOX),
            FrameLocalization(4, 0.60, self._BOX),
            FrameLocalization(5, 0.61, self._BOX),
            FrameLocalization(6, 0.30, (300, 100, 54, 54)),
        ]

        assert stabilize_sora_localizations(detections, provenance=True) == [self._BOX] * 7

    def test_transition_prefers_low_score_match_at_a_confirmed_position(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        other_box = (500, 300, 150, 54)
        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
            FrameLocalization(5, 0.20, (250, 180, 54, 54)),
            FrameLocalization(6, 0.52, self._BOX),
            FrameLocalization(7, 0.61, other_box),
            FrameLocalization(8, 0.62, other_box),
            FrameLocalization(9, 0.63, other_box),
            FrameLocalization(10, 0.61, other_box),
            FrameLocalization(11, 0.62, other_box),
        ]

        stabilized = stabilize_sora_localizations(detections, provenance=True)

        assert stabilized[6] == self._BOX
        assert stabilized[7:] == [other_box] * 5

    def test_transition_without_a_match_keeps_previous_stable_position(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        other_box = (500, 300, 150, 54)
        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
            FrameLocalization(5, 0.20, (250, 180, 54, 54)),
            FrameLocalization(6, 0.20, (300, 200, 54, 54)),
            FrameLocalization(7, 0.61, other_box),
            FrameLocalization(8, 0.62, other_box),
            FrameLocalization(9, 0.63, other_box),
            FrameLocalization(10, 0.61, other_box),
            FrameLocalization(11, 0.62, other_box),
        ]

        stabilized = stabilize_sora_localizations(detections, provenance=True)

        assert stabilized[5:7] == [self._BOX, self._BOX]

    def test_unproven_weak_run_is_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.62, self._BOX),
            FrameLocalization(2, 0.63, self._BOX),
            FrameLocalization(3, 0.62, self._BOX),
            FrameLocalization(4, 0.61, self._BOX),
        ]

        assert stabilize_sora_localizations(detections, provenance=False) == [None] * 5

    def test_strong_recurring_visual_run_needs_no_metadata(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.61, self._BOX),
            FrameLocalization(1, 0.66, self._BOX),
            FrameLocalization(2, 0.62, self._BOX),
            FrameLocalization(3, 0.61, self._BOX),
            FrameLocalization(4, 0.62, self._BOX),
        ]

        assert stabilize_sora_localizations(detections, provenance=False) == [self._BOX] * 5

    def test_isolated_lookalikes_at_different_positions_are_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.70, (10, 10, 150, 54)),
            FrameLocalization(1, 0.70, (400, 200, 150, 54)),
            FrameLocalization(2, 0.70, (650, 400, 150, 54)),
        ]

        assert stabilize_sora_localizations(detections, provenance=True) == [None, None, None]

    def test_short_dropout_between_matching_boxes_is_filled(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_sora_localizations

        detections = [
            FrameLocalization(0, 0.66, self._BOX),
            FrameLocalization(1, 0.20, (500, 300, 54, 54)),
            FrameLocalization(2, 0.67, self._BOX),
            FrameLocalization(3, 0.66, self._BOX),
            FrameLocalization(4, 0.66, self._BOX),
            FrameLocalization(5, 0.66, self._BOX),
        ]

        assert stabilize_sora_localizations(detections, provenance=False) == [self._BOX] * 6


class TestVeoTemporalArbiter:
    _BOX = (1132, 572, 56, 56)

    def test_eleven_frame_lookalike_run_is_too_short(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_veo_localizations

        detections = [FrameLocalization(index, 0.70, self._BOX) for index in range(11)]

        assert stabilize_veo_localizations(detections, provenance=False) == [None] * 11

    def test_strong_fixed_run_covers_video_without_metadata(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_veo_localizations

        detections = [FrameLocalization(index, 0.60, self._BOX) for index in range(12)]
        detections.extend(FrameLocalization(index, 0.20, (300, 200, 48, 48)) for index in range(12, 15))

        assert stabilize_veo_localizations(detections, provenance=False) == [self._BOX] * 15

    def test_google_provenance_accepts_recurring_low_contrast_diamond(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_veo_localizations

        detections = [FrameLocalization(index, 0.47, self._BOX) for index in range(12)]

        assert stabilize_veo_localizations(detections, provenance=True) == [self._BOX] * 12

    def test_unproven_weak_run_is_rejected(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_veo_localizations

        detections = [FrameLocalization(index, 0.52, self._BOX) for index in range(12)]

        assert stabilize_veo_localizations(detections, provenance=False) == [None] * 12


class TestByteDanceTemporalArbiter:
    _SEEDANCE_BOX = (1110, 610, 90, 66)
    _DOLA_BOX = (1160, 680, 96, 22)

    def test_seedance_requires_twelve_recurring_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_seedance_localizations

        detections = [FrameLocalization(index, 0.50, self._SEEDANCE_BOX) for index in range(11)]

        assert stabilize_seedance_localizations(detections, provenance=False) == [None] * 11

    def test_seedance_strong_run_covers_low_contrast_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_seedance_localizations

        detections = [FrameLocalization(index, 0.45, self._SEEDANCE_BOX) for index in range(12)]
        detections.extend(FrameLocalization(index, 0.20, (200, 100, 80, 60)) for index in range(12, 15))

        assert stabilize_seedance_localizations(detections, provenance=False) == [self._SEEDANCE_BOX] * 15

    def test_seedance_rejects_a_slowly_drifting_scene_detail(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_seedance_localizations

        detections = [FrameLocalization(index, 0.46, (1110 - index * 3, 610, 90, 66)) for index in range(14)]

        assert stabilize_seedance_localizations(detections, provenance=False) == [None] * 14

    def test_dola_requires_twelve_recurring_frames(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_dola_localizations

        detections = [FrameLocalization(index, 0.60, self._DOLA_BOX) for index in range(11)]

        assert stabilize_dola_localizations(detections, provenance=True) == [None] * 11

    def test_dola_provenance_accepts_recurring_low_contrast_text(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_dola_localizations

        detections = [FrameLocalization(index, 0.49, self._DOLA_BOX) for index in range(12)]

        assert stabilize_dola_localizations(detections, provenance=True) == [self._DOLA_BOX] * 12

    def test_dola_without_provenance_needs_a_strong_frame(self):
        from remove_ai_watermarks.video_visible import FrameLocalization, stabilize_dola_localizations

        detections = [FrameLocalization(index, 0.51, self._DOLA_BOX) for index in range(12)]

        assert stabilize_dola_localizations(detections, provenance=False) == [None] * 12

    def test_bytedance_provenance_requires_ai_source_type(self):
        from remove_ai_watermarks.video_visible import has_bytedance_video_provenance

        assert has_bytedance_video_provenance(
            {
                "issuer": "BytePlus (ByteDance)",
                "source_type": "trainedAlgorithmicMedia (AI-generated)",
            }
        )
        assert not has_bytedance_video_provenance({"issuer": "BytePlus (ByteDance)"})


class TestVideoVisibleApi:
    def test_removes_stable_sora_run_and_writes_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (4, 4, 20, 8)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.66, box) for index in range(5)),
        )
        monkeypatch.setattr(video_visible, "scan_sora_video", lambda _source: scan)

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **_kwargs: object,
        ) -> int:
            assert regions == [box] * 5
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 5

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output)

        assert result.output == output
        assert result.detected_frames == 5
        assert result.removed_frames == 5
        assert result.remaining_metadata == {}

    def test_no_stable_mark_writes_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=(
                FrameLocalization(0, 0.70, (1, 1, 20, 8)),
                FrameLocalization(1, 0.70, (30, 30, 20, 8)),
                FrameLocalization(2, 0.70, (1, 30, 20, 8)),
            ),
        )
        monkeypatch.setattr(video_visible, "scan_sora_video", lambda _source: scan)

        result = remove_video_visible(source, output)

        assert result.output is None
        assert result.removed_frames == 0
        assert not output.exists()

    def test_dispatches_veo_detector_and_uses_tighter_mask(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (4, 4, 20, 20)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.60, box) for index in range(12)),
        )
        monkeypatch.setattr(video_visible, "scan_veo_video", lambda _source: scan)

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [box] * 12
            assert kwargs["padding_fraction"] == 0.18
            assert kwargs["mask_style"] == "veo"
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 12

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output, mark="veo")

        assert result.output == output
        assert result.mark == "veo"
        assert result.detected_frames == 12
        assert result.removed_frames == 12

    @pytest.mark.parametrize(
        ("mark", "scan_name", "mask_style"),
        [
            ("seedance", "scan_seedance_video", "box"),
            ("dola", "scan_dola_video", "box"),
        ],
    )
    def test_dispatches_bytedance_detectors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mark: str,
        scan_name: str,
        mask_style: str,
    ):
        from remove_ai_watermarks import video_visible
        from remove_ai_watermarks.video import remove_video_visible
        from remove_ai_watermarks.video_visible import FrameLocalization, VideoScan

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        box = (40, 40, 20, 12)
        scan = VideoScan(
            width=64,
            height=64,
            fps=24.0,
            detections=tuple(FrameLocalization(index, 0.60, box) for index in range(12)),
        )
        monkeypatch.setattr(video_visible, scan_name, lambda _source: scan)

        def fake_encode(
            _source: Path,
            target: Path,
            _scan: VideoScan,
            regions: list[tuple[int, int, int, int] | None],
            **kwargs: object,
        ) -> int:
            assert regions == [box] * 12
            assert kwargs["mask_style"] == mask_style
            target.write_bytes(_MP4_FTYP + _box(b"mdat", _VIDEO_PAYLOAD))
            return 12

        monkeypatch.setattr(video_visible, "encode_clean_video", fake_encode)

        result = remove_video_visible(source, output, mark=mark)

        assert result.output == output
        assert result.mark == mark
        assert result.detected_frames == 12
        assert result.removed_frames == 12


class TestVideoVisibleCli:
    def test_help(self):
        result = CliRunner().invoke(main, ["video", "visible", "--help"])

        assert result.exit_code == 0, result.output
        assert "temporally stable" in result.output
        assert "sora|veo|seedance|dola" in result.output

    def test_reports_removed_frames(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import video
        from remove_ai_watermarks.video import VideoVisibleResult

        source = _video_with_c2pa(tmp_path / "source.mp4")
        output = tmp_path / "clean.mp4"
        monkeypatch.setattr(
            video,
            "remove_video_visible",
            lambda *_args, **_kwargs: VideoVisibleResult(
                source=source,
                output=output,
                mark="sora",
                total_frames=12,
                detected_frames=10,
                removed_frames=10,
                remaining_metadata={},
            ),
        )

        result = CliRunner().invoke(main, ["video", "visible", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "10/12 frames" in result.output

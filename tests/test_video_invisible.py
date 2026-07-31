"""Regression tests for the video SynthID removal engine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from remove_ai_watermarks import video_encoding, video_invisible
from remove_ai_watermarks.video_synthid import DEFAULT_VIDEO_SYNTHID_NOISE_STD

if TYPE_CHECKING:
    from pathlib import Path


def test_availability_requires_both_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_invisible,
        "find_spec",
        lambda name: object() if name == "torch" else None,
    )

    assert video_invisible.is_available() is False


def test_regeneration_rejects_noise_outside_unit_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        video_invisible.regenerate_video_candidate(
            tmp_path / "source.mp4",
            tmp_path / "candidate.mp4",
            noise_std=1.01,
        )


def test_encoder_command_discards_metadata_and_copies_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "candidate.mp4"

    monkeypatch.setattr(video_encoding.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    profile = video_encoding.VideoEncodeProfile(
        pixel_format="yuv420p",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        time_base="1/90000",
    )

    command = video_encoding.raw_video_command(
        source,
        output,
        width=8,
        height=8,
        fps=2.0,
        strip_metadata=True,
        crf=18,
        profile=profile,
    )

    metadata_index = command.index("-map_metadata")
    assert command[metadata_index + 1] == "-1"
    audio_codec_index = command.index("-c:a")
    assert command[audio_codec_index + 1] == "copy"
    output_pixel_format_index = command.index("-pix_fmt", command.index("-c:v"))
    assert command[output_pixel_format_index + 1] == "yuv420p"
    assert command[command.index("-color_range") + 1] == "tv"
    assert command[command.index("-colorspace") + 1] == "bt709"
    assert command[command.index("-color_trc") + 1] == "bt709"
    assert command[command.index("-color_primaries") + 1] == "bt709"
    assert command[command.index("-enc_time_base:v") + 1] == "1/90000"
    assert command[command.index("-video_track_timescale") + 1] == "90000"
    assert command[command.index("-x264-params") + 1] == (
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"
    )
    assert "pipe:0" in command
    assert "-shortest" not in command


def test_timestamped_encoder_reads_nut_and_passes_pts_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "candidate.mp4"
    monkeypatch.setattr(video_encoding.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    command = video_encoding.raw_video_command(
        source,
        output,
        width=8,
        height=8,
        fps=24.0,
        strip_metadata=True,
        crf=18,
        profile=video_encoding.VideoEncodeProfile(time_base="1/90000"),
        timestamped_input=True,
        copy_input_timestamps=True,
    )

    assert "-copyts" in command
    assert command[command.index("-f") : command.index("-f") + 4] == ["-f", "nut", "-i", "pipe:0"]
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-avoid_negative_ts") + 1] == "disabled"
    assert command[command.index("-enc_time_base:v") + 1] == "1/90000"


def test_probe_encode_profile_preserves_supported_8_bit_properties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(video_encoding.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_encoding.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":[{"pix_fmt":"yuvj422p","color_range":"tv",'
                '"color_space":"bt709","color_transfer":"bt709",'
                '"color_primaries":"bt709","time_base":"2/180000",'
                '"start_pts":180000,"bits_per_raw_sample":"8"}]}'
            ),
            stderr="",
        ),
    )

    profile = video_encoding.probe_video_encode_profile(source)

    assert profile == video_encoding.VideoEncodeProfile(
        pixel_format="yuv422p",
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
        time_base="1/90000",
        start_pts=180000,
        source_pixel_format="yuvj422p",
        component_depth=8,
    )


def test_probe_encode_profile_uses_compatible_defaults_without_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_encoding.shutil, "which", lambda _name: None)

    assert video_encoding.probe_video_encode_profile(tmp_path / "source.mp4") == (video_encoding.VideoEncodeProfile())


def test_probe_video_timestamps_uses_best_effort_pts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(video_encoding.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_encoding.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="0.000000\n0.041667\n",
            stderr="",
        ),
    )

    assert video_encoding.probe_video_timestamps(source) == (0.0, 0.041667)


def test_default_noise_matches_full_clip_oracle_floor() -> None:
    assert DEFAULT_VIDEO_SYNTHID_NOISE_STD == 0.15


def test_stream_batches_consumes_only_one_batch_ahead() -> None:
    consumed: list[int] = []

    def values():
        for value in range(5):
            consumed.append(value)
            yield value

    batches = video_invisible._stream_batches(values(), 2)

    assert next(iter(batches)) == [0, 1]
    assert consumed == [0, 1]

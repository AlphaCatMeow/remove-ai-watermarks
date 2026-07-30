"""Regression tests for the video SynthID candidate engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from remove_ai_watermarks import video_encoding, video_invisible

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

    command = video_encoding.raw_video_command(
        source,
        output,
        width=8,
        height=8,
        fps=2.0,
        strip_metadata=True,
        crf=18,
    )

    metadata_index = command.index("-map_metadata")
    assert command[metadata_index + 1] == "-1"
    audio_codec_index = command.index("-c:a")
    assert command[audio_codec_index + 1] == "copy"
    assert "pipe:0" in command


def test_stream_batches_consumes_only_one_batch_ahead() -> None:
    consumed: list[int] = []

    def values():
        for value in range(5):
            consumed.append(value)
            yield value

    batches = video_invisible._stream_batches(values(), 2)

    assert next(iter(batches)) == [0, 1]
    assert consumed == [0, 1]

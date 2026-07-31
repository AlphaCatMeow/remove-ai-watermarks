"""Shared ffmpeg raw-video encoding helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

log = logging.getLogger(__name__)


@contextmanager
def atomic_video_output(output: Path) -> Generator[Path]:
    """Yield a sibling temporary path and publish it only after success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        dir=output.parent,
        delete=False,
    ) as stream:
        temporary_output = Path(stream.name)
    try:
        yield temporary_output
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)


def _video_codec_args(suffix: str, *, crf: int) -> list[str]:
    if suffix == ".webm":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]


def raw_video_command(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    strip_metadata: bool,
    crf: int,
) -> list[str]:
    """Build an ffmpeg command that accepts BGR frames on standard input."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Video processing requires ffmpeg on PATH")
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
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        *_video_codec_args(output.suffix.lower(), crf=crf),
        "-c:a",
        "copy",
        "-map_metadata",
        "-1" if strip_metadata else "1",
        "-map_chapters",
        "-1" if strip_metadata else "1",
    ]
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output))
    return command


def start_raw_video_encoder(command: list[str]) -> subprocess.Popen[bytes]:
    """Start ffmpeg and validate its raw-frame pipes."""
    log.info("Starting ffmpeg video encode: command=%s", command)
    process = subprocess.Popen(  # noqa: S603
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("Could not open ffmpeg pipes")
    return process


def finish_raw_video_encoder(
    process: subprocess.Popen[bytes],
    output: Path,
    *,
    operation: str,
) -> None:
    """Close the frame stream and raise when ffmpeg rejects the encode."""
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("ffmpeg pipes are unavailable")
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    log.info("ffmpeg %s finished: status=%s stderr=%s", operation, return_code, stderr)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed to encode {output}: {stderr.strip()[:500]}")


def abort_raw_video_encoder(process: subprocess.Popen[bytes]) -> None:
    """Stop an incomplete ffmpeg encode."""
    process.kill()
    process.wait()

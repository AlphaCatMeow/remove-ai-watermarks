"""Build oracle-gated video regeneration candidates for SynthID research.

This is a research harness, not a shipped removal command. Google does not
publish a local video SynthID decoder, so the script cannot label a candidate as
clean. It produces:

* a re-encode control with the same duration, frame rate, dimensions, and codec;
* one VAE-regenerated video per requested latent-noise level;
* paired fidelity and temporal-residual measurements;
* a CSV column for the external Gemini SynthID verdict.

The control is load-bearing. If it reads clean, the experiment is invalid:
resize, frame-rate conversion, or H.264 compression already silenced the oracle,
so a VAE candidate cannot be credited with removal.

The regeneration attack follows the general encode, perturb, reconstruct family
from WatermarkAttacker (NeurIPS 2024). A single spatial latent-noise sample is
shared by every frame. Independent per-frame noise creates avoidable flicker and
does not test the video-specific question.

Run with the project's GPU extra:

    uv run --extra gpu python scripts/video_synthid_sweep.py input.mp4 -o out/

Then upload ``control.mp4`` and each candidate in separate Gemini chats, invoke
the built-in SynthID verifier (``@synthid``), and use the question printed by
the script. Do not follow the verdict with an adversarial prompt asking the chat
model to reinterpret the detector: that switches back to ordinary reasoning.
Only a control-positive, candidate-negative pair from the built-in verifier is
removal evidence.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
import cv2
import numpy as np

from remove_ai_watermarks.video_invisible import (
    _decode_frame_latents,
    _encode_frame_latents,
    _fit_size,
    _pick_device,
    _shared_latent_noise,
    build_temporal_reference,
    encode_video_frames,
    paired_psnr,
    read_sampled_frames,
    temporal_residual_ratio,
)
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_VAE,
    VIDEO_SYNTHID_LATENT_MULTIPLE,
    VIDEO_SYNTHID_VERIFICATION_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


def _parse_noise_levels(values: str) -> tuple[float, ...]:
    levels = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    if not levels:
        raise click.BadParameter("At least one noise level is required")
    if any(not 0.0 <= value <= 1.0 for value in levels):
        raise click.BadParameter("Noise levels must be between 0 and 1")
    return levels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, rows: Sequence[dict[str, str]]) -> Path:
    path = output_dir / "sweep.csv"
    fieldnames = [
        "variant",
        "noise_std",
        "psnr_db",
        "temporal_residual_ratio",
        "file",
        "sha256",
        "synthid_oracle",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--noise-levels", default="0,0.05,0.1,0.15", show_default=True)
@click.option("--duration", type=click.FloatRange(min=0.1), default=2.0, show_default=True)
@click.option("--fps", type=click.FloatRange(min=1.0), default=DEFAULT_VIDEO_SYNTHID_FPS, show_default=True)
@click.option(
    "--long-side",
    type=click.IntRange(min=VIDEO_SYNTHID_LATENT_MULTIPLE),
    default=DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--model", default=DEFAULT_VIDEO_SYNTHID_VAE, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto", show_default=True)
def main(
    source: Path,
    output_dir: Path,
    noise_levels: str,
    duration: float,
    fps: float,
    long_side: int,
    batch_size: int,
    seed: int,
    model: str,
    device: str,
) -> None:
    """Generate VAE video candidates from the prefix of SOURCE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch
    from diffusers import AutoencoderKL

    levels = _parse_noise_levels(noise_levels)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {source}")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    size = _fit_size(width, height, long_side)
    frames, effective_fps = read_sampled_frames(source, duration=duration, output_fps=fps, size=size)
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "control.mp4"
    encode_video_frames(frames, source, control_path, fps=effective_fps)
    rows: list[dict[str, str]] = [
        {
            "variant": "control",
            "noise_std": "",
            "psnr_db": "inf",
            "temporal_residual_ratio": "1",
            "file": control_path.name,
            "sha256": _sha256(control_path),
            "synthid_oracle": "",
        }
    ]

    resolved_device = _pick_device(device)
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    log.info("Loading %s on %s", model, resolved_device)
    vae = AutoencoderKL.from_pretrained(model, torch_dtype=dtype).to(resolved_device)
    vae.eval()
    vae.enable_slicing()

    log.info("Encoding source frames")
    latent_batches = _encode_frame_latents(
        frames,
        vae=vae,
        device=resolved_device,
        batch_size=batch_size,
    )
    first_latents = latent_batches[0]
    shared_noise = _shared_latent_noise(
        first_latents.shape[1:],
        seed=seed,
        device=resolved_device,
        dtype=first_latents.dtype,
    )
    reference_stack = np.stack(frames)
    temporal_maps, temporal_baseline = build_temporal_reference(frames)
    for level in levels:
        log.info("Decoding latent noise %.4f", level)
        regenerated = _decode_frame_latents(
            latent_batches,
            vae=vae,
            noise_std=level,
            shared_noise=shared_noise,
        )
        output_path = output_dir / f"vae-noise-{level:.4f}.mp4"
        encode_video_frames(
            regenerated,
            source,
            output_path,
            fps=effective_fps,
        )
        psnr = paired_psnr(reference_stack, np.stack(regenerated))
        temporal_ratio = temporal_residual_ratio(regenerated, temporal_maps, temporal_baseline)
        rows.append(
            {
                "variant": "vae",
                "noise_std": f"{level:.4f}",
                "psnr_db": f"{psnr:.4f}",
                "temporal_residual_ratio": f"{temporal_ratio:.4f}",
                "file": output_path.name,
                "sha256": _sha256(output_path),
                "synthid_oracle": "",
            }
        )

    manifest = _write_manifest(output_dir, rows)
    log.info("Wrote %s", manifest)
    log.info(
        "Verify control.mp4 first with Gemini's built-in SynthID verifier and this question: %s",
        VIDEO_SYNTHID_VERIFICATION_PROMPT,
    )


if __name__ == "__main__":
    main()

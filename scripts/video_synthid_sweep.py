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

Then upload ``control.mp4`` and each candidate through Gemini's SynthID
verification flow. Some eligible versions expose an explicit ``@synthid``
trigger. A generic chat answer that says it lacks a decoder is not an oracle
verdict. Only a control-positive, candidate-negative pair is removal evidence.
"""

from __future__ import annotations

# torch/diffusers/cv2 expose incomplete types at this boundary. Pure helpers
# remain annotated while third-party tensor and image calls are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false
import csv
import hashlib
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

log = logging.getLogger(__name__)

DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
_LATENT_MULTIPLE = 8


def _fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Fit dimensions to ``long_side`` while preserving aspect and VAE alignment."""
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    if long_side < _LATENT_MULTIPLE:
        raise ValueError(f"Long side must be at least {_LATENT_MULTIPLE}")
    scale = long_side / max(width, height)
    fitted_width = max(_LATENT_MULTIPLE, round(width * scale) // _LATENT_MULTIPLE * _LATENT_MULTIPLE)
    fitted_height = max(_LATENT_MULTIPLE, round(height * scale) // _LATENT_MULTIPLE * _LATENT_MULTIPLE)
    return fitted_width, fitted_height


def _parse_noise_levels(values: str) -> tuple[float, ...]:
    levels = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    if not levels:
        raise click.BadParameter("At least one noise level is required")
    if any(not 0.0 <= value <= 1.0 for value in levels):
        raise click.BadParameter("Noise levels must be between 0 and 1")
    return levels


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _shared_latent_noise(
    spatial_shape: Sequence[int],
    *,
    seed: int,
    device: str,
    dtype: Any,
) -> Any:
    """Return one deterministic spatial noise field for reuse across time."""
    import torch

    if len(spatial_shape) != 3 or any(size <= 0 for size in spatial_shape):
        raise ValueError("Expected a positive CHW latent shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, *spatial_shape), generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return paired PSNR over uint8 frame stacks."""
    if reference.shape != candidate.shape:
        raise ValueError("PSNR inputs must have matching shapes")
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _backward_map(current_gray: np.ndarray, previous_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a remap from a previous frame into current coordinates."""
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return grid_x + flow[..., 0], grid_y + flow[..., 1]


def _backward_warp(image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply a precomputed backward optical-flow map."""
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _temporal_reference(
    reference: Sequence[np.ndarray],
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], float]:
    """Precompute source motion maps and its mean residual."""
    if len(reference) < 2:
        raise ValueError("Temporal metric needs at least two frames")
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    reference_residuals: list[float] = []
    for index in range(1, len(reference)):
        current_gray = cv2.cvtColor(reference[index], cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(reference[index - 1], cv2.COLOR_BGR2GRAY)
        frame_maps = _backward_map(current_gray, previous_gray)
        maps.append(frame_maps)
        warped_reference = _backward_warp(reference[index - 1], frame_maps)
        reference_residuals.append(
            float(np.mean(np.abs(reference[index].astype(np.float32) - warped_reference.astype(np.float32))))
        )
    return tuple(maps), float(np.mean(reference_residuals))


def _temporal_residual_ratio(
    candidate: Sequence[np.ndarray],
    maps: Sequence[tuple[np.ndarray, np.ndarray]],
    baseline: float,
) -> float:
    """Measure candidate flicker against a precomputed source residual."""
    if len(candidate) != len(maps) + 1:
        raise ValueError("Temporal metric needs one map per adjacent frame pair")
    candidate_residuals: list[float] = []
    for index, frame_maps in enumerate(maps, start=1):
        warped_candidate = _backward_warp(candidate[index - 1], frame_maps)
        candidate_residuals.append(
            float(np.mean(np.abs(candidate[index].astype(np.float32) - warped_candidate.astype(np.float32))))
        )
    measured = float(np.mean(candidate_residuals))
    return measured / max(baseline, 1e-6)


def _read_frames(
    source: Path,
    *,
    duration: float,
    output_fps: float,
    size: tuple[int, int],
) -> tuple[list[np.ndarray], float]:
    """Read a uniformly sampled prefix and resize it to the experiment geometry."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        capture.release()
        raise ValueError(f"Video has no usable frame rate: {source}")
    effective_fps = min(output_fps, source_fps)
    sample_period = 1.0 / effective_fps
    next_sample_time = 0.0
    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if timestamp + 1e-9 >= duration:
                break
            if timestamp + 1e-9 >= next_sample_time:
                frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4))
                next_sample_time += sample_period
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("The selected clip produced fewer than two frames")
    return frames, effective_fps


def _frame_batches(frames: Sequence[np.ndarray], batch_size: int) -> Iterable[Sequence[np.ndarray]]:
    for start in range(0, len(frames), batch_size):
        yield frames[start : start + batch_size]


def _encode_frame_latents(
    frames: Sequence[np.ndarray],
    *,
    vae: Any,
    device: str,
    batch_size: int,
) -> list[Any]:
    """Encode source frames once so every noise level reuses identical latents."""
    import torch

    latent_batches: list[Any] = []
    scaling_factor = float(vae.config.scaling_factor)
    with torch.inference_mode():
        for batch in _frame_batches(frames, batch_size):
            rgb = np.stack([frame[:, :, ::-1] for frame in batch])
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
            tensor = tensor.to(device=device, dtype=vae.dtype) / 127.5 - 1.0
            latents = vae.encode(tensor).latent_dist.mode() * scaling_factor
            latent_batches.append(latents)
    return latent_batches


def _decode_frame_latents(
    latent_batches: Sequence[Any],
    *,
    vae: Any,
    noise_std: float,
    shared_noise: Any,
) -> list[np.ndarray]:
    """Decode cached latents with one perturbation shared across time."""
    import torch

    output: list[np.ndarray] = []
    scaling_factor = float(vae.config.scaling_factor)
    with torch.inference_mode():
        for latents in latent_batches:
            perturbed = latents + noise_std * shared_noise.expand(latents.shape[0], -1, -1, -1)
            decoded = vae.decode(perturbed / scaling_factor).sample
            decoded = ((decoded / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
            output.extend(np.ascontiguousarray(frame[:, :, ::-1]) for frame in decoded)
    return output


def _write_png_frames(frames: Sequence[np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        path = directory / f"{index:06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Failed to write frame: {path}")


def _encode_video(
    frames: Sequence[np.ndarray],
    source: Path,
    output: Path,
    *,
    fps: float,
    duration: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required on PATH")
    with tempfile.TemporaryDirectory(prefix="video-synthid-") as temp_dir:
        frame_dir = Path(temp_dir)
        _write_png_frames(frames, frame_dir)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:.8g}",
            "-i",
            str(frame_dir / "%06d.png"),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-t",
            f"{duration:.8g}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
        log.info("Encoding %s", output.name)
        subprocess.run(command, check=True)  # noqa: S603


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
@click.option("--noise-levels", default="0,0.025,0.05,0.1", show_default=True)
@click.option("--duration", type=click.FloatRange(min=0.1), default=2.0, show_default=True)
@click.option("--fps", type=click.FloatRange(min=1.0), default=12.0, show_default=True)
@click.option("--long-side", type=click.IntRange(min=_LATENT_MULTIPLE), default=512, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--model", default=DEFAULT_VAE, show_default=True)
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
    frames, effective_fps = _read_frames(source, duration=duration, output_fps=fps, size=size)
    effective_duration = len(frames) / effective_fps

    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "control.mp4"
    _encode_video(frames, source, control_path, fps=effective_fps, duration=effective_duration)
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
    temporal_maps, temporal_baseline = _temporal_reference(frames)
    for level in levels:
        log.info("Decoding latent noise %.4f", level)
        regenerated = _decode_frame_latents(
            latent_batches,
            vae=vae,
            noise_std=level,
            shared_noise=shared_noise,
        )
        output_path = output_dir / f"vae-noise-{level:.4f}.mp4"
        _encode_video(
            regenerated,
            source,
            output_path,
            fps=effective_fps,
            duration=effective_duration,
        )
        psnr = _psnr(reference_stack, np.stack(regenerated))
        temporal_ratio = _temporal_residual_ratio(regenerated, temporal_maps, temporal_baseline)
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
    log.info("Verify control.mp4 first in Gemini's SynthID flow; stop if the control is not detected.")


if __name__ == "__main__":
    main()

"""VAE regeneration for externally verified video SynthID candidates.

Google does not publish a local video SynthID decoder. This module therefore
regenerates pixels and measures fidelity, but never labels its output clean.
Callers must verify the candidate with Google's matching content-verification
flow.
"""

from __future__ import annotations

# torch/diffusers/cv2 expose incomplete types at this boundary. Pure helpers
# remain annotated while third-party tensor and image calls are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false
import logging
import math
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks.video_encoding import (
    abort_raw_video_encoder,
    atomic_video_output,
    finish_raw_video_encoder,
    raw_video_command,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    DEFAULT_VIDEO_SYNTHID_VAE,
    VIDEO_SYNTHID_LATENT_MULTIPLE,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegenerationMetrics:
    """Measured properties of one regenerated video candidate."""

    frames: int
    fps: float
    width: int
    height: int
    psnr_db: float
    temporal_residual_ratio: float


def is_available() -> bool:
    """Return whether the optional VAE runtime can be imported."""
    return find_spec("torch") is not None and find_spec("diffusers") is not None


def _fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Fit dimensions to ``long_side`` while preserving aspect and VAE alignment."""
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    if long_side < VIDEO_SYNTHID_LATENT_MULTIPLE:
        raise ValueError(f"Long side must be at least {VIDEO_SYNTHID_LATENT_MULTIPLE}")
    scale = long_side / max(width, height)
    fitted_width = max(
        VIDEO_SYNTHID_LATENT_MULTIPLE,
        round(width * scale) // VIDEO_SYNTHID_LATENT_MULTIPLE * VIDEO_SYNTHID_LATENT_MULTIPLE,
    )
    fitted_height = max(
        VIDEO_SYNTHID_LATENT_MULTIPLE,
        round(height * scale) // VIDEO_SYNTHID_LATENT_MULTIPLE * VIDEO_SYNTHID_LATENT_MULTIPLE,
    )
    return fitted_width, fitted_height


def _pick_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return requested


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


def paired_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
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


def build_temporal_reference(
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


def temporal_residual_ratio(
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


def _probe_video(source: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has no usable dimensions: {source}")
    if source_fps <= 0.0:
        raise ValueError(f"Video has no usable frame rate: {source}")
    return width, height, source_fps


def read_sampled_frames(
    source: Path,
    *,
    duration: float | None,
    output_fps: float,
    size: tuple[int, int],
) -> tuple[list[np.ndarray], float]:
    """Read uniformly sampled frames and resize them to the VAE geometry."""
    _width, _height, source_fps = _probe_video(source)
    effective_fps = min(output_fps, source_fps)
    frames = list(
        _iter_sampled_frames(
            source,
            source_fps=source_fps,
            duration=duration,
            effective_fps=effective_fps,
            size=size,
        )
    )
    if len(frames) < 2:
        raise ValueError("The selected clip produced fewer than two frames")
    return frames, effective_fps


def _iter_sampled_frames(
    source: Path,
    *,
    source_fps: float,
    duration: float | None,
    effective_fps: float,
    size: tuple[int, int],
) -> Iterable[np.ndarray]:
    """Yield uniformly sampled frames without retaining the full video."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    sample_period = 1.0 / effective_fps
    next_sample_time = 0.0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if duration is not None and timestamp + 1e-9 >= duration:
                break
            if timestamp + 1e-9 >= next_sample_time:
                yield cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                next_sample_time += sample_period
            frame_index += 1
    finally:
        capture.release()


def _frame_batches(frames: Sequence[np.ndarray], batch_size: int) -> Iterable[Sequence[np.ndarray]]:
    for start in range(0, len(frames), batch_size):
        yield frames[start : start + batch_size]


def _stream_batches(frames: Iterable[np.ndarray], batch_size: int) -> Iterable[list[np.ndarray]]:
    batch: list[np.ndarray] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_frame_latents(
    frames: Sequence[np.ndarray],
    *,
    vae: Any,
    device: str,
    batch_size: int,
) -> list[Any]:
    """Encode source frames once so every candidate can reuse identical latents."""
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


def encode_video_frames(
    frames: Sequence[np.ndarray],
    source: Path,
    output: Path,
    *,
    fps: float,
) -> None:
    """Encode regenerated frames, copy audio, and omit source metadata."""
    if not frames:
        raise ValueError("At least one frame is required for video encoding")
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = start_raw_video_encoder(
        raw_video_command(
            source,
            output,
            width=width,
            height=height,
            fps=fps,
            strip_metadata=True,
            crf=18,
        )
    )
    frame_pipe = process.stdin
    if frame_pipe is None:
        abort_raw_video_encoder(process)
        raise RuntimeError("Could not open ffmpeg input pipe")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Video frames must have matching dimensions")
            frame_pipe.write(frame.tobytes())
        finish_raw_video_encoder(process, output, operation="SynthID candidate encode")
    except Exception:
        if process.poll() is None:
            abort_raw_video_encoder(process)
        raise


def regenerate_video_candidate(
    source: Path,
    output: Path,
    *,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    duration: float | None = None,
) -> RegenerationMetrics:
    """Regenerate video pixels and return fidelity metrics.

    This function does not verify SynthID. Its output is an oracle candidate,
    not a locally proven clean file.
    """
    if not 0.0 <= noise_std <= 1.0:
        raise ValueError("noise_std must be between 0 and 1")
    if fps < 1.0:
        raise ValueError("fps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if not is_available():
        raise RuntimeError("Video SynthID regeneration requires the gpu extra")

    import torch
    from diffusers import AutoencoderKL

    width, height, source_fps = _probe_video(source)
    size = _fit_size(width, height, long_side)
    effective_fps = min(fps, source_fps)

    resolved_device = _pick_device(device)
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    log.info("Loading %s on %s", model, resolved_device)
    vae = AutoencoderKL.from_pretrained(model, torch_dtype=dtype).to(resolved_device)
    vae.eval()
    vae.enable_slicing()

    with atomic_video_output(output) as temporary_output:
        process = start_raw_video_encoder(
            raw_video_command(
                source,
                temporary_output,
                width=size[0],
                height=size[1],
                fps=effective_fps,
                strip_metadata=True,
                crf=18,
            )
        )
        frame_pipe = process.stdin
        if frame_pipe is None:
            abort_raw_video_encoder(process)
            raise RuntimeError("Could not open ffmpeg input pipe")
        frame_count = 0
        squared_error = 0.0
        pixel_count = 0
        temporal_baseline = 0.0
        temporal_candidate = 0.0
        previous_reference: np.ndarray | None = None
        previous_candidate: np.ndarray | None = None
        shared_noise: Any | None = None
        try:
            sampled_frames = _iter_sampled_frames(
                source,
                source_fps=source_fps,
                duration=duration,
                effective_fps=effective_fps,
                size=size,
            )
            for frames in _stream_batches(sampled_frames, batch_size):
                latent_batches = _encode_frame_latents(
                    frames,
                    vae=vae,
                    device=resolved_device,
                    batch_size=batch_size,
                )
                latents = latent_batches[0]
                if shared_noise is None:
                    shared_noise = _shared_latent_noise(
                        latents.shape[1:],
                        seed=seed,
                        device=resolved_device,
                        dtype=latents.dtype,
                    )
                regenerated = _decode_frame_latents(
                    latent_batches,
                    vae=vae,
                    noise_std=noise_std,
                    shared_noise=shared_noise,
                )
                for reference, candidate in zip(frames, regenerated, strict=True):
                    frame_pipe.write(candidate.tobytes())
                    difference = reference.astype(np.float32) - candidate.astype(np.float32)
                    squared_error += float(np.sum(difference * difference, dtype=np.float64))
                    pixel_count += reference.size
                    if previous_reference is not None and previous_candidate is not None:
                        current_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
                        previous_gray = cv2.cvtColor(previous_reference, cv2.COLOR_BGR2GRAY)
                        frame_maps = _backward_map(current_gray, previous_gray)
                        warped_reference = _backward_warp(previous_reference, frame_maps)
                        warped_candidate = _backward_warp(previous_candidate, frame_maps)
                        temporal_baseline += float(
                            np.mean(np.abs(reference.astype(np.float32) - warped_reference.astype(np.float32)))
                        )
                        temporal_candidate += float(
                            np.mean(np.abs(candidate.astype(np.float32) - warped_candidate.astype(np.float32)))
                        )
                    previous_reference = reference
                    previous_candidate = candidate
                    frame_count += 1
            if frame_count < 2:
                raise ValueError("The selected clip produced fewer than two frames")
            finish_raw_video_encoder(
                process,
                temporary_output,
                operation="SynthID candidate encode",
            )
        except Exception:
            if process.poll() is None:
                abort_raw_video_encoder(process)
            raise

        mse = squared_error / pixel_count
        return RegenerationMetrics(
            frames=frame_count,
            fps=effective_fps,
            width=size[0],
            height=size[1],
            psnr_db=math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse)),
            temporal_residual_ratio=temporal_candidate / max(temporal_baseline, 1e-6),
        )

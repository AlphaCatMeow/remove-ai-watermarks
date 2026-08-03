"""Project-native orchestration for diffusion-based pixel regeneration."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from PIL import Image

from remove_ai_watermarks._internal.watermark_profiles import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROFILE,
    PROFILE_CFG,
    PROFILE_CHOICES,
    PROFILE_STEPS,
    QWEN_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    normalize_profile,
    resolve_seed,
    resolve_steps,
    resolve_strength,
)
from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

# Both two-stage profiles share the face stage, the four-step schedule, CFG 1.0, the
# fixed model stack and the native-resolution contract; only the global model differs.
_ZIMAGE_STACKS = {
    QWEN_ZIMAGE_PROFILE: "Qwen-Image-2512 and Z-Image",
    SDXL_ZIMAGE_PROFILE: "SDXL and Z-Image",
}

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

_HAS_DIFFUSERS = module_available("diffusers")


def is_watermark_removal_available() -> bool:
    """Return whether the standard diffusion runtime can be imported."""
    return _HAS_TORCH and _HAS_DIFFUSERS


def _ensure_watermark_deps() -> None:
    if not is_watermark_removal_available():
        raise ImportError(
            "Invisible watermark regeneration requires the 'diffusion' extra. Install remove-ai-watermarks[diffusion]."
        )


def _has_nvidia_gpu() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def try_empty_device_cache(device: str) -> None:
    """Ask Torch to release cached accelerator memory when the backend supports it.

    Moved here when ``img2img_runner`` was deleted: the runner and its MPS recovery
    path went with the CPU/MPS profiles, leaving this as that module's only content.
    Silent by design -- it runs in cleanup paths where a raise would replace the real
    error.
    """
    if not _HAS_TORCH:
        return
    backend = getattr(torch, device, None)  # type: ignore[union-attr]
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        with contextlib.suppress(Exception):
            empty_cache()


def _backend_works(device: str) -> bool:
    try:
        probe = torch.tensor([1.0], device=device)  # type: ignore[union-attr]
        _ = probe + probe
    except (AssertionError, RuntimeError):
        return False
    return True


def get_device() -> str:
    """Select CUDA, XPU, MPS, or CPU in that order when each backend is usable."""
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available() and _backend_works("cuda"):  # type: ignore[union-attr]
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available() and _backend_works("xpu"):
        return "xpu"
    if _has_nvidia_gpu():
        logger.warning("NVIDIA GPU detected, but the installed PyTorch build has no working CUDA backend")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class WatermarkRemover:
    """Load one regeneration profile and write a metadata-clean raster output."""

    DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    _DEVICES = frozenset({"cuda"})

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: Any = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
        pipeline: str = DEFAULT_PROFILE,
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        self.model_profile = normalize_profile(pipeline)
        if self.model_profile not in PROFILE_CHOICES:
            raise ValueError(f"Unsupported pipeline '{pipeline}'. Use one of: {', '.join(PROFILE_CHOICES)}.")
        if model_id is not None:
            raise ValueError(
                f"The {self.model_profile} profile uses a fixed {_ZIMAGE_STACKS[self.model_profile]} model stack."
            )
        self.model_id = (
            "Qwen/Qwen-Image-2512 + Tongyi-MAI/Z-Image-Turbo"
            if self.model_profile == QWEN_ZIMAGE_PROFILE
            else f"{DEFAULT_MODEL_ID} + Tongyi-MAI/Z-Image-Turbo"
        )
        _ensure_watermark_deps()
        selected_device = (device or get_device()).casefold()
        self.device = get_device() if selected_device == "auto" else selected_device
        # CUDA is a precondition of the object, not of the run. Both profiles raise on
        # any other device, so accepting one here only defers a guaranteed failure to
        # model-load time, several layers down and under the wrong profile's name.
        if self.device not in self._DEVICES:
            raise ValueError(
                f"Invisible-watermark removal is CUDA-only, so '{device}' cannot run it. "
                "Both remaining profiles need an NVIDIA GPU. Visible-mark removal and "
                "every identify command still run on CPU."
            )

        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif self.model_profile == SDXL_ZIMAGE_PROFILE:
            # SDXL ships fp16 weights and an fp16-safe VAE; bf16 would give up the
            # variant without buying anything on this architecture.
            self.torch_dtype = torch.float16  # type: ignore[union-attr]
        else:
            self.torch_dtype = torch.bfloat16  # type: ignore[union-attr]

        self.cpu_offload = cpu_offload
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._qwen_zimage_pipeline: Any = None

    def _set_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            with contextlib.suppress(Exception):
                self._progress_callback(message)

    def preload(self, *, global_only: bool = False) -> None:
        """Materialize the selected model stack before the first request."""
        self._load_qwen_zimage_pipeline().preload(global_only=global_only)

    def _load_qwen_zimage_pipeline(self) -> Any:
        if self._qwen_zimage_pipeline is None:
            if getattr(self, "model_profile", QWEN_ZIMAGE_PROFILE) == SDXL_ZIMAGE_PROFILE:
                from remove_ai_watermarks._internal.sdxl_zimage_pipeline import (
                    SdxlZImagePipeline as _Pipeline,
                )
            else:
                from remove_ai_watermarks._internal.qwen_zimage_pipeline import (
                    QwenZImagePipeline as _Pipeline,
                )

            self._qwen_zimage_pipeline = _Pipeline(
                device=self.device,
                torch_dtype=self.torch_dtype,
                hf_token=self.hf_token,
                progress_callback=self._progress_callback,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                keep_face_models_on_device=False if self.cpu_offload else None,
                keep_global_models_on_device=False if self.cpu_offload else None,
            )
        return self._qwen_zimage_pipeline

    def _run_qwen_zimage(
        self,
        init_image: Image.Image,
        strength: float,
        seed: int | None,
        *,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
    ) -> Image.Image:
        return self._load_qwen_zimage_pipeline().run(
            init_image,
            strength=strength,
            seed=seed,
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    def _generate(
        self,
        image: Image.Image,
        strength: float,
        seed: int | None,
        *,
        tile: bool,
        tile_size: int,
        tile_overlap: int,
    ) -> Image.Image:
        return self._run_qwen_zimage(
            image,
            strength,
            seed,
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    def _write_output(self, image: Image.Image, output_path: Path) -> None:
        import numpy as np

        from remove_ai_watermarks import image_io

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        if not image_io.imwrite(str(output_path), bgr):
            image.save(output_path)
        from remove_ai_watermarks.metadata import remove_ai_metadata

        remove_ai_metadata(output_path, output_path, keep_standard=True)

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        vendor: str | None = None,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
        region: tuple[int, int, int, int] | None = None,
        region_feather: int = 64,
    ) -> Path:
        """Regenerate image pixels and write the result without AI metadata."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        destination = output_path or image_path
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")

        resolved_strength = resolve_strength(strength, vendor, self.model_profile, size=source.size)
        if not 0.0 <= resolved_strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {resolved_strength}")

        # Both profiles are distilled four-step schedules at CFG 1.0. Anything else is
        # a caller error rather than a knob, so it is rejected instead of coerced.
        steps = resolve_steps(num_inference_steps)
        if steps != PROFILE_STEPS:
            raise ValueError(f"The {self.model_profile} profile requires {PROFILE_STEPS} steps.")
        if guidance_scale is not None and guidance_scale != PROFILE_CFG:
            raise ValueError(f"The {self.model_profile} profile requires CFG {PROFILE_CFG}.")

        result = self._generate(
            source,
            resolved_strength,
            resolve_seed(seed),
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        if region is not None:
            import numpy as np

            from remove_ai_watermarks._internal.tiling import feather_region_composite

            if result.size != source.size:
                result = result.resize(source.size, Image.Resampling.LANCZOS)
            merged = feather_region_composite(
                np.asarray(source),
                np.asarray(result.convert("RGB")),
                region,
                feather=region_feather,
            )
            result = Image.fromarray(merged)

        self._write_output(result, destination)
        return destination

    def remove_watermark_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        strength: float | None = None,
        num_inference_steps: int | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> list[Path]:
        """Process matching files in a directory, logging and continuing on failures."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        candidates = sorted(path for path in input_dir.iterdir() if path.suffix.casefold() in extensions)
        for source in candidates:
            try:
                outputs.append(self.remove_watermark(source, output_dir / source.name, strength, num_inference_steps))
            except Exception as error:
                logger.error("Failed to process %s: %s", source, error)
            finally:
                try_empty_device_cache(self.device)
        return outputs


def remove_watermark(
    image_path: Path,
    output_path: Path | None = None,
    strength: float | None = None,
    model_id: str | None = None,
    device: str | None = None,
    hf_token: str | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> Path:
    """Convenience wrapper using the default ControlNet profile."""
    from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

    remover = WatermarkRemover(model_id=model_id, device=device, hf_token=hf_token)
    return remover.remove_watermark(
        image_path,
        output_path,
        strength,
        vendor=vendor_for_strength(image_path),
        region=region,
    )

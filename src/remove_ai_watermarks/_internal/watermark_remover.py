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
    CONTROLNET_CANNY_MODEL,
    DEFAULT_MODEL_ID,
    DEFAULT_STRENGTH,
    QWEN_MODEL_ID,
    QWEN_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    normalize_profile,
    resolve_seed,
    resolve_steps,
    resolve_strength,
    viable_steps,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

# Both two-stage profiles share the face stage, the four-step schedule, CFG 1.0, the
# fixed model stack and the native-resolution contract; only the global model differs.
_ZIMAGE_PROFILES = frozenset({QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE})
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

try:
    from diffusers import AutoPipelineForImage2Image as AutoImg2ImgPipeline

    _HAS_DIFFUSERS = True
except ImportError:
    AutoImg2ImgPipeline = None  # type: ignore[assignment,misc]
    _HAS_DIFFUSERS = False

_SDXL_FP16_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
_DEGENERATE_THRESHOLD = 1.0
_CANNY_LOW = 100
_CANNY_HIGH = 200
_CONTROLNET_PROMPT = "best quality, high quality, sharp, detailed, photographic"
_CONTROLNET_NEGATIVE = "blurry, lowres, deformed, distorted text, garbled text, watermark, jpeg artifacts"
_QWEN_PROMPT = "high quality, sharp, detailed, faithful to the original"
_QWEN_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"


def is_watermark_removal_available() -> bool:
    """Return whether the standard diffusion runtime can be imported."""
    return _HAS_TORCH and _HAS_DIFFUSERS


def _ensure_watermark_deps() -> None:
    if not is_watermark_removal_available():
        raise ImportError(
            "Invisible watermark regeneration requires the 'diffusion' extra. Install remove-ai-watermarks[diffusion]."
        )


def _needs_fp16_vae_fix(model_id: str, default_model_id: str, is_fp16: bool) -> bool:
    """Return whether the default SDXL pipeline needs the overflow-safe VAE."""
    return is_fp16 and model_id == default_model_id


def _is_degenerate_image(image: Image.Image) -> bool:
    """Detect the uniform near-black output produced by an fp16 decode collapse."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(pixels.mean()) < _DEGENERATE_THRESHOLD and float(pixels.std()) < _DEGENERATE_THRESHOLD


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


def _detect_cuda_index_url() -> str:
    """Return a PyTorch wheel index compatible with the reported CUDA runtime."""
    try:
        report = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "https://download.pytorch.org/whl/cu121"
    import re

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", report)
    if match is None:
        return "https://download.pytorch.org/whl/cu121"
    return f"https://download.pytorch.org/whl/cu{match.group(1)}{match.group(2)}"


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


def _make_seed_generator(device: str, seed: int) -> Any:
    """Create a deterministic generator, using CPU when device RNG is unavailable."""
    try:
        return torch.Generator(device=device).manual_seed(seed)  # type: ignore[union-attr]
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(seed)  # type: ignore[union-attr]


def _qwen_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to Qwen's 16-pixel latent grid."""
    return max(16, width - width % 16), max(16, height - height % 16)


def _build_qwen_kwargs(
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    true_cfg_scale: float,
    generator: Any,
) -> dict[str, Any]:
    """Build the Qwen img2img call without importing its optional pipeline class."""
    width, height = _qwen_target_size(image.width, image.height)
    return {
        "prompt": _QWEN_PROMPT,
        "negative_prompt": _QWEN_NEGATIVE,
        "image": image,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "generator": generator,
        "width": width,
        "height": height,
    }


class WatermarkRemover:
    """Load one regeneration profile and write a metadata-clean raster output."""

    DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    DEFAULT_STRENGTH = DEFAULT_STRENGTH
    CONTROLNET_CANNY_MODEL = CONTROLNET_CANNY_MODEL
    _DEVICES = frozenset({"cpu", "mps", "cuda", "xpu"})

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: Any = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
        pipeline: str = "controlnet",
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        requested_model = model_id or self.DEFAULT_MODEL_ID
        self.model_profile = normalize_profile(pipeline)
        if self.model_profile in _ZIMAGE_PROFILES and model_id not in {None, self.DEFAULT_MODEL_ID}:
            raise ValueError(
                f"The {self.model_profile} profile uses a fixed {_ZIMAGE_STACKS[self.model_profile]} model stack."
            )
        self.model_id = (
            "Qwen/Qwen-Image-2512 + Tongyi-MAI/Z-Image-Turbo"
            if self.model_profile == QWEN_ZIMAGE_PROFILE
            else requested_model
        )
        _ensure_watermark_deps()
        selected_device = (device or get_device()).casefold()
        self.device = get_device() if selected_device == "auto" else selected_device
        if self.device not in self._DEVICES:
            raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, mps, cuda, xpu.")

        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif self.device in {"cpu", "mps"}:
            self.torch_dtype = torch.float32  # type: ignore[union-attr]
        elif self.model_profile == SDXL_ZIMAGE_PROFILE:
            # SDXL ships fp16 weights and an fp16-safe VAE; bf16 would give up the
            # variant without buying anything on this architecture.
            self.torch_dtype = torch.float16  # type: ignore[union-attr]
        elif self.model_profile in {"qwen", QWEN_ZIMAGE_PROFILE}:
            self.torch_dtype = torch.bfloat16  # type: ignore[union-attr]
        else:
            self.torch_dtype = torch.float16  # type: ignore[union-attr]

        self.cpu_offload = cpu_offload
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._pipeline: Any = None
        self._controlnet_pipeline: Any = None
        self._qwen_pipeline: Any = None
        self._qwen_zimage_pipeline: Any = None

    def _set_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            with contextlib.suppress(Exception):
                self._progress_callback(message)

    def preload(self, *, global_only: bool = False) -> None:
        """Materialize the selected model stack before the first request."""
        if self.model_profile in _ZIMAGE_PROFILES:
            self._load_qwen_zimage_pipeline().preload(global_only=global_only)
        elif self.model_profile == "qwen":
            self._load_qwen_pipeline()
        elif self.model_profile == "controlnet":
            self._load_controlnet_pipeline()
        else:
            self._load_pipeline()

    def _base_load_kwargs(self) -> dict[str, Any]:
        options: dict[str, Any] = {"torch_dtype": self.torch_dtype}
        if self.hf_token:
            options["token"] = self.hf_token
        return options

    def _load_from_pretrained(self, cls: Any, model_id: str, **kwargs: Any) -> Any:
        if self.torch_dtype == torch.float16:  # type: ignore[union-attr]
            try:
                return cls.from_pretrained(model_id, variant="fp16", **kwargs)
            except Exception as error:
                logger.info("Model %s has no usable fp16 variant (%s); using default weights", model_id, error)
        return cls.from_pretrained(model_id, **kwargs)

    def _maybe_add_fp16_vae(self, options: dict[str, Any]) -> None:
        if not _needs_fp16_vae_fix(
            self.model_id,
            self.DEFAULT_MODEL_ID,
            self.torch_dtype == torch.float16,  # type: ignore[union-attr]
        ):
            return
        from diffusers import AutoencoderKL

        options["vae"] = AutoencoderKL.from_pretrained(_SDXL_FP16_VAE_ID, torch_dtype=torch.float16)

    @staticmethod
    def _disable_sdxl_watermarker(options: dict[str, Any]) -> None:
        options["add_watermarker"] = False

    def _move_to_device_and_optimize(self, pipeline: Any) -> Any:
        if self.cpu_offload and self.device == "cuda":
            offload = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise RuntimeError("CPU offload was requested, but this pipeline does not support it.")
            offload(device="cuda")
        else:
            try:
                pipeline = pipeline.to(self.device)
            except (RuntimeError, AssertionError) as error:
                if self.device == "cuda":
                    raise RuntimeError(
                        f"Failed to move model to CUDA ({error}). Install a compatible PyTorch wheel from "
                        f"{_detect_cuda_index_url()}."
                    ) from error
                raise
        optimize = getattr(pipeline, "enable_xformers_memory_efficient_attention", None)
        if callable(optimize):
            with contextlib.suppress(Exception):
                optimize()
        if self.device == "mps":
            slice_attention = getattr(pipeline, "enable_attention_slicing", None)
            if callable(slice_attention):
                with contextlib.suppress(Exception):
                    slice_attention("max")
        return pipeline

    def _sdxl_options(self) -> dict[str, Any]:
        options = self._base_load_kwargs()
        self._disable_sdxl_watermarker(options)
        self._maybe_add_fp16_vae(options)
        return options

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            options = self._sdxl_options()
            options.update(safety_checker=None, requires_safety_checker=False)
            loaded = self._load_from_pretrained(AutoImg2ImgPipeline, self.model_id, **options)
            self._pipeline = self._move_to_device_and_optimize(loaded)
        return self._pipeline

    def _load_controlnet_pipeline(self) -> Any:
        if self._controlnet_pipeline is None:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

            controlnet = self._load_from_pretrained(
                ControlNetModel,
                CONTROLNET_CANNY_MODEL,
                torch_dtype=self.torch_dtype,
            )
            options = self._sdxl_options()
            options["controlnet"] = controlnet
            loaded = self._load_from_pretrained(
                StableDiffusionXLControlNetImg2ImgPipeline,
                self.model_id,
                **options,
            )
            self._controlnet_pipeline = self._move_to_device_and_optimize(loaded)
        return self._controlnet_pipeline

    def _load_qwen_pipeline(self) -> Any:
        if self._qwen_pipeline is None:
            try:
                from diffusers import QwenImageImg2ImgPipeline
            except ImportError as error:
                raise ImportError("The qwen profile requires Diffusers with QwenImageImg2ImgPipeline.") from error
            model_id = QWEN_MODEL_ID if self.model_id == self.DEFAULT_MODEL_ID else self.model_id
            loaded = QwenImageImg2ImgPipeline.from_pretrained(model_id, **self._base_load_kwargs())
            self._qwen_pipeline = self._move_to_device_and_optimize(loaded)
        return self._qwen_pipeline

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

    def _reload_on_cpu(self, cache_name: str, loader: Callable[[], Any]) -> Any:
        self.device = "cpu"
        self.torch_dtype = torch.float32  # type: ignore[union-attr]
        setattr(self, cache_name, None)
        return loader()

    def _run_img2img(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        from remove_ai_watermarks._internal.img2img_runner import run_img2img_with_mps_fallback

        output, device = run_img2img_with_mps_fallback(
            self._load_pipeline,
            init_image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            self.device,
            self._set_progress,
            reload_on_cpu=lambda: self._reload_on_cpu("_pipeline", self._load_pipeline),
        )
        self.device = device
        return output

    def _build_canny_control_image(self, init_image: Image.Image) -> Image.Image:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.asarray(init_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
        return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))

    def _run_controlnet(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        from remove_ai_watermarks._internal.img2img_runner import run_img2img_with_mps_fallback

        extras = {
            "prompt": _CONTROLNET_PROMPT,
            "negative_prompt": _CONTROLNET_NEGATIVE,
            "control_image": self._build_canny_control_image(init_image),
            "controlnet_conditioning_scale": float(self.controlnet_conditioning_scale),
        }
        output, device = run_img2img_with_mps_fallback(
            self._load_controlnet_pipeline,
            init_image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            self.device,
            self._set_progress,
            reload_on_cpu=lambda: self._reload_on_cpu("_controlnet_pipeline", self._load_controlnet_pipeline),
            extra_kwargs=extras,
        )
        self.device = device
        return output

    def _run_qwen(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        response = self._load_qwen_pipeline()(
            **_build_qwen_kwargs(init_image, strength, num_inference_steps, guidance_scale, generator)
        )
        return response.images[0]

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
        steps: int,
        guidance: float,
        generator: Any,
        seed: int | None,
        *,
        tile: bool,
        tile_size: int,
        tile_overlap: int,
    ) -> Image.Image:
        if self.model_profile in _ZIMAGE_PROFILES:
            return self._run_qwen_zimage(
                image,
                strength,
                seed,
                tile=tile,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

        runner = {
            "qwen": self._run_qwen,
            "controlnet": self._run_controlnet,
        }.get(self.model_profile, self._run_img2img)
        if tile and max(image.size) > tile_size:
            from remove_ai_watermarks._internal.tiling import run_tiled

            return run_tiled(
                lambda crop: runner(crop, strength, steps, guidance, generator),
                image,
                tile_size,
                tile_overlap,
                self._set_progress,
            )
        return runner(image, strength, steps, guidance, generator)

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

        if self.model_profile == QWEN_ZIMAGE_PROFILE:
            from remove_ai_watermarks._internal.qwen_zimage_pipeline import resolution_adaptive_denoise

            resolved_strength = strength if strength is not None else resolution_adaptive_denoise(*source.size)
        else:
            resolved_strength = resolve_strength(strength, vendor, self.model_profile)
        if not 0.0 <= resolved_strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {resolved_strength}")

        resolved_seed = resolve_seed(seed, self.model_profile)
        steps = resolve_steps(num_inference_steps, self.model_profile)
        guidance = 1.0 if guidance_scale is None and self.model_profile in _ZIMAGE_PROFILES else guidance_scale or 7.5
        if self.model_profile in _ZIMAGE_PROFILES:
            if steps != 4:
                raise ValueError(f"The {self.model_profile} profile requires 4 steps.")
            if guidance != 1.0:
                raise ValueError(f"The {self.model_profile} profile requires CFG 1.0.")
        else:
            steps = viable_steps(steps, resolved_strength)

        generator = None
        if resolved_seed is not None and _HAS_TORCH:
            generator = _make_seed_generator(self.device, resolved_seed)
        result = self._generate(
            source,
            resolved_strength,
            steps,
            guidance,
            generator,
            resolved_seed,
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        if self.torch_dtype == torch.float16 and _is_degenerate_image(result):  # type: ignore[union-attr]
            self.torch_dtype = torch.float32  # type: ignore[union-attr]
            self._pipeline = self._controlnet_pipeline = self._qwen_pipeline = self._qwen_zimage_pipeline = None
            result = self._generate(
                source,
                resolved_strength,
                steps,
                guidance,
                generator,
                resolved_seed,
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
        from remove_ai_watermarks._internal.img2img_runner import try_empty_device_cache

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

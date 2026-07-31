"""Execute Diffusers img2img calls and recover from an MPS runtime failure."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal.progress import is_mps_error, make_pipeline_progress

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

logger = logging.getLogger(__name__)


def _pipeline_arguments(
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    step_callback: Any,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "prompt": "",
        "image": image,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    arguments.update(overrides or {})
    if step_callback is not None:
        arguments.update(callback=step_callback, callback_steps=1)
    return arguments


def _invoke(pipeline: Any, arguments: dict[str, Any]) -> Image.Image:
    response = pipeline(**arguments)
    return response.images[0]


def run_img2img(
    pipeline: Any,
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
    extra_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """Run one img2img request and report denoising progress when supported."""
    callback, started, finished, launch_monitor = make_pipeline_progress(
        max(1, int(num_inference_steps * strength)), device, set_progress
    )
    launch_monitor()
    arguments = _pipeline_arguments(
        image, strength, num_inference_steps, guidance_scale, generator, callback, extra_kwargs
    )
    try:
        try:
            return _invoke(pipeline, arguments)
        except TypeError as error:
            if "callback" not in str(error):
                raise
            started.set()
            arguments.pop("callback", None)
            arguments.pop("callback_steps", None)
            return _invoke(pipeline, arguments)
    finally:
        started.set()
        finished.set()


def run_img2img_with_mps_fallback(
    load_pipeline: Callable[[], Any],
    image: Image.Image,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any,
    device: str,
    set_progress: Callable[[str], None],
    *,
    reload_on_cpu: Callable[[], Any],
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Image.Image, str]:
    """Retry an MPS-specific failure once with a freshly loaded CPU pipeline."""
    try:
        output = run_img2img(
            load_pipeline(),
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            generator,
            device,
            set_progress,
            extra_kwargs,
        )
        return output, device
    except RuntimeError as error:
        if device != "mps" or not is_mps_error(error):
            raise
        logger.warning("MPS execution failed (%s); retrying on CPU", error)
        set_progress("MPS execution failed; retrying on CPU...")
        try_empty_device_cache("mps")
        output = run_img2img(
            reload_on_cpu(),
            image,
            strength,
            num_inference_steps,
            guidance_scale,
            None,
            "cpu",
            set_progress,
            extra_kwargs,
        )
        return output, "cpu"


def try_empty_device_cache(device: str) -> None:
    """Ask Torch to release cached accelerator memory when the backend supports it."""
    with contextlib.suppress(Exception):
        import torch

        backend = getattr(torch, device, None)
        empty_cache = getattr(backend, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

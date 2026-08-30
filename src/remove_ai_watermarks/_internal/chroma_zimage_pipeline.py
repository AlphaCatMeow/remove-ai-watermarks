"""The chroma-zimage recipe with a Chroma1 global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`TwoStageZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the profiles.

Pieces bound to this architecture: the diffusers ``ChromaImg2ImgPipeline``
(no DiffSynth here), the neutral faithful-regeneration prompt the floors were
calibrated with, guidance 5.0, and the step-count compensation -- diffusers
truncates the step COUNT (``int(steps * strength)``), so the requested count is
scaled to always spend four effective denoising steps, the same semantics as
``sdxl_zimage_pipeline.requested_steps``. Strength is bound to it too: the flat
vendor floors in ``watermark_profiles`` come from the 2026-08-29/30 oracle
calibration (docs/chroma1-engine-research.md) and do not transfer to another
prompt, guidance, or effective step count.

No Canny conditioning: the floors were measured on a plain strength-controlled
img2img pass, so that is the calibrated path.
"""

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import TwoStageZImagePipeline

log = logging.getLogger(__name__)

CHROMA_MODEL_ID = "lodestones/Chroma1-HD"

# The Chroma1 global stage spends four effective denoising steps at every
# strength. The calibration ladder (docs/chroma1-engine-research.md) held this
# fixed; a different count is a different engine.
CHROMA_STEPS = 4
CHROMA_GUIDANCE = 5.0

# The neutral faithful-regeneration prompt the floors were measured with. It is
# deliberately NOT the canny-stage prompt the Qwen/SDXL profiles use: those were
# never calibrated against Chroma1, and swapping prompts silently moves the
# oracle boundaries.
CHROMA_PROMPT = "high quality, sharp, detailed, faithful to the original"
CHROMA_NEGATIVE = "blurry, lowres, distorted text, garbled text, artifacts"

# The latent grid the calibration floors were generated on. The source is floored
# to this grid, generated, then resized back to the exact input size -- the same
# shape the prototype scripts used.
_LATENT_GRID = 16


def requested_steps(effective_steps: int, strength: float) -> int:
    """Spend ``effective_steps`` regardless of Diffusers' step-count truncation.

    Diffusers img2img truncates the step COUNT (``init_timestep = int(steps *
    strength)``), so at the floors this profile uses a naive request would run
    zero or one steps. Scale the request so the effective count is exact.
    """
    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


def chroma_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to the latent grid without changing aspect."""
    return max(_LATENT_GRID, (width // _LATENT_GRID) * _LATENT_GRID), max(
        _LATENT_GRID, (height // _LATENT_GRID) * _LATENT_GRID
    )


@dataclass
class ChromaZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the Chroma1 global stage plus the inherited face stage."""

    profile_name: ClassVar[str] = "chroma-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._chroma_pipe: Any = None

    def _load_global(self) -> Any:
        if self._chroma_pipe is not None:
            return self._chroma_pipe
        self._require_cuda()
        try:
            import torch
            from diffusers import ChromaImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "The chroma-zimage pipeline needs the optional dependency group. "
                "Install: pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc

        self._progress("Loading Chroma1-HD (bf16)...")
        token = {"token": self.hf_token} if self.hf_token else {}
        pipe = ChromaImg2ImgPipeline.from_pretrained(CHROMA_MODEL_ID, torch_dtype=torch.bfloat16, **token)
        pipe = pipe.to(self.device)
        self._chroma_pipe = pipe
        return pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_global()
        target = chroma_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        steps = requested_steps(CHROMA_STEPS, strength)
        self._progress(f"Running Chroma1 pass: strength={strength:.4f}, steps={CHROMA_STEPS} of {steps}...")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=CHROMA_PROMPT,
            negative_prompt=CHROMA_NEGATIVE,
            image=prepared,
            width=target[0],
            height=target[1],
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=CHROMA_GUIDANCE,
            generator=generator,
        ).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

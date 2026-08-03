"""Project-owned configuration for invisible-watermark regeneration profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
QWEN_MODEL_ID = "Qwen/Qwen-Image"
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

SDXL_PROFILE = "sdxl"
QWEN_ZIMAGE_PROFILE = "qwen-zimage"
SDXL_ZIMAGE_PROFILE = "sdxl-zimage"

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_PATTERN = "sdxl_lightning_4step_lora.safetensors"

OPENAI_STRENGTH = 0.10
GEMINI_STRENGTH = 0.15
UNKNOWN_STRENGTH = GEMINI_STRENGTH
DEFAULT_STRENGTH = UNKNOWN_STRENGTH

QWEN_OPENAI_STRENGTH = 0.10
QWEN_GEMINI_STRENGTH = 0.25
QWEN_UNKNOWN_STRENGTH = QWEN_GEMINI_STRENGTH

# sdxl-zimage runs the qwen-zimage recipe on an SDXL global stage, and strength is
# architecture-bound: at Qwen's 0.154 an SDXL global pass leaves SynthID on a native
# 2816x1536 Gemini original, while 0.20, 0.25 and 0.30 all read clean in the Gemini
# app. 0.25 keeps a rung of margin over that boundary, which the historical SDXL
# certification argues for -- it recorded 0.20 as DETECTED against Gemini on an
# older SDXL pipeline. OpenAI is the easier oracle: the profile already cleared
# openai.com/verify at 0.1102, so 0.15 sits above what was verified rather than on
# it. Unknown follows Gemini, the stricter of the two.
#
# Unlike qwen-zimage this is a flat vendor policy rather than a resolution curve,
# because flat values are what was measured. Every verdict above comes from a fixed
# strength at one size; no size dependence has been established for this stage.
SDXL_ZIMAGE_OPENAI_STRENGTH = 0.15
SDXL_ZIMAGE_GEMINI_STRENGTH = 0.25
SDXL_ZIMAGE_UNKNOWN_STRENGTH = SDXL_ZIMAGE_GEMINI_STRENGTH


@dataclass(frozen=True)
class _StrengthPolicy:
    unknown: float
    by_vendor: dict[str, float]

    def choose(self, vendor: str | None) -> float:
        return self.by_vendor.get((vendor or "").casefold(), self.unknown)


_STANDARD_POLICY = _StrengthPolicy(
    unknown=UNKNOWN_STRENGTH,
    by_vendor={"openai": OPENAI_STRENGTH, "google": GEMINI_STRENGTH},
)
_QWEN_POLICY = _StrengthPolicy(
    unknown=QWEN_UNKNOWN_STRENGTH,
    by_vendor={"openai": QWEN_OPENAI_STRENGTH, "google": QWEN_GEMINI_STRENGTH},
)
_SDXL_ZIMAGE_POLICY = _StrengthPolicy(
    unknown=SDXL_ZIMAGE_UNKNOWN_STRENGTH,
    by_vendor={"openai": SDXL_ZIMAGE_OPENAI_STRENGTH, "google": SDXL_ZIMAGE_GEMINI_STRENGTH},
)
_ALIASES = {
    "default": SDXL_PROFILE,
    "qwen_zimage": QWEN_ZIMAGE_PROFILE,
    "sdxl_zimage": SDXL_ZIMAGE_PROFILE,
}
_FOUR_STEP_PROFILES = frozenset({QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE})


def normalize_profile(profile: str) -> str:
    """Normalize spelling and resolve compatibility aliases."""
    value = profile.strip().casefold()
    return _ALIASES.get(value, value)


def resolve_steps(num_inference_steps: int | None, pipeline: str) -> int:
    """Return an explicit step count or the selected profile's default."""
    if num_inference_steps is not None:
        return num_inference_steps
    return 4 if normalize_profile(pipeline) in _FOUR_STEP_PROFILES else 50


def resolve_seed(seed: int | None, pipeline: str) -> int | None:
    """Keep the fixed four-step Z-Image profiles reproducible by default."""
    if seed is not None:
        return seed
    return 0 if normalize_profile(pipeline) in _FOUR_STEP_PROFILES else None


def strength_default_help() -> str:
    """Describe the live default policy without duplicating its values."""
    return (
        f"vendor-adaptive (OpenAI {OPENAI_STRENGTH} / Google {GEMINI_STRENGTH} / "
        f"unknown {UNKNOWN_STRENGTH}, from the C2PA issuer; qwen-zimage instead uses "
        "resolution-adaptive denoise)"
    )


def resolve_strength(strength: float | None, vendor: str | None = None, pipeline: str | None = None) -> float:
    """Resolve a user override or the calibrated policy for a profile and vendor."""
    if strength is not None:
        return strength
    profile = normalize_profile(pipeline) if pipeline is not None else ""
    if profile == "qwen":
        policy = _QWEN_POLICY
    elif profile == SDXL_ZIMAGE_PROFILE:
        policy = _SDXL_ZIMAGE_POLICY
    else:
        policy = _STANDARD_POLICY
    return policy.choose(vendor)


def viable_steps(num_inference_steps: int, strength: float) -> int:
    """Ensure Diffusers receives at least one effective img2img denoising step."""
    if strength <= 0 or int(num_inference_steps * strength) >= 1:
        return num_inference_steps
    return math.ceil(1.0 / strength)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google"] | None:
    """Select the strength cohort using the input's SynthID provenance proxy."""
    try:
        from remove_ai_watermarks.metadata import synthid_source

        evidence = (synthid_source(image_path) or "").casefold()
    except Exception:
        return None
    if "google" in evidence:
        return "google"
    if "openai" in evidence:
        return "openai"
    return None

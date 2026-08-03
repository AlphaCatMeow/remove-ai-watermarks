"""Tests for cross-platform and cross-device compatibility.

Verifies that device detection, MPS fallback, and platform-specific
code paths work correctly on CPU, MPS (macOS), and CUDA (Linux/Windows).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from remove_ai_watermarks._internal.utils import get_image_format, is_supported_format
from remove_ai_watermarks._internal.watermark_profiles import (
    PROFILE_CHOICES,
    SDXL_ZIMAGE_GEMINI_STRENGTH,
    SDXL_ZIMAGE_OPENAI_STRENGTH,
    SDXL_ZIMAGE_UNKNOWN_STRENGTH,
    normalize_profile,
    resolve_strength,
    strength_default_help,
)
from remove_ai_watermarks._internal.watermark_remover import get_device, is_watermark_removal_available

# ── Device detection ────────────────────────────────────────────────


class TestDeviceDetection:
    """Tests for get_device() across platforms."""

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cpu", "mps", "cuda", "xpu")

    def test_cpu_fallback_when_no_gpu(self):
        """On CI / machines without GPU, should fall back to cpu or mps."""
        device = get_device()
        # Just verify it doesn't crash and returns a valid string
        assert isinstance(device, str)

    @patch("remove_ai_watermarks._internal.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"

    def test_xpu_selected_when_available(self):
        """An XPU-enabled torch (no CUDA) routes to the Intel GPU backend.

        The whole torch module is mocked so the smoke-test ops succeed without
        any real device; cuda must read False so the cuda branch is skipped.
        """
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.xpu.is_available.return_value = True
        with patch("remove_ai_watermarks._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "xpu"
        fake_torch.tensor.assert_called_with([1.0], device="xpu")

    def test_non_cuda_devices_are_refused_at_construction(self):
        """CUDA is a precondition of the object, not of the run.

        Both remaining profiles raise on any other device, so accepting cpu/mps/xpu
        here only defers a guaranteed failure to model-load time - several layers down,
        after the dependency check and the pipeline import, under a message naming
        whichever profile the internal pipeline happens to be.
        """
        if not is_watermark_removal_available():
            pytest.skip("torch/diffusers not installed")
        import torch

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

        for device in ("cpu", "mps", "xpu"):
            with pytest.raises(ValueError, match="CUDA-only"):
                WatermarkRemover(device=device)

        remover = WatermarkRemover(device="cuda")
        assert remover.device == "cuda"
        assert remover.torch_dtype == torch.bfloat16


class TestEmptyDeviceCache:
    """try_empty_device_cache is all that remains of the img2img runner.

    Its module lost run_img2img and the MPS fallback along with the CPU/MPS profiles;
    both surviving profiles are CUDA-only, so there is no MPS failure left to recover
    from. The helper must stay silent on a backend that cannot empty a cache, because
    it runs in cleanup paths where a raise would replace the real error.
    """

    def test_unknown_backend_is_a_silent_no_op(self):
        from remove_ai_watermarks._internal.watermark_remover import try_empty_device_cache

        try_empty_device_cache("cpu")
        try_empty_device_cache("definitely-not-a-backend")


class TestModelProfiles:
    """Only the two CUDA-only two-stage profiles remain."""

    def test_canonical_profiles_unchanged(self):
        assert normalize_profile("qwen-zimage") == "qwen-zimage"
        assert normalize_profile("sdxl-zimage") == "sdxl-zimage"

    def test_underscore_spellings_resolve(self):
        assert normalize_profile("qwen_zimage") == "qwen-zimage"
        assert normalize_profile("  SDXL_ZImage ") == "sdxl-zimage"

    def test_retired_names_no_longer_resolve_to_a_profile(self):
        """default/sdxl/controlnet/qwen were removed, not aliased onward.

        Silently mapping them at the alias layer would route an old script into a
        profile it never asked for; the remover raises on the unknown name instead.
        """
        for retired in ("default", "sdxl", "controlnet", "qwen"):
            assert normalize_profile(retired) not in PROFILE_CHOICES


class TestNoReembeddedWatermark:
    """F2 regression: the SDXL global stage must disable the diffusers watermarker.

    diffusers stamps an open "Stable Diffusion XL" DWT-DCT watermark onto every SDXL
    output whenever ``invisible-watermark`` is installed. A watermark REMOVER that left
    it on would replace one detectable AI watermark (SynthID) with another -- the
    cleaned output re-reads as AI. The ControlNet sub-model load must NOT receive the
    kwarg, since it is not a pipeline and does not accept it.

    Only sdxl-zimage carries an SDXL pipeline now; qwen-zimage's global stage is
    DiffSynth, which has no such watermarker.
    """

    def test_sdxl_global_stage_disables_watermarker(self, monkeypatch: pytest.MonkeyPatch):
        if not is_watermark_removal_available():
            pytest.skip("torch/diffusers not installed")
        import diffusers

        from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        calls: dict[str, dict] = {}

        def record(name):
            def fake(*_args, **kwargs):
                calls[name] = kwargs
                return MagicMock()

            return fake

        monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", record("controlnet"))
        monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", record("vae"))
        monkeypatch.setattr(diffusers.StableDiffusionXLControlNetImg2ImgPipeline, "from_pretrained", record("pipeline"))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "lora.safetensors")
        # from_config would otherwise resolve the mock's config as a repo id.
        monkeypatch.setattr(diffusers.EulerDiscreteScheduler, "from_config", lambda *a, **k: MagicMock())

        pipeline = SdxlZImagePipeline(device="cuda", torch_dtype=None)
        monkeypatch.setattr(type(pipeline), "_require_cuda", lambda self: None)
        pipeline._load_sdxl()

        assert calls["pipeline"].get("add_watermarker") is False
        assert "add_watermarker" not in calls["controlnet"]


class TestResolveStrength:
    """resolve_strength answers for sdxl-zimage and defers for qwen-zimage."""

    def test_qwen_zimage_answers_from_the_resolution_curve(self):
        """The function is total: it owns both policies rather than returning None.

        qwen-zimage picks strength from image area, so it takes the size. Returning
        None for it would push that branch onto every caller and leave one of the two
        strength policies living outside this module. The vendor is ignored here on
        purpose - the curve, not the issuer, is what was calibrated.
        """
        assert resolve_strength(None, "google", "qwen-zimage", size=(2000, 1850)) == pytest.approx(0.154)
        assert resolve_strength(None, None, "qwen-zimage", size=(600, 500)) == pytest.approx(0.084)

    def test_qwen_zimage_without_a_size_fails_loudly(self):
        """A missing size must not silently fall back to some vendor value."""
        with pytest.raises(ValueError, match="size is required"):
            resolve_strength(None, "google", "qwen-zimage")

    def test_sdxl_zimage_uses_its_flat_vendor_ladder(self):

        assert SDXL_ZIMAGE_OPENAI_STRENGTH == 0.15
        assert SDXL_ZIMAGE_GEMINI_STRENGTH == 0.25
        assert SDXL_ZIMAGE_UNKNOWN_STRENGTH == SDXL_ZIMAGE_GEMINI_STRENGTH
        assert resolve_strength(None, "openai", "sdxl-zimage") == SDXL_ZIMAGE_OPENAI_STRENGTH
        assert resolve_strength(None, "google", "sdxl-zimage") == SDXL_ZIMAGE_GEMINI_STRENGTH
        # An unrecognised issuer takes the stricter Gemini value, not the OpenAI one.
        assert resolve_strength(None, "adobe", "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH
        assert resolve_strength(None, None, "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH

    def test_strength_default_help_derives_from_constants(self):

        h = strength_default_help()
        assert str(SDXL_ZIMAGE_OPENAI_STRENGTH) in h
        assert str(SDXL_ZIMAGE_GEMINI_STRENGTH) in h

    def test_explicit_value_overrides_vendor(self):

        assert resolve_strength(0.3, "openai", "sdxl-zimage") == 0.3
        assert resolve_strength(0.3, None, "qwen-zimage") == 0.3

    def test_explicit_zero_is_respected_not_treated_as_unset(self):
        # 0.0 is falsy but explicit -- it must not fall through to the vendor default
        # (the old `strength or DEFAULT` bug would have). Range validation lives in
        # remove_watermark, not here.

        assert resolve_strength(0.0, "google", "sdxl-zimage") == 0.0
        assert resolve_strength(0.0, None, "qwen-zimage") == 0.0


class TestVendorForStrength:
    """vendor_for_strength normalizes the C2PA SynthID proxy to openai/google/None."""

    @staticmethod
    def _patch(value):
        return patch("remove_ai_watermarks.metadata.synthid_source", return_value=value)

    def test_openai(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI"):
            assert vendor_for_strength(Path("x.png")) == "openai"

    def test_google(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_both_issuers_google_wins(self):
        # The more-robust watermark wins -> safer (higher) strength.
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI, Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_none_when_no_synthid_source(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with self._patch(None):
            assert vendor_for_strength(Path("x.png")) is None

    def test_unreadable_metadata_is_none(self):
        from remove_ai_watermarks._internal.watermark_profiles import vendor_for_strength

        with patch("remove_ai_watermarks.metadata.synthid_source", side_effect=OSError):
            assert vendor_for_strength(Path("x.png")) is None


# ── Format utilities ────────────────────────────────────────────────


class TestFormatUtils:
    """Tests for utils.py format helpers."""

    def test_supported_png(self, tmp_path):
        assert is_supported_format(tmp_path / "test.png")

    def test_supported_jpg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpg")

    def test_supported_jpeg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpeg")

    def test_supported_webp(self, tmp_path):
        assert is_supported_format(tmp_path / "test.webp")

    def test_unsupported_bmp(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.bmp")

    def test_unsupported_gif(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.gif")

    def test_get_format_png(self, tmp_path):
        assert get_image_format(tmp_path / "x.png") == "PNG"

    def test_get_format_jpg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpg") == "JPEG"

    def test_get_format_jpeg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpeg") == "JPEG"

    def test_get_format_webp_defaults_png(self, tmp_path):
        # .webp falls through to PNG in current implementation
        assert get_image_format(tmp_path / "x.webp") == "PNG"


# ── Availability checks ────────────────────────────────────────────


class TestAvailability:
    """Tests for dependency availability checks."""

    def test_watermark_removal_available(self):
        # Reflects the actual environment: True iff torch + diffusers (the gpu
        # extra) are importable. The default+dev CI env has no diffusers, so this
        # must not assume the full stack is present.
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_watermark_removal_available() is expected

    def test_invisible_is_available(self):
        import importlib.util

        from remove_ai_watermarks.invisible_engine import is_available

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_available() is expected


# ── Platform-specific path handling ─────────────────────────────────


class TestPlatformPaths:
    """Verify path handling works on current platform."""

    def test_pathlib_works_for_assets(self):
        from pathlib import Path

        asset_dir = Path(__file__).parent.parent / "src" / "remove_ai_watermarks" / "assets"
        assert (asset_dir / "gemini_bg_48.png").exists()
        assert (asset_dir / "gemini_bg_96.png").exists()

    def test_asset_loading_works(self):
        """Verify embedded assets load correctly (critical for packaging)."""
        from remove_ai_watermarks.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # If we get here without error, asset loading works
        assert engine._alpha_small.shape == (48, 48)
        assert engine._alpha_large.shape == (96, 96)

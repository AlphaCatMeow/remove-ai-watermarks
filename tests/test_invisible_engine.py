"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from remove_ai_watermarks.invisible_engine import InvisibleEngine, _target_size, is_available


class TestIsAvailable:
    """Tests for dependency checking."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_available_reflects_dependencies(self):
        """is_available() is True iff torch + diffusers (the diffusion extra) import.

        Must not assume the full stack: the default+dev CI env has no diffusers.
        """
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_available() is expected


class TestInvisibleEngineInit:
    """Tests for InvisibleEngine construction (no GPU required)."""

    def test_default_model_id(self):
        # SDXL base became the default in May 2026 (defeats SynthID v2).
        assert InvisibleEngine.DEFAULT_MODEL_ID == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_preload_forwards_global_only(self):
        engine = object.__new__(InvisibleEngine)
        engine._remover = SimpleNamespace(preload=lambda **kwargs: setattr(engine, "_preload_kwargs", kwargs))

        engine.preload(global_only=True)

        assert engine._preload_kwargs == {"global_only": True}


class TestNativeOutputSize:
    """Model-side latent-grid rounding must not change the public output size."""

    def test_no_polish_restores_native_non_multiple_of_eight_size(self, tmp_path):
        engine = object.__new__(InvisibleEngine)

        def _remove_watermark(image_path, output_path=None, **_kwargs):
            out = output_path or image_path.with_stem(image_path.stem + "_clean")
            # Model-side latent-grid rounding: 18px becomes 16px.
            Image.open(image_path).crop((0, 0, 24, 16)).save(out)
            return out

        engine._remover = SimpleNamespace(remove_watermark=_remove_watermark)
        engine._progress_callback = None
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        Image.new("RGB", (24, 18), (128, 128, 128)).save(src)

        engine.remove_watermark(src, out, min_resolution=0, adaptive_polish=False)

        assert Image.open(out).size == (24, 18)


class TestTargetSize:
    """Regression guard for the native-resolution decision (issues #10 / #15).

    max_resolution=0 must NOT downscale -- the forced downscale->upscale
    round-trip was the quality loss in #10, and downscaling at all let SynthID
    survive in #15 (the native SDXL pass at strength ~0.05 is what defeats it).
    """

    def test_native_default_no_downscale(self):
        # The default (0) means native resolution: no resize, regardless of size.
        assert _target_size(4096, 4096, 0) is None
        assert _target_size(123, 456, 0) is None

    def test_negative_cap_treated_as_native(self):
        assert _target_size(4096, 4096, -1) is None

    def test_cap_below_long_side_downscales(self):
        # 2000x1000, cap 1024 -> long side scaled to 1024, aspect preserved.
        assert _target_size(2000, 1000, 1024) == (1024, 512)

    def test_cap_uses_long_side_for_portrait(self):
        # Portrait: height is the long side, so it drives the ratio.
        assert _target_size(1000, 2000, 1024) == (512, 1024)

    def test_cap_at_or_above_long_side_no_downscale(self):
        # Already within the cap (and exactly equal) -> no resize.
        assert _target_size(800, 600, 1024) is None
        assert _target_size(1024, 768, 1024) is None

    def test_integer_truncation_matches_pil_call_site(self):
        # 1254x1254 (the gpt-image sample) capped at 1000: int(1254*1000/1254)=1000.
        assert _target_size(1254, 1254, 1000) == (1000, 1000)
        # Non-divisible ratio truncates toward zero like int() at the call site.
        assert _target_size(1000, 333, 500) == (500, 166)

    def test_extreme_aspect_ratio_clamps_short_side_to_one(self):
        # 5000x3 capped at 1024: int(3 * 1024/5000) = 0 would crash resize();
        # the short side must clamp to 1, never 0.
        assert _target_size(5000, 3, 1024) == (1024, 1)
        assert _target_size(3, 5000, 1024) == (1, 1024)

    # ── min_resolution floor (small inputs upscaled so SDXL runs near 1024) ──

    def test_floor_default_off(self):
        # min_resolution defaults to 0 -> no upscale, preserving legacy behavior.
        assert _target_size(381, 512, 0) is None

    def test_floor_upscales_small_input(self):
        # 381x512 portrait, floor 1024 -> long side 512 scaled up to 1024 (x2).
        assert _target_size(381, 512, 0, 1024) == (762, 1024)
        # Landscape: width is the long side.
        assert _target_size(512, 381, 0, 1024) == (1024, 762)

    def test_floor_rounds_short_side(self):
        # 333x500, floor 1024: ratio 2.048 -> 333*2.048=681.98 rounds to 682.
        assert _target_size(333, 500, 0, 1024) == (682, 1024)

    def test_floor_no_op_at_or_above_floor(self):
        # Long side already >= floor -> no upscale (and no cap set -> native).
        assert _target_size(1024, 768, 0, 1024) is None
        assert _target_size(2000, 1000, 0, 1024) is None

    def test_cap_takes_precedence_over_floor(self):
        # A huge input with both set: the cap downscales; the floor never fires.
        assert _target_size(2000, 1000, 1024, 1024) == (1024, 512)

    def test_floor_skipped_on_min_above_max_misconfig(self):
        # min(1024) > max(800) is a misconfig: the floor must not upscale above the
        # cap, so it is skipped and the (within-cap) input stays native.
        assert _target_size(500, 400, 800, 1024) is None


class TestEsrganUpscale:
    """Branches of InvisibleEngine._esrgan_upscale (no diffusion model loaded).

    A SimpleNamespace stands in for the engine so we exercise the helper without
    constructing a real InvisibleEngine (which would load WatermarkRemover).
    """

    @staticmethod
    def _fake_engine():
        from types import SimpleNamespace

        return SimpleNamespace(_remover=SimpleNamespace(device="cpu"))

    @staticmethod
    def _pil(w=120, h=80):
        import numpy as np
        from PIL import Image

        return Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8))

    def test_falls_back_to_lanczos_when_extra_absent(self, monkeypatch):
        import numpy as np
        from PIL import Image

        from remove_ai_watermarks import upscaler

        monkeypatch.setattr(upscaler, "is_available", lambda: False)
        img = self._pil()
        out = InvisibleEngine._esrgan_upscale(self._fake_engine(), img, (1024, 683))
        assert out.size == (1024, 683)
        # Identical to a plain Lanczos resize (the fallback path).
        assert np.array_equal(np.asarray(out), np.asarray(img.resize((1024, 683), Image.Resampling.LANCZOS)))

    def test_resizes_esrgan_output_to_exact_target(self, monkeypatch):
        import cv2

        from remove_ai_watermarks import upscaler

        monkeypatch.setattr(upscaler, "is_available", lambda: True)

        # Fake a 2x upscale that does NOT match the requested target; the helper must
        # resize it to the exact target.
        def _fake_upscale(bgr, device=None):
            return cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

        monkeypatch.setattr(upscaler, "upscale", _fake_upscale)
        out = InvisibleEngine._esrgan_upscale(self._fake_engine(), self._pil(), (1024, 683))
        assert out.size == (1024, 683)

    def test_falls_back_to_lanczos_when_upscale_raises(self, monkeypatch):
        import numpy as np
        from PIL import Image

        from remove_ai_watermarks import upscaler

        monkeypatch.setattr(upscaler, "is_available", lambda: True)

        def _boom(bgr, device=None):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(upscaler, "upscale", _boom)
        img = self._pil()
        out = InvisibleEngine._esrgan_upscale(self._fake_engine(), img, (512, 341))
        assert out.size == (512, 341)
        assert np.array_equal(np.asarray(out), np.asarray(img.resize((512, 341), Image.Resampling.LANCZOS)))


class TestCannyControlImage:
    """The ControlNet canny conditioning image builder (pure cv2/numpy; behind the gpu
    extra since it lives on WatermarkRemover). Skips when torch/diffusers are absent."""

    def test_edge_map_is_3channel_rgb(self):
        if not is_available():
            pytest.skip("diffusion extra (torch/diffusers) not installed")
        import numpy as np

        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8))
        # The method uses no instance state, so call it unbound with a dummy self.
        out = WatermarkRemover._build_canny_control_image(None, img)  # type: ignore[arg-type]
        arr = np.array(out)
        assert out.mode == "RGB"
        assert arr.shape == (64, 80, 3)
        assert arr.max() <= 255

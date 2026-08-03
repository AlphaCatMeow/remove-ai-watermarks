"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

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

        engine.remove_watermark(src, out, adaptive_polish=False)

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

    def test_a_small_input_is_left_at_native_size(self):
        """No minimum-resolution floor: only the cap can move geometry."""
        assert _target_size(381, 512, 0) is None
        assert _target_size(381, 512, 4096) is None


class TestEngineDoesNotFabricateAModelId:
    """The engine must forward model_id untouched, including None.

    It used to substitute DEFAULT_MODEL_ID for None. Once the remover tightened its
    "you may not override the fixed stack" check from `not in {None, DEFAULT_MODEL_ID}`
    to `is not None`, that substitution made EVERY InvisibleEngine construction raise -
    and no test saw it, because the library tests build WatermarkRemover directly while
    the engine tests mock it. A deployed Modal worker caught it instead.
    """

    def test_none_stays_none(self):
        from unittest.mock import patch

        import remove_ai_watermarks.invisible_engine as engine_module

        with patch("remove_ai_watermarks._internal.watermark_remover.WatermarkRemover") as remover:
            engine_module.InvisibleEngine(pipeline="qwen-zimage")
        assert remover.call_args.kwargs["model_id"] is None

    def test_an_explicit_model_id_still_reaches_the_remover_to_be_rejected(self):
        from unittest.mock import patch

        import remove_ai_watermarks.invisible_engine as engine_module

        with patch("remove_ai_watermarks._internal.watermark_remover.WatermarkRemover") as remover:
            engine_module.InvisibleEngine(model_id="org/custom", pipeline="qwen-zimage")
        assert remover.call_args.kwargs["model_id"] == "org/custom"

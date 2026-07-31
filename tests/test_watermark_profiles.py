"""Pure tests for the strength/steps profile helpers (no model, no torch needed)."""

from __future__ import annotations

import pytest

from remove_ai_watermarks._internal.watermark_profiles import resolve_strength, viable_steps


class TestViableSteps:
    """Guards the crash found by the release smoke matrix on 2026-07-19.

    diffusers derives its img2img timesteps as ``int(steps * strength)``. When that
    rounds to zero the pipeline builds an empty tensor and dies deep inside attention
    with "cannot reshape tensor of 0 elements into shape [0, -1, 1, 512]". At the
    default strength 0.15 that was every ``--steps`` below 7, reachable with entirely
    valid CLI arguments and no special flags.
    """

    @pytest.mark.parametrize(
        ("steps", "strength"),
        [(1, 0.15), (2, 0.15), (5, 0.15), (6, 0.15), (5, 0.10), (9, 0.10), (1, 0.5)],
    )
    def test_never_returns_a_count_that_denoises_zero_steps(self, steps: int, strength: float):
        assert int(viable_steps(steps, strength) * strength) >= 1

    @pytest.mark.parametrize(
        ("steps", "strength"),
        [(50, 0.15), (20, 0.15), (7, 0.15), (10, 0.10), (2, 0.5), (50, 1.0)],
    )
    def test_leaves_a_workable_count_untouched(self, steps: int, strength: float):
        assert viable_steps(steps, strength) == steps

    def test_raises_only_to_the_minimum_needed(self):
        # strength 0.15 needs 7 (int(7*0.15)==1); it must not jump to some larger default.
        assert viable_steps(5, 0.15) == 7
        assert viable_steps(1, 0.10) == 10

    def test_the_vendor_defaults_all_have_a_reachable_floor(self):
        for vendor in (None, "openai", "google"):
            strength = resolve_strength(None, vendor)
            assert int(viable_steps(1, strength) * strength) >= 1

    @pytest.mark.parametrize("strength", [0.0, -0.1])
    def test_a_non_positive_strength_cannot_loop_or_divide_by_zero(self, strength: float):
        # No denoising is possible at all here; return the caller's value rather than
        # dividing by zero or spinning.
        assert viable_steps(20, strength) == 20

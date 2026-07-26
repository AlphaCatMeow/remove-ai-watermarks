"""Tests for the Qwen 2512 + Z-Image SynthID removal profile."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image


def _mock_watermark_runtime_deps(monkeypatch):
    """Bypass optional GPU imports while testing Qwen Z-Image routing."""
    from remove_ai_watermarks.noai import watermark_remover

    fake_torch = MagicMock()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    monkeypatch.setattr(watermark_remover, "torch", fake_torch)
    monkeypatch.setattr(watermark_remover, "_HAS_TORCH", False)
    monkeypatch.setattr(watermark_remover, "is_watermark_removal_available", lambda: True)


def test_resolution_adaptive_denoise_matches_reference_formula():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import resolution_adaptive_denoise

    # The reference node maps 0.30 MP to the lower bound and 3.70 MP to the
    # upper bound. Level 6 adds one fifth of the configured upward spread.
    assert resolution_adaptive_denoise(600, 500, adaptive_level=6) == pytest.approx(0.084)
    assert resolution_adaptive_denoise(2000, 1850, adaptive_level=6) == pytest.approx(0.154)


def test_largest_face_denoise_matches_reference_formula():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import largest_face_denoise

    image_size = (1000, 1000)
    assert largest_face_denoise([(0, 0, 300, 100)], image_size) == 0.10
    assert largest_face_denoise([(0, 0, 150, 100)], image_size) == 0.05
    assert largest_face_denoise([(0, 0, 900, 900)], image_size) == 0.28


def test_global_kwargs_use_lightning_and_diffsynth_controlnet_shape():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import build_global_kwargs

    image = Image.new("RGB", (1122, 1402))
    kwargs = build_global_kwargs(image, strength=0.11, seed=7, controlnet_input="CONTROL")

    assert kwargs["input_image"].size == (1120, 1392)
    assert kwargs["blockwise_controlnet_inputs"] == ["CONTROL"]
    assert kwargs["denoising_strength"] == 0.11
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["seed"] == 7
    assert kwargs["width"] == 1120
    assert kwargs["height"] == 1392
    assert kwargs["exponential_shift_mu"] == pytest.approx(math.log(3.0))


def test_face_kwargs_use_zimage_reference_settings():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import build_face_kwargs

    crop = Image.new("RGB", (713, 941))
    kwargs = build_face_kwargs(crop, strength=0.17, seed=9)

    assert kwargs["input_image"].size == (704, 928)
    assert kwargs["denoising_strength"] == 0.17
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["cfg_scale"] == 1.0
    assert kwargs["seed"] == 9
    assert kwargs["width"] == 704
    assert kwargs["height"] == 928


def test_canny_control_image_matches_reference_thresholds():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import build_canny_control_image

    source = np.zeros((64, 80, 3), dtype=np.uint8)
    source[:, 40:] = 255
    result = np.asarray(build_canny_control_image(Image.fromarray(source)))

    assert result.shape == (64, 80, 3)
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])
    assert result.max() == 255


def test_yunet_download_targets_verified_lfs_artifact():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import (
        YUNET_MODEL_SHA256,
        YUNET_MODEL_URL,
        YUNET_SCORE_THRESHOLD,
    )

    assert YUNET_MODEL_URL.startswith("https://media.githubusercontent.com/media/opencv/opencv_zoo/")
    assert YUNET_MODEL_SHA256 == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    # YuNet scores are not calibrated like the upstream YOLO detector's scores.
    # A 0.2 YuNet threshold admitted background and decorative false positives,
    # multiplying the serial Z-Image face-stage cost on crowded scenes.
    assert pytest.approx(0.5) == YUNET_SCORE_THRESHOLD


def test_resident_face_models_disable_vram_offload():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import (
        QwenZImagePipeline,
        _pin_vram_managed_models,
        resolve_face_model_residency,
    )

    config = QwenZImagePipeline._zimage_vram_config()

    assert config["offload_device"] == "cpu"
    assert config["onload_device"] == "cpu"
    assert config["preparing_device"] == "cuda"
    assert config["computation_device"] == "cuda"
    assert resolve_face_model_residency(None, total_memory_gib=79.2) is True
    assert resolve_face_model_residency(None, total_memory_gib=39.5) is False
    assert resolve_face_model_residency(False, total_memory_gib=79.2) is False
    assert resolve_face_model_residency(True, total_memory_gib=39.5) is True

    class ManagedModule:
        offload_dtype = "bf16"
        offload_device = "cpu"
        onload_dtype = "bf16"
        onload_device = "cpu"
        preparing_dtype = "bf16"
        preparing_device = "cuda"
        computation_dtype = "bf16"
        computation_device = "cuda"

        def modules(self):
            return [self]

    class Pipe:
        text_encoder = ManagedModule()
        dit = ManagedModule()
        vae_encoder = ManagedModule()
        vae_decoder = ManagedModule()

        def __init__(self):
            self.loaded = None

        def load_models_to_device(self, names):
            self.loaded = names

    pipe = Pipe()
    _pin_vram_managed_models(pipe)

    assert pipe.loaded == ["text_encoder", "dit", "vae_encoder", "vae_decoder"]
    assert pipe.dit.offload_device == "cuda"
    assert pipe.dit.onload_device == "cuda"
    assert pipe.dit.preparing_device == "cuda"


def test_static_prompt_cache_reuses_embeddings_without_caching_image_edits():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import _cache_static_prompt_embeddings

    class PromptUnit:
        output_params = ("prompt_embeds",)

        def __init__(self):
            self.calls = 0

        def process(self, _pipe, prompt, edit_image=None):
            self.calls += 1
            return {"prompt_embeds": [object()], "prompt": prompt, "edit_image": edit_image}

    unit = PromptUnit()
    pipe = MagicMock()
    pipe.units = [unit]

    assert _cache_static_prompt_embeddings(pipe, ("prompt_embeds",)) is True
    first = unit.process(pipe, "constant")
    second = unit.process(pipe, "constant")
    different = unit.process(pipe, "different")
    edited_first = unit.process(pipe, "constant", edit_image=object())
    edited_second = unit.process(pipe, "constant", edit_image=object())

    assert first is second
    assert first is not different
    assert edited_first is not edited_second
    assert unit.calls == 4


def test_sam_pixels_match_model_dtype_without_casting_boxes():
    import torch

    from remove_ai_watermarks.noai.qwen_zimage_pipeline import _prepare_sam_inputs

    class Inputs(dict[str, torch.Tensor]):
        def to(self, device: str):
            return Inputs({name: value.to(device) for name, value in self.items()})

    inputs = Inputs(
        {
            "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "input_boxes": torch.zeros((1, 1, 4), dtype=torch.float32),
        }
    )

    prepared = _prepare_sam_inputs(inputs, "cpu", torch.bfloat16)

    assert prepared["pixel_values"].dtype == torch.bfloat16
    assert prepared["input_boxes"].dtype == torch.float32


def test_sam_prompts_match_impact_center_and_clip_masks_to_boxes():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import (
        _clip_sam_masks_to_boxes,
        _sam_point_prompts,
    )

    boxes = [(2, 3, 8, 9), (10, 4, 16, 12)]
    points, labels = _sam_point_prompts(boxes)
    masks = [np.full((14, 18), 255, dtype=np.uint8) for _box in boxes]

    clipped = _clip_sam_masks_to_boxes(masks, boxes, (18, 14))

    assert points == [[[[5.0, 6.0]], [[13.0, 8.0]]]]
    assert labels == [[[1], [1]]]
    assert np.count_nonzero(clipped[0]) == 36
    assert np.count_nonzero(clipped[1]) == 48
    assert clipped[0][2, 2] == 0
    assert clipped[0][3, 2] == 255


def test_sam_proposal_selection_matches_impact_sub_threshold():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import _select_sam_masks

    masks = np.zeros((2, 3, 8, 8), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    masks[0, 1, 4:7, 4:7] = 1.0
    masks[0, 2, :, :] = 1.0
    masks[1, 0, :, :] = 1.0
    masks[1, 1, 1:5, 1:5] = 1.0
    masks[1, 2, 2:4, 2:5] = 1.0
    scores = np.asarray(
        [
            [0.95, 0.94, 0.50],
            [0.50, 0.70, 0.90],
        ],
        dtype=np.float32,
    )

    selected = _select_sam_masks(masks, scores)

    # The first face unions both proposals over 0.93. The second has none over
    # 0.93, so it falls back to its single highest-IoU proposal.
    assert np.count_nonzero(selected[0]) == 13
    assert np.count_nonzero(selected[1]) == 6
    assert selected[0][6, 6] == 255
    assert selected[1][1, 1] == 0


def test_sam_bfloat16_outputs_convert_to_numpy_float32():
    import torch

    from remove_ai_watermarks.noai.qwen_zimage_pipeline import _sam_outputs_to_numpy

    masks = torch.ones((1, 2, 3, 4, 4), dtype=torch.bfloat16)
    scores = torch.tensor([[[0.95, 0.75, 0.50], [0.99, 0.80, 0.60]]], dtype=torch.bfloat16)

    mask_array, score_array = _sam_outputs_to_numpy(masks, scores)

    assert mask_array.dtype == np.float32
    assert score_array.dtype == np.float32
    assert score_array[0, 0, 0] == pytest.approx(0.94921875)


def test_face_composite_preserves_every_pixel_outside_mask():
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import composite_face

    base = np.full((32, 32, 3), 10, dtype=np.uint8)
    detail = np.full((32, 32, 3), 240, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 255

    result = composite_face(base, detail, mask, feather=0)

    assert np.array_equal(result[:12], base[:12])
    assert np.array_equal(result[:, :12], base[:, :12])
    assert np.all(result[12:20, 12:20] == 240)


def test_profile_defaults_to_four_global_steps():
    from remove_ai_watermarks.noai.watermark_profiles import (
        normalize_profile,
        resolve_seed,
        resolve_steps,
    )

    assert normalize_profile("qwen-zimage") == "qwen-zimage"
    assert resolve_steps(None, "qwen-zimage") == 4
    assert resolve_steps(None, "controlnet") == 50
    assert resolve_steps(12, "qwen-zimage") == 12
    assert resolve_seed(None, "qwen-zimage") == 0
    assert resolve_seed(None, "controlnet") is None
    assert resolve_seed(17, "qwen-zimage") == 17


def test_cli_exposes_qwen_zimage_profile():
    from remove_ai_watermarks.cli import _PIPELINE_CHOICES

    assert "qwen-zimage" in _PIPELINE_CHOICES


def test_cli_qwen_zimage_keeps_upstream_postprocess_default(tmp_image_path, monkeypatch):
    from remove_ai_watermarks import cli

    mock_engine = MagicMock()
    mock_engine.remove_watermark.return_value = tmp_image_path
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: True)
    monkeypatch.setattr("remove_ai_watermarks.invisible_engine.InvisibleEngine", MagicMock(return_value=mock_engine))

    result = CliRunner().invoke(
        cli.main,
        ["invisible", str(tmp_image_path), "--pipeline", "qwen-zimage", "--force"],
    )

    assert result.exit_code == 0, result.output
    assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is False
    assert mock_engine.remove_watermark.call_args.kwargs["seed"] == 0

    result = CliRunner().invoke(
        cli.main,
        [
            "invisible",
            str(tmp_image_path),
            "--pipeline",
            "qwen-zimage",
            "--adaptive-polish",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is True


def test_watermark_remover_dispatches_to_full_pipeline(tmp_path, monkeypatch):
    from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (64, 48), (20, 30, 40)).save(source)

    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (64, 48), (50, 60, 70))
    remover = WatermarkRemover(device="cpu", pipeline="qwen-zimage")
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)
    assert remover.model_id == "Qwen/Qwen-Image-2512 + Tongyi-MAI/Z-Image-Turbo"

    remover.remove_watermark(
        source,
        output,
    )

    runtime.run.assert_called_once()
    _, kwargs = runtime.run.call_args
    assert kwargs["strength"] == pytest.approx(0.084)
    assert kwargs["seed"] == 0
    assert output.exists()


def test_watermark_remover_dispatches_qwen_tiling_to_full_pipeline(tmp_path, monkeypatch):
    from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (96, 80), (20, 30, 40)).save(source)

    runtime = MagicMock()
    runtime.run.return_value = Image.new("RGB", (96, 80), (50, 60, 70))
    remover = WatermarkRemover(device="cpu", pipeline="qwen-zimage")
    monkeypatch.setattr(remover, "_load_qwen_zimage_pipeline", lambda: runtime)

    remover.remove_watermark(
        source,
        output,
        seed=0,
        tile=True,
        tile_size=64,
        tile_overlap=16,
    )

    runtime.run.assert_called_once()
    _, kwargs = runtime.run.call_args
    assert kwargs["seed"] == 0
    assert kwargs["tile"] is True
    assert kwargs["tile_size"] == 64
    assert kwargs["tile_overlap"] == 16
    assert output.exists()


def test_qwen_tiling_runs_global_tiles_then_one_full_frame_face_stage(monkeypatch):
    from remove_ai_watermarks.noai.qwen_zimage_pipeline import (
        QwenZImagePipeline,
        resolution_adaptive_denoise,
    )
    from remove_ai_watermarks.noai.tiling import plan_tiles

    image = Image.new("RGB", (1500, 1500), (20, 30, 40))
    runtime = QwenZImagePipeline(device="cuda", torch_dtype="bf16")
    monkeypatch.setattr(runtime, "_require_cuda", lambda: None)

    global_calls = []

    def fake_global(tile, strength, seed):
        global_calls.append((tile.size, strength, seed))
        return tile

    face_stage = MagicMock(return_value=image)
    monkeypatch.setattr(runtime, "_run_global", fake_global)
    monkeypatch.setattr(runtime, "_run_faces", face_stage)
    monkeypatch.setattr(
        "remove_ai_watermarks.noai.qwen_zimage_pipeline.detect_faces",
        lambda _image: [(100, 100, 300, 300)],
    )
    monkeypatch.setattr(runtime, "_sam_masks", lambda _image, _boxes: [np.ones((1500, 1500), dtype=np.uint8)])

    result = runtime.run(
        image,
        strength=None,
        seed=0,
        tile=True,
        tile_size=1024,
        tile_overlap=128,
    )

    expected_tiles = plan_tiles(1500, 1500, 1024, 128)
    assert len(global_calls) == len(expected_tiles) == 4
    assert all(size == (1024, 1024) for size, _strength, _seed in global_calls)
    assert all(strength == pytest.approx(resolution_adaptive_denoise(1500, 1500)) for _, strength, _ in global_calls)
    assert all(seed == 0 for _, _, seed in global_calls)
    face_stage.assert_called_once()
    assert face_stage.call_args.args[0] is image
    assert face_stage.call_args.args[1].size == image.size
    assert result.size == image.size


def test_qwen_zimage_rejects_runtime_knobs_that_change_fixed_graph(tmp_path, monkeypatch):
    from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

    _mock_watermark_runtime_deps(monkeypatch)
    with pytest.raises(ValueError, match="fixed Qwen-Image-2512"):
        WatermarkRemover(model_id="custom/model", device="cpu", pipeline="qwen-zimage")

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 48)).save(source)
    remover = WatermarkRemover(device="cpu", pipeline="qwen-zimage")
    with pytest.raises(ValueError, match=r"CFG 1\.0"):
        remover.remove_watermark(source, guidance_scale=2.0)
    with pytest.raises(ValueError, match="4-step Lightning"):
        remover.remove_watermark(source, num_inference_steps=8)


def test_invisible_engine_uses_qwen_zimage_step_default(tmp_image_path, tmp_path):
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    engine = InvisibleEngine.__new__(InvisibleEngine)
    engine._progress_callback = None
    engine._remover = MagicMock(model_profile="qwen-zimage")
    engine._remover.remove_watermark.return_value = tmp_path / "clean.png"

    engine.remove_watermark(
        tmp_image_path,
        tmp_path / "clean.png",
        min_resolution=0,
    )

    assert engine._remover.remove_watermark.call_args.kwargs["num_inference_steps"] == 4

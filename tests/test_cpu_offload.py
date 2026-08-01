"""Unit tests for the --cpu-offload device-placement branch.

``WatermarkRemover._move_to_device_and_optimize`` chooses between a full
``pipeline.to("cuda")`` and ``enable_model_cpu_offload()``. The placement
decision is exercised with a mock pipeline and an uninitialized remover, so the
core CI matrix needs no diffusion dependency, model download, or GPU.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover


def _remover(device: str, cpu_offload: bool) -> WatermarkRemover:
    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.device = device
    remover.cpu_offload = cpu_offload
    remover._progress_callback = None
    return remover


class TestCpuOffloadPlacement:
    def test_offload_enabled_on_cuda_streams_instead_of_moving(self):
        remover = _remover("cuda", cpu_offload=True)
        pipeline = Mock()

        returned = remover._move_to_device_and_optimize(pipeline)

        pipeline.enable_model_cpu_offload.assert_called_once_with(device="cuda")
        pipeline.to.assert_not_called()
        # Offload leaves the pipeline object in place (accelerate hooks handle it).
        assert returned is pipeline

    def test_no_offload_moves_whole_pipeline_to_cuda(self):
        remover = _remover("cuda", cpu_offload=False)
        pipeline = Mock()

        remover._move_to_device_and_optimize(pipeline)

        pipeline.to.assert_called_once_with("cuda")
        pipeline.enable_model_cpu_offload.assert_not_called()

    def test_offload_flag_ignored_off_cuda(self):
        # The flag is CUDA-only: on cpu it must still be a plain .to("cpu").
        remover = _remover("cpu", cpu_offload=True)
        pipeline = Mock()

        remover._move_to_device_and_optimize(pipeline)

        pipeline.to.assert_called_once_with("cpu")
        pipeline.enable_model_cpu_offload.assert_not_called()

    def test_offload_fails_loudly_when_pipeline_lacks_support(self):
        remover = _remover("cuda", cpu_offload=True)
        pipeline = Mock(spec=["to"])

        with pytest.raises(RuntimeError, match="does not support"):
            remover._move_to_device_and_optimize(pipeline)

        pipeline.to.assert_not_called()

    def test_qwen_zimage_forces_face_stack_offload(self):
        remover = _remover("cuda", cpu_offload=True)
        remover.torch_dtype = object()
        remover.hf_token = None
        remover.controlnet_conditioning_scale = 1.0
        remover._qwen_zimage_pipeline = None

        runtime = remover._load_qwen_zimage_pipeline()

        assert runtime.keep_face_models_on_device is False

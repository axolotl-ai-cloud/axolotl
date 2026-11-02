"""Tests for the gpu-marker helpers in tests/conftest.py."""

import pytest

from tests.conftest import (
    is_blackwell_skip_reason,
    is_gpu_skip_reason,
    skip_marker_text,
)


@pytest.mark.parametrize(
    "reason",
    [
        "CUDA required",
        "requires two CUDA GPUs",
        "FLA kernels require CUDA",
        "Requires CUDA GPU",
        "needs cuda",
        "requires at least 16 GiB free GPU memory",
        "Need >=2 GPUs for FSDP2 multi-rank tests",
        "varlen_attn needs CUDA",
    ],
)
def test_gpu_reasons_match(reason):
    assert is_gpu_skip_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "swanlab package not installed",
        "xformers not available",
        "torchao required",
        "deep_ep not installed",
        "Not running in CI cache preload",
        "cudatoolkit missing",
        "gpuless",
        "",
        None,
    ],
)
def test_non_gpu_reasons_do_not_match(reason):
    assert not is_gpu_skip_reason(reason)


def test_skip_marker_text_reads_reason_and_string_conditions():
    mark = pytest.mark.skipif("not torch.cuda.is_available()", reason="needs it").mark
    text = skip_marker_text(mark)
    assert "cuda" in text and "needs it" in text
    assert (
        skip_marker_text(pytest.mark.skipif(True, reason="CUDA required").mark)
        == "CUDA required"
    )
    assert skip_marker_text(pytest.mark.skip(reason="flaky").mark) == "flaky"


@pytest.mark.parametrize(
    "reason",
    [
        "CUDA sm100+ required",
        "test requires sm>=100",
        "grouped NVFP4 training backends require Blackwell (sm100/sm120)",
        "grouped fp4 needs sm100/sm120 (Blackwell)",
        "requires an SM_100 GPU",
    ],
)
def test_blackwell_reasons_match(reason):
    assert is_blackwell_skip_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+)",
        "grouped NVFP4 needs CUDA + a resolvable fp4 forward backend (Marlin sm80+ / "
        "DeepGEMM sm90+ / CUTLASS sm120)",
        "marlin W4A16 ext (sm80+, nvcc) required",
        "test requires h100 or newer GPU",
        "CUDA required",
        "sm1000 is not an arch",
        "",
        None,
    ],
)
def test_non_blackwell_reasons_do_not_match(reason):
    assert not is_blackwell_skip_reason(reason)

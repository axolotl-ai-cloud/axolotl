"""``marlin_w4a16_available`` needs the nvcc that the JIT build runs, not just a CUDA_HOME."""

import pytest
import torch
import torch.utils.cpp_extension

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels

from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (  # noqa: E402
    marlin_w4a16_available,
)


@pytest.fixture
def sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    # the probe is cached per process; keep the faked results out of other tests
    marlin_w4a16_available.cache_clear()
    yield
    marlin_w4a16_available.cache_clear()


def test_runtime_only_cuda_home_is_unavailable(sm120, monkeypatch, tmp_path):
    (tmp_path / "lib64").mkdir()
    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", str(tmp_path))
    assert not marlin_w4a16_available()


@pytest.mark.parametrize("mode,available", [(0o755, True), (0o644, False)])
def test_cuda_home_with_nvcc_needs_an_executable(
    sm120, monkeypatch, tmp_path, mode, available
):
    (tmp_path / "bin").mkdir()
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.touch()
    nvcc.chmod(mode)
    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", str(tmp_path))
    assert marlin_w4a16_available() is available


def test_no_cuda_home_is_unavailable(sm120, monkeypatch):
    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", None)
    assert not marlin_w4a16_available()

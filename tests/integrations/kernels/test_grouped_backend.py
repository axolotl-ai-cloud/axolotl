"""CPU-only tests for the grouped NVFP4 base-GEMM backend override resolution."""

import pytest

from axolotl.integrations.kernels.libs.scattermoe_lora import grouped_train as gt


@pytest.fixture(autouse=True)
def _reset_override():
    gt.set_grouped_backend_override(None)
    yield
    gt.set_grouped_backend_override(None)


def test_dequant_forces_fallback():
    gt.set_grouped_backend_override("dequant")
    assert gt._train_backend("nvfp4") is None  # forced chunked-dequant fallback


def test_unavailable_override_falls_back_to_auto():
    # On CPU no fused backend is available, so a forced 'marlin' warns and auto-selects -> None.
    gt.set_grouped_backend_override("marlin")
    assert gt._train_backend("nvfp4") == gt._auto_backend()


def test_non_nvfp4_mode_returns_none():
    gt.set_grouped_backend_override("marlin")
    assert gt._train_backend("fp8") is None


def test_override_setter_normalizes():
    gt.set_grouped_backend_override("MARLIN")
    assert gt._BACKEND_OVERRIDE == "marlin"
    gt.set_grouped_backend_override(None)
    assert gt._BACKEND_OVERRIDE is None

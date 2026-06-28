"""CPU-only tests for the grouped NVFP4 base-GEMM backend override resolution."""

import pytest

from axolotl.integrations.kernels.libs.scattermoe_lora import (
    dequant_grouped as dgq,
    grouped_train as gt,
)


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
    assert gt.RUNTIME.grouped_backend == "marlin"
    gt.set_grouped_backend_override(None)
    assert gt.RUNTIME.grouped_backend is None


@pytest.mark.parametrize(
    "major,expected_order",
    [
        (12, ("marlin", "cutlass", "deepgemm")),  # consumer Blackwell sm120
        (10, ("deepgemm", "marlin")),  # datacenter Blackwell sm100
        (9, ("deepgemm", "marlin")),  # Hopper sm90
        (8, ("marlin",)),  # Ampere/Ada sm80/sm89 — Marlin is the only fused W4A16 path
    ],
)
def test_auto_backend_arch_order(monkeypatch, major, expected_order):
    """The auto-select probe order is arch-aware: each GPU class gets its tuned default first."""
    probed = []

    def fake_available(name):
        probed.append(name)
        return False  # force it to probe the whole order

    monkeypatch.setattr(gt, "_backend_available", fake_available)
    monkeypatch.setattr(gt.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gt.torch.cuda, "get_device_capability", lambda: (major, 0))
    assert gt._auto_backend() is None
    assert tuple(probed) == expected_order


class _FakeDG:
    m_grouped_fp8_fp4_gemm_nt_contiguous = object()


@pytest.mark.parametrize(
    "major,expected",
    [
        (8, False),  # Ampere/Ada -> Marlin
        (
            9,
            False,
        ),  # Hopper sm90: fp8xfp4 kernel asserts arch_major==10 -> must fall back to Marlin
        (
            10,
            True,
        ),  # datacenter Blackwell sm100 (B200): the only arch the fp4 kernel runs on
        (12, False),  # consumer Blackwell sm120 -> CUTLASS/Marlin, not DeepGEMM
    ],
)
def test_deepgemm_available_is_sm100_only(monkeypatch, major, expected):
    """Regression: deepgemm_grouped_available() must gate to sm100. The kernel symbol resolves on
    sm90 too, but calling it asserts arch_major==10 mid-forward; gating up front routes sm90 to
    Marlin instead of crashing."""
    monkeypatch.setattr(dgq.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dgq.torch.cuda, "get_device_capability", lambda: (major, 0))
    monkeypatch.setattr(
        dgq, "_dg", lambda: _FakeDG()
    )  # avoid needing a real kernel/GPU on sm100
    assert dgq.deepgemm_grouped_available() is expected


def test_deepgemm_unavailable_without_cuda(monkeypatch):
    monkeypatch.setattr(dgq.torch.cuda, "is_available", lambda: False)
    assert dgq.deepgemm_grouped_available() is False

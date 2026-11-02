"""Tests for the setup-time skip conditions in tests/cuda_gates.py."""

import pytest
import torch

from tests.conftest import skip_marker_text
from tests.cuda_gates import AtSetup, cuda_capability_below, cuda_major_not_in


@pytest.fixture
def device_queries(monkeypatch):
    queries = []

    def get_device_capability():
        queries.append("capability")
        return (9, 0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    return queries


def test_device_is_queried_only_when_evaluated(device_queries):
    below_sm89 = cuda_capability_below(8, 9)
    not_blackwell = cuda_major_not_in(10, 12)
    assert device_queries == []

    assert not below_sm89
    assert not_blackwell
    assert not below_sm89
    assert device_queries == ["capability", "capability"]


def test_or_with_an_eager_guard_stays_deferred(device_queries):
    condition = not torch.cuda.is_available() or cuda_capability_below(8, 9)
    assert isinstance(condition, AtSetup)
    assert device_queries == []


@pytest.mark.parametrize(
    "capability,below_sm89,below_sm100,not_blackwell",
    [
        ((8, 6), True, True, True),
        ((8, 9), False, True, True),
        ((9, 0), False, True, True),
        ((10, 0), False, False, False),
        ((11, 0), False, False, True),
        ((12, 0), False, False, False),
    ],
)
def test_capability_comparisons(
    monkeypatch, capability, below_sm89, below_sm100, not_blackwell
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert bool(cuda_capability_below(8, 9)) is below_sm89
    assert bool(cuda_capability_below(10)) is below_sm100
    assert bool(cuda_major_not_in(10, 12)) is not_blackwell


def test_without_cuda_conditions_skip(monkeypatch):
    def get_device_capability():
        raise AssertionError("queried the device without CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    assert cuda_capability_below(8, 9)
    assert cuda_major_not_in(10, 12)


def test_gpu_marker_heuristics_see_only_the_reason():
    reason = "grouped fp4 needs sm100/sm120 (Blackwell)"
    mark = pytest.mark.skipif(cuda_major_not_in(10, 12), reason=reason).mark
    assert skip_marker_text(mark) == reason

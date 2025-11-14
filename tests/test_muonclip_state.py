"""Tests for MuonClip state helpers."""

import torch

from axolotl.muonclip.state import MuonStateStore


def test_state_store_allocates_buffers_like_param():
    param = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
    store = MuonStateStore()

    state = store.get_or_create(param, with_rms=True)

    assert state.momentum.shape == param.shape
    assert torch.all(state.momentum == 0)
    assert state.momentum.device == param.device
    assert state.rms is not None
    assert state.rms.shape == param.shape


def test_state_store_reuses_existing_entries():
    param = torch.nn.Parameter(torch.ones(4))
    store = MuonStateStore()

    first = store.get_or_create(param)
    first.momentum.add_(1)

    second = store.get_or_create(param)
    assert torch.all(second.momentum == 1)
    assert first is second


def test_state_dict_round_trip_cpu():
    param = torch.nn.Parameter(torch.ones(4))
    store = MuonStateStore(device=torch.device("cpu"))
    state = store.get_or_create(param, with_rms=True)
    state.momentum.add_(2)
    state.rms.add_(3)

    payload = store.state_dict()

    new_store = MuonStateStore(device=torch.device("cpu"))
    new_store.get_or_create(param, with_rms=True)
    new_store.load_state_dict(payload)
    restored = new_store.get_or_create(param, with_rms=True)
    assert torch.allclose(restored.momentum, torch.full_like(restored.momentum, 2))
    assert torch.allclose(restored.rms, torch.full_like(restored.rms, 3))

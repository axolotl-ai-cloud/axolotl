"""Tests for ZeRO/FSDP gather helpers."""

import torch

from axolotl.muonclip.zero_utils import gather_full_param


def test_gather_full_param_noop_on_cpu():
    param = torch.nn.Parameter(torch.randn(2, 2))
    before = param.clone()
    with gather_full_param(param) as full, torch.no_grad():
        assert full is param
        full.add_(1.0)
    assert torch.allclose(param, before + 1.0)

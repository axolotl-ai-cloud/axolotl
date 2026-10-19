"""Tied expert modules (DiffusionGemma encoder/decoder) share one grouped-fp4 weight cache."""

import torch
from torch import nn

from axolotl.integrations.kernels.libs.scattermoe_lora.experts import _fp4_weight_cache


class _Experts(nn.Module):
    def __init__(self, gate_up: nn.Parameter):
        super().__init__()
        self.gate_up_proj = gate_up


def test_tied_modules_share_one_cache_and_untied_modules_do_not():
    shared = nn.Parameter(torch.zeros(2, 4, 3), requires_grad=False)
    encoder, decoder = _Experts(shared), _Experts(shared)
    other = _Experts(nn.Parameter(torch.zeros(2, 4, 3), requires_grad=False))

    cache = _fp4_weight_cache(encoder, encoder.gate_up_proj)
    # The marlin backend frees qdata after filling the first module's cache; the
    # tied module must find the repacked weight there.
    cache["marlin_gate_up"] = "repacked"

    assert _fp4_weight_cache(decoder, decoder.gate_up_proj) is cache
    assert _fp4_weight_cache(encoder, encoder.gate_up_proj) is cache
    assert _fp4_weight_cache(other, other.gate_up_proj) is not cache
    assert _fp4_weight_cache(other, other.gate_up_proj) == {}


def test_module_cache_survives_a_fresh_weight_tensor():
    # FSDP re-gathers a fresh weight tensor each step; the module keeps its cache.
    module = _Experts(nn.Parameter(torch.zeros(1, 2, 2), requires_grad=False))
    cache = _fp4_weight_cache(module, module.gate_up_proj)
    assert _fp4_weight_cache(module, torch.zeros(1, 2, 2)) is cache

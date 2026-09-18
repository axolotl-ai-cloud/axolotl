"""selective_expert_weights must route staged torchao NF4 experts to their parametrization."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
    selective_expert_weights,
)


def test_torchao_nf4_experts_dispatch_to_parametrization():
    pytest.importorskip("torchao")
    from axolotl.utils.nf4 import quantize_torchao_nf4

    weight = torch.randn(4, 64, 128)
    data, transform = quantize_torchao_nf4(weight, chunk_size=16384)
    experts = nn.Module()
    experts.num_experts = 4
    experts.gate_up_proj = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(
        experts, "gate_up_proj", transform, unsafe=True
    )

    active = torch.tensor([3, 1])
    selected = selective_expert_weights(experts, "gate_up_proj", active)

    assert selected.shape == (2, 64, 128)
    torch.testing.assert_close(selected, transform(data)[active], rtol=0, atol=0)

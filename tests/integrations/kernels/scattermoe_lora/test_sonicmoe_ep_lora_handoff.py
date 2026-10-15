# SPDX-License-Identifier: Apache-2.0
"""The SonicMoE fast path hands PEFT-layout LoRA to the raw base; the ScatterMoE EP forward must see it."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from axolotl.integrations.kernels.libs.scattermoe_lora import (
    experts as ex,  # noqa: E402
)


def _experts(e_local=2, e_global=8, rank=4, hidden=6, inter=3):
    module = SimpleNamespace(
        gate_up_proj=torch.zeros(e_local, 2 * inter, hidden),
        down_proj=torch.zeros(e_local, hidden, inter),
        num_experts=e_local,
        num_experts_global=e_global,
        local_expert_offset=2,
    )
    wrapper = SimpleNamespace(
        r={"default": rank}, active_adapters=["default"], _ep_lora_sharded=True
    )
    module._sonicmoe_lora = {
        "gate_up_proj": (
            torch.randn(e_local * rank, hidden),
            torch.randn(2 * inter, rank * e_local),
            0.5,
        ),
        "down_proj": (
            torch.randn(e_local * rank, inter),
            torch.randn(hidden, rank * e_local),
            0.5,
        ),
    }
    module._sonicmoe_lora_wrappers = {"gate_up_proj": wrapper, "down_proj": wrapper}
    return module


def test_handed_lora_reaches_the_merge_aware_lookup():
    module = _experts()
    gup, down = ex._ep_local_peft_lora(module)
    assert gup[0].shape == (8, 6) and gup[1].shape == (6, 8) and gup[2] == 0.5
    assert down[0].shape == (8, 3) and down[1].shape == (6, 8)


def test_handed_lora_reaches_the_scattermoe_forward_in_its_layout():
    module = _experts()
    converted = ex._handed_lora_as_scattermoe(module)
    assert set(converted) == {"gate_up_proj", "down_proj"}
    sm_A, sm_B, scaling = converted["gate_up_proj"]
    assert torch.equal(sm_A, module._sonicmoe_lora["gate_up_proj"][0])
    assert sm_B.shape == (6, 8) and scaling == 0.5
    module._sonicmoe_lora = None
    assert ex._handed_lora_as_scattermoe(module) is None

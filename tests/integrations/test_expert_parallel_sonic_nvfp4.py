"""CPU routing coverage for SonicMoE's DeepEP local kernel."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import experts_fn


def _native_weight():
    pytest.importorskip("torchao")
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    return NVFP4Tensor.to_nvfp4(torch.randn(16, 16, dtype=torch.bfloat16))


def test_sonicmoe_local_routes_native_weights_to_merge_aware_ep(monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora import experts as scatter

    native = _native_weight()
    experts = SimpleNamespace(
        gate_up_proj=native,
        down_proj=native,
        num_experts=2,
    )
    received = {}

    def merge_aware(module, hidden_states, topk_idx, topk_weights):
        received.update(
            module=module,
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        return hidden_states + 1

    monkeypatch.setattr(scatter, "scattermoe_experts_forward_ep", merge_aware)
    monkeypatch.setattr(
        "axolotl.integrations.kernels.libs.sonicmoe.experts.sonicmoe_experts_forward_with_lora",
        lambda *_: pytest.fail(
            "native experts must not enter SonicMoE's EP-unsupported path"
        ),
    )
    hidden = torch.randn(3, 16)
    local_ids = torch.tensor([[0, -1], [1, 0], [-1, 1]])
    weights = torch.rand(3, 2)

    actual = experts_fn._sonicmoe_local(experts, hidden, local_ids, weights)

    assert actual is not hidden
    assert received["module"] is experts
    assert received["hidden_states"] is hidden
    assert received["topk_idx"] is local_ids
    assert received["topk_weights"] is weights


def test_sonicmoe_local_keeps_dense_sentinel_and_bucket_behavior(monkeypatch):
    experts = SimpleNamespace(
        gate_up_proj=torch.randn(2, 32, 16),
        down_proj=torch.randn(2, 16, 16),
        num_experts=2,
    )
    received = {}

    def sonic(module, hidden_states, topk_idx, topk_weights):
        received.update(
            module=module,
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        return hidden_states

    monkeypatch.setattr(
        "axolotl.integrations.kernels.libs.sonicmoe.experts.sonicmoe_experts_forward_with_lora",
        sonic,
    )
    hidden = torch.randn(3, 16)
    local_ids = torch.tensor([[0, -1], [1, 0], [-1, 1]])
    weights = torch.rand(3, 2)

    actual = experts_fn._sonicmoe_local(experts, hidden, local_ids, weights)

    assert actual.shape == hidden.shape
    assert received["module"] is experts
    assert received["hidden_states"].shape[0] == 1024
    assert torch.all(received["topk_idx"][:3][local_ids < 0] == 2)
    assert torch.equal(received["topk_weights"][:3], weights)

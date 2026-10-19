"""CPU routing coverage for SonicMoE's expert-parallel local forward."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe import experts as sonic


def _native_weight():
    pytest.importorskip("torchao")
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    return NVFP4Tensor.to_nvfp4(torch.randn(16, 16, dtype=torch.bfloat16))


def _ep_experts(weight):
    return SimpleNamespace(
        gate_up_proj=weight, down_proj=weight, num_experts=2, num_experts_global=4
    )


def test_sonicmoe_ep_routes_native_weights_to_merge_aware_ep(monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora import experts as scatter

    experts = _ep_experts(_native_weight())
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
        sonic,
        "_sonicmoe_forward",
        lambda *_: pytest.fail("native experts must not enter the CUTLASS path"),
    )
    hidden = torch.randn(3, 16)
    local_ids = torch.tensor([[0, 2], [1, 0], [2, 1]])
    weights = torch.rand(3, 2)

    actual = sonic.sonicmoe_experts_forward_with_lora(
        experts, hidden, local_ids, weights
    )

    assert actual is not hidden
    assert received["module"] is experts
    assert received["hidden_states"] is hidden
    assert received["topk_idx"] is local_ids
    assert received["topk_weights"] is weights


def test_sonicmoe_ep_keeps_dense_sentinel_and_bucket_behavior(monkeypatch):
    experts = _ep_experts(torch.randn(2, 32, 16))
    experts.down_proj = torch.randn(2, 16, 16)
    received = {}

    def cutlass(module, hidden_states, topk_idx, topk_weights):
        received.update(
            module=module,
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        return hidden_states

    monkeypatch.setattr(sonic, "_sonicmoe_forward", cutlass)
    hidden = torch.randn(3, 16)
    local_ids = torch.tensor([[0, 2], [1, 0], [2, 1]])
    weights = torch.rand(3, 2)

    actual = sonic.sonicmoe_experts_forward_with_lora(
        experts, hidden, local_ids, weights
    )

    assert actual.shape == hidden.shape
    assert received["module"] is experts
    assert received["hidden_states"].shape[0] == 1024
    assert torch.equal(received["topk_idx"][:3], local_ids)
    # pad rows carry the local sentinel id, so every GEMM range drops them
    assert torch.all(received["topk_idx"][3:] == experts.num_experts)
    assert torch.equal(received["topk_weights"][:3], weights)
    assert torch.all(received["topk_weights"][3:] == 0)


def test_sonicmoe_ep_native_empty_batch_preserves_raw_routing(monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora import experts as scatter

    experts = _ep_experts(_native_weight())
    received = {}

    def merge_aware(module, hidden_states, topk_idx, topk_weights):
        received.update(ids=topk_idx, weights=topk_weights)
        return hidden_states * 0

    monkeypatch.setattr(scatter, "scattermoe_experts_forward_ep", merge_aware)
    hidden = torch.randn(0, 16)
    ids = torch.empty((0, 2), dtype=torch.long)
    weights = torch.empty((0, 2))

    output = sonic.sonicmoe_experts_forward_with_lora(experts, hidden, ids, weights)

    assert output.shape == hidden.shape
    assert received["ids"] is ids
    assert received["weights"] is weights


def test_sonicmoe_dense_module_skips_ep_forward(monkeypatch):
    experts = SimpleNamespace(has_gate=True, num_experts=2)
    monkeypatch.setattr(
        sonic,
        "_sonicmoe_ep_forward",
        lambda *_: pytest.fail("a module without num_experts_global is not EP-sharded"),
    )
    with pytest.raises(ValueError, match="CUDA"):
        sonic.sonicmoe_experts_forward_with_lora(
            experts,
            torch.zeros(2, 4),
            torch.zeros(2, 1, dtype=torch.long),
            torch.ones(2, 1),
        )

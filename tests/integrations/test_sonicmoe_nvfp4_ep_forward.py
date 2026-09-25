# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""CPU coverage for merge-aware NVFP4 expert-parallel local forwards."""

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
    grouped_moe_merge_aware_ep_forward,
    set_merge_aware_enabled,
)


def _nvfp4_experts(experts, rows, columns):
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    dense = torch.randn(experts, rows, columns) * 0.1
    packed, scales, pts = [], [], []
    for expert in dense:
        per_tensor_scale = per_tensor_amax_to_scale(expert.abs().max())
        weight = NVFP4Tensor.to_nvfp4(expert, per_tensor_scale=per_tensor_scale)
        packed.append(weight.qdata)
        scales.append(weight.scale)
        pts.append(per_tensor_scale)
    return NVFP4Tensor(
        torch.stack(packed),
        torch.stack(scales),
        16,
        torch.float32,
        per_tensor_scale=torch.stack(pts).reshape(experts, 1, 1),
    )


def _slice_nvfp4(weight, start, stop):
    return type(weight)(
        weight.qdata[start:stop],
        weight.scale[start:stop],
        weight.block_size,
        weight.orig_dtype,
        per_tensor_scale=weight.per_tensor_scale[start:stop],
    )


def _factors(experts, output, input_size, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            rank * experts, input_size, generator=generator, requires_grad=True
        ),
        torch.randn(output, rank * experts, generator=generator, requires_grad=True),
    )


def _slice_factors(A, B, start, stop, experts, rank):
    return (
        A[start * rank : stop * rank].detach().clone().requires_grad_(True),
        B.reshape(B.shape[0], rank, experts)[:, :, start:stop]
        .reshape(B.shape[0], rank * (stop - start))
        .detach()
        .clone()
        .requires_grad_(True),
    )


def _route(tokens, top_k, experts):
    generator = torch.Generator().manual_seed(19)
    ids = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)]
    )
    weights = torch.rand(tokens, top_k, generator=generator)
    return ids, weights / weights.sum(dim=-1, keepdim=True)


@pytest.fixture(autouse=True)
def _merge_aware():
    set_merge_aware_enabled(True)
    yield
    set_merge_aware_enabled(False)


def _forward(x, ids, weights, w1, w2, lora1, lora2, experts):
    return grouped_moe_merge_aware_ep_forward(
        x,
        ids,
        weights,
        w1,
        None,
        w2,
        None,
        lora1,
        lora2,
        experts,
        act="silu",
        concat=True,
        scaling1=0.7,
        scaling2=0.4,
    )


def test_ep_partitions_preserve_merge_aware_output_and_factor_grads():
    pytest.importorskip("torchao")
    torch.manual_seed(5)
    experts, hidden, intermediate, rank, tokens, top_k = 4, 16, 16, 2, 11, 2
    w1 = _nvfp4_experts(experts, 2 * intermediate, hidden)
    w2 = _nvfp4_experts(experts, hidden, intermediate)
    A1, B1 = _factors(experts, 2 * intermediate, hidden, rank, seed=1)
    A2, B2 = _factors(experts, hidden, intermediate, rank, seed=2)
    ids, weights = _route(tokens, top_k, experts)
    x = torch.randn(tokens, hidden, requires_grad=True)
    cotangent = torch.randn(tokens, hidden)

    full = _forward(x, ids, weights, w1, w2, (A1, B1), (A2, B2), experts)
    full.backward(cotangent)
    full_grads = [A1.grad.clone(), B1.grad.clone(), A2.grad.clone(), B2.grad.clone()]
    full_x_grad = x.grad.clone()

    output = torch.zeros_like(full)
    x_grad = torch.zeros_like(x)
    partition_grads = [torch.zeros_like(grad) for grad in full_grads]
    for partition in range(2):
        start, stop = partition * 2, (partition + 1) * 2
        local_ids = torch.where(
            (ids >= start) & (ids < stop), ids - start, torch.full_like(ids, -1)
        )
        factors = [
            *_slice_factors(A1, B1, start, stop, experts, rank),
            *_slice_factors(A2, B2, start, stop, experts, rank),
        ]
        local_x = x.detach().clone().requires_grad_(True)
        local = _forward(
            local_x,
            local_ids,
            weights,
            _slice_nvfp4(w1, start, stop),
            _slice_nvfp4(w2, start, stop),
            tuple(factors[:2]),
            tuple(factors[2:]),
            stop - start,
        )
        local.backward(cotangent)
        output += local.detach()
        x_grad += local_x.grad
        for factor_index, (target, factor) in enumerate(
            zip(partition_grads, factors, strict=True)
        ):
            if factor_index in (1, 3):
                target.reshape(target.shape[0], rank, experts)[:, :, start:stop].copy_(
                    factor.grad.reshape(target.shape[0], rank, stop - start)
                )
            else:
                target[start * rank : stop * rank].copy_(factor.grad)

    assert torch.allclose(output, full.detach(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(x_grad, full_x_grad, rtol=1e-5, atol=1e-6)
    for expected, actual in zip(full_grads, partition_grads, strict=True):
        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_ep_partial_adapter_and_all_sentinel_output_are_differentiable():
    pytest.importorskip("torchao")
    experts, hidden, intermediate, rank = 2, 16, 16, 2
    w1 = _nvfp4_experts(experts, 2 * intermediate, hidden)
    w2 = _nvfp4_experts(experts, hidden, intermediate)
    A1, B1 = _factors(experts, 2 * intermediate, hidden, rank, seed=3)
    active_x = torch.randn(3, hidden, requires_grad=True)
    active_ids = torch.tensor([[0, 1], [1, 0], [0, 0]])
    active_weights = torch.rand(3, 2)
    active = _forward(
        active_x, active_ids, active_weights, w1, w2, (A1, B1), None, experts
    )
    active.square().sum().backward()
    assert A1.grad is not None and A1.grad.norm() > 0
    assert B1.grad is not None and B1.grad.norm() > 0

    x = torch.randn(3, hidden, requires_grad=True)
    ids = torch.full((3, 2), -1, dtype=torch.long)
    weights = torch.rand(3, 2)
    output = _forward(x, ids, weights, w1, w2, (A1, B1), None, experts)
    assert output.requires_grad
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x.grad))


def test_ep_merge_aware_fallback_warns_once_and_marks_module():
    from types import SimpleNamespace

    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _warn_merge_aware_ep_unsupported,
    )

    experts = SimpleNamespace()
    with pytest.warns(RuntimeWarning, match="deployment parity is not guaranteed"):
        _warn_merge_aware_ep_unsupported(experts, "no local adapters")
    _warn_merge_aware_ep_unsupported(experts, "no local adapters")
    assert experts._axolotl_merge_aware_unsupported is True


def test_frozen_ep_experts_do_not_mark_merge_aware_unsafe(monkeypatch):
    from types import SimpleNamespace

    import axolotl.integrations.kernels.libs.scattermoe_lora.experts as expert_module

    experts = SimpleNamespace()
    monkeypatch.setattr(expert_module, "_ep_local_peft_lora", lambda _: (None, None))
    output, reason = expert_module._ep_merge_aware_forward(experts, None, None, None)
    assert output is None
    assert reason is None
    assert not hasattr(experts, "_axolotl_merge_aware_unsupported")


def test_ep_merge_aware_dynamic_activation_falls_back_before_native_forward(
    monkeypatch,
):
    from types import SimpleNamespace

    import axolotl.integrations.kernels.libs.scattermoe_lora.experts as expert_module

    weight = SimpleNamespace(act_quant_kwargs=object())
    experts = SimpleNamespace(gate_up_proj=weight, down_proj=weight)
    monkeypatch.setattr(
        expert_module,
        "_ep_local_peft_lora",
        lambda _: ((object(), object(), 1.0), None),
    )
    monkeypatch.setattr(expert_module, "is_nvfp4_param", lambda _: True)

    output, reason = expert_module._ep_merge_aware_forward(experts, None, None, None)

    assert output is None
    assert reason == "dynamic activation quantization"


def test_ep_merge_aware_sharded_factor_access_falls_back(monkeypatch):
    from types import SimpleNamespace

    import axolotl.integrations.kernels.libs.scattermoe_lora.experts as expert_module

    experts = SimpleNamespace()
    monkeypatch.setattr(expert_module, "_ep_adapter_unsupported_reason", lambda _: None)
    monkeypatch.setattr(
        expert_module,
        "_ep_factor_access_reason",
        lambda _: "FSDP-sharded LoRA factors outside their materialized forward",
    )
    monkeypatch.setattr(
        expert_module,
        "_ep_local_peft_lora",
        lambda _: pytest.fail("unsafe factors must not be read"),
    )

    output, reason = expert_module._ep_merge_aware_forward(experts, None, None, None)

    assert output is None
    assert reason == "FSDP-sharded LoRA factors outside their materialized forward"


@pytest.mark.parametrize("pts_kind", ["none", "scalar", "per_expert"])
@pytest.mark.parametrize("ep_size", [2, 4])
def test_ep_fresh_snap_commutes_with_expert_sharding(pts_kind, ep_size):
    """Each local expert partition must retain the full fresh-grid bytes."""
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        fake_quant_nvfp4_dispatch,
        quantize_nvfp4_merge,
    )

    torch.manual_seed(56)
    experts, rows, columns = 8, 32, 32
    weights = (torch.randn(experts, rows, columns) * 0.05).bfloat16()
    scales = {
        "none": None,
        "scalar": torch.tensor(0.7),
        "per_expert": torch.linspace(0.2, 0.9, experts),
    }[pts_kind]
    packed, block_scale = quantize_nvfp4_merge(weights, scales, scale_mode="fresh")
    snapped = fake_quant_nvfp4_dispatch(weights, scales)

    local_experts = experts // ep_size
    for rank in range(ep_size):
        start, stop = rank * local_experts, (rank + 1) * local_experts
        local_scales = (
            scales
            if scales is None or scales.numel() == 1
            else scales[start:stop].contiguous()
        )
        local_packed, local_block_scale = quantize_nvfp4_merge(
            weights[start:stop].contiguous(), local_scales, scale_mode="fresh"
        )
        assert torch.equal(local_packed, packed[start:stop])
        assert torch.equal(local_block_scale, block_scale[start:stop])
        assert torch.equal(
            fake_quant_nvfp4_dispatch(weights[start:stop].contiguous(), local_scales),
            snapped[start:stop],
        )


def test_ep_fallback_marker_clears_merge_metadata(tmp_path):
    """An EP local fallback invalidates the adapter's merge-aware certificate."""
    import json
    from types import SimpleNamespace

    from axolotl.integrations.kernels.merge_aware_callback import (
        MergeAwareScheduleCallback,
    )

    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts._axolotl_merge_aware_unsupported = True
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    config_path = checkpoint / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 2, "nvfp4_merge_aware": {"v": 1}}))
    callback = MergeAwareScheduleCallback()
    callback._enabled = True
    state = SimpleNamespace(global_step=1, max_steps=1, is_world_process_zero=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        callback.on_save(SimpleNamespace(output_dir=tmp_path), state, None, model=model)

    assert "nvfp4_merge_aware" not in json.loads(config_path.read_text())

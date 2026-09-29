"""Actual FSDP2 SonicMoE packed-expert LoRA training oracle."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
    patch_nvfp4_fsdp,
)
from axolotl.integrations.kernels.libs.sonicmoe.experts_lora_fastpath import (
    patch_paramwrapper_sonicmoe_fastpath,
)
from axolotl.integrations.kernels.libs.sonicmoe.lora import get_lora_params_from_wrapper
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import dequantize_expert_weight
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
    _merge_aware_wfq,
    grouped_moe_reference_forward,
    set_merge_aware_enabled,
)
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import quantize_nvfp4_merge
from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching


def _pack_experts(shape):
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    packed, block_scales, per_tensor_scales = [], [], []
    for _ in range(shape[0]):
        dense = torch.randn(*shape[1:], device="cuda", dtype=torch.bfloat16) * 0.04
        per_tensor_scale = per_tensor_amax_to_scale(dense.abs().max())
        assert per_tensor_scale.isfinite()
        native = NVFP4Tensor.to_nvfp4(
            dense.contiguous(), block_size=16, per_tensor_scale=per_tensor_scale
        )
        packed.append(native.qdata)
        block_scales.append(native.scale)
        per_tensor_scales.append(per_tensor_scale)
    return NVFP4Tensor(
        torch.stack(packed),
        torch.stack(block_scales),
        16,
        torch.bfloat16,
        per_tensor_scale=torch.stack(per_tensor_scales).reshape(-1, 1, 1),
    )


class _PackedSonicExperts(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_experts = 2
        self.num_experts_global = 2
        self.has_gate = True
        self.has_bias = False
        self.is_concatenated = True
        self.is_transposed = False
        self.config = SimpleNamespace(
            _experts_implementation="sonicmoe", hidden_act="silu"
        )
        torch.manual_seed(19)
        self.gate_up_proj = nn.Parameter(
            _pack_experts((self.num_experts, 32, 16)),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            _pack_experts((self.num_experts, 16, 16)),
            requires_grad=False,
        )
        self._sonicmoe_lora = None

    def forward(self, hidden_states):
        lora = self._sonicmoe_lora
        assert lora is not None, "Sonic PEFT fastpath did not materialize adapters"
        weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=torch.long,
            device=hidden_states.device,
        )
        gate_lora, gate_scaling = _grouped_lora(lora, "gate_up_proj")
        down_lora, down_scaling = _grouped_lora(lora, "down_proj")
        return grouped_moe_reference_forward(
            hidden_states,
            ids,
            weights,
            self.gate_up_proj,
            None,
            self.down_proj,
            None,
            gate_lora,
            down_lora,
            self.num_experts,
            act="silu",
            backend="dequant",
            concat=True,
            scaling1=gate_scaling,
            scaling2=down_scaling,
        )


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedSonicExperts()

    def forward(self, hidden_states):
        return self.experts(hidden_states)


def _grouped_lora(lora, name):
    A, B, scaling = lora[name]
    return (A, B), scaling


def _unwrap_local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _clone_native_components(qdata, scale, per_tensor_scale):
    return (
        qdata.detach().clone(),
        scale.detach().clone(),
        None if per_tensor_scale is None else per_tensor_scale.detach().clone(),
    )


def _full_bytes(weight):
    assert type(weight).__name__ == "NVFP4Tensor"
    return _clone_native_components(weight.qdata, weight.scale, weight.per_tensor_scale)


def _assert_frozen_bytes(weight, expected):
    qdata, scale, per_tensor_scale = _full_bytes(weight)
    assert torch.equal(qdata, expected[0])
    assert torch.equal(scale, expected[1])
    if expected[2] is None:
        assert per_tensor_scale is None
    else:
        assert torch.equal(per_tensor_scale, expected[2])


def _merged(weight, lora, name):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    A, B, scaling = lora[name]
    A, B = _unwrap_local(A), _unwrap_local(B)
    weight = _unwrap_local(weight)
    E, dim1, dim2 = weight.shape
    rank = A.shape[0] // E
    delta = (
        torch.bmm(
            B.reshape(dim1, rank, E).permute(2, 0, 1).float(),
            A.reshape(E, rank, dim2).float(),
        )
        * scaling
    )
    base = weight.dequantize()
    effective = (base.float() + delta).to(base.dtype)
    packed, scale = quantize_nvfp4_merge(
        effective, weight.per_tensor_scale.reshape(-1), scale_mode="fresh"
    )
    packed, scale, per_tensor_scale = _clone_native_components(
        packed, scale, weight.per_tensor_scale
    )
    return NVFP4Tensor(
        packed,
        scale,
        16,
        weight.orig_dtype,
        per_tensor_scale=per_tensor_scale,
    )


def _same_or_max_abs(left, right):
    return torch.equal(left, right), (left.float() - right.float()).abs().max().item()


def _sonic_forward(hidden, w1, w2, lora1=None, lora2=None):
    grouped_lora1 = None if lora1 is None else lora1[:2]
    grouped_lora2 = None if lora2 is None else lora2[:2]
    return grouped_moe_reference_forward(
        hidden,
        torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=torch.long,
            device=hidden.device,
        ),
        torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]],
            dtype=hidden.dtype,
            device=hidden.device,
        ),
        w1,
        None,
        w2,
        None,
        grouped_lora1,
        grouped_lora2,
        2,
        act="silu",
        backend="dequant",
        concat=True,
        scaling1=1.0 if lora1 is None else lora1[2],
        scaling2=1.0 if lora2 is None else lora2[2],
    )


def _merge_aware_operand(weight, lora, name):
    A, B, scaling = lora[name]
    return _merge_aware_wfq(
        dequantize_expert_weight(weight),
        _unwrap_local(A),
        _unwrap_local(B),
        scaling,
        weight.per_tensor_scale,
    )


def _assert_full_lora_shape(weight, lora, name):
    A, B, _ = lora[name]
    E, dim1, dim2 = weight.shape
    assert A.ndim == B.ndim == 2
    assert A.shape[1] == dim2
    assert A.shape[0] % E == 0
    assert B.shape == (dim1, A.shape[0])


def _shard(model):
    wrappers = []
    current = model.base_model.model.experts
    while hasattr(current, "base_layer"):
        wrappers.append(current)
        current = current.base_layer
    raw = current
    fully_shard(raw)
    for wrapper in reversed(wrappers):
        for factors in (wrapper.lora_A, wrapper.lora_B):
            fully_shard(factors["default"])
        fully_shard(wrapper)
    fully_shard(model)
    return raw, wrappers


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        from peft import LoraConfig, get_peft_model

        patch_nvfp4_fsdp()
        patch_peft_target_parameters_matching()
        patch_paramwrapper_sonicmoe_fastpath()
        model = get_peft_model(
            _Model().cuda(),
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=[],
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                lora_dropout=0.0,
                bias="none",
            ),
        )
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.data.normal_(mean=0.0, std=0.05)
        raw, wrappers = _shard(model)
        factor_owners = [
            factors["default"]
            for wrapper in wrappers
            for factors in (wrapper.lora_A, wrapper.lora_B)
        ]
        assert isinstance(raw.gate_up_proj, DTensor)
        assert isinstance(raw.down_proj, DTensor)
        assert all(
            isinstance(factors["default"].weight, DTensor)
            for wrapper in wrappers
            for factors in (wrapper.lora_A, wrapper.lora_B)
        )
        raw.unshard()
        before = {
            "gate_up_proj": _full_bytes(raw.gate_up_proj),
            "down_proj": _full_bytes(raw.down_proj),
        }
        raw.reshard()
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        assert trainable and all(
            "lora_" in name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        optimizer = torch.optim.SGD(trainable, lr=0.05)
        hidden = (
            torch.tensor(
                [[0.1] * 16, [0.2] * 16, [-0.1] * 16, [0.3] * 16],
                device="cuda",
                dtype=torch.bfloat16,
            )
            + local_rank * 0.03125
        )
        local_before = [parameter.detach().clone() for parameter in trainable]
        set_merge_aware_enabled(True)
        loss = model(hidden).float().square().mean()
        loss.backward()
        for parameter in trainable:
            gradient = _unwrap_local(parameter.grad)
            assert gradient is not None and torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient).item() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        changed = torch.tensor(
            any(
                not torch.equal(before_, after)
                for before_, after in zip(local_before, trainable, strict=True)
            ),
            device="cuda",
            dtype=torch.int32,
        )
        dist.all_reduce(changed, op=dist.ReduceOp.SUM)
        assert changed.item() == dist.get_world_size()

        raw.unshard()
        for name, original in before.items():
            _assert_frozen_bytes(getattr(raw, name), original)
        for wrapper in wrappers:
            wrapper.unshard()
        for owner in factor_owners:
            owner.unshard()
        lora = {}
        for wrapper in wrappers:
            A, B, scaling = get_lora_params_from_wrapper(wrapper)
            assert A is not None and B is not None
            lora[wrapper.parameter_name] = (
                _unwrap_local(A).detach().clone(),
                _unwrap_local(B).detach().clone(),
                scaling,
            )
        assert set(lora) == {"gate_up_proj", "down_proj"}
        _assert_full_lora_shape(raw.gate_up_proj, lora, "gate_up_proj")
        _assert_full_lora_shape(raw.down_proj, lora, "down_proj")
        with torch.no_grad():
            merged_w1 = _merged(raw.gate_up_proj, lora, "gate_up_proj")
            merged_w2 = _merged(raw.down_proj, lora, "down_proj")
            runtime_base_w1 = dequantize_expert_weight(raw.gate_up_proj)
            runtime_base_w2 = dequantize_expert_weight(raw.down_proj)
            canonical_base_w1 = raw.gate_up_proj.dequantize()
            canonical_base_w2 = raw.down_proj.dequantize()
            runtime_snapped_w1 = _merge_aware_operand(
                raw.gate_up_proj, lora, "gate_up_proj"
            )
            runtime_snapped_w2 = _merge_aware_operand(raw.down_proj, lora, "down_proj")
            runtime_snapshot = _sonic_forward(
                hidden,
                raw.gate_up_proj,
                raw.down_proj,
                lora["gate_up_proj"],
                lora["down_proj"],
            )
            merged = _sonic_forward(hidden, merged_w1, merged_w2)
            diagnostics = {
                "base_gate": _same_or_max_abs(runtime_base_w1, canonical_base_w1),
                "base_down": _same_or_max_abs(runtime_base_w2, canonical_base_w2),
                "snap_gate": _same_or_max_abs(
                    runtime_snapped_w1, merged_w1.dequantize()
                ),
                "snap_down": _same_or_max_abs(
                    runtime_snapped_w2, merged_w2.dequantize()
                ),
                "snapshot_merged": _same_or_max_abs(runtime_snapshot, merged),
            }
        for owner in reversed(factor_owners):
            owner.reshard()
        for wrapper in reversed(wrappers):
            wrapper.reshard()
        raw.reshard()

        with torch.no_grad():
            trained = model(hidden)
        equal, difference = _same_or_max_abs(trained, merged)
        assert equal, (
            "standalone merged packed experts diverged "
            f"(max_abs_diff={difference}; diagnostics={diagnostics})"
        )
        print(f"SONICMOE_NVFP4_FSDP2_PASS rank={dist.get_rank()}", flush=True)
    finally:
        set_merge_aware_enabled(False)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

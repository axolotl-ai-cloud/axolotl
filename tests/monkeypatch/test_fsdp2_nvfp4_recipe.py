# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""NVFP4 recipe preservation during FSDP2 shard reconstruction."""

import copy
from types import SimpleNamespace

import pytest
import torch

import axolotl.monkeypatch.accelerate.fsdp2 as fsdp2
from axolotl.monkeypatch.accelerate.fsdp2 import (
    _rebuild_nvfp4_like,
    _restore_non_persistent_buffers,
    _state_dict_entry,
)

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor


def test_rebuild_nvfp4_like_preserves_complete_native_recipe():
    qdata = torch.randint(0, 255, (2, 16, 8), dtype=torch.uint8)
    scale = torch.ones((2, 16, 1), dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.tensor([[[0.3]], [[0.7]]])
    act_per_tensor_scale = torch.tensor([[[0.125]], [[0.25]]])
    recipe = NVFP4Tensor(
        qdata,
        scale,
        16,
        torch.float32,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=False,
        use_triton_kernel=False,
        act_quant_kwargs={"block_size": 16},
    )

    rebuilt = _rebuild_nvfp4_like(
        recipe,
        recipe.qdata[:1].clone(),
        recipe.scale[:1].clone(),
        recipe.per_tensor_scale[1:].clone(),
        recipe.act_per_tensor_scale[1:].clone(),
    )

    assert rebuilt.block_size == recipe.block_size
    assert rebuilt.orig_dtype == recipe.orig_dtype
    assert torch.equal(rebuilt.per_tensor_scale, recipe.per_tensor_scale[1:])
    assert torch.equal(rebuilt.act_per_tensor_scale, recipe.act_per_tensor_scale[1:])
    assert rebuilt.is_swizzled_scales is recipe.is_swizzled_scales
    assert rebuilt.use_triton_kernel is recipe.use_triton_kernel
    assert rebuilt.act_quant_kwargs == recipe.act_quant_kwargs


def test_state_dict_broadcast_entries_preserve_parameter_and_buffer_roles():
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([3.0]))
            self.trainable_alias = self.trainable
            self.frozen = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
            self.register_buffer("count", torch.tensor([7], dtype=torch.int64))
            self.register_buffer("count_alias", self.count)
            self.register_buffer("enabled", torch.tensor([True], dtype=torch.bool))

    source = Source()
    parameter_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in source.named_parameters(remove_duplicate=False)
    }
    buffer_names = {name for name, _ in source.named_buffers(remove_duplicate=False)}
    entries = {
        name: _state_dict_entry(
            value.clone(), name, parameter_requires_grad, buffer_names
        )
        for name, value in source.state_dict().items()
    }
    target = Source().to("meta")
    target.load_state_dict(entries, assign=True)

    assert isinstance(entries["trainable"], torch.nn.Parameter)
    assert entries["trainable"].requires_grad
    assert isinstance(entries["trainable_alias"], torch.nn.Parameter)
    assert entries["trainable_alias"].requires_grad
    assert isinstance(entries["frozen"], torch.nn.Parameter)
    assert not entries["frozen"].requires_grad
    assert not isinstance(entries["count"], torch.nn.Parameter)
    assert not isinstance(entries["count_alias"], torch.nn.Parameter)
    assert entries["count"].dtype is torch.int64
    assert not isinstance(entries["enabled"], torch.nn.Parameter)
    assert entries["enabled"].dtype is torch.bool
    assert target.count.dtype is torch.int64
    assert target.enabled.dtype is torch.bool


def test_full_state_broadcast_preserves_buffer_and_parameter_roles(monkeypatch):
    class State(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([3.0]))
            self.trainable_alias = self.trainable
            self.frozen = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
            self.register_buffer("count", torch.tensor([7], dtype=torch.int64))
            self.register_buffer("count_alias", self.count)
            self.register_buffer("enabled", torch.tensor([True], dtype=torch.bool))

    source = State()
    target = State().to("meta")
    real_device = torch.device
    monkeypatch.setattr(
        fsdp2.torch,
        "device",
        lambda name: real_device("cpu") if name == "cuda" else real_device(name),
    )
    monkeypatch.setattr(fsdp2.dist, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fsdp2, "log_gpu_memory_usage", lambda *_args, **_kwargs: None)

    fsdp2.fsdp2_load_full_state_dict(
        SimpleNamespace(is_main_process=True), target, source.state_dict()
    )

    assert target.trainable.requires_grad
    assert target.trainable_alias.requires_grad
    assert not target.frozen.requires_grad
    assert target.count.dtype is torch.int64
    assert target.count_alias.dtype is torch.int64
    assert target.enabled.dtype is torch.bool
    torch.testing.assert_close(target.trainable, source.trainable)
    torch.testing.assert_close(target.frozen, source.frozen)
    torch.testing.assert_close(target.count, source.count)
    torch.testing.assert_close(target.enabled, source.enabled)


def test_nonpersistent_buffer_restoration_broadcasts_source_values_to_meta_rank(
    monkeypatch,
):
    class State(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "rope", torch.tensor([0.25, 0.5], dtype=torch.float32), persistent=False
            )
            self.register_buffer("rope_alias", self.rope, persistent=False)
            self.register_buffer(
                "mask", torch.tensor([True, False], dtype=torch.bool), persistent=False
            )

    source = State()
    target = State().to("meta")
    target.rope_alias = torch.empty_like(target.rope, device="meta")
    assert source.rope is source.rope_alias
    assert target.rope is not target.rope_alias
    source_buffers = copy.deepcopy(dict(source.named_buffers(remove_duplicate=False)))
    target_buffers = copy.deepcopy(dict(target.named_buffers(remove_duplicate=False)))
    assert all(buffer.is_meta for buffer in target_buffers.values())

    source_payloads = []
    calls = []
    is_source = True
    replay_index = 0

    def broadcast(tensor, *_args, **_kwargs):
        nonlocal replay_index
        if tensor.is_meta:
            raise AssertionError("broadcast destination must be materialized")
        calls.append("source" if is_source else "target")
        if is_source:
            source_payloads.append(tensor.clone())
        else:
            tensor.copy_(source_payloads[replay_index])
            replay_index += 1

    monkeypatch.setattr(fsdp2.dist, "broadcast", broadcast)
    device = torch.device("cpu")
    _restore_non_persistent_buffers(
        source, source_buffers, SimpleNamespace(is_main_process=True, device=device)
    )
    is_source = False
    _restore_non_persistent_buffers(
        target, target_buffers, SimpleNamespace(is_main_process=False, device=device)
    )

    assert calls == ["source"] * 3 + ["target"] * 3
    torch.testing.assert_close(target.rope, source.rope)
    torch.testing.assert_close(target.rope_alias, source.rope_alias)
    torch.testing.assert_close(target.mask, source.mask)
    assert "rope" in target._non_persistent_buffers_set
    assert "rope_alias" in target._non_persistent_buffers_set
    assert "mask" in target._non_persistent_buffers_set

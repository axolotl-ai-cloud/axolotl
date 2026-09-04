"""Regression tests for ``materialize_trainable_meta_params``.

With ``fsdp_config.cpu_ram_efficient_loading`` under LoRA, non-rank-0 base params stay on
meta and PEFT lands the freshly created adapter params there too. accelerate's
``Accelerator._prepare_fsdp2`` then remaps the optimizer's params onto the new sharded
DTensors by ``data_ptr()``, and every meta tensor reports 0 — so on those ranks every
optimizer slot collapses onto one param and their LoRA shards never get stepped, leaving
exactly ``1/world_size`` of each ``lora_B``'s rows nonzero in the saved adapter.
"""

import os
import types
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

import axolotl.loaders.utils as utils_mod
from axolotl.loaders.utils import materialize_trainable_meta_params


class _MixedDeviceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable_meta = torch.nn.Parameter(torch.empty(4, 3, device="meta"))
        self.frozen_meta = torch.nn.Parameter(
            torch.empty(4, 3, device="meta"), requires_grad=False
        )
        self.trainable_cpu = torch.nn.Parameter(torch.randn(4, 3))


class _NestedMetaAdapter(torch.nn.Module):
    """Nested like a real PEFT adapter: the params hang off submodules, not the root."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Linear(4, 8, bias=False, device="meta")
        self.lora_B = torch.nn.Linear(8, 4, bias=False, device="meta")


class _Packed(torch.nn.Parameter):
    """Stand-in for an opaque tensor subclass (bnb Params4bit, torchao NVFP4Tensor, ...)."""


class _DTensorModel(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        local = torch.empty(4, 3, device="meta")
        dtensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
        self.sharded = torch.nn.Parameter(dtensor)


def _accelerate_data_ptr_keys(model: torch.nn.Module) -> dict[str, int]:
    return Accelerator._get_named_parameters(  # pylint: disable=protected-access
        types.SimpleNamespace(fp8_backend=None), model, drop_refs=True
    )


def test_materializes_trainable_meta_params_only():
    model = _MixedDeviceModel()
    cpu_param = model.trainable_cpu
    cpu_data_ptr = model.trainable_cpu.data_ptr()

    materialized = materialize_trainable_meta_params(model)

    assert materialized == ["trainable_meta"]
    assert model.trainable_meta.device.type == "cpu"
    assert not model.trainable_meta.is_meta
    assert torch.equal(model.trainable_meta.detach(), torch.zeros(4, 3))
    assert model.trainable_meta.requires_grad

    # frozen base params staying on meta is the CPU-RAM saving this whole path exists for
    assert model.frozen_meta.is_meta
    assert not model.frozen_meta.requires_grad

    assert model.trainable_cpu is cpu_param
    assert model.trainable_cpu.data_ptr() == cpu_data_ptr


def test_second_call_is_a_no_op():
    model = _MixedDeviceModel()

    assert materialize_trainable_meta_params(model) == ["trainable_meta"]
    assert materialize_trainable_meta_params(model) == []


def test_data_ptr_keys_collide_on_meta_and_separate_after():
    """Pins the accelerate mechanism the helper works around."""
    model = _NestedMetaAdapter()

    before = _accelerate_data_ptr_keys(model)
    assert before["lora_A.weight"] == before["lora_B.weight"] == 0, (
        "accelerate's Accelerator._prepare_fsdp2 keys its FSDP2 optimizer param remap on "
        f"data_ptr(), and every meta tensor reports 0 — got {before}. If this assertion "
        "fails, accelerate fixed the collision upstream and "
        "materialize_trainable_meta_params may be droppable."
    )

    assert sorted(materialize_trainable_meta_params(model)) == [
        "lora_A.weight",
        "lora_B.weight",
    ]

    after = _accelerate_data_ptr_keys(model)
    assert after["lora_A.weight"] != after["lora_B.weight"], (
        "each trainable param needs its own data_ptr() once materialized on CPU — "
        f"got {after}. accelerate's data_ptr-keyed FSDP2 optimizer remap otherwise maps "
        "every slot onto the same param and those ranks' LoRA shards are never stepped."
    )
    assert 0 not in after.values()


class TestMaterializeDTensorParams:
    """Single-process gloo mesh, like TestShardingSingleRank in test_expert_parallel.py."""

    def setup_method(self):
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29561")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_materializes_dtensor_local_shard_in_place(self):
        mesh = init_device_mesh("cpu", (1,))
        model = _DTensorModel(mesh)
        placements_before = model.sharded.placements
        mesh_before = model.sharded.device_mesh

        materialized = materialize_trainable_meta_params(model)

        assert materialized == ["sharded"]
        assert isinstance(model.sharded, DTensor)
        assert not model.sharded.is_meta
        assert model.sharded.placements == placements_before
        assert model.sharded.device_mesh is mesh_before
        assert model.sharded._local_tensor.device.type == "cpu"

        keys = _accelerate_data_ptr_keys(model)
        assert keys["sharded"] != 0


def test_unsupported_tensor_subclass_is_skipped_and_warned():
    model = torch.nn.Module()
    model.packed = _Packed(torch.empty(2, 2, device="meta"))
    assert model.packed.requires_grad
    assert model.packed.is_meta

    with patch.object(utils_mod.LOG, "warning") as mock_warning:
        materialized = materialize_trainable_meta_params(model)

    assert materialized == []
    assert model.packed.is_meta
    assert type(model.packed) is _Packed

    mock_warning.assert_called_once()
    message = mock_warning.call_args[0][0]
    assert "packed" in message and "_Packed" in message
    # the helper only runs on non-rank-0, so the main-process default would drop this
    assert mock_warning.call_args[1]["main_process_only"] is False


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    ],
)
def test_dtype_is_preserved(dtype):
    model = torch.nn.Module()
    model.p = torch.nn.Parameter(
        torch.empty(4, 3, device="meta", dtype=dtype), requires_grad=True
    )

    materialized = materialize_trainable_meta_params(model)

    assert materialized == ["p"]
    assert model.p.device.type == "cpu"
    assert model.p.dtype == dtype
    assert model.p.data_ptr() != 0


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"), reason="torch build lacks float4_e2m1fn_x2"
)
def test_fp4_dtype_falls_back_to_empty_like():
    model = torch.nn.Module()
    model.p = torch.nn.Parameter(
        torch.empty(4, 3, device="meta", dtype=torch.float4_e2m1fn_x2),
        requires_grad=True,
    )

    materialized = materialize_trainable_meta_params(model)

    assert materialized == ["p"]
    assert model.p.device.type == "cpu"
    assert model.p.dtype == torch.float4_e2m1fn_x2
    assert model.p.data_ptr() != 0

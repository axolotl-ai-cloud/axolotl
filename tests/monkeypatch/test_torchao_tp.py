"""CPU checks for native NVFP4 tensor-parallel component slicing."""

import pytest
import torch

import axolotl.monkeypatch.torchao_tp as torchao_tp
from axolotl.monkeypatch.torchao_tp import (
    materialize_native_nvfp4_tp,
    native_nvfp4_tp_shard,
    preflight_native_nvfp4_tp,
)


def _nvfp4(rows=256, columns=128, *, swizzled=False):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    generator = torch.Generator().manual_seed(rows * 10_000 + columns)
    value = torch.randn(rows, columns, generator=generator, dtype=torch.float32)
    return NVFP4Tensor.to_nvfp4(
        value,
        per_tensor_scale=torch.tensor(1.0),
        is_swizzled_scales=swizzled,
    )


@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("dim", [0, 1])
def test_native_nvfp4_tp_slice_preserves_component_parity(swizzled, dim):
    tensor = _nvfp4(swizzled=swizzled)
    shards = [native_nvfp4_tp_shard(tensor, dim, rank, 2) for rank in range(2)]
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": dim}, rank, 2)[
            "weight"
        ]
        for rank in range(2)
    ]

    rebuilt = torch.cat([item.dequantize() for item in local], dim=dim)
    torch.testing.assert_close(rebuilt, tensor.dequantize(), rtol=0, atol=0)
    assert all(type(item).__name__ == "NVFP4Tensor" for item in local)
    assert all(item.per_tensor_scale is not tensor.per_tensor_scale for item in local)
    assert all(
        item.qdata.untyped_storage().nbytes() == item.qdata.numel() for item in local
    )
    assert all(
        item.scale.untyped_storage().nbytes() == item.scale.numel() for item in local
    )
    assert (
        [(item.start, item.end) for item in shards] == [(0, 128), (128, 256)]
        if dim == 0
        else [(0, 64), (64, 128)]
    )


def test_native_nvfp4_tp_rejects_unaligned_input_blocks_without_mutation():
    tensor = _nvfp4(columns=80)
    qdata = tensor.qdata.clone()
    scale = tensor.scale.clone()
    with pytest.raises(ValueError, match="align to 16"):
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 1}, 0, 2)
    torch.testing.assert_close(tensor.qdata, qdata)
    torch.testing.assert_close(tensor.scale, scale)


def test_native_nvfp4_tp_logicalizes_swizzled_scales_for_output_shards():
    tensor = _nvfp4(rows=192, swizzled=True)
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 0}, rank, 2)[
            "weight"
        ]
        for rank in range(2)
    ]
    assert all(item.is_swizzled_scales for item in local)
    torch.testing.assert_close(
        torch.cat([item.dequantize() for item in local]),
        tensor.dequantize(),
        rtol=0,
        atol=0,
    )


def test_native_nvfp4_tp_preflight_rejects_alias_placement_disagreement_without_mutation():
    tensor = _nvfp4()
    with pytest.raises(ValueError, match="aliases disagree"):
        preflight_native_nvfp4_tp(
            [("left.weight", tensor), ("right.weight", tensor)],
            {"left.weight": 0, "right.weight": 1},
            0,
            2,
        )
    assert tensor.shape == (256, 128)


def test_native_nvfp4_tp_rejects_invalid_dimension_without_mutation():
    tensor = _nvfp4()
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native_nvfp4_tp_shard(tensor, 2, 0, 2)
    assert tensor.shape == (256, 128)


def test_native_nvfp4_tp_materializes_tied_alias_once_with_owned_storage():
    tensor = _nvfp4()
    shards = materialize_native_nvfp4_tp(
        [("left.weight", tensor), ("right.weight", tensor)],
        {"left.weight": 0, "right.weight": 0},
        0,
        2,
    )
    assert shards["left.weight"] is shards["right.weight"]
    assert (
        shards["left.weight"].qdata.untyped_storage().data_ptr()
        != tensor.qdata.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("dim, expected", [(-2, 0), (-1, 1)])
def test_native_nvfp4_tp_accepts_standard_negative_dimensions(dim, expected):
    tensor = _nvfp4()
    assert native_nvfp4_tp_shard(tensor, dim, 0, 2).dim == expected


def test_native_nvfp4_tp_allows_block_aligned_swizzled_input_shards():
    tensor = _nvfp4(columns=96, swizzled=True)
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 1}, rank, 3)[
            "weight"
        ]
        for rank in range(3)
    ]
    assert [item.shape[1] for item in local] == [32, 32, 32]
    torch.testing.assert_close(
        torch.cat([item.dequantize() for item in local], dim=1),
        tensor.dequantize(),
        rtol=0,
        atol=0,
    )


def test_native_nvfp4_tp_preflight_does_not_slice_before_later_failure(monkeypatch):
    valid = _nvfp4()
    invalid = _nvfp4(columns=80)
    calls = []
    original = torchao_tp.slice_native_nvfp4_tp

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(torchao_tp, "slice_native_nvfp4_tp", record)
    with pytest.raises(ValueError, match="align to 16"):
        materialize_native_nvfp4_tp(
            [("valid.weight", valid), ("invalid.weight", invalid)],
            {"valid.weight": 0, "invalid.weight": 1},
            0,
            2,
        )
    assert calls == []

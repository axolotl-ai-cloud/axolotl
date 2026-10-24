"""Storage order and tensor ownership during sharded expert merges."""

import weakref
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch
from transformers.core_model_loading import (
    Concatenate,
    MergeModulelist,
    WeightConverter,
)

from axolotl.cli.utils import lora_merge


def _converter():
    return WeightConverter(
        source_patterns=[
            "mlp.experts.*.gate_proj.weight",
            "mlp.experts.*.up_proj.weight",
        ],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )


def test_read_shard_uses_storage_order(tmp_path):
    path = tmp_path / "model.safetensors"
    tensors = {
        "a": torch.ones(2, dtype=torch.float32),
        "z": torch.ones(2, dtype=torch.float64),
    }
    safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})
    with safetensors.safe_open(path, framework="pt") as reader:
        storage_order = reader.offset_keys()
        assert storage_order != reader.keys()
    loaded, metadata = lora_merge._read_shard(path)
    assert list(loaded) == storage_order
    assert metadata == {"format": "pt"}
    for key in tensors:
        torch.testing.assert_close(loaded[key], tensors[key], rtol=0, atol=0)


@pytest.mark.parametrize("num_shards", [1, 2])
def test_prefetch_releases_yielded_shard(monkeypatch, num_shards):
    monkeypatch.setattr(
        lora_merge,
        "_read_shard",
        lambda path: ({"weight": torch.ones(2)}, {}),
    )
    reader = lora_merge._prefetched_shards(
        [Path(f"shard-{i}") for i in range(num_shards)]
    )
    try:
        _, tensors, _ = next(reader)
        ref = weakref.ref(tensors["weight"])
        del tensors
        assert ref() is None
    finally:
        reader.close()


def test_fusion_releases_sources_and_stacks_before_merge(monkeypatch):
    experts = [10, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 11]
    shard = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": (
            torch.full((4, 8), layer * 100 + expert + offset, dtype=torch.float32)
        )
        for layer in (10, 2)
        for expert in experts
        for projection, offset in (("gate_proj", 0), ("up_proj", 20))
    }
    refs = {key: weakref.ref(tensor) for key, tensor in shard.items()}
    expected = {
        f"model.layers.{layer}.mlp.experts.gate_up_proj": torch.cat(
            [
                torch.stack(
                    [
                        shard[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"]
                        for e in range(12)
                    ]
                )
                for proj in ("gate_proj", "up_proj")
            ],
            dim=1,
        )
        for layer in (10, 2)
    }
    stack_refs = []
    original_stack = torch.stack
    original_merge = lora_merge._merge_tensor_with_lora
    merged_keys = []

    def stack(*args, **kwargs):
        tensor = original_stack(*args, **kwargs)
        stack_refs.append(weakref.ref(tensor))
        return tensor

    def merge(tensor, key, *args, **kwargs):
        prefix = key.removesuffix("gate_up_proj")
        assert all(
            ref() is None for name, ref in refs.items() if name.startswith(prefix)
        )
        assert all(ref() is None for ref in stack_refs)
        merged_keys.append(key)
        return original_merge(tensor, key, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", stack)
    monkeypatch.setattr(lora_merge, "_merge_tensor_with_lora", merge)
    result, count, processed = lora_merge._fuse_and_unfuse_with_merge(
        shard,
        [_converter()],
        {},
        1.0,
        {"r": 2, "lora_alpha": 2},
        "cpu",
        expected_num_experts=12,
        inplace=True,
    )
    assert result is shard
    assert merged_keys == list(expected)
    assert count == 0
    assert processed == set(refs) | set(expected)
    assert set(result) == set(expected)
    for key in expected:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)


def test_carry_owns_sources_until_completing_shard(monkeypatch):
    prefix = "model.layers.0.mlp.experts."
    first = {
        f"{prefix}0.{proj}.weight": torch.ones(4, 8)
        for proj in ("gate_proj", "up_proj")
    }
    refs = [weakref.ref(tensor) for tensor in first.values()]
    carry = lora_merge._FusedExpertCarry()
    args = ([_converter()], {}, 1.0, {"r": 2, "lora_alpha": 2}, "cpu")
    result, count, processed = lora_merge._fuse_and_unfuse_with_merge(
        first, *args, expected_num_experts=2, carry=carry, inplace=True
    )
    assert result is first
    assert not first and count == 0 and not processed
    assert all(ref() is not None for ref in refs)

    second = {
        f"{prefix}1.{proj}.weight": torch.full((4, 8), 2.0)
        for proj in ("gate_proj", "up_proj")
    }
    refs.extend(weakref.ref(tensor) for tensor in second.values())
    original_merge = lora_merge._merge_tensor_with_lora

    def merge(*args, **kwargs):
        assert all(ref() is None for ref in refs)
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(lora_merge, "_merge_tensor_with_lora", merge)
    result, _, _ = lora_merge._fuse_and_unfuse_with_merge(
        second, *args, expected_num_experts=2, carry=carry, inplace=True
    )
    carry.assert_drained()
    assert set(result) == {prefix + "gate_up_proj"}
    expected = torch.stack([torch.ones(8, 8), torch.full((8, 8), 2.0)])
    torch.testing.assert_close(
        result[prefix + "gate_up_proj"], expected, rtol=0, atol=0
    )

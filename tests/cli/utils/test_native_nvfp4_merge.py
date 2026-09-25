"""Native TorchAO NVFP4 safetensors writer tests."""

import pytest
import torch

pytest.importorskip("torchao")

from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: E402
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
)
from torchao.prototype.safetensors.safetensors_support import (  # noqa: E402
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)

from axolotl.cli.utils.native_nvfp4_merge import (  # noqa: E402
    has_native_nvfp4_weights,
    merge_native_nvfp4_shard,
)
from axolotl.monkeypatch.torchao_nvfp4_merge import (  # noqa: E402
    capture_native_nvfp4_recipe,
)


@pytest.mark.parametrize(
    "dynamic,supplied_scale", [(False, False), (True, False), (True, True)]
)
def test_native_writer_preserves_recipe_and_untouched_metadata(dynamic, supplied_scale):
    torch.manual_seed(21)
    kwargs = {}
    if dynamic:
        kwargs["act_quant_kwargs"] = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=True
        )
        if supplied_scale:
            kwargs["act_per_tensor_scale"] = torch.tensor(1.75)
    weight = NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16),
        per_tensor_scale=torch.tensor(1.125),
        **kwargs,
    )
    untouched = NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16), per_tensor_scale=torch.tensor(0.75)
    )
    tensors, metadata = flatten_tensor_state_dict(
        {"model.proj.weight": weight, "model.other.weight": untouched}
    )
    original = {key: value.clone() for key, value in tensors.items()}
    original_metadata = dict(metadata)
    delta = torch.randn(32, 32, dtype=torch.float32) * 0.01

    def merge(dense, name):
        if name == "model.proj.weight":
            return (dense.float() + delta).to(torch.bfloat16), True
        return dense, False

    output, output_metadata, count = merge_native_nvfp4_shard(tensors, metadata, merge)
    assert count == 1
    assert has_native_nvfp4_weights(metadata)
    assert metadata == original_metadata
    for key, value in original.items():
        assert torch.equal(tensors[key], value)
    rebuilt, leftover = unflatten_tensor_state_dict(output, output_metadata)
    assert not leftover
    expected = capture_native_nvfp4_recipe(weight).quantize(
        (weight.dequantize().float() + delta).to(torch.bfloat16)
    )
    actual = rebuilt["model.proj.weight"]
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.act_quant_kwargs == expected.act_quant_kwargs
    if expected.act_per_tensor_scale is None:
        assert actual.act_per_tensor_scale is None
    else:
        assert torch.equal(actual.act_per_tensor_scale, expected.act_per_tensor_scale)
    assert torch.equal(rebuilt["model.other.weight"].qdata, untouched.qdata)
    assert (
        output_metadata["model.other.weight"] == original_metadata["model.other.weight"]
    )


def test_native_writer_root_weight_and_incomplete_components():
    weight = NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16), per_tensor_scale=torch.tensor(1.0)
    )
    nested_tensors, nested_metadata = flatten_tensor_state_dict(
        {"model.weight": weight}
    )
    tensors = {
        key.removeprefix("model."): value for key, value in nested_tensors.items()
    }
    metadata = {
        "weight": nested_metadata["model.weight"],
        "tensor_names": '["weight"]',
    }

    output, output_metadata, count = merge_native_nvfp4_shard(
        tensors, metadata, lambda dense, _: (dense + 0.01, True)
    )
    assert count == 1
    assert "_weight_qdata" in output
    assert "weight" in output_metadata

    incomplete = dict(tensors)
    incomplete.pop("_weight_qdata")
    output, output_metadata, count = merge_native_nvfp4_shard(
        incomplete, metadata, lambda dense, _: (dense + 0.01, True)
    )
    assert count == 0
    assert output == incomplete
    assert output_metadata == metadata


def test_native_writer_dequantizes_merged_and_untouched_weights():
    weight = NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16), per_tensor_scale=torch.tensor(1.0)
    )
    tensors, metadata = flatten_tensor_state_dict({"model.weight": weight})
    output, output_metadata, count = merge_native_nvfp4_shard(
        tensors, metadata, lambda dense, _: (dense + 0.01, True), dequant=True
    )
    assert count == 1
    assert "model.weight" in output
    assert "model._weight_qdata" not in output
    assert output_metadata["model.weight"] == '{"_type": "Tensor"}'

    output, output_metadata, count = merge_native_nvfp4_shard(
        tensors, metadata, lambda dense, _: (dense, False), dequant=True
    )
    assert count == 0
    assert torch.equal(output["model.weight"], weight.dequantize())
    assert output_metadata["model.weight"] == '{"_type": "Tensor"}'


@pytest.mark.parametrize("dequant", [False, True])
def test_merge_lora_sharded_efficient_matches_native_recipe(tmp_path, dequant):
    import json

    import safetensors
    import safetensors.torch

    from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

    torch.manual_seed(91)
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    base_dir.mkdir()
    adapter_dir.mkdir()
    key = "model.layers.0.self_attn.q_proj.weight"
    weight = NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16), per_tensor_scale=torch.tensor(1.125)
    )
    flattened, metadata = flatten_tensor_state_dict(
        {
            key: weight,
            "model.embed_tokens.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        }
    )
    safetensors.torch.save_file(
        flattened, base_dir / "model.safetensors", metadata=metadata
    )
    (base_dir / "config.json").write_text("{}")
    a = torch.randn(8, 32, dtype=torch.float32) * 0.02
    b = torch.randn(32, 8, dtype=torch.float32) * 0.02
    safetensors.torch.save_file(
        {
            f"base_model.model.{key[:-7]}.lora_A.weight": a,
            f"base_model.model.{key[:-7]}.lora_B.weight": b,
        },
        adapter_dir / "adapter_model.safetensors",
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16, "peft_type": "LORA"})
    )

    merge_lora_sharded_efficient(
        base_dir, adapter_dir, output_dir, device="cpu", dequant=dequant
    )
    with safetensors.safe_open(output_dir / "model.safetensors", framework="pt") as f:
        result_metadata = f.metadata()
        result_tensors = {name: f.get_tensor(name) for name in f.keys()}
    rebuilt, leftover = unflatten_tensor_state_dict(result_tensors, result_metadata)
    assert not leftover
    effective = (weight.dequantize().float() + 2 * (b @ a)).to(torch.bfloat16)
    if dequant:
        assert torch.equal(rebuilt[key], effective)
        assert not has_native_nvfp4_weights(result_metadata)
        return
    expected = capture_native_nvfp4_recipe(weight).quantize(effective)
    actual = rebuilt[key]
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.act_quant_kwargs == expected.act_quant_kwargs

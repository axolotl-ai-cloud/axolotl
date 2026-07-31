"""CPU-only tests for the bake, transformers-bitnet packer and parity gates."""

import copy
import json
import re
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import bake, hf_bitnet, parity
from axolotl.integrations.ternary.export.hf_bitnet import _modules_to_not_convert
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import SwapEntry, SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

bitnet = pytest.importorskip("transformers.integrations.bitnet")


def _tiny_llama(dtype: torch.dtype = torch.bfloat16) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).to(dtype)


def _write_master(
    directory, bake_weights: bool = True, max_shard_size: str = "5GB"
) -> SwapManifest:
    model = _tiny_llama()
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": {}})
    )
    if bake_weights:
        for name, module in iter_ternary_modules(model):
            module._post_training(model, name)
    model.save_pretrained(directory, max_shard_size=max_shard_size)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


def _random_codes(out_features: int, in_features: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -1, 2, (out_features, in_features), generator=generator, dtype=torch.int8
    )


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("master")
    return directory, _write_master(directory)


@pytest.fixture(scope="module")
def packed(master, tmp_path_factory):
    master_dir, manifest = master
    output = tmp_path_factory.mktemp("packed") / "hf_bitnet"
    return hf_bitnet.export_hf_bitnet(master_dir, output, manifest), manifest


# --------------------------------------------------------------------------- packing


@pytest.mark.parametrize(
    "shape", [(4, 1), (8, 3), (64, 64), (128, 37), (12, 5), (4096, 8)]
)
def test_pack_unpack_roundtrip(shape):
    codes = _random_codes(*shape, seed=shape[0])

    packed = hf_bitnet.pack_hf_bitnet(codes)

    assert packed.shape == (shape[0] // 4, shape[1])
    assert packed.dtype == torch.uint8
    assert torch.equal(hf_bitnet.unpack_hf_bitnet(packed, shape[0]), codes)


def test_pack_layout_is_row_strided():
    codes = torch.tensor([[1], [-1], [0], [1]], dtype=torch.int8)

    packed = hf_bitnet.pack_hf_bitnet(codes)

    # lane i holds row i * (out // 4): (1+1) | (0)<<2 | (1)<<4 | (2)<<6
    assert packed.shape == (1, 1)
    assert int(packed[0, 0]) == 2 | (0 << 2) | (1 << 4) | (2 << 6)


def test_pack_matches_transformers_reference():
    codes = _random_codes(16, 5, seed=3)

    packed = hf_bitnet.pack_hf_bitnet(codes)
    reference = bitnet.pack_weights(codes.clone())

    assert torch.equal(packed, reference)


def test_unpack_matches_transformers_reference():
    codes = _random_codes(32, 7, seed=4)
    packed = hf_bitnet.pack_hf_bitnet(codes)

    reference = bitnet.unpack_weights(packed, dtype=torch.float32)

    assert torch.equal(reference, codes.to(torch.float32))
    assert torch.equal(hf_bitnet.unpack_hf_bitnet(packed, 32), codes)


def test_pack_rejects_rows_not_divisible_by_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        hf_bitnet.pack_hf_bitnet(_random_codes(6, 3))


def test_pack_rejects_non_2d():
    with pytest.raises(ValueError, match="2D weight"):
        hf_bitnet.pack_hf_bitnet(torch.zeros(4, 4, 4, dtype=torch.int8))


def test_unpack_rejects_wrong_out_features():
    packed = hf_bitnet.pack_hf_bitnet(_random_codes(8, 3))

    with pytest.raises(ValueError, match="not 12"):
        hf_bitnet.unpack_hf_bitnet(packed, 12)


# ------------------------------------------------------------------------------ bake


def test_bake_weight_yields_exactly_three_values():
    weight = torch.randn(32, 64)

    baked = bake.bake_weight(weight)

    magnitudes = baked.abs().unique()
    assert magnitudes.numel() == 2 and float(magnitudes[0]) == 0.0
    scale = float(magnitudes[1])
    assert scale == float(quant.f16_round_scale(quant.absmean_scale(weight)))


def test_bake_weight_is_idempotent():
    weight = torch.randn(32, 64)

    once = bake.bake_weight(weight)
    twice = bake.bake_weight(once)

    assert twice is once
    assert torch.equal(twice, once)


def test_bake_weight_idempotent_in_bf16():
    weight = torch.randn(32, 64, dtype=torch.bfloat16)

    once = bake.bake_weight(weight)

    assert torch.equal(bake.bake_weight(once), once)


def test_derive_codes_and_scale_recovers_the_quantizer():
    weight = torch.randn(32, 64)
    scale = quant.absmean_scale(weight)
    baked = bake.bake_weight(weight)

    codes, recovered = bake.derive_codes_and_scale(baked)

    assert torch.equal(codes, quant.ternary_codes(weight, scale))
    assert torch.equal(recovered, quant.f16_round_scale(scale))
    assert torch.equal(quant.dequantize_codes(codes, recovered, baked.dtype), baked)


def test_derive_codes_and_scale_recovers_a_bf16_master():
    baked = bake.bake_weight(torch.randn(32, 64, dtype=torch.bfloat16))

    codes, scale = bake.derive_codes_and_scale(baked)

    assert torch.equal(quant.dequantize_codes(codes, scale, baked.dtype), baked)


def test_derive_codes_and_scale_rejects_a_latent_weight():
    with pytest.raises(ValueError, match="not baked"):
        bake.derive_codes_and_scale(torch.randn(8, 8))


def test_derive_codes_and_scale_on_all_zero_weight():
    codes, scale = bake.derive_codes_and_scale(torch.zeros(8, 8))

    assert int(codes.abs().sum()) == 0
    assert float(scale) == pytest.approx(quant.SCALE_EPS)


def test_derive_codes_and_scale_group_mode():
    weight = torch.randn(8, 64)
    baked = bake.bake_weight(weight, group_size=16)

    codes, scale = bake.derive_codes_and_scale(baked, group_size=16)

    assert scale.shape == (8, 4)
    assert torch.equal(
        codes, quant.ternary_codes(weight, quant.absmean_scale(weight, 16))
    )
    assert torch.equal(quant.dequantize_codes(codes, scale, baked.dtype), baked)


def test_bake_state_dict_touches_only_manifest_entries():
    manifest = SwapManifest(
        model_type="llama",
        entries=[SwapEntry("layer.q_proj", 8, 8, "q_proj", "absmean")],
    )
    state = {
        "layer.q_proj.weight": torch.randn(8, 8),
        "other.weight": torch.randn(4, 4),
    }

    baked = bake.bake_state_dict(state, manifest)

    assert baked["other.weight"] is state["other.weight"]
    assert baked["layer.q_proj.weight"].abs().unique().numel() == 2


def test_bake_state_dict_missing_entry_raises():
    manifest = SwapManifest(
        model_type="llama", entries=[SwapEntry("missing", 8, 8, "q_proj", "absmean")]
    )

    with pytest.raises(KeyError, match="missing.weight"):
        bake.bake_state_dict({}, manifest)


def test_bake_directory_bakes_a_latent_checkpoint(tmp_path):
    directory = tmp_path / "latent"
    manifest = _write_master(directory, bake_weights=False)
    latent = bake.load_tensors(directory)

    bake.bake_directory(directory)
    baked = bake.load_tensors(directory)

    key = f"{manifest.entries[0].name}.weight"
    assert baked[key].abs().unique().numel() == 2
    assert not torch.equal(baked[key], latent[key])
    assert torch.equal(baked["model.norm.weight"], latent["model.norm.weight"])


def test_bake_directory_is_idempotent(tmp_path):
    directory = tmp_path / "latent"
    _write_master(directory, bake_weights=False)

    bake.bake_directory(directory)
    first = (directory / bake.SAFETENSORS_NAME).read_bytes()
    bake.bake_directory(directory)

    assert (directory / bake.SAFETENSORS_NAME).read_bytes() == first


def test_bake_directory_writes_manifest_and_stamp(tmp_path):
    source = tmp_path / "latent"
    manifest = _write_master(source, bake_weights=False)
    (source / "ternary_manifest.json").unlink()
    destination = tmp_path / "baked"

    bake.bake_directory(source, destination, manifest)

    assert SwapManifest.load(destination).entries == manifest.entries
    config = json.loads((destination / "config.json").read_text())
    assert config[bake.QUANTIZER_METADATA_KEY]["scheme"] == "ternary"
    assert config[bake.QUANTIZER_METADATA_KEY]["format"] == "master_bf16"
    assert (destination / "generation_config.json").is_file()


def test_bake_directory_reports_a_missing_weight(tmp_path):
    directory = tmp_path / "latent"
    manifest = _write_master(directory, bake_weights=False)
    manifest.entries.append(SwapEntry("model.ghost", 8, 8, "ghost", "absmean"))

    with pytest.raises(KeyError, match="model.ghost.weight"):
        bake.bake_directory(directory, manifest=manifest)


def test_write_quantizer_metadata_without_config(tmp_path):
    assert not bake.write_quantizer_metadata(tmp_path, SwapManifest(model_type="llama"))


def test_load_master_returns_tensors_and_manifest(master):
    master_dir, manifest = master

    tensors, loaded = bake.load_master(master_dir)

    assert loaded.entries == manifest.entries
    assert f"{manifest.entries[0].name}.weight" in tensors


def test_shard_paths_without_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="no model weights"):
        bake.shard_paths(tmp_path)


# ---------------------------------------------------------------------------- export


def test_export_writes_packed_weights_and_scales(packed):
    artifact, manifest = packed
    tensors = bake.load_tensors(artifact)

    for entry in manifest.entries:
        weight = tensors[f"{entry.name}.weight"]
        scale = tensors[f"{entry.name}.weight_scale"]
        assert weight.dtype == torch.uint8
        assert weight.shape == (entry.out_features // 4, entry.in_features)
        assert scale.dtype == torch.float32 and scale.shape == (1,)
    assert tensors["lm_head.weight"].dtype == torch.bfloat16


def test_export_stamps_quantization_config(packed):
    artifact, _ = packed
    config = json.loads((artifact / "config.json").read_text())

    assert config["quantization_config"] == {
        "quant_method": "bitnet",
        "linear_class": "bitlinear",
        "quantization_mode": "offline",
        "use_rms_norm": False,
        "rms_norm_eps": 1e-6,
        # anchored so the prefix match cannot also spare `...gate_proj`
        "modules_to_not_convert": ["lm_head$"],
    }
    assert config[bake.QUANTIZER_METADATA_KEY]["format"] == "hf_bitnet"


def test_export_weight_scale_is_the_reciprocal_absmean(packed, master):
    artifact, manifest = packed
    master_tensors = bake.load_tensors(master[0])
    tensors = bake.load_tensors(artifact)

    name = manifest.entries[0].name
    scale = float(master_tensors[f"{name}.weight"].abs().amax())

    assert float(tensors[f"{name}.weight_scale"]) == pytest.approx(
        1.0 / scale, rel=1e-6
    )


def test_export_rejects_group_scales(master, tmp_path):
    master_dir, manifest = master
    grouped = SwapManifest(
        model_type=manifest.model_type,
        entries=manifest.entries,
        weight_scale="group",
        group_size=32,
    )

    with pytest.raises(ValueError, match="scalar weight_scale"):
        hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "out", grouped)


def test_export_rejects_unsupported_architecture(master, tmp_path):
    master_dir, manifest = master
    foreign = SwapManifest(model_type="gpt2", entries=manifest.entries)

    with pytest.raises(ValueError, match="model_type 'gpt2'"):
        hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "out", foreign)


def test_export_refuses_to_overwrite_the_master(master):
    master_dir, manifest = master

    with pytest.raises(ValueError, match="not over it"):
        hf_bitnet.export_hf_bitnet(master_dir, master_dir, manifest)


def test_export_reports_a_missing_master_weight(master, tmp_path):
    master_dir, manifest = master
    incomplete = SwapManifest(
        model_type=manifest.model_type,
        entries=[*manifest.entries, SwapEntry("model.ghost", 8, 8, "ghost", "absmean")],
        kept_fp=manifest.kept_fp,
    )

    with pytest.raises(KeyError, match="model.ghost.weight"):
        hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "out", incomplete)


def test_export_rejects_a_latent_master(tmp_path):
    master_dir = tmp_path / "latent"
    manifest = _write_master(master_dir, bake_weights=False)

    with pytest.raises(ValueError, match="still holds latent weights"):
        hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "out", manifest)


def test_export_preserves_sharding(tmp_path):
    master_dir = tmp_path / "sharded"
    manifest = _write_master(master_dir, max_shard_size="20KB")
    assert len(bake.shard_paths(master_dir)) > 1

    artifact = hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "packed", manifest)

    index = json.loads((artifact / bake.SAFETENSORS_INDEX_NAME).read_text())
    tensors = bake.load_tensors(artifact)
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    assert parity.run_parity_gate(master_dir, artifact, "hf_bitnet", manifest).passed


def test_bake_directory_preserves_sharding(tmp_path):
    master_dir = tmp_path / "sharded"
    manifest = _write_master(master_dir, bake_weights=False, max_shard_size="20KB")
    shards = bake.shard_paths(master_dir)
    assert len(shards) > 1

    bake.bake_directory(master_dir)

    assert bake.shard_paths(master_dir) == shards
    key = f"{manifest.entries[0].name}.weight"
    assert bake.load_tensors(master_dir)[key].abs().unique().numel() == 2


def test_bitlinear_forward_matches_the_master(packed, master):
    artifact, manifest = packed
    entry = next(e for e in manifest.entries if e.family == "q_proj")
    master_weight = bake.load_tensors(master[0])[f"{entry.name}.weight"].float()
    tensors = bake.load_tensors(artifact)

    linear = bitnet.BitLinear(
        entry.in_features, entry.out_features, bias=False, dtype=torch.float32
    )
    linear.weight = tensors[f"{entry.name}.weight"]
    linear.weight_scale = tensors[f"{entry.name}.weight_scale"]
    torch.manual_seed(0)
    x = torch.randn(4, entry.in_features)
    reference = F.linear(x, master_weight)

    with torch.no_grad():
        correct = linear(x)
        # the reciprocal orientation is load-bearing: BitLinear divides by weight_scale
        linear.weight_scale = tensors[f"{entry.name}.weight_scale"].reciprocal()
        flipped = linear(x)

    error = float((correct - reference).norm() / reference.norm())
    flipped_error = float((flipped - reference).norm() / reference.norm())
    assert error < 0.05
    assert flipped_error > 10 * error


# ---------------------------------------------------------------------------- parity


def test_parity_gate_passes_end_to_end(packed, master):
    artifact, manifest = packed

    report = parity.run_parity_gate(master[0], artifact, "hf_bitnet", manifest)

    assert report.passed
    assert report.tensors_checked == len(manifest.entries)
    assert report.code_mismatches == 0
    assert report.max_dequant_error == 0.0
    # the weight path is a deterministic graph fed bit-identical weights
    assert report.smoke.weight_path_logit_delta == 0.0
    # the deployment quantizer is compared to the training one on the activations
    # themselves, and its end-to-end effect is reported without being gated
    assert report.smoke.runtime_code_mismatch <= parity.RUNTIME_CODE_MISMATCH_TOL
    assert report.smoke.runtime_dequant_steps <= parity.RUNTIME_DEQUANT_STEP_TOL
    assert report.smoke.runtime_code_range_matches
    assert report.smoke.cross_quantizer_logit_delta is not None
    assert report.smoke.probes_compared == len(manifest.entries)


def test_smoke_eval_separates_the_weight_path_from_the_runtime_quantizer(
    packed, master
):
    """One end-to-end number cannot gate both; the weight path must stay exact."""
    artifact, manifest = packed

    smoke = parity.run_smoke_eval(master[0], artifact, "hf_bitnet")

    assert smoke.passed
    assert smoke.weight_path_logit_delta == 0.0
    assert smoke.cross_quantizer_logit_delta > smoke.weight_path_logit_delta

    without_acts = copy.deepcopy(manifest)
    without_acts.activation_bits = None
    without_acts.save(master[0])
    try:
        weight_only = parity.run_smoke_eval(master[0], artifact, "hf_bitnet")
    finally:
        manifest.save(master[0])
    assert weight_only.passed
    assert weight_only.weight_path_logit_delta == 0.0
    # no activation quantizer runs at all, so there is nothing to compare
    assert weight_only.cross_quantizer_logit_delta is None
    assert weight_only.runtime_code_mismatch is None


def test_bitlinear_activation_quant_is_not_the_training_one():
    """BitLinear floors `amax`; training floors the scale — they diverge near zero."""
    torch.manual_seed(0)
    tiny = torch.randn(4, 8) * 1e-6

    assert float(quant.act_quant(tiny, 1.0).abs().max()) == 0.0
    assert float(parity.bitlinear_act_quant(tiny).abs().max()) > 0.0


def test_smoke_eval_is_skipped_for_a_format_with_no_in_process_runtime(master):
    assert parity.run_smoke_eval(master[0], master[0], "gguf_tq2_0") is None


def test_parity_gate_passes_on_the_master_itself(master):
    master_dir, manifest = master

    report = parity.run_parity_gate(master_dir, master_dir, "master_bf16", manifest)

    assert report.passed and report.tensors_checked == len(manifest.entries)


def test_parity_gate_catches_a_corrupted_code(packed, master, tmp_path):
    artifact, manifest = packed
    corrupted = tmp_path / "corrupted"
    shutil.copytree(artifact, corrupted)
    entry = manifest.entries[0]
    tensors = bake.load_tensors(corrupted)
    codes = hf_bitnet.unpack_hf_bitnet(
        tensors[f"{entry.name}.weight"], entry.out_features
    )
    codes[0, 0] = 1 - codes[0, 0]
    tensors[f"{entry.name}.weight"] = hf_bitnet.pack_hf_bitnet(codes)
    bake.save_shard(tensors, corrupted / bake.SAFETENSORS_NAME)

    report = parity.run_parity_gate(master[0], corrupted, "hf_bitnet", manifest)

    assert not report.passed
    assert report.code_mismatches == 1
    assert any("codes differ" in failure for failure in report.failures)


def test_parity_gate_catches_a_drifted_scale(packed, master, tmp_path):
    artifact, manifest = packed
    drifted = tmp_path / "drifted"
    shutil.copytree(artifact, drifted)
    tensors = bake.load_tensors(drifted)
    key = f"{manifest.entries[0].name}.weight_scale"
    tensors[key] = tensors[key] * 1.01
    bake.save_shard(tensors, drifted / bake.SAFETENSORS_NAME)

    report = parity.run_parity_gate(master[0], drifted, "hf_bitnet", manifest)

    assert not report.passed
    assert report.code_mismatches == 0
    assert any("exceeds the f16 bound" in failure for failure in report.failures)


def test_parity_gate_reports_a_missing_tensor(packed, master, tmp_path):
    artifact, manifest = packed
    trimmed = tmp_path / "trimmed"
    shutil.copytree(artifact, trimmed)
    tensors = bake.load_tensors(trimmed)
    del tensors[f"{manifest.entries[0].name}.weight_scale"]
    bake.save_shard(tensors, trimmed / bake.SAFETENSORS_NAME)

    report = parity.run_parity_gate(master[0], trimmed, "hf_bitnet", manifest)

    assert not report.passed
    assert report.tensors_checked == len(manifest.entries) - 1
    assert any("is missing from the hf_bitnet" in f for f in report.failures)


def test_parity_gate_fails_on_a_latent_master(tmp_path):
    master_dir = tmp_path / "latent"
    manifest = _write_master(master_dir, bake_weights=False)

    report = parity.run_parity_gate(master_dir, master_dir, "master_bf16", manifest)

    assert not report.passed
    assert any("could not unpack" in failure for failure in report.failures)


def test_parity_gate_rejects_an_unregistered_format(master):
    master_dir, manifest = master

    with pytest.raises(ValueError, match="no parity unpacker"):
        parity.run_parity_gate(master_dir, master_dir, "gguf_tq2_0", manifest)


def test_check_code_parity_counts_differences():
    baked = bake.bake_weight(torch.randn(8, 8))
    codes, _ = bake.derive_codes_and_scale(baked)
    wrong = codes.clone()
    wrong[0, 0] = 1 - wrong[0, 0]
    wrong[1, 1] = 1 - wrong[1, 1]

    assert parity.check_code_parity(baked, codes) == 0
    assert parity.check_code_parity(baked, wrong) == 2


def test_check_code_parity_rejects_a_shape_mismatch():
    baked = bake.bake_weight(torch.randn(8, 8))

    with pytest.raises(ValueError, match="shape mismatch"):
        parity.check_code_parity(baked, torch.zeros(4, 8, dtype=torch.int8))


def test_check_dequant_error_bounds_f16_rounding():
    baked = bake.bake_weight(torch.randn(8, 8))
    codes, scale = bake.derive_codes_and_scale(baked)

    exact, bound = parity.check_dequant_error(
        baked, quant.dequantize_codes(codes, scale, torch.float32)
    )
    within, _ = parity.check_dequant_error(
        baked, quant.dequantize_codes(codes, scale * (1 + 2**-12), torch.float32)
    )
    beyond, _ = parity.check_dequant_error(
        baked, quant.dequantize_codes(codes, scale * (1 + 2**-9), torch.float32)
    )

    assert exact == 0.0
    assert bound == pytest.approx(float(scale) * 2**-11)
    assert 0.0 < within <= bound < beyond


def test_parity_report_passed_property():
    report = parity.ParityReport(format="hf_bitnet")
    assert report.passed

    report.max_dequant_error = 1e-3
    report.dequant_error_bound = 1e-4
    assert not report.passed

    report.dequant_error_bound = 1e-2
    assert report.passed

    report.failures.append("boom")
    assert not report.passed


def test_run_smoke_eval_without_a_model_directory(tmp_path, master):
    _, manifest = master
    manifest.save(tmp_path)

    assert parity.run_smoke_eval(tmp_path, tmp_path, "hf_bitnet") is None


def test_run_smoke_eval_unknown_format_returns_none(master):
    master_dir, _ = master

    assert parity.run_smoke_eval(master_dir, master_dir, "i2_s") is None


def test_transformers_loads_the_packed_checkpoint(packed, master):
    artifact, _ = packed

    quantized = AutoModelForCausalLM.from_pretrained(artifact, dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(master[0], dtype=torch.float32)
    input_ids = (torch.arange(16) % 128).unsqueeze(0)
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual = quantized(input_ids=input_ids).logits

    assert type(quantized.model.layers[0].self_attn.q_proj).__name__ == "BitLinear"
    assert type(quantized.lm_head).__name__ == "Linear"
    # BitLinear additionally quantizes activations to int8, so parity is statistical
    assert float((expected - actual).abs().max() / expected.abs().max()) < 0.05


# ------------------------------------------- deep model + injected packer faults


def _deep_master(directory, layers: int = 10, hidden: int = 256) -> SwapManifest:
    """A deep swapped model: chaotic enough that ulp-level drift reaches O(1) logits."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": {}})
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


@pytest.fixture(name="deep", scope="module")
def fixture_deep(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deep_master")
    manifest = _deep_master(directory)
    artifact = hf_bitnet.export_hf_bitnet(
        directory, tmp_path_factory.mktemp("deep_packed") / "hf_bitnet", manifest
    )
    return directory, artifact, manifest


def _repack(artifact: Path, destination: Path, transform) -> Path:
    """Copy a packed artifact, rewriting each ternary tensor with `transform`."""
    shutil.copytree(artifact, destination)
    tensors = bake.load_tensors(destination)
    for key in list(tensors):
        if key.endswith(f".{hf_bitnet.WEIGHT_SCALE_SUFFIX}"):
            transform(tensors, key[: -len(hf_bitnet.WEIGHT_SCALE_SUFFIX) - 1])
    bake.save_shard(tensors, destination / bake.SAFETENSORS_NAME)
    return destination


def test_deep_model_passes_both_gates(deep):
    """A correct packer must clear the gates however deep and damaged the model is."""
    master_dir, artifact, manifest = deep

    report = parity.run_parity_gate(master_dir, artifact, "hf_bitnet", manifest)

    assert report.passed, report.failures
    assert report.smoke.weight_path_logit_delta == 0.0
    assert report.smoke.runtime_code_range_matches
    assert report.smoke.runtime_code_mismatch <= parity.RUNTIME_CODE_MISMATCH_TOL
    # the cross-quantizer delta is the chaos this gate deliberately does not bound
    assert report.smoke.cross_quantizer_logit_delta > parity.WEIGHT_PATH_LOGIT_TOL


def test_a_transposed_packing_fails_the_weight_path_gate(deep, tmp_path):
    master_dir, artifact, _ = deep

    def flip_rows(tensors, name):
        packed = tensors[f"{name}.weight"]
        tensors[f"{name}.weight"] = packed.flip(0).contiguous()

    corrupted = _repack(artifact, tmp_path / "transposed", flip_rows)
    smoke = parity.run_smoke_eval(master_dir, corrupted, "hf_bitnet")

    assert not smoke.passed
    assert "weight path" in smoke.failures[0]
    assert smoke.weight_path_logit_delta > parity.WEIGHT_PATH_LOGIT_TOL


def test_an_inverted_weight_scale_fails_the_weight_path_gate(deep, tmp_path):
    """`hf_bitnet` stores `1 / s`; writing `s` leaves the codes right and the model wrong."""
    master_dir, artifact, _ = deep

    def uninvert(tensors, name):
        key = f"{name}.{hf_bitnet.WEIGHT_SCALE_SUFFIX}"
        tensors[key] = tensors[key].reciprocal()

    corrupted = _repack(artifact, tmp_path / "inverted", uninvert)
    smoke = parity.run_smoke_eval(master_dir, corrupted, "hf_bitnet")

    assert not smoke.passed
    assert "weight path" in smoke.failures[0]


def test_a_wrong_clamp_range_fails_the_runtime_gate(monkeypatch):
    """[-127, 126] moves too few codes to show as a fraction; it truncates the extreme."""
    torch.manual_seed(0)
    activations = [torch.randn(4, 512) for _ in range(3)]
    monkeypatch.setattr(
        parity,
        "bitlinear_act_quant_int8",
        lambda x: _act_quant_int8_with(x, low=-127, high=126),
    )
    report = parity.SmokeReport(format="hf_bitnet")

    report.gate_runtime_quantizer(activations)

    assert not report.passed
    assert any("clamp-range" in failure for failure in report.failures)
    assert report.runtime_code_range_matches is False
    # the fraction alone would have waved it through
    assert report.runtime_code_mismatch <= parity.RUNTIME_CODE_MISMATCH_TOL


def test_an_inverted_activation_scale_fails_the_runtime_gate(monkeypatch):
    torch.manual_seed(0)
    activations = [torch.randn(4, 512) for _ in range(3)]
    monkeypatch.setattr(
        parity,
        "bitlinear_act_quant_int8",
        lambda x: _act_quant_int8_with(x, inverted=True),
    )
    report = parity.SmokeReport(format="hf_bitnet")

    report.gate_runtime_quantizer(activations)

    assert not report.passed
    assert report.runtime_code_mismatch > parity.RUNTIME_CODE_MISMATCH_TOL


def test_the_real_runtime_quantizer_clears_the_gate_on_the_same_activations():
    torch.manual_seed(0)
    activations = [torch.randn(4, 512) for _ in range(3)]
    report = parity.SmokeReport(format="hf_bitnet")

    report.gate_runtime_quantizer(activations)

    assert report.passed, report.failures
    assert report.probes_compared == 3


def _act_quant_int8_with(
    x: torch.Tensor,
    low: int = -128,
    high: int = quant.ACT_QMAX,
    inverted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`bitlinear_act_quant_int8` with a deliberately wrong clamp range or scale."""
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(quant.SCALE_EPS)
    scale = amax / quant.ACT_QMAX if inverted else quant.ACT_QMAX / amax
    codes = (x.float() * scale).round().clamp_(low, high)
    return codes.to(torch.int8), scale


# ------------------------------------------- kept-FP islands through transformers

QWEN35_LINEARS = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.linear_attn.in_proj_qkv",
    "model.language_model.layers.1.linear_attn.in_proj_a",
    "model.language_model.layers.1.linear_attn.in_proj_b",
    "model.language_model.layers.1.linear_attn.in_proj_z",
    "model.language_model.layers.1.linear_attn.out_proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
    "model.visual.merger.linear_fc1",
    "lm_head",
)


def _mixed_master(directory):
    """A master with kept-FP *block* Linears — `v_proj` and `down_proj` islands."""
    torch.manual_seed(0)
    model = _tiny_llama()
    manifest = convert_model(
        model,
        DictDefault(
            {
                "output_dir": str(directory),
                "ternary": {
                    "target_modules": [
                        r".*\.self_attn\.(q|k|o)_proj",
                        r".*\.mlp\.(gate|up)_proj",
                    ],
                    "keep_fp_modules": [
                        r".*\.self_attn\.v_proj",
                        r".*\.mlp\.down_proj",
                    ],
                },
            }
        ),
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


def test_modules_to_not_convert_lists_every_kept_linear(tmp_path):
    manifest = _mixed_master(tmp_path / "master")

    patterns = hf_bitnet._modules_to_not_convert(manifest)

    # one anchored absolute pattern per kept module, plus re-rooting suffixes
    assert len(patterns) >= len(set(manifest.kept_fp))
    for name in set(manifest.kept_fp):
        assert f"{re.escape(name)}$" in patterns
    # the always-FP head is in there alongside the block islands
    assert any(pattern.startswith("lm_head") for pattern in patterns)
    assert any("v_proj" in pattern for pattern in patterns)
    assert any("down_proj" in pattern for pattern in patterns)


def test_the_patterns_match_transformers_own_matcher(tmp_path):
    """`should_convert_module` is the consumer; the list is only correct if it agrees."""
    from transformers.quantizers.quantizers_utils import should_convert_module

    manifest = _mixed_master(tmp_path / "master")
    patterns = hf_bitnet._modules_to_not_convert(manifest)

    for name in manifest.kept_fp:
        assert not should_convert_module(name, patterns), name
    for entry in manifest.entries:
        assert should_convert_module(entry.name, patterns), entry.name


def test_the_patterns_do_not_spare_a_longer_sibling_name():
    """`layers.1` must not shield `layers.11`, so the patterns are end-anchored."""
    from transformers.quantizers.quantizers_utils import should_convert_module

    manifest = SwapManifest(
        model_type="llama", kept_fp=["model.layers.1.mlp.down_proj"]
    )
    patterns = hf_bitnet._modules_to_not_convert(manifest)

    assert not should_convert_module("model.layers.1.mlp.down_proj", patterns)
    assert should_convert_module("model.layers.11.mlp.down_proj", patterns)


def test_the_packed_config_always_carries_the_kept_list(tmp_path):
    """Regression: a master with kept-FP block Linears must never ship without it."""
    manifest = _mixed_master(tmp_path / "master")

    artifact = hf_bitnet.export_hf_bitnet(
        tmp_path / "master", tmp_path / "packed", manifest
    )

    quant_config = json.loads((artifact / "config.json").read_text())[
        "quantization_config"
    ]
    islands = [
        name
        for name in manifest.kept_fp
        if "lm_head" not in name and "embed" not in name
    ]
    assert islands, "the fixture is meant to have kept-FP block Linears"
    assert quant_config["modules_to_not_convert"]
    assert len(quant_config["modules_to_not_convert"]) >= len(set(manifest.kept_fp))


def test_transformers_leaves_the_kept_islands_unwrapped(tmp_path):
    """End to end: the kept-FP Linears must survive the bitnet quantizer as `nn.Linear`.

    Without the not-convert list they would be wrapped in `BitLinear`, their missing
    `weight_scale` freshly initialized and their full-precision weights read as packed
    codes — a silently wrong model that still loads.
    """
    manifest = _mixed_master(tmp_path / "master")
    artifact = hf_bitnet.export_hf_bitnet(
        tmp_path / "master", tmp_path / "packed", manifest
    )

    loaded = AutoModelForCausalLM.from_pretrained(artifact, dtype=torch.bfloat16)

    kinds = {name: type(module).__name__ for name, module in loaded.named_modules()}
    for entry in manifest.entries:
        assert kinds[entry.name].endswith("BitLinear"), entry.name
    for name in manifest.kept_fp:
        assert kinds[name] == "Linear", name


# --------------------------------------------------------------- qwen3_5 preset


def test_qwen3_5_is_a_supported_pack_target():
    assert "qwen3_5" in hf_bitnet.SUPPORTED_MODEL_TYPES


def test_qwen3_5_moe_is_a_supported_pack_target():
    assert "qwen3_5_moe" in hf_bitnet.SUPPORTED_MODEL_TYPES


def test_the_qwen3_5_preset_splits_the_hybrid_the_way_the_probe_did():
    """Full attention and dense MLPs ternary; linear attention and the tower FP."""
    from axolotl.integrations.ternary.swap import ALWAYS_KEEP_FP, resolve_preset

    preset = resolve_preset("qwen3_5")
    targets = [re.compile(pattern) for pattern in preset.target_modules]
    keeps = [re.compile(pattern) for pattern in preset.keep_fp_modules]
    protected = [re.compile(pattern) for pattern in ALWAYS_KEEP_FP]

    targeted, kept = [], []
    for name in QWEN35_LINEARS:
        is_target = any(pattern.fullmatch(name) for pattern in targets)
        is_keep = any(pattern.fullmatch(name) for pattern in keeps)
        assert not (is_target and is_keep), f"{name} matches both lists"
        if any(pattern.fullmatch(name) for pattern in protected):
            kept.append(name)
            continue
        assert is_target or is_keep, f"{name} matches neither list"
        (targeted if is_target else kept).append(name)

    assert (
        targeted
        == [name for name in QWEN35_LINEARS if "self_attn" in name or "mlp" in name][:7]
    )
    assert all("linear_attn" in n or "visual" in n or "lm_head" in n for n in kept)


def test_the_qwen3_5_preset_keeps_the_vision_tower_whole():
    from axolotl.integrations.ternary.swap import resolve_preset

    keeps = [re.compile(p) for p in resolve_preset("qwen3_5").keep_fp_modules]
    tower = [name for name in QWEN35_LINEARS if name.startswith("model.visual.")]

    assert tower
    for name in tower:
        assert any(pattern.fullmatch(name) for pattern in keeps), name


# ------------------------------------------- re-rooted trees (the qwen3_5 failure)


def _nested_manifest() -> SwapManifest:
    """A master saved from a multimodal tree: the decoder sits under `language_model`."""
    prefix = "model.language_model.layers.0"
    return SwapManifest(
        model_type="qwen3_5",
        entries=[
            SwapEntry(f"{prefix}.self_attn.q_proj", 64, 64, "q_proj", "absmean"),
            SwapEntry(f"{prefix}.mlp.gate_proj", 64, 128, "gate_proj", "absmean"),
        ],
        kept_fp=[
            f"{prefix}.linear_attn.in_proj_qkv",
            f"{prefix}.linear_attn.out_proj",
            "model.visual.blocks.0.attn.qkv",
            "lm_head",
        ],
    )


def _reroot(name: str) -> str:
    """The same module as a text-only causal LM names it at load."""
    return name.replace("model.language_model.", "model.")


def test_kept_modules_survive_a_re_rooted_reload():
    """The real failure: saved under `model.language_model.*`, loaded as `model.*`.

    An absolute pattern misses, the module is wrapped in BitLinear, its `weight_scale`
    is freshly initialized and its FP weights are read as packed codes.
    """
    from transformers.quantizers.quantizers_utils import should_convert_module

    manifest = _nested_manifest()
    patterns = _modules_to_not_convert(manifest)

    for name in manifest.kept_fp:
        assert not should_convert_module(name, patterns), f"saved tree: {name}"
        assert not should_convert_module(_reroot(name), patterns), (
            f"loaded tree: {name}"
        )
    for entry in manifest.entries:
        assert should_convert_module(entry.name, patterns)
        assert should_convert_module(_reroot(entry.name), patterns)


def test_the_absolute_pattern_alone_would_miss_the_re_rooted_module():
    """Pins why the suffix exists, so nobody 'simplifies' it back."""
    from transformers.quantizers.quantizers_utils import should_convert_module

    kept = "model.language_model.layers.0.linear_attn.out_proj"

    absolute_only = [f"{re.escape(kept)}$"]
    assert should_convert_module(_reroot(kept), absolute_only), "premise changed"
    assert not should_convert_module(
        _reroot(kept),
        _modules_to_not_convert(SwapManifest(model_type="qwen3_5", kept_fp=[kept])),
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "model.language_model.layers.0.linear_attn.out_proj",
            "layers.0.linear_attn.out_proj",
        ),
        ("model.layers.11.mlp.down_proj", "layers.11.mlp.down_proj"),
        ("model.visual.blocks.3.attn.qkv", "blocks.3.attn.qkv"),
        ("lm_head", None),
        ("model.visual.merger.linear_fc1", None),
    ],
)
def test_the_re_rooting_suffix_starts_at_the_stack_index(name, expected):
    assert hf_bitnet._rerooting_suffix(name) == expected


def test_a_suffix_that_would_swallow_a_packed_tensor_is_not_emitted():
    """A keep must never spare a module whose weights are actually packed."""
    manifest = SwapManifest(
        model_type="llama",
        entries=[
            SwapEntry("vision.layers.0.mlp.down_proj", 64, 64, "down_proj", "absmean")
        ],
        kept_fp=["text.layers.0.mlp.down_proj"],
    )

    patterns = _modules_to_not_convert(manifest)

    assert "layers.0.mlp.down_proj" not in patterns
    assert any(pattern.endswith("$") for pattern in patterns)


def test_every_packed_tensor_stays_convertible_under_both_trees():
    """The invariant the suffix guard protects, checked with transformers' matcher."""
    from transformers.quantizers.quantizers_utils import should_convert_module

    manifest = _nested_manifest()
    patterns = _modules_to_not_convert(manifest)

    for entry in manifest.entries:
        for name in (entry.name, _reroot(entry.name)):
            assert should_convert_module(name, patterns), name

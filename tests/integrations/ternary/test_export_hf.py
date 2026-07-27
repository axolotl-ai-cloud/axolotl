"""CPU-only tests for the bake, transformers-bitnet packer and parity gates."""

import copy
import json
import shutil

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import bake, hf_bitnet, parity
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
    # the smoke eval runs BitLinear's own activation quantizer against the master's,
    # so it measures the deployment mismatch rather than re-checking the codes
    assert 0.0 < report.smoke_max_logit_delta <= parity.SMOKE_LOGIT_TOL


def test_smoke_eval_measures_the_runtime_activation_quantizer(packed, master):
    """Substituting the same codes through the same dequant can never fail; this can."""
    artifact, manifest = packed

    delta = parity.run_smoke_eval(master[0], artifact, "hf_bitnet")

    assert delta > 0.0
    manifest_without_acts = copy.deepcopy(manifest)
    manifest_without_acts.activation_bits = None
    manifest_without_acts.save(master[0])
    try:
        assert parity.run_smoke_eval(master[0], artifact, "hf_bitnet") == 0.0
    finally:
        manifest.save(master[0])


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

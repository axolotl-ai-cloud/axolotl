"""A `codebook: binary` master through the transformers-bitnet container.

`{-s, +s}` is a subset of `{-s, 0, +s}`, so `hf_bitnet` carries a binary master with no
special case at all — the two-bit lanes simply never take the zero value. That is worth
a test rather than an assumption: the packer, the gates and `BitLinear` itself all have
code paths that could have assumed a zero exists somewhere in the tensor.
"""

import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from axolotl.integrations.ternary.export import bake, hf_bitnet, parity

from .test_binary_codebook import write_binary_master

pytest.importorskip("transformers.integrations.bitnet")


@pytest.fixture(name="master", scope="module")
def fixture_master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("binary_hf_master")
    return directory, write_binary_master(directory, formats=["master_bf16"])


@pytest.fixture(name="packed", scope="module")
def fixture_packed(master, tmp_path_factory):
    master_dir, manifest = master
    output = tmp_path_factory.mktemp("binary_hf_packed") / "hf_bitnet"
    return hf_bitnet.export_hf_bitnet(master_dir, output, manifest), manifest


def test_the_packed_lanes_never_take_the_zero_code(packed):
    artifact, manifest = packed
    tensors = bake.load_tensors(artifact)

    for entry in manifest.entries:
        codes = hf_bitnet.unpack_hf_bitnet(
            tensors[f"{entry.name}.weight"], entry.out_features
        )
        assert set(codes.reshape(-1).tolist()) == {-1, 1}


def test_the_scale_companion_is_the_reciprocal_magnitude(packed, master):
    artifact, manifest = packed
    master_tensors = bake.load_tensors(master[0])
    tensors = bake.load_tensors(artifact)

    for entry in manifest.entries:
        magnitude = float(master_tensors[f"{entry.name}.weight"].abs().amax())
        assert float(
            tensors[f"{entry.name}.{hf_bitnet.WEIGHT_SCALE_SUFFIX}"]
        ) == pytest.approx(1.0 / magnitude, rel=1e-6)


def test_the_container_costs_two_bits_for_one_bit_of_information(packed, master):
    """Which is why `sign_plane` exists; the packer warns rather than refuses."""
    artifact, manifest = packed
    entry = manifest.entries[0]
    weights = entry.out_features * entry.in_features

    payload = bake.load_tensors(artifact)[f"{entry.name}.weight"]

    assert payload.numel() * 8 == 2 * weights


def test_the_export_warns_that_a_sign_codebook_is_padded(master, tmp_path, caplog):
    master_dir, manifest = master
    declared = type(manifest)(
        model_type=manifest.model_type,
        entries=manifest.entries,
        kept_fp=manifest.kept_fp,
        quantizer={**manifest.quantizer, "codebook": "binary"},
    )

    with caplog.at_level("WARNING"):
        hf_bitnet.export_hf_bitnet(master_dir, tmp_path / "warned", declared)

    assert "two-bit code per weight" in caplog.text


def test_the_parity_gate_passes_on_a_binary_master(packed, master):
    artifact, manifest = packed

    report = parity.run_parity_gate(master[0], artifact, "hf_bitnet", manifest)

    assert report.passed
    assert report.tensors_checked == len(manifest.entries)
    assert report.code_mismatches == 0
    assert report.smoke.weight_path_logit_delta == 0.0


def test_transformers_loads_the_packed_binary_checkpoint(packed, master):
    artifact, _ = packed

    quantized = AutoModelForCausalLM.from_pretrained(artifact, dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(master[0], dtype=torch.float32)
    input_ids = (torch.arange(16) % 64).unsqueeze(0)
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual = quantized(input_ids=input_ids).logits

    assert type(quantized.model.layers[0].self_attn.q_proj).__name__ == "BitLinear"
    assert (
        json.loads((artifact / "config.json").read_text())["quantization_config"][
            "quant_method"
        ]
        == "bitnet"
    )
    # BitLinear additionally quantizes activations to int8, so parity is statistical
    assert float((expected - actual).abs().max() / expected.abs().max()) < 0.05

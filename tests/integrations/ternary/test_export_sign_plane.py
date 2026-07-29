"""`sign_plane`: one bit per weight, the packed container for `codebook: binary`.

It is the one-plane member of the bit-plane family and the only format that is *not*
safe to point at an arbitrary master: it has no bit pattern for a zero, so a ternary
checkpoint handed to it would come back with every zeroed weight at ±s. Every gate here
exists to make that impossible rather than unlikely.
"""

import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import bake, bitplanes, parity, run_export
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

from .test_binary_codebook import write_binary_master

SIGN_KEY = bitplanes.SIGN_PLANE_SUFFIX
SCALE_KEY = bitplanes.SCALES_SUFFIX


def _ternary_master(directory) -> SwapManifest:
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": {}})
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


@pytest.fixture(name="master", scope="module")
def fixture_master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("binary_master")
    return directory, write_binary_master(
        directory, formats=["master_bf16", "sign_plane"]
    )


@pytest.fixture(name="packed", scope="module")
def fixture_packed(master, tmp_path_factory):
    master_dir, manifest = master
    output = tmp_path_factory.mktemp("sign_plane") / "sign_plane"
    return bitplanes.export_sign_plane(master_dir, output, manifest), manifest


# ------------------------------------------------------------------ micro-fixture


def test_the_bits_are_exactly_one_per_weight_lsb_first():
    """Every byte spelled out: a mixed row, an all-positive row, and a padded tail."""
    codes = torch.tensor(
        [
            [1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int8,
    )

    packed = bitplanes.pack_sign_plane(codes)

    # 24 weights, flat row-major, LSB first: positive at 0, 3, 4, 5, 8, 9, 11 and
    # then every weight of the second row
    assert packed.tolist() == [0x39, 0xFB, 0xFF]
    assert torch.equal(bitplanes.unpack_sign_plane(packed, (2, 12)), codes)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (8, 8), (5, 13), (64, 64)])
def test_the_round_trip_is_exact_for_any_shape(shape):
    torch.manual_seed(0)
    codes = torch.where(torch.randn(shape) < 0, -1, 1).to(torch.int8)

    packed = bitplanes.pack_sign_plane(codes)

    assert packed.numel() == -(-shape[0] * shape[1] // 8)
    assert torch.equal(bitplanes.unpack_sign_plane(packed, shape), codes)


def test_the_padding_bits_of_the_last_byte_are_not_weights():
    """They are zero on write, but a decoder must ignore them rather than read `-1`s."""
    codes = torch.ones(1, 3, dtype=torch.int8)

    packed = bitplanes.pack_sign_plane(codes)

    assert packed.tolist() == [0x07]
    assert torch.equal(bitplanes.unpack_sign_plane(packed, (1, 3)), codes)


def test_packing_a_zero_is_refused():
    codes = torch.tensor([[1, 0, -1, 1]], dtype=torch.int8)

    with pytest.raises(ValueError, match="no bit pattern for a zero"):
        bitplanes.pack_sign_plane(codes)


def test_packing_a_non_2d_tensor_is_refused():
    with pytest.raises(ValueError, match="expected a 2D weight"):
        bitplanes.pack_sign_plane(torch.ones(2, 2, 2, dtype=torch.int8))


def test_unpacking_the_wrong_byte_count_is_refused():
    with pytest.raises(ValueError, match="expected 8"):
        bitplanes.unpack_sign_plane(torch.zeros(4, dtype=torch.uint8), (8, 8))


def test_decoding_a_per_row_scale_is_refused():
    packed = bitplanes.pack_sign_plane(torch.ones(4, 8, dtype=torch.int8))

    with pytest.raises(ValueError, match="one scale for the whole tensor"):
        bitplanes.decode_sign_plane(packed, torch.ones(4, 2), (4, 8))


def test_the_decoded_pair_reconstructs_the_master_weight():
    torch.manual_seed(0)
    baked = quant.fake_quant_weight_binary(torch.randn(4, 16))
    codes, scale = quant.baked_binary_codes_and_scale(baked)

    decoded_codes, decoded_scale = bitplanes.decode_sign_plane(
        bitplanes.pack_sign_plane(codes), scale.reshape(1), (4, 16)
    )

    assert torch.equal(
        quant.dequantize_codes(decoded_codes, decoded_scale, torch.float32), baked
    )


# ------------------------------------------------------------------- accounting


def test_one_bit_per_weight_plus_one_scale_for_the_tensor():
    assert bitplanes.payload_bytes("sign_plane", (64, 64)) == 512
    assert bitplanes.bits_per_weight("sign_plane", (64, 64), with_scales=False) == 1.0
    assert bitplanes.scale_bytes("sign_plane", (64, 64)) == 4


def test_the_scale_overhead_vanishes_as_the_tensor_grows():
    """A per-tensor scale is 32 bits total, unlike the two-plane grids' per-row pair."""
    small = bitplanes.bits_per_weight("sign_plane", (8, 8))
    large = bitplanes.bits_per_weight("sign_plane", (4096, 4096))

    assert 1.0 < large < small
    assert large == pytest.approx(1.0, abs=1e-5)


def test_it_is_the_cheapest_container_in_the_family():
    shape = (256, 256)

    assert bitplanes.bits_per_weight("sign_plane", shape) < bitplanes.bits_per_weight(
        "fv5", shape
    )


# ----------------------------------------------------------------------- export


def test_the_export_writes_bits_a_scale_and_a_record(packed, master):
    artifact, manifest = packed
    tensors = bake.load_tensors(artifact)
    master_tensors = bake.load_tensors(master[0])

    for entry in manifest.entries:
        payload = tensors[f"{entry.name}.{SIGN_KEY}"]
        scale = tensors[f"{entry.name}.{SCALE_KEY}"]
        weights = entry.out_features * entry.in_features
        assert payload.dtype is torch.uint8
        assert payload.numel() == -(-weights // 8)
        assert scale.dtype is torch.float32 and scale.shape == (1,)
        assert float(scale) == float(
            master_tensors[f"{entry.name}.weight"].abs().amax()
        )
        assert f"{entry.name}.weight" not in tensors
    # everything the swap left alone is copied through untouched
    assert torch.equal(tensors["lm_head.weight"], master_tensors["lm_head.weight"])


def test_the_record_reports_the_codebook_and_the_measured_bits(packed):
    artifact, manifest = packed

    record = json.loads(
        (artifact / bitplanes.SIGN_PLANE_RECORD_FILENAME).read_text(encoding="utf-8")
    )

    assert record["format"] == "sign_plane"
    assert record["codebook"] == "binary"
    assert record["payload_bits_per_weight"] == 1.0
    assert 1.0 < record["bits_per_weight"] < 1.01
    assert set(record["tensors"]) == {entry.name for entry in manifest.entries}
    for tensor in record["tensors"].values():
        assert tensor["scale_bytes"] == 4
        assert len(tensor["sha256"]) == 64


def test_the_export_stamps_the_artifact_format(packed):
    artifact, _ = packed

    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))

    assert config[bake.QUANTIZER_METADATA_KEY]["format"] == "sign_plane"


def test_the_export_refuses_to_overwrite_the_master(master):
    master_dir, manifest = master

    with pytest.raises(ValueError, match="beside the master"):
        bitplanes.export_sign_plane(master_dir, master_dir, manifest)


def test_a_ternary_master_is_refused_tensor_by_tensor(tmp_path):
    """The zeros are the whole point: they have nowhere to go in this container.

    The manifest is relabelled `binary` so the values themselves have to carry the
    refusal: a mislabelled stamp is the one case the up-front gate cannot catch.
    """
    master_dir = tmp_path / "ternary"
    manifest = _ternary_master(master_dir)
    mislabelled = SwapManifest(
        model_type=manifest.model_type,
        entries=manifest.entries,
        codebook="binary",
        quantizer={**manifest.quantizer, "codebook": "binary"},
    )

    with pytest.raises(ValueError, match="not on the binary grid"):
        bitplanes.export_sign_plane(master_dir, tmp_path / "packed", mislabelled)


def test_a_manifest_that_declares_ternary_is_refused_up_front(master, tmp_path):
    """Once the manifest carries the codebook, the refusal costs no tensor reads."""
    master_dir, manifest = master
    declared = SwapManifest(
        model_type=manifest.model_type,
        entries=manifest.entries,
        quantizer={**manifest.quantizer, "codebook": "ternary"},
    )

    with pytest.raises(ValueError, match="packed form of ternary.codebook: binary"):
        bitplanes.export_sign_plane(master_dir, tmp_path / "declared", declared)


def test_a_per_row_scale_mode_is_refused(master, tmp_path):
    master_dir, manifest = master
    rowwise = SwapManifest(
        model_type=manifest.model_type,
        entries=manifest.entries,
        weight_scale="learnable_row",
        codebook=manifest.codebook,
    )

    with pytest.raises(ValueError, match="one scale per tensor"):
        bitplanes.export_sign_plane(master_dir, tmp_path / "rowwise", rowwise)


def test_a_missing_master_weight_is_reported(master, tmp_path):
    master_dir, manifest = master
    extra = SwapManifest(
        model_type=manifest.model_type,
        entries=[*manifest.entries, _missing_entry(manifest)],
        codebook=manifest.codebook,
    )

    with pytest.raises(KeyError, match="missing from the master"):
        bitplanes.export_sign_plane(master_dir, tmp_path / "missing", extra)


def _missing_entry(manifest: SwapManifest):
    entry = manifest.entries[0]
    return type(entry)(
        name="model.layers.9.mlp.up_proj",
        in_features=entry.in_features,
        out_features=entry.out_features,
        family=entry.family,
        weight_scale=entry.weight_scale,
    )


# ------------------------------------------------------------------------ parity


def test_the_parity_gate_passes_end_to_end(packed, master):
    artifact, manifest = packed

    report = parity.run_parity_gate(master[0], artifact, "sign_plane", manifest)

    assert report.passed
    assert report.tensors_checked == len(manifest.entries)
    assert report.code_mismatches == 0
    assert report.max_dequant_error == 0.0


def test_the_gate_catches_a_flipped_bit(packed, master, tmp_path):
    artifact, manifest = packed
    corrupted = tmp_path / "corrupted"
    corrupted.mkdir()
    for path in bake.shard_paths(artifact):
        tensors, metadata = bake.load_shard(path)
        key = f"{manifest.entries[0].name}.{SIGN_KEY}"
        tensors[key] = tensors[key].clone()
        tensors[key][0] ^= 1
        bake.save_shard(tensors, corrupted / path.name, metadata)

    report = parity.run_parity_gate(master[0], corrupted, "sign_plane", manifest)

    assert not report.passed
    assert "codes differ" in " ".join(report.failures)


def test_the_gate_catches_a_drifted_scale(packed, master, tmp_path):
    artifact, manifest = packed
    corrupted = tmp_path / "drifted"
    corrupted.mkdir()
    for path in bake.shard_paths(artifact):
        tensors, metadata = bake.load_shard(path)
        key = f"{manifest.entries[0].name}.{SCALE_KEY}"
        tensors[key] = tensors[key] * 1.5
        bake.save_shard(tensors, corrupted / path.name, metadata)

    report = parity.run_parity_gate(master[0], corrupted, "sign_plane", manifest)

    assert not report.passed


def test_the_byte_gate_rejects_a_ternary_master_directly():
    torch.manual_seed(0)
    ternary = quant.fake_quant_weight(torch.randn(8, 16))
    codes = torch.where(ternary < 0, -1, 1).to(torch.int8)
    payload = bitplanes.pack_sign_plane(codes)

    failures = parity.gate_bitplane_bytes(
        "sign_plane", payload, torch.ones(1), ternary, "absmean"
    )

    assert any("not on the binary grid" in failure for failure in failures)


# ---------------------------------------------------------------- the export path


def test_run_export_writes_and_gates_the_container(master, tmp_path):
    master_dir, manifest = master
    cfg = DictDefault(
        {
            "output_dir": str(master_dir),
            "ternary": {
                "codebook": "binary",
                "export": {"formats": ["sign_plane"]},
            },
        }
    )

    artifacts = run_export(
        cfg, master_dir=master_dir, output_dir=tmp_path, manifest=manifest
    )

    assert artifacts["sign_plane"] == tmp_path / "sign_plane"
    assert (tmp_path / "sign_plane" / bitplanes.SIGN_PLANE_RECORD_FILENAME).is_file()

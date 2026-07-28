"""CPU-only tests for the mask+sign export format."""

import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.export import _gate_artifact, bake, parity, run_export
from axolotl.integrations.ternary.export.mask_sign import (
    HEADER_BYTES,
    MASK_SIGN_MAGIC,
    MASK_SIGN_VERSION,
    PACKED_SUFFIX,
    RECORD_FILENAME,
    bits_per_weight,
    decode_mask_sign,
    export_mask_sign,
    pack_mask_sign,
    unpack_mask_sign,
)
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import SwapEntry, SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

# 2x5 codes whose every packed byte is asserted below
FIXTURE_CODES = torch.tensor(
    [[1, 0, -1, 1, 0], [0, 0, 1, -1, -1]],
    dtype=torch.int8,
)
FIXTURE_SCALE = torch.tensor(0.5)

# header (32) + zero bitmap (ceil(10/8)) + sign plane (ceil(6/8))
# fmt: off
FIXTURE_BYTES = [
    0x4D, 0x53, 0x47, 0x4E,  # b"MSGN"
    0x01, 0x00, 0x00, 0x00,  # version 1
    0x02, 0x00, 0x00, 0x00,  # rows 2
    0x05, 0x00, 0x00, 0x00,  # cols 5
    0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # 6 nonzeros
    0x00, 0x38,  # f16 0.5
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # header padding
    0x8D,  # zero bitmap, weights 0-7:  1 0 1 1 0 0 0 1 (LSB first)
    0x03,  # zero bitmap, weights 8-9:  1 1
    0x0D,  # sign plane, 6 nonzeros:    1 0 1 1 0 0 (1 = +1)
]
# fmt: on


def _codes(shape, zero_fraction: float, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    draw = torch.rand(shape, generator=generator)
    signs = torch.where(torch.rand(shape, generator=generator) < 0.5, -1, 1)
    return torch.where(draw < zero_fraction, 0, signs).to(torch.int8)


def _write_master(directory, **ternary):
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
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": ternary})
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    return manifest


@pytest.fixture(name="master", scope="module")
def fixture_master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mask_sign_master")
    return directory, _write_master(directory)


# ------------------------------------------------------------- byte-level layout


def test_micro_fixture_packs_to_the_documented_bytes():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE)

    assert packed.dtype == torch.uint8
    assert packed.tolist() == FIXTURE_BYTES


def test_micro_fixture_round_trips():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE)

    codes, scale = decode_mask_sign(packed, (2, 5))

    assert torch.equal(codes, FIXTURE_CODES)
    assert float(scale) == 0.5


def test_header_fields_are_where_the_docstring_says():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE)
    head = bytes(packed[:HEADER_BYTES].tolist())

    assert head[:4] == MASK_SIGN_MAGIC
    assert int.from_bytes(head[4:8], "little") == MASK_SIGN_VERSION
    assert int.from_bytes(head[8:12], "little") == 2
    assert int.from_bytes(head[12:16], "little") == 5
    assert int.from_bytes(head[16:24], "little") == 6


def test_packed_length_is_header_plus_two_planes():
    codes = _codes((16, 32), zero_fraction=0.5)

    packed = pack_mask_sign(codes, torch.tensor(0.25))

    nnz = int((codes != 0).sum())
    assert packed.numel() == HEADER_BYTES + -(-codes.numel() // 8) + -(-nnz // 8)


# --------------------------------------------------------------------- roundtrip


@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (4, 8), (16, 33), (64, 64)])
@pytest.mark.parametrize("zero_fraction", [0.0, 0.3, 0.62, 1.0])
def test_roundtrip_is_exact(shape, zero_fraction):
    codes = _codes(shape, zero_fraction)
    scale = torch.tensor(0.0234375)

    packed = pack_mask_sign(codes, scale)
    unpacked, decoded_scale = decode_mask_sign(packed, shape)

    assert torch.equal(unpacked, codes)
    assert unpacked.dtype == torch.int8
    assert float(decoded_scale) == float(scale)


def test_all_zero_tensor_has_no_sign_plane():
    codes = torch.zeros(8, 8, dtype=torch.int8)

    packed = pack_mask_sign(codes, torch.tensor(1.0))

    assert packed.numel() == HEADER_BYTES + 8
    assert torch.equal(unpack_mask_sign(packed, (8, 8)), codes)


def test_all_nonzero_tensor_fills_both_planes():
    codes = torch.full((8, 8), -1, dtype=torch.int8)

    packed = pack_mask_sign(codes, torch.tensor(1.0))

    assert packed.numel() == HEADER_BYTES + 8 + 8
    assert torch.equal(unpack_mask_sign(packed, (8, 8)), codes)
    # every weight nonzero, every sign negative
    assert packed[HEADER_BYTES : HEADER_BYTES + 8].tolist() == [0xFF] * 8
    assert packed[HEADER_BYTES + 8 :].tolist() == [0x00] * 8


def test_scale_survives_the_f16_slot_exactly():
    # the master's s16 is already f16-rounded, so the slot is lossless for it
    scale = torch.tensor(0.123).to(torch.float16).to(torch.float32)

    packed = pack_mask_sign(FIXTURE_CODES, scale)

    assert float(decode_mask_sign(packed, (2, 5))[1]) == float(scale)


# --------------------------------------------------------------- bit accounting


@pytest.mark.parametrize("zero_fraction", [0.0, 0.25, 0.5, 0.626, 0.9, 1.0])
def test_bits_per_weight_matches_the_measured_payload(zero_fraction):
    codes = _codes((64, 64), zero_fraction)

    packed = pack_mask_sign(codes, torch.tensor(0.5))

    measured = (packed.numel() - HEADER_BYTES) * 8 / codes.numel()
    # the two planes pad to a byte each, so the measurement runs up to 14 bits high
    assert measured == pytest.approx(bits_per_weight(codes), abs=14 / codes.numel())
    assert bits_per_weight(codes) == pytest.approx(
        2.0 - float((codes == 0).to(torch.float32).mean()), abs=1e-9
    )


def test_bits_per_weight_is_bounded_by_the_2_bit_formats():
    dense = torch.full((32, 32), 1, dtype=torch.int8)
    sparse = torch.zeros(32, 32, dtype=torch.int8)

    assert bits_per_weight(dense) == 2.0
    assert bits_per_weight(sparse) == 1.0
    assert bits_per_weight(torch.zeros(0, dtype=torch.int8)) == 0.0


# ----------------------------------------------------------- rejected and broken


def test_pack_rejects_non_ternary_codes():
    with pytest.raises(ValueError, match=r"\{-1, 0, \+1\}"):
        pack_mask_sign(torch.tensor([[2, 0]], dtype=torch.int8), torch.tensor(1.0))


def test_pack_rejects_a_non_2d_weight():
    with pytest.raises(ValueError, match="2D weight"):
        pack_mask_sign(torch.zeros(4, dtype=torch.int8), torch.tensor(1.0))


def test_pack_rejects_a_multi_valued_scale():
    with pytest.raises(ValueError, match="one scale per tensor"):
        pack_mask_sign(FIXTURE_CODES, torch.tensor([1.0, 2.0]))


def test_unpack_rejects_a_foreign_header():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE).clone()
    packed[0] = ord("X")

    with pytest.raises(ValueError, match="not a mask_sign tensor"):
        unpack_mask_sign(packed, (2, 5))


def test_unpack_rejects_a_future_version():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE).clone()
    packed[4] = MASK_SIGN_VERSION + 1

    with pytest.raises(ValueError, match="is not readable by version"):
        unpack_mask_sign(packed, (2, 5))


def test_unpack_rejects_a_shape_disagreement():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE)

    with pytest.raises(ValueError, match="holds a 2x5 weight"):
        unpack_mask_sign(packed, (5, 2))


def test_unpack_rejects_a_truncated_payload():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE)

    with pytest.raises(ValueError, match="bytes, expected"):
        unpack_mask_sign(packed[:-1], (2, 5))

    with pytest.raises(ValueError, match="at least 32 bytes"):
        unpack_mask_sign(packed[:8], (2, 5))


def test_unpack_rejects_a_corrupted_bitmap():
    packed = pack_mask_sign(FIXTURE_CODES, FIXTURE_SCALE).clone()
    packed[HEADER_BYTES] ^= 0b10  # a zero weight becomes nonzero

    with pytest.raises(ValueError, match="the header claims"):
        unpack_mask_sign(packed, (2, 5))


def test_unpack_rejects_a_wrongly_typed_tensor():
    with pytest.raises(ValueError, match="flat uint8"):
        unpack_mask_sign(torch.zeros(64, dtype=torch.int8), (2, 5))


# ------------------------------------------------------------- export pipeline


def test_export_writes_packed_tensors_and_a_record(master, tmp_path):
    master_dir, manifest = master

    artifact = export_mask_sign(master_dir, tmp_path / "mask_sign", manifest)

    tensors = bake.load_tensors(artifact)
    entry = manifest.entries[0]
    assert f"{entry.name}.{PACKED_SUFFIX}" in tensors
    assert f"{entry.name}.weight" not in tensors
    assert tensors[f"{entry.name}.{PACKED_SUFFIX}"].dtype == torch.uint8
    # embeddings, head and norms come through untouched
    assert "model.embed_tokens.weight" in tensors

    record = json.loads((artifact / RECORD_FILENAME).read_text())
    assert record["format"] == "mask_sign"
    assert record["tensor_count"] == len(manifest.entries)
    assert set(record["tensors"]) == {entry.name for entry in manifest.entries}
    assert 1.0 <= record["bits_per_weight"] <= 2.0
    assert record["stored_bits_per_weight"] >= record["bits_per_weight"]


def test_export_record_digests_match_the_written_bytes(master, tmp_path):
    import hashlib

    master_dir, manifest = master

    artifact = export_mask_sign(master_dir, tmp_path / "digest", manifest)

    tensors = bake.load_tensors(artifact)
    record = json.loads((artifact / RECORD_FILENAME).read_text())
    for name, entry_record in record["tensors"].items():
        payload = tensors[f"{name}.{PACKED_SUFFIX}"]
        assert entry_record["bytes"] == payload.numel()
        assert (
            hashlib.sha256(payload.numpy().tobytes()).hexdigest()
            == entry_record["sha256"]
        )


def test_export_stamps_the_quantizer_metadata(master, tmp_path):
    master_dir, manifest = master

    artifact = export_mask_sign(master_dir, tmp_path / "stamped", manifest)

    config = json.loads((artifact / "config.json").read_text())
    stamp = config[bake.QUANTIZER_METADATA_KEY]
    assert stamp["format"] == "mask_sign"
    assert stamp["scheme"] == "ternary"
    assert (artifact / "ternary_manifest.json").is_file()


def test_run_export_writes_and_gates_mask_sign(master, tmp_path):
    master_dir, _ = master

    artifacts = run_export(
        DictDefault(
            {
                "output_dir": str(master_dir),
                "ternary": {"export": {"formats": ["mask_sign"]}},
            }
        ),
        output_dir=tmp_path,
    )

    assert artifacts["mask_sign"] == tmp_path / "mask_sign"
    assert (artifacts["mask_sign"] / "model.safetensors").is_file()


def test_mask_sign_is_gated_by_re_reading_it(master):
    assert "mask_sign" in parity.EXTRACTORS
    assert "mask_sign" not in parity.BLOCK_GATED_FORMATS
    assert "mask_sign" in parity.BLOCK_DECODERS


def test_parity_gate_passes_on_a_fresh_export(master, tmp_path):
    master_dir, manifest = master
    artifact = export_mask_sign(master_dir, tmp_path / "gated", manifest)

    report = parity.run_parity_gate(master_dir, artifact, "mask_sign", manifest)

    assert report.passed
    assert report.code_mismatches == 0
    assert report.tensors_checked == len(manifest.entries)
    assert report.max_dequant_error == 0.0


def test_corrupted_artifact_fails_the_gate(master, tmp_path):
    master_dir, manifest = master
    artifact = export_mask_sign(master_dir, tmp_path / "corrupt", manifest)
    tensors = bake.load_tensors(artifact)
    key = f"{manifest.entries[0].name}.{PACKED_SUFFIX}"
    # flip a sign bit: the popcount check still passes, the codes do not
    tensors[key][-1] ^= 0b1
    bake.save_shard(tensors, artifact / "model.safetensors")

    with pytest.raises(RuntimeError, match="codes differ from the master"):
        _gate_artifact("mask_sign", master_dir, artifact, manifest)


def test_export_refuses_to_overwrite_the_master(master):
    master_dir, manifest = master

    with pytest.raises(ValueError, match="beside the master"):
        export_mask_sign(master_dir, master_dir, manifest)


def test_export_rejects_a_non_per_tensor_scale(master, tmp_path):
    master_dir, manifest = master
    grouped = SwapManifest(
        model_type=manifest.model_type,
        entries=list(manifest.entries),
        weight_scale="group",
        group_size=64,
    )

    with pytest.raises(ValueError, match="cannot represent"):
        export_mask_sign(master_dir, tmp_path / "grouped", grouped)


def test_export_rejects_a_latent_master(tmp_path):
    master_dir = tmp_path / "latent"
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(master_dir), "ternary": {}})
    )
    model.save_pretrained(master_dir)
    manifest.save(master_dir)

    with pytest.raises(ValueError, match="still holds latent weights"):
        export_mask_sign(master_dir, tmp_path / "packed", manifest)


def test_export_reports_a_missing_weight(master, tmp_path):
    master_dir, manifest = master
    extended = SwapManifest(
        model_type=manifest.model_type,
        entries=[
            *manifest.entries,
            SwapEntry(
                name="model.layers.9.mlp.up_proj",
                in_features=64,
                out_features=128,
                family="up_proj",
                weight_scale="absmean",
            ),
        ],
    )

    with pytest.raises(KeyError, match="missing from the master"):
        export_mask_sign(master_dir, tmp_path / "missing", extended)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mask_sign CUDA round-trip")
def test_roundtrip_on_cuda():
    codes = _codes((32, 64), zero_fraction=0.6).cuda()
    scale = torch.tensor(0.5, device="cuda")

    packed = pack_mask_sign(codes, scale)
    unpacked, decoded_scale = decode_mask_sign(packed, (32, 64))

    assert packed.device.type == "cuda"
    assert torch.equal(unpacked, codes)
    assert float(decoded_scale) == 0.5

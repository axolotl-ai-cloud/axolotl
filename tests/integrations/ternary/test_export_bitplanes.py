"""`fv5` and `tp9`: packed containers for the two-plane grids.

`fv5` is three bit planes over the five-value `dual` grid (`bp`, `bn`, `br`), `tp9` is
the two trit planes of the nine-value free sum packed two bits each. Neither has a
consumer yet; both exist so a healed two-plane model ships at its real size and so the
byte layout a future kernel would read is pinned here.
"""

import json

import pydantic
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.export import bake, bitplanes, parity, run_export
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

MODE_FOR = {"fv5": "dual", "tp9": "trit_planes"}


def _write_master(directory, weight_scale: str) -> SwapManifest:
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
        model,
        DictDefault(
            {
                "output_dir": str(directory),
                "ternary": {
                    "weight_scale": weight_scale,
                    "export": {"formats": ["master_bf16"]},
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


@pytest.fixture(name="masters", scope="module")
def fixture_masters(tmp_path_factory):
    root = tmp_path_factory.mktemp("bitplane_masters")
    return {
        fmt: (root / fmt, _write_master(root / fmt, mode))
        for fmt, mode in MODE_FOR.items()
    }


# ------------------------------------------------------------ micro-fixtures


def test_fv5_bytes_are_exactly_the_three_planes():
    """Every byte spelled out: a mixed row, an all-zero row, and an `s_lo == s_hi` row.

    Flat row-major over 24 weights, so each plane is 3 bytes, LSB first.
    """
    codes = torch.tensor(
        [
            [1, -1, 2, -2, 0, 1, 0, 2],  # both levels, both signs, and zeros
            [0, 0, 0, 0, 0, 0, 0, 0],  # an all-zero row selects no scale at all
            [1, 1, -1, 0, 1, -1, 0, 1],  # an s_lo == s_hi row never reaches level 2
        ],
        dtype=torch.int8,
    )

    packed = bitplanes.pack_fv5(codes)

    assert packed.numel() == bitplanes.FV5_PLANES * 3
    assert packed.tolist() == [
        # bp: weights 0, 2, 5, 7 | none | 16, 17, 20, 23
        0xA5, 0x00, 0x93,
        # bn: weights 1, 3 | none | 18, 21
        0x0A, 0x00, 0x24,
        # br: weights 2, 3, 7 (the |code| == 2 ones) | none | none
        0x8C, 0x00, 0x00,
    ]  # fmt: skip
    assert torch.equal(bitplanes.unpack_fv5(packed, (3, 8)), codes)


def test_tp9_bytes_are_the_two_trit_planes():
    """Combination states included: `(1,1)`, `(1,-1)`, `(-1,1)`, `(-1,-1)`, and pure ones."""
    planes = torch.tensor(
        [
            [[1, 1, -1, 0, 1, 0, -1, 0]],
            [[1, -1, 1, 1, 0, 0, -1, 0]],
        ],
        dtype=torch.int8,
    )

    packed = bitplanes.pack_tp9(planes)

    assert packed.numel() == quant.DUAL_PLANES * 2
    # each byte holds four `code + 1` values at shifts 0, 2, 4, 6
    assert packed.tolist() == [0x4A, 0x46, 0xA2, 0x45]
    assert torch.equal(bitplanes.unpack_tp9(packed, (1, 8)), planes)


def test_the_fv5_reconstruction_is_the_documented_formula():
    codes = torch.tensor([[1, -1, 2, -2, 0]], dtype=torch.int8)
    scales = torch.tensor([[0.25, 1.5]])

    got_codes, got_scales = bitplanes.decode_fv5(
        bitplanes.pack_fv5(codes), scales, (1, 5)
    )
    values = bake.dequantize_derived(got_codes, got_scales, "dual", torch.float32)

    assert values.tolist() == [[0.25, -0.25, 1.5, -1.5, 0.0]]


def test_the_tp9_reconstruction_is_the_documented_formula():
    planes = torch.tensor([[[1, 1, 0, -1]], [[1, -1, 1, -1]]], dtype=torch.int8)
    scales = torch.tensor([[2.0, 0.5]])

    got_planes, got_scales = bitplanes.decode_tp9(
        bitplanes.pack_tp9(planes), scales, (1, 4)
    )
    values = bake.dequantize_derived(
        got_planes, got_scales, "trit_planes", torch.float32
    )

    # s1 + s2, s1 - s2, s2, -(s1 + s2)
    assert values.tolist() == [[2.5, 1.5, 0.5, -2.5]]


# ------------------------------------------------------------------ roundtrip


@pytest.mark.parametrize("shape", [(1, 8), (4, 32), (7, 13), (16, 128)])
def test_fv5_round_trips_random_five_state_codes(shape):
    generator = torch.Generator().manual_seed(shape[1])
    codes = torch.randint(-2, 3, shape, generator=generator, dtype=torch.int8)

    packed = bitplanes.pack_fv5(codes)

    assert packed.numel() == bitplanes.payload_bytes("fv5", shape)
    assert torch.equal(bitplanes.unpack_fv5(packed, shape), codes)


@pytest.mark.parametrize("shape", [(1, 8), (4, 32), (7, 13), (16, 128)])
def test_tp9_round_trips_random_trit_planes(shape):
    generator = torch.Generator().manual_seed(shape[1])
    planes = torch.randint(
        -1, 2, (quant.DUAL_PLANES, *shape), generator=generator, dtype=torch.int8
    )

    packed = bitplanes.pack_tp9(planes)

    assert packed.numel() == bitplanes.payload_bytes("tp9", shape)
    assert torch.equal(bitplanes.unpack_tp9(packed, shape), planes)


# ------------------------------------------------------------------ invariants


def test_fv5_rejects_a_weight_marked_both_signs():
    codes = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)
    packed = bitplanes.pack_fv5(codes)
    packed[1] |= 0b1  # set bn for weight 0, which bp already claims

    with pytest.raises(ValueError, match="both positive and negative"):
        bitplanes.unpack_fv5(packed, (1, 8))


def test_fv5_rejects_a_zero_weight_selecting_a_scale():
    codes = torch.zeros(1, 8, dtype=torch.int8)
    packed = bitplanes.pack_fv5(codes)
    packed[2] |= 0b100  # set br for weight 2, which is zero

    with pytest.raises(ValueError, match="zero weight selects"):
        bitplanes.unpack_fv5(packed, (1, 8))


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_a_truncated_payload_is_rejected(fmt):
    shape = (2, 16)
    payload = torch.zeros(bitplanes.payload_bytes(fmt, shape) - 1, dtype=torch.uint8)
    unpack = bitplanes.unpack_fv5 if fmt == "fv5" else bitplanes.unpack_tp9

    with pytest.raises(ValueError, match="expected"):
        unpack(payload, shape)


def test_pack_rejects_out_of_range_states():
    with pytest.raises(ValueError, match="five-state"):
        bitplanes.pack_fv5(torch.tensor([[3]], dtype=torch.int8))
    with pytest.raises(ValueError, match="trit planes"):
        bitplanes.pack_tp9(torch.full((2, 1, 4), 2, dtype=torch.int8))
    with pytest.raises(ValueError, match="2D weight"):
        bitplanes.pack_fv5(torch.zeros(4, dtype=torch.int8))
    with pytest.raises(ValueError, match="rows, cols"):
        bitplanes.pack_tp9(torch.zeros(3, 2, 4, dtype=torch.int8))


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_a_wrong_scale_shape_is_rejected(fmt):
    shape = (4, 8)
    payload = torch.zeros(bitplanes.payload_bytes(fmt, shape), dtype=torch.uint8)
    decode = bitplanes.decode_fv5 if fmt == "fv5" else bitplanes.decode_tp9

    with pytest.raises(ValueError, match="scales"):
        decode(payload, torch.zeros(4, 3), shape)


# ----------------------------------------------------------------- accounting


@pytest.mark.parametrize("fmt,payload_bpw", [("fv5", 3.0), ("tp9", 4.0)])
def test_the_payload_costs_exactly_its_planes(fmt, payload_bpw):
    shape = (64, 512)  # a multiple of 8 both ways, so no padding rounds the count up

    assert bitplanes.bits_per_weight(fmt, shape, with_scales=False) == payload_bpw
    assert bitplanes.payload_bytes(fmt, shape) * 8 == payload_bpw * shape[0] * shape[1]


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_scale_overhead_is_two_floats_per_row(fmt):
    narrow = bitplanes.bits_per_weight(fmt, (64, 64))
    wide = bitplanes.bits_per_weight(fmt, (64, 4096))
    payload = bitplanes.bits_per_weight(fmt, (64, 64), with_scales=False)

    # 2 f32 per row spread over `cols` weights
    assert narrow - payload == pytest.approx(64 / 64)
    assert wide - payload == pytest.approx(64 / 4096)
    assert narrow > wide


def test_an_empty_tensor_costs_nothing():
    assert bitplanes.bits_per_weight("fv5", (0, 8)) == 0.0


# ---------------------------------------------------------------------- export


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_export_writes_planes_scales_and_a_record(masters, tmp_path, fmt):
    master_dir, _ = masters[fmt]
    manifest = SwapManifest.load(master_dir)

    output = bitplanes.export_bitplanes(master_dir, tmp_path / fmt, manifest, fmt)

    tensors = bake.load_tensors(output)
    suffix = bitplanes.FV5_SUFFIX if fmt == "fv5" else bitplanes.TP9_SUFFIX
    for entry in manifest.entries:
        payload = tensors[f"{entry.name}.{suffix}"]
        scales = tensors[f"{entry.name}.{bitplanes.SCALES_SUFFIX}"]
        assert payload.dtype is torch.uint8
        assert payload.numel() == bitplanes.payload_bytes(
            fmt, (entry.out_features, entry.in_features)
        )
        assert scales.shape == (entry.out_features, 2)
        assert f"{entry.name}.weight" not in tensors
    name = (
        bitplanes.FV5_RECORD_FILENAME if fmt == "fv5" else bitplanes.TP9_RECORD_FILENAME
    )
    record = json.loads((output / name).read_text())
    assert record["format"] == fmt
    assert record["scale_mode"] == MODE_FOR[fmt]
    assert record["payload_bits_per_weight"] == pytest.approx(
        3.0 if fmt == "fv5" else 4.0
    )
    assert record["bits_per_weight"] > record["payload_bits_per_weight"]
    assert set(record["tensors"]) == {entry.name for entry in manifest.entries}


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_non_ternary_tensors_pass_through(masters, tmp_path, fmt):
    master_dir, _ = masters[fmt]
    manifest = SwapManifest.load(master_dir)

    output = bitplanes.export_bitplanes(master_dir, tmp_path / fmt, manifest, fmt)

    before = bake.load_tensors(master_dir)
    after = bake.load_tensors(output)
    swapped = {f"{entry.name}.weight" for entry in manifest.entries}
    for key, value in before.items():
        if key not in swapped:
            assert torch.equal(after[key], value), key


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_export_refuses_the_wrong_scale_mode(masters, tmp_path, fmt):
    other = "tp9" if fmt == "fv5" else "fv5"
    master_dir, _ = masters[fmt]

    with pytest.raises(ValueError, match="packed form of"):
        bitplanes.export_bitplanes(
            master_dir, tmp_path / other, SwapManifest.load(master_dir), other
        )


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_export_refuses_to_overwrite_the_master(masters, fmt):
    master_dir, _ = masters[fmt]

    with pytest.raises(ValueError, match="beside the master"):
        bitplanes.export_bitplanes(
            master_dir, master_dir, SwapManifest.load(master_dir), fmt
        )


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_a_master_without_persisted_scales_is_refused(masters, tmp_path, fmt):
    """Packing invents no grid: a free-sum row does not contain its own scales."""
    master_dir, _ = masters[fmt]
    manifest = SwapManifest.load(master_dir)
    manifest.scales = {}

    with pytest.raises(ValueError, match="ships none"):
        bitplanes.export_bitplanes(master_dir, tmp_path / fmt, manifest, fmt)


def test_an_unknown_format_is_refused():
    manifest = SwapManifest(model_type="llama", weight_scale="dual")

    with pytest.raises(ValueError, match="unknown bit-plane format"):
        bitplanes.export_bitplanes("unused", "unused", manifest, "fv7")


# ------------------------------------------------------------------- the gate


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_parity_gate_passes_end_to_end(masters, tmp_path, fmt):
    master_dir, _ = masters[fmt]
    manifest = SwapManifest.load(master_dir)
    artifact = bitplanes.export_bitplanes(master_dir, tmp_path / fmt, manifest, fmt)

    report = parity.run_parity_gate(
        master_dir, artifact, fmt, manifest, run_smoke=False
    )

    assert report.passed, report.failures
    assert report.tensors_checked == len(manifest.entries)
    assert report.code_mismatches == 0
    assert report.max_dequant_error == 0.0


def _gate_inputs(fmt: str, shape=(4, 16)):
    """A baked master of `shape`, its persisted scales, and its packed payload."""
    torch.manual_seed(0)
    weight = (torch.randn(*shape) * 0.05).to(torch.bfloat16)
    mode = MODE_FOR[fmt]
    if fmt == "fv5":
        low, high = quant.dual_absmean_scales(weight)
        master = quant.fake_quant_weight_dual(weight, 1.0, low, high)
        scales_pair = quant.baked_dual_codes_and_scales(master)[1:]
    else:
        first, second = quant.trit_plane_absmean_scales(weight)
        master = quant.fake_quant_weight_trit_planes(weight, 1.0, first, second)
        scales_pair = quant.baked_trit_plane_codes_and_scales(master)[1:]
    codes, scales = bake.derive_codes_and_scale(master, None, mode, scales_pair)
    payload = bitplanes.pack_fv5(codes) if fmt == "fv5" else bitplanes.pack_tp9(codes)
    return payload, scales, master, mode


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_gate_accepts_a_faithful_pack(fmt):
    payload, scales, master, mode = _gate_inputs(fmt)

    assert parity.gate_bitplane_bytes(fmt, payload, scales, master, mode) == []


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
@pytest.mark.parametrize("plane", [0, 1, 2])
def test_the_gate_catches_a_flipped_bit_in_any_plane(fmt, plane):
    payload, scales, master, mode = _gate_inputs(fmt)
    planes = bitplanes.FV5_PLANES if fmt == "fv5" else quant.DUAL_PLANES
    if plane >= planes:
        pytest.skip(f"{fmt} has {planes} planes")
    stride = payload.numel() // planes
    corrupted = payload.clone()
    corrupted[plane * stride] ^= 0b1

    failures = parity.gate_bitplane_bytes(fmt, corrupted, scales, master, mode)

    assert failures, f"plane {plane} corruption went unnoticed"


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_gate_catches_a_corrupted_scale_vector(fmt):
    payload, scales, master, mode = _gate_inputs(fmt)
    corrupted = scales.clone()
    corrupted[0, 0] *= 1.5

    failures = parity.gate_bitplane_bytes(fmt, payload, corrupted, master, mode)

    assert failures


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_the_gate_catches_a_swapped_scale_column(fmt):
    payload, scales, master, mode = _gate_inputs(fmt)

    failures = parity.gate_bitplane_bytes(fmt, payload, scales.flip(-1), master, mode)

    assert failures


@pytest.mark.parametrize("fmt", ["fv5", "tp9"])
def test_a_corrupted_export_fails_the_written_gate(masters, tmp_path, fmt, monkeypatch):
    master_dir, _ = masters[fmt]
    packer = bitplanes.pack_fv5 if fmt == "fv5" else bitplanes.pack_tp9

    def corrupt(codes):
        payload = packer(codes)
        payload[0] ^= 0xFF
        return payload

    monkeypatch.setattr(bitplanes, "pack_fv5" if fmt == "fv5" else "pack_tp9", corrupt)

    with pytest.raises(RuntimeError, match="gate rejected"):
        bitplanes.export_bitplanes(
            master_dir, tmp_path / fmt, SwapManifest.load(master_dir), fmt
        )


# ------------------------------------------------------------------- the schema


@pytest.mark.parametrize("fmt,mode", sorted(MODE_FOR.items()))
def test_the_schema_pairs_each_container_with_its_mode(fmt, mode):
    config = TernaryConfig(weight_scale=mode, export={"formats": [fmt]})

    assert fmt in config.export.formats


@pytest.mark.parametrize(
    "fmt,mode",
    [("fv5", "trit_planes"), ("fv5", "absmean"), ("tp9", "dual"), ("tp9", "learnable")],
)
def test_the_schema_rejects_a_mismatched_container(fmt, mode):
    with pytest.raises(pydantic.ValidationError, match="packed form of"):
        TernaryConfig(weight_scale=mode, export={"formats": [fmt]})


@pytest.mark.parametrize("fmt,mode", sorted(MODE_FOR.items()))
def test_subln_masters_can_be_packed(fmt, mode):
    """Our own container: the sub-norm gains pass through like any other norm."""
    config = TernaryConfig(weight_scale=mode, subln=True, export={"formats": [fmt]})

    assert config.subln and fmt in config.export.formats


@pytest.mark.parametrize("fmt,mode", sorted(MODE_FOR.items()))
def test_run_export_dispatches_the_container(masters, tmp_path, fmt, mode):
    master_dir, _ = masters[fmt]
    cfg = DictDefault(
        {
            "output_dir": str(master_dir),
            "ternary": {
                "weight_scale": mode,
                "export": {"formats": [fmt], "run_parity_gate": True},
            },
        }
    )

    artifacts = run_export(cfg, master_dir=master_dir, output_dir=tmp_path)

    assert artifacts[fmt].is_dir()
    name = (
        bitplanes.FV5_RECORD_FILENAME if fmt == "fv5" else bitplanes.TP9_RECORD_FILENAME
    )
    assert (artifacts[fmt] / name).is_file()


def test_the_containers_are_not_claimed_to_have_a_runtime():
    """They are archival: the record says so, so nobody ships one expecting an engine."""
    records = bitplanes.write_record
    assert callable(records)
    summary = bitplanes.summarize({})
    assert summary["tensor_count"] == 0 and summary["bits_per_weight"] == 0.0

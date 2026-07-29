"""`onebitllms_bf16`: handing a healed master to axolotl's `use_onebitllms` path.

`BitNetLinear` is a QAT layer that re-derives its own absmean grid every forward, so a
checkpoint feeds it a *latent*, not a grid. Storing the master verbatim keeps the codes
but shrinks every scale by the density — the trap these tests exist to pin.
"""

import json

import pydantic
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.export import bake, onebitllms, parity, run_export
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import SwapEntry, SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

FORMAT = "onebitllms_bf16"


def _ternary_tensor(rows: int, cols: int, zero_fraction: float, scale: float = 0.05):
    """A baked master tensor with a chosen zero fraction, exactly."""
    total = rows * cols
    zeros = round(zero_fraction * total)
    codes = torch.ones(total)
    codes[:zeros] = 0.0
    codes[zeros : zeros + (total - zeros) // 2] = -1.0
    codes = codes[torch.randperm(total, generator=torch.Generator().manual_seed(0))]
    values = (codes * scale).reshape(rows, cols)
    return values.to(torch.bfloat16), codes.reshape(rows, cols).to(torch.int8)


def _write_master(directory, **ternary) -> SwapManifest:
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
        model, DictDefault({"output_dir": str(directory), "ternary": ternary})
    )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


@pytest.fixture(name="master", scope="module")
def fixture_master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("onebitllms_master")
    return directory, _write_master(directory)


# ------------------------------------------------------------- inflation math


@pytest.mark.parametrize(
    "zero_fraction,expected",
    [(0.0, 1.0), (0.25, 4 / 3), (1 / 3, 1.5), (0.6, 2.5), (0.75, 4.0)],
)
def test_the_inflation_factor_is_one_over_the_density(zero_fraction, expected):
    rows, cols = 8, 100
    _, codes = _ternary_tensor(rows, cols, zero_fraction)

    got_zero, factor = onebitllms.inflation_factor(codes)

    # the crafted count is rounded to whole weights, so the fraction lands within one
    assert got_zero == pytest.approx(zero_fraction, abs=1 / (rows * cols))
    assert factor == pytest.approx(1.0 / (1.0 - got_zero))
    assert factor == pytest.approx(expected, rel=1e-2)


def test_an_all_zero_tensor_does_not_divide_by_zero():
    codes = torch.zeros(4, 8, dtype=torch.int8)

    zero_fraction, factor = onebitllms.inflation_factor(codes)

    assert (zero_fraction, factor) == (1.0, 1.0)
    inflated, _, _ = onebitllms.inflate_master_weight(
        torch.zeros(4, 8), torch.tensor(0.05), codes
    )
    assert not int(inflated.abs().sum())
    # and their quantizer maps it straight back to zero
    got_codes, _ = onebitllms.onebitllms_quantize(inflated)
    assert not int(got_codes.abs().sum())


def test_an_empty_tensor_is_handled():
    assert onebitllms.inflation_factor(torch.zeros(0, dtype=torch.int8)) == (0.0, 1.0)


@pytest.mark.parametrize("zero_fraction", [0.0, 1 / 3, 0.6])
def test_the_inflated_absmean_is_the_master_scale(zero_fraction):
    """The whole point: `mean|code · s / (1 - z)| = s`, so their scale is `1 / s`."""
    scale = 0.0625  # exact in bf16, so the check is not measuring storage rounding
    master, codes = _ternary_tensor(8, 96, zero_fraction, scale=scale)

    inflated, _, factor = onebitllms.inflate_master_weight(
        master, torch.tensor(scale), codes
    )

    assert float(inflated.float().abs().mean()) == pytest.approx(scale, rel=1e-3)
    assert float(inflated.float().abs().max()) == pytest.approx(
        scale * factor, rel=1e-2
    )


# ------------------------------------------------- their quantizer's fixed point


@pytest.mark.parametrize("zero_fraction", [0.0, 0.2, 1 / 3, 0.6, 0.9])
def test_the_inflated_latent_is_a_fixed_point_of_their_quantizer(zero_fraction):
    master, codes = _ternary_tensor(16, 128, zero_fraction)
    scale = torch.tensor(float(master.float().abs().max()))

    inflated, _, _ = onebitllms.inflate_master_weight(master, scale, codes)
    got_codes, got_scale = onebitllms.onebitllms_quantize(inflated)

    assert torch.equal(got_codes, codes)
    assert float(got_scale) == pytest.approx(float(scale), rel=1e-2)
    dequantized = got_codes.float() * got_scale
    assert float((dequantized - master.float()).abs().max()) <= float(scale) * 1e-2


@pytest.mark.parametrize("seed", range(4))
def test_codes_survive_on_random_ternary_tensors(seed):
    torch.manual_seed(seed)
    weight = (torch.randn(12, 64) * 0.02).to(torch.bfloat16)
    master = quant.fake_quant_weight(weight, 1.0)
    codes, scale = quant.baked_codes_and_scale(master)

    inflated, _, _ = onebitllms.inflate_master_weight(master, scale, codes)

    assert torch.equal(onebitllms.onebitllms_quantize(inflated)[0], codes)


def test_the_raw_master_shrinks_by_the_density():
    """The failure mode: their quantizer keeps the codes and loses the scale."""
    master, codes = _ternary_tensor(16, 128, 1 / 3)
    scale = float(master.float().abs().max())

    got_codes, got_scale = onebitllms.onebitllms_quantize(master)

    assert torch.equal(got_codes, codes)
    assert float(got_scale) == pytest.approx(scale * (2 / 3), rel=1e-2)


# --------------------------------------------------------------- the export dir


def test_the_export_writes_a_loadable_checkpoint(master, tmp_path):
    master_dir, manifest = master

    output = onebitllms.export_onebitllms(master_dir, tmp_path / FORMAT, manifest)

    assert (output / "config.json").is_file()
    assert (output / "model.safetensors").is_file()
    tensors = bake.load_tensors(output)
    for entry in manifest.entries:
        stored = tensors[f"{entry.name}.weight"]
        assert stored.dtype is torch.bfloat16
        master_weight = bake.load_tensors(master_dir)[f"{entry.name}.weight"]
        # the latent is the master scaled up, never the master itself
        assert not torch.equal(stored, master_weight)
        codes, scale = quant.baked_codes_and_scale(master_weight)
        got_codes, got_scale = onebitllms.onebitllms_quantize(stored)
        assert torch.equal(got_codes, codes)
        assert float(got_scale) == pytest.approx(float(scale), rel=1e-2)


def test_non_ternary_tensors_pass_through(master, tmp_path):
    master_dir, manifest = master

    output = onebitllms.export_onebitllms(master_dir, tmp_path / FORMAT, manifest)

    before = bake.load_tensors(master_dir)
    after = bake.load_tensors(output)
    swapped = {f"{entry.name}.weight" for entry in manifest.entries}
    assert swapped
    for key, value in before.items():
        if key not in swapped:
            assert torch.equal(after[key], value), key


def test_the_record_lists_every_inflation_factor(master, tmp_path):
    master_dir, manifest = master

    output = onebitllms.export_onebitllms(master_dir, tmp_path / FORMAT, manifest)

    record = json.loads((output / onebitllms.RECORD_FILENAME).read_text())
    assert record["variant"] == FORMAT
    assert set(record["tensors"]) == {entry.name for entry in manifest.entries}
    for stats in record["tensors"].values():
        assert stats["inflation"] == pytest.approx(1 / (1 - stats["zero_fraction"]))
    stamp = json.loads((output / "config.json").read_text())["ternary_quantizer"]
    assert stamp["format"] == FORMAT


def test_the_export_refuses_to_overwrite_the_master(master):
    master_dir, manifest = master

    with pytest.raises(ValueError, match="beside the master"):
        onebitllms.export_onebitllms(master_dir, master_dir, manifest)


def test_a_latent_master_is_refused(tmp_path):
    directory = tmp_path / "latent"
    manifest = _write_master(directory)
    tensors = bake.load_tensors(directory)
    key = f"{manifest.entries[0].name}.weight"
    tensors[key] = torch.randn_like(tensors[key])
    bake.save_shard(tensors, directory / "model.safetensors")

    with pytest.raises(ValueError, match="still holds latent weights"):
        onebitllms.export_onebitllms(directory, tmp_path / FORMAT, manifest)


@pytest.mark.parametrize(
    "weight_scale,group_size",
    [("group", 32), ("learnable_row", None), ("dual", None), ("trit_planes", None)],
)
def test_multi_scale_modes_are_refused(weight_scale, group_size):
    manifest = SwapManifest(
        model_type="llama",
        entries=[SwapEntry("proj", 64, 8, "proj", weight_scale, group_size)],
        weight_scale=weight_scale,
        group_size=group_size,
    )

    with pytest.raises(ValueError, match="cannot represent"):
        onebitllms.export_onebitllms("unused", "unused", manifest)


def test_subln_is_refused():
    manifest = SwapManifest(model_type="llama", weight_scale="absmean", subln=True)

    with pytest.raises(ValueError, match="subln"):
        onebitllms.export_onebitllms("unused", "unused", manifest)


def test_kept_fp_linears_are_warned_about(caplog):
    manifest = SwapManifest(
        model_type="llama",
        weight_scale="absmean",
        kept_fp=["lm_head", "model.embed_tokens", "model.layers.0.mlp.gate"],
    )

    with caplog.at_level("WARNING"):
        onebitllms._warn_about_extra_quantization(manifest)

    assert "will still be quantized" in caplog.text
    assert "model.layers.0.mlp.gate" in caplog.text


# ---------------------------------------------------------------- the gate


def test_the_parity_gate_passes_on_a_faithful_export(master, tmp_path):
    master_dir, manifest = master
    artifact = onebitllms.export_onebitllms(master_dir, tmp_path / FORMAT, manifest)

    report = parity.run_parity_gate(master_dir, artifact, FORMAT, manifest)

    assert report.passed, report.failures
    assert report.code_mismatches == 0
    assert report.tensors_checked == len(manifest.entries)
    assert report.max_dequant_error <= report.dequant_error_bound


def test_the_gate_fails_on_the_raw_master(master, tmp_path):
    """The trap: shipping the master itself keeps the codes and loses 33% of the scale."""
    master_dir, manifest = master
    copied = tmp_path / "uninflated"
    bake.copy_aux_files(master_dir, copied)
    tensors = bake.load_tensors(master_dir)
    bake.save_shard(tensors, copied / "model.safetensors")

    report = parity.run_parity_gate(master_dir, copied, FORMAT, manifest)

    assert not report.passed
    assert report.code_mismatches == 0, "the codes survive — only the scale is lost"
    assert report.max_dequant_error > report.dequant_error_bound
    assert any("dequant error" in failure for failure in report.failures)


def test_the_gate_uses_a_latent_relative_bound():
    assert FORMAT in parity.LATENT_SCALE_FORMATS
    assert parity._dequant_rtol(FORMAT) == parity.LATENT_SCALE_RTOL
    assert parity._dequant_rtol("master_bf16") == parity.F16_HALF_ULP


def test_run_export_writes_and_gates_the_format(master, tmp_path):
    master_dir, _ = master
    cfg = DictDefault(
        {
            "output_dir": str(master_dir),
            "ternary": {"export": {"formats": [FORMAT], "run_parity_gate": True}},
        }
    )

    artifacts = run_export(cfg, master_dir=master_dir, output_dir=tmp_path)

    assert artifacts[FORMAT].is_dir()
    assert (artifacts[FORMAT] / onebitllms.RECORD_FILENAME).is_file()


# ---------------------------------------------------------------- the schema


def test_the_schema_accepts_the_format_for_per_tensor_modes():
    for weight_scale in sorted(onebitllms.SUPPORTED_SCALE_MODES):
        config = TernaryConfig(weight_scale=weight_scale, export={"formats": [FORMAT]})
        assert FORMAT in config.export.formats


@pytest.mark.parametrize(
    "weight_scale,group_size",
    [("group", 32), ("learnable_row", None), ("dual", None), ("trit_planes", None)],
)
def test_the_schema_rejects_multi_scale_modes(weight_scale, group_size):
    with pytest.raises(pydantic.ValidationError, match="cannot be represented"):
        TernaryConfig(
            weight_scale=weight_scale,
            group_size=group_size,
            export={"formats": [FORMAT]},
        )


def test_the_schema_rejects_subln():
    with pytest.raises(pydantic.ValidationError, match="subln"):
        TernaryConfig(subln=True, export={"formats": [FORMAT]})


# ------------------------------------------------------- the real package (GPU)


requires_onebitllms = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="onebitllms' BitNetLinear needs CUDA and triton",
)


@requires_onebitllms
def test_the_real_bitnet_linear_reproduces_the_master_function(master, tmp_path):
    """Load the export the way `use_onebitllms` does and compare against the master.

    `replace_linear_with_bitnet_linear` swaps every `nn.Linear` but `lm_head`, so the
    reference is the master run with its own ternary quantizer over the same set.
    """
    pytest.importorskip("onebitllms")
    from onebitllms import replace_linear_with_bitnet_linear

    master_dir, manifest = master
    artifact = onebitllms.export_onebitllms(master_dir, tmp_path / FORMAT, manifest)

    reference = LlamaForCausalLM.from_pretrained(master_dir, dtype=torch.bfloat16)
    swapped = LlamaForCausalLM.from_pretrained(artifact, dtype=torch.bfloat16).cuda()
    replace_linear_with_bitnet_linear(swapped)

    ids = torch.arange(8).unsqueeze(0) % 64
    with torch.no_grad():
        want = reference(input_ids=ids).logits.float()
        got = swapped(input_ids=ids.cuda()).logits.float().cpu()

    # the master already *is* its own ternary grid, so their quantizer is the identity
    # on it — any gap here is the inflation failing to survive their re-derivation
    assert float((got - want).abs().max()) < 0.05 * float(want.abs().max())

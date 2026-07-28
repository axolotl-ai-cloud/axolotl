"""CPU-only tests for the seams between the wave-2 features.

Each feature is tested on its own elsewhere; what is pinned here is the wiring
between them — the PTQ init writing scales a row/dual module can read back, the
export pipeline reading a row/dual master on its own grid, and the sub-norm guards
firing from inside `convert_model` and `run_export` rather than only when a caller
remembers to invoke them.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import TernaryPlugin, quant
from axolotl.integrations.ternary.args import (
    SUBLN_UNSUPPORTED_FORMATS,
    TernaryConfig,
    TernaryDistillConfig,
)
from axolotl.integrations.ternary.export import bake, run_export
from axolotl.integrations.ternary.export.gguf_tq import _check_per_tensor_scale
from axolotl.integrations.ternary.export.hf_bitnet import export_hf_bitnet
from axolotl.integrations.ternary.modules import SUBLN_ATTR, iter_ternary_modules
from axolotl.integrations.ternary.ptq.ternary_fit import dequantize_fit, fit_ternary
from axolotl.integrations.ternary.subln import (
    SUBLN_UNSUPPORTED_FORMATS as SUBLN_REASONS,
)
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

SCALE_MODES = ["absmean", "learnable", "group", "learnable_row", "dual"]
# a fitting init only lands in a mode that keeps the scale it solved; the statistic
# modes re-derive absmean from the ternary latent and throw it away (see args.py)
SCALE_KEEPING_MODES = ["learnable", "learnable_row", "dual"]


def _tiny_llama(dtype: torch.dtype = torch.float32) -> LlamaForCausalLM:
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


def _cfg(weight_scale: str = "absmean", **ternary) -> DictDefault:
    ternary["weight_scale"] = weight_scale
    if weight_scale == "group":
        ternary.setdefault("group_size", 32)
    if weight_scale != "absmean" or ternary.get("subln"):
        ternary.setdefault("export", {"formats": ["master_bf16"]})
    return DictDefault({"ternary": ternary})


def _master(directory, weight_scale: str = "absmean", **ternary) -> SwapManifest:
    model = _tiny_llama(torch.bfloat16)
    cfg = _cfg(weight_scale, **ternary)
    cfg["output_dir"] = str(directory)
    manifest = convert_model(model, cfg)
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    return manifest


# ------------------------------------------------- init x scale-mode (builder A x B)


@pytest.mark.parametrize("weight_scale", SCALE_KEEPING_MODES)
def test_ternary_fit_init_writes_scales_every_mode_can_read_back(weight_scale):
    """`write_latent` has to reach the dual pair, which no `scale.reshape` can."""
    model = _tiny_llama()

    convert_model(model, _cfg(weight_scale, init="ternary_fit"))

    for _, module in iter_ternary_modules(model):
        assert module.is_baked()
        # the module's own quantizer reproduces the fitted latent exactly
        assert torch.equal(module.baked_weight(), module.weight.detach())


@pytest.mark.parametrize("weight_scale", SCALE_KEEPING_MODES)
def test_ternary_fit_init_lands_on_the_solvers_own_reconstruction(weight_scale):
    model = _tiny_llama()
    group_size = 32 if weight_scale == "group" else None
    name, linear = next(
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("layers.0.self_attn.q_proj")
    )
    codes, scale = fit_ternary(
        linear.weight.detach(), scale_mode=weight_scale, group_size=group_size
    )
    expected = dequantize_fit(codes, scale, linear.weight.dtype)

    convert_model(model, _cfg(weight_scale, init="ternary_fit"))

    assert torch.equal(model.get_submodule(name).weight.detach(), expected)


@pytest.mark.parametrize("weight_scale", SCALE_KEEPING_MODES)
def test_the_init_is_idempotent_across_a_reconversion(weight_scale):
    model = _tiny_llama()
    convert_model(model, _cfg(weight_scale, init="ternary_fit"))
    fitted = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(model)
    }

    reloaded = _tiny_llama()
    convert_model(reloaded, _cfg(weight_scale, init="ternary_fit"))
    for name, module in iter_ternary_modules(reloaded):
        module.load_state_dict({"weight": fitted[name]}, strict=False)
        module._detect_baked()
        module.refresh_scale_from_weight()
        assert torch.equal(module.baked_weight(), fitted[name])


# ------------------------------------------------------- bake x scale-mode (builder B)


@pytest.mark.parametrize("weight_scale", SCALE_MODES)
def test_baking_a_master_on_its_own_grid_is_a_no_op(weight_scale, tmp_path):
    manifest = _master(tmp_path / weight_scale, weight_scale)
    before = bake.load_tensors(tmp_path / weight_scale)

    bake.bake_directory(tmp_path / weight_scale, manifest=manifest)

    after = bake.load_tensors(tmp_path / weight_scale)
    for entry in manifest.entries:
        key = f"{entry.name}.weight"
        assert torch.equal(after[key], before[key]), key


@pytest.mark.parametrize("weight_scale", ["learnable_row", "dual"])
def test_a_per_tensor_read_of_a_row_master_would_have_collapsed_it(
    weight_scale, tmp_path
):
    """Pins the corruption the `weight_scale` dispatch exists to prevent."""
    manifest = _master(tmp_path / weight_scale, weight_scale)
    weight = bake.load_tensors(tmp_path / weight_scale)[
        f"{manifest.entries[0].name}.weight"
    ]

    magnitudes = len(torch.unique(weight.abs()))
    collapsed = bake.bake_weight(weight)

    assert magnitudes > 2
    assert len(torch.unique(collapsed.abs())) == 2
    assert torch.equal(bake.bake_weight(weight, None, weight_scale), weight)


@pytest.mark.parametrize("weight_scale", SCALE_MODES)
def test_derive_codes_and_scale_round_trips_a_master(weight_scale, tmp_path):
    manifest = _master(tmp_path / weight_scale, weight_scale)
    entry = manifest.entries[0]
    weight = bake.load_tensors(tmp_path / weight_scale)[f"{entry.name}.weight"]

    codes, scale = bake.derive_codes_and_scale(
        weight, entry.group_size, entry.weight_scale
    )
    dequantized = bake.dequantize_derived(codes, scale, entry.weight_scale)

    assert torch.equal(dequantized.to(weight.dtype), weight)
    if weight_scale == "dual":
        assert scale.shape == (entry.out_features, 2)
        assert int(codes.abs().amax()) <= 2


# ---------------------------------------------------- export pipeline x scale-mode


@pytest.mark.parametrize("weight_scale", SCALE_MODES)
def test_master_export_and_its_parity_gate_accept_every_scale_mode(
    weight_scale, tmp_path
):
    directory = tmp_path / weight_scale
    _master(directory, weight_scale)

    artifacts = run_export(
        DictDefault({"output_dir": str(directory)}), formats=["master_bf16"]
    )

    assert artifacts["master_bf16"] == directory


@pytest.mark.parametrize("weight_scale", ["learnable_row", "dual"])
def test_the_packers_refuse_a_non_per_tensor_manifest(weight_scale, tmp_path):
    manifest = _master(tmp_path / weight_scale, weight_scale)

    with pytest.raises(ValueError, match="cannot represent"):
        export_hf_bitnet(tmp_path / weight_scale, tmp_path / "packed", manifest)
    with pytest.raises(ValueError, match="cannot be represented"):
        _check_per_tensor_scale(manifest)


def test_the_dual_manifest_stamps_a_five_state_codebook():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg("dual"))

    assert manifest.quantizer["codes"] == [-2, -1, 0, 1, 2]
    assert manifest.quantizer["scheme"] == "dual_ternary"


def test_a_ternary_manifest_still_stamps_three_codes():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg("absmean"))

    assert manifest.quantizer["codes"] == [-1, 0, 1]
    assert manifest.quantizer["scheme"] == "ternary"


# ------------------------------------------------------------- subln seams (builder D)


def test_convert_model_refuses_a_subln_master_when_subln_is_off(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    convert_model(
        model, DictDefault({**_cfg("absmean", subln=True), "output_dir": str(master)})
    )
    model.save_pretrained(master)
    reloaded = LlamaForCausalLM.from_pretrained(master)

    with pytest.raises(ValueError, match="ternary.subln: true"):
        convert_model(reloaded, _cfg("absmean"))


@pytest.mark.parametrize("weight_scale", ["dual", "learnable_row", "group"])
def test_convert_model_refuses_a_master_read_through_the_wrong_grid(
    weight_scale, tmp_path
):
    """The baked probe just misses on the wrong grid; nothing else would say so."""
    directory = tmp_path / weight_scale
    _master(directory, weight_scale)
    reloaded = LlamaForCausalLM.from_pretrained(directory)

    with pytest.raises(ValueError, match="baked on the"):
        convert_model(
            reloaded, DictDefault({**_cfg("absmean"), "base_model": str(directory)})
        )


@pytest.mark.parametrize("weight_scale", ["dual", "learnable_row"])
def test_a_master_reloaded_on_its_own_grid_is_still_baked(weight_scale, tmp_path):
    directory = tmp_path / weight_scale
    _master(directory, weight_scale)
    reloaded = LlamaForCausalLM.from_pretrained(directory)

    convert_model(
        reloaded,
        DictDefault({**_cfg(weight_scale), "base_model": str(directory)}),
    )

    assert all(module.is_baked() for _, module in iter_ternary_modules(reloaded))


def test_the_grid_guard_ignores_a_group_size_the_master_agrees_on(tmp_path):
    directory = tmp_path / "group"
    _master(directory, "group")
    reloaded = LlamaForCausalLM.from_pretrained(directory)

    convert_model(
        reloaded, DictDefault({**_cfg("group"), "base_model": str(directory)})
    )

    with pytest.raises(ValueError, match="ternary.group_size: 32"):
        convert_model(
            LlamaForCausalLM.from_pretrained(directory),
            DictDefault(
                {
                    **_cfg("group", group_size=64),
                    "base_model": str(directory),
                }
            ),
        )


def test_run_export_refuses_a_format_that_would_drop_the_gains(tmp_path):
    directory = tmp_path / "master"
    manifest = _master(directory, "absmean", subln=True)
    assert manifest.subln_modules

    with pytest.raises(ValueError, match="cannot carry them"):
        run_export(DictDefault({"output_dir": str(directory)}), formats=["hf_bitnet"])


def test_run_export_still_writes_the_master_for_a_subln_manifest(tmp_path):
    directory = tmp_path / "master"
    _master(directory, "absmean", subln=True)

    artifacts = run_export(
        DictDefault({"output_dir": str(directory)}), formats=["master_bf16"]
    )

    assert artifacts["master_bf16"] == directory
    assert any(
        key.endswith(f"{SUBLN_ATTR}.weight") for key in bake.load_tensors(directory)
    )


def test_the_schema_and_the_export_guard_refuse_the_same_formats():
    """A format the schema lets through would only fail after training."""
    assert set(SUBLN_REASONS) == set(SUBLN_UNSUPPORTED_FORMATS)


# ------------------------------------------------------- anchored distill (builder C)


def test_get_training_args_forwards_the_anchored_schedule():
    plugin = TernaryPlugin()
    cfg = DictDefault(
        {
            "base_model": "meta-llama/Llama-3.2-1B",
            "ternary": {
                "distill": {
                    "mode": "inprocess",
                    "schedule": "anchored",
                    "anchor_start": 0.8,
                }
            },
        }
    )

    args = plugin.get_training_args(cfg)

    assert args["ternary_distill_schedule"] == "anchored"
    assert args["ternary_distill_anchor_start"] == 0.8


def test_the_forwarded_keys_are_fields_of_the_training_args_mixin():
    from axolotl.integrations.ternary.distill import TernaryDistillTrainingArgsMixin

    plugin = TernaryPlugin()
    cfg = DictDefault(
        {
            "base_model": "meta-llama/Llama-3.2-1B",
            "ternary": {"distill": {"mode": "inprocess"}},
        }
    )

    assert set(plugin.get_training_args(cfg)) <= set(
        TernaryDistillTrainingArgsMixin.__dataclass_fields__
    )


def test_the_anchored_schedule_warns_where_only_kd_plugin_would_read_it(caplog):
    """The schema accepts the pair; only the in-process trainer honours the anchor."""
    plugin = TernaryPlugin()
    cfg = {
        "plugins": ["axolotl.integrations.kd.KDPlugin"],
        "save_only_model": True,
        "ternary": {"distill": {"mode": "kd_plugin", "schedule": "anchored"}},
    }

    with caplog.at_level("WARNING"):
        plugin.register(cfg)

    assert any("schedule: anchored" in record.message for record in caplog.records)


def test_the_default_schedule_is_forwarded_unchanged():
    plugin = TernaryPlugin()
    cfg = DictDefault(
        {"base_model": "m", "ternary": {"distill": {"mode": "inprocess"}}}
    )

    assert plugin.get_training_args(cfg)["ternary_distill_schedule"] == (
        TernaryDistillConfig().schedule
    )


# --------------------------------------------------------------- config-level guards


@pytest.mark.parametrize("weight_scale", ["learnable_row", "dual"])
def test_the_schema_keeps_row_modes_off_the_packed_formats(weight_scale):
    with pytest.raises(ValueError, match="cannot be represented"):
        TernaryConfig(weight_scale=weight_scale)


def test_a_dual_run_still_bakes_to_a_five_value_grid(tmp_path):
    manifest = _master(tmp_path / "dual", "dual")
    weights = bake.load_tensors(tmp_path / "dual")

    for entry in manifest.entries:
        weight = weights[f"{entry.name}.weight"]
        assert quant.baked_dual_codes_and_scales(weight) is not None

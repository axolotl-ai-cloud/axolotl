"""CPU-only tests for BitDistill-style sub-norm insertion."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, Trainer

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.modules import SUBLN_ATTR, iter_ternary_modules
from axolotl.integrations.ternary.subln import (
    SUBLN_CONFIG_KEY,
    SUBLN_EPS,
    SUBLN_FAMILIES,
    SUBLN_UNSUPPORTED_FORMATS,
    TernarySubNorm,
    assert_subln_exportable,
    assert_subln_reloadable,
    has_subln_marker,
    insert_subln,
)
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

LAYERS = 2
HIDDEN = 64
INTERMEDIATE = 128


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _cfg(**ternary) -> DictDefault:
    ternary.setdefault("subln", True)
    ternary.setdefault("export", {"formats": ["master_bf16"]})
    return DictDefault({"ternary": ternary})


def _target_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "layers" in name
    ]


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + SUBLN_EPS)


def _manifest(subln: bool = True) -> SwapManifest:
    return SwapManifest(
        model_type="llama",
        subln=subln,
        subln_modules=[f"model.layers.0.self_attn.o_proj.{SUBLN_ATTR}"],
    )


def test_insertion_count_matches_the_family_counts():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert manifest.subln is True
    assert manifest.subln_modules == [
        f"model.layers.{layer}.{parent}.{SUBLN_ATTR}"
        for layer in range(LAYERS)
        for parent in ("self_attn.o_proj", "mlp.down_proj")
    ]
    assert len(manifest.subln_modules) == LAYERS * len(SUBLN_FAMILIES)


def test_only_the_two_families_carry_a_norm():
    model = _tiny_llama()

    convert_model(model, _cfg())

    widths = {
        name: None if module.sub_norm is None else module.sub_norm.num_features
        for name, module in iter_ternary_modules(model)
    }
    assert widths["model.layers.0.self_attn.o_proj"] == HIDDEN
    assert widths["model.layers.0.mlp.down_proj"] == INTERMEDIATE
    assert all(
        width is None
        for name, width in widths.items()
        if name.rpartition(".")[2] not in SUBLN_FAMILIES
    )


def test_the_norm_is_a_parameter_under_the_linear_scope():
    model = _tiny_llama()

    convert_model(model, _cfg())

    state = model.state_dict()
    assert f"model.layers.0.self_attn.o_proj.{SUBLN_ATTR}.weight" in state
    assert f"model.layers.0.self_attn.q_proj.{SUBLN_ATTR}.weight" not in state
    assert torch.equal(
        state[f"model.layers.0.mlp.down_proj.{SUBLN_ATTR}.weight"],
        torch.ones(INTERMEDIATE),
    )


def test_forward_changes_only_the_targeted_modules_inputs():
    model = _tiny_llama()
    convert_model(model, _cfg(fused_fake_quant=False))
    modules = dict(iter_ternary_modules(model))
    for module in modules.values():
        module.set_lambda(0.0)
    x = 4.0 * torch.randn(3, HIDDEN)

    untouched = modules["model.layers.0.self_attn.q_proj"]
    normalized = modules["model.layers.0.self_attn.o_proj"]

    torch.testing.assert_close(untouched(x), F.linear(x, untouched.weight))
    torch.testing.assert_close(
        normalized(x), F.linear(_rms(x), normalized.weight), rtol=1e-5, atol=1e-5
    )


def test_unit_gain_is_not_the_identity_at_lambda_zero():
    """The documented behavior: an RMSNorm at unit gain still rescales every token."""
    model = _tiny_llama()
    convert_model(model, _cfg(fused_fake_quant=False))
    module = dict(iter_ternary_modules(model))["model.layers.0.self_attn.o_proj"]
    module.set_lambda(0.0)
    x = 4.0 * torch.randn(3, HIDDEN)

    unquantized = F.linear(x, module.weight)

    assert not torch.allclose(module(x), unquantized, rtol=1e-2, atol=1e-2)
    # the change is exactly a per-token gain of 1 / rms(x) on the projection's output
    gain = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + SUBLN_EPS)
    torch.testing.assert_close(module(x), gain * unquantized, rtol=1e-5, atol=1e-5)


def test_a_unit_rms_input_is_the_only_fixed_point():
    module = TernarySubNorm(HIDDEN)
    x = torch.randn(3, HIDDEN)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))

    torch.testing.assert_close(module(x), x, rtol=1e-4, atol=1e-4)


def test_the_norm_runs_before_the_activation_quantizer():
    model = _tiny_llama()
    convert_model(model, _cfg(fused_fake_quant=False, activation_bits=8))
    module = dict(iter_ternary_modules(model))["model.layers.0.mlp.down_proj"]
    x = 4.0 * torch.randn(3, INTERMEDIATE)

    expected = F.linear(
        quant.act_quant(_rms(x)), quant.fake_quant_weight(module.weight.detach(), 1.0)
    )

    torch.testing.assert_close(module(x), expected)


def test_the_gain_trains_through_the_quantized_forward():
    model = _tiny_llama()
    convert_model(model, _cfg(fused_fake_quant=False))
    ids = torch.randint(0, 128, (2, 8))

    model(input_ids=ids, labels=ids).loss.backward()

    gain = model.get_submodule(f"model.layers.0.mlp.down_proj.{SUBLN_ATTR}").weight
    assert gain.grad is not None
    assert gain.grad.abs().sum() > 0


def test_insert_is_idempotent():
    model = _tiny_llama()
    names = _target_names(model)

    first = insert_subln(model, _cfg(), names)
    parked = model.get_submodule(first[0])
    second = insert_subln(model, _cfg(), names)

    assert first == second
    assert model.get_submodule(first[0]) is parked
    assert sum("sub_norm" in key for key in model.state_dict()) == len(first)


def test_insert_rejects_a_target_that_is_not_a_linear():
    model = _tiny_llama()

    with pytest.raises(ValueError, match="LlamaMLP"):
        insert_subln(model, _cfg(), ["model.layers.0.mlp"])

    with pytest.raises(ValueError, match="not a module"):
        insert_subln(model, _cfg(), ["model.layers.0.mlp.w2"])


def test_insert_requires_every_family_to_match():
    model = _tiny_llama()
    attention = [name for name in _target_names(model) if "self_attn" in name]

    with pytest.raises(ValueError, match="down_proj"):
        insert_subln(model, _cfg(), [n for n in attention if "o_proj" in n])

    with pytest.raises(ValueError, match=r"\['o_proj', 'down_proj'\]"):
        insert_subln(model, _cfg(), [n for n in attention if "o_proj" not in n])


def test_insert_refuses_a_config_with_subln_off():
    model = _tiny_llama()

    with pytest.raises(ValueError, match="ternary.subln: true"):
        insert_subln(model, _cfg(subln=False), _target_names(model))


def test_insertion_stamps_the_model_config():
    model = _tiny_llama()
    assert has_subln_marker(model.config) is False

    convert_model(model, _cfg())

    assert has_subln_marker(model.config) is True


def test_the_gain_is_kept_out_of_the_weight_decay_group():
    """The `_norm` in the attribute name is what excludes it; a rename would decay it."""
    model = _tiny_llama()
    convert_model(model, _cfg())

    decayed = Trainer.get_decay_parameter_names(None, model)

    assert not [name for name in decayed if SUBLN_ATTR in name]
    assert "model.layers.0.self_attn.o_proj.weight" in decayed


def test_master_roundtrip_through_the_plugin_preserves_the_gains(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    convert_model(model, DictDefault({**_cfg(), "output_dir": str(master)}))
    with torch.no_grad():
        for _, module in iter_ternary_modules(model):
            if module.sub_norm is not None:
                module.sub_norm.weight.normal_(mean=1.0, std=0.2)
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    model.save_pretrained(master)

    stock = LlamaForCausalLM.from_pretrained(master)
    # verified against transformers: the gains come back as UNEXPECTED keys and are
    # dropped, which is why the plugin has to restore them itself
    assert not [key for key in stock.state_dict() if SUBLN_ATTR in key]
    assert has_subln_marker(stock.config) is True

    reloaded = LlamaForCausalLM.from_pretrained(master)
    manifest = convert_model(
        reloaded, DictDefault({**_cfg(), "base_model": str(master)})
    )

    assert manifest.subln_modules
    restored = reloaded.state_dict()
    assert set(restored) == set(saved)
    for key, value in saved.items():
        assert torch.equal(restored[key], value), key


def test_a_reloaded_subln_master_stays_a_bake_fixed_point(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    convert_model(model, DictDefault({**_cfg(), "output_dir": str(master)}))
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(master)
    saved = {key: value.clone() for key, value in model.state_dict().items()}

    reloaded = LlamaForCausalLM.from_pretrained(master)
    convert_model(reloaded, DictDefault({**_cfg(), "base_model": str(master)}))
    for name, module in iter_ternary_modules(reloaded):
        assert module.is_baked()
        module._post_training(reloaded, name)

    for key, value in reloaded.state_dict().items():
        assert torch.equal(value, saved[key]), key


def test_reloading_a_marked_master_without_subln_is_refused(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    convert_model(model, DictDefault({**_cfg(), "output_dir": str(master)}))
    model.save_pretrained(master)
    reloaded = LlamaForCausalLM.from_pretrained(master)

    assert_subln_reloadable(reloaded, _cfg())
    with pytest.raises(ValueError, match="dropped them as unexpected keys"):
        assert_subln_reloadable(reloaded, _cfg(subln=False))


def test_restoring_the_gains_needs_a_local_master(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    convert_model(model, DictDefault({**_cfg(), "output_dir": str(master)}))
    model.save_pretrained(master)
    reloaded = LlamaForCausalLM.from_pretrained(master)
    reloaded.config._name_or_path = "some-org/some-master"

    with pytest.raises(ValueError, match="could not be found"):
        convert_model(reloaded, _cfg())


def test_a_master_missing_its_gains_is_refused(tmp_path):
    master = tmp_path / "master"
    model = _tiny_llama()
    model.config.ternary_subln = True
    model.save_pretrained(master)
    reloaded = LlamaForCausalLM.from_pretrained(master)

    with pytest.raises(ValueError, match="missing from its shards"):
        convert_model(reloaded, DictDefault({**_cfg(), "base_model": str(master)}))


@pytest.mark.parametrize("fmt", sorted(SUBLN_UNSUPPORTED_FORMATS))
def test_export_refuses_the_formats_with_no_slot_for_the_gains(fmt):
    with pytest.raises(ValueError, match=fmt):
        assert_subln_exportable(_manifest(), fmt)

    assert_subln_exportable(_manifest(subln=False), fmt)


@pytest.mark.parametrize("fmt", ["master_bf16", "mask_sign"])
def test_export_allows_the_formats_that_copy_the_gains_through(fmt):
    assert_subln_exportable(_manifest(), fmt)


def test_the_config_marker_is_the_export_gate_key():
    assert SUBLN_CONFIG_KEY == "ternary_subln"
    assert set(SUBLN_UNSUPPORTED_FORMATS) == {
        "gguf_tq1_0",
        "gguf_tq2_0",
        "i2_s",
        "hf_bitnet",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_norm_reduces_in_fp32_on_gpu():
    """bf16 activations must not lose the sum of squares to accumulation error."""
    torch.manual_seed(0)
    x = 8.0 * torch.randn(4, HIDDEN)
    reference = _rms(x)
    norm = TernarySubNorm(HIDDEN, device="cuda", dtype=torch.bfloat16)

    out = norm(x.to("cuda", torch.bfloat16))

    torch.testing.assert_close(out.float().cpu(), reference, rtol=8e-3, atol=8e-3)

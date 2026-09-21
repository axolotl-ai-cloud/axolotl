"""Resolve nested expert adapters by identity, independent of shape and wrapper depth."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.tuners.lora.layer import ParamWrapper
from transformers.core_model_loading import MergeModulelist, WeightConverter

from axolotl.cli.utils.lora_merge import (
    _find_param_wrapper_lora,
    _fuse_and_unfuse_with_merge,
    _key_has_lora,
    _merge_tensor_with_lora,
    _param_wrapper_target,
    merge_lora_sharded_efficient,
)
from axolotl.cli.utils.param_wrapper_merge import (
    build_param_wrapper_map,
    strip_base_layers,
)


class ExpertModel(torch.nn.Module):
    def __init__(self, shapes, is_transposed=False, reverse=False):
        super().__init__()
        self.experts = torch.nn.Module()
        self.experts.is_transposed = is_transposed
        items = list(enumerate(shapes))
        for index, shape in reversed(items) if reverse else items:
            self.experts.register_parameter(
                f"p{index}", torch.nn.Parameter(torch.randn(shape))
            )


def make_adapter(shapes, is_transposed=False, reverse=False, use_rslora=False):
    torch.manual_seed(15)
    base_model = ExpertModel(shapes, is_transposed, reverse)
    base = {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
    }
    with torch.device("meta"):
        meta = ExpertModel(shapes, is_transposed, reverse)
    config = LoraConfig(
        r=2,
        lora_alpha=7,
        target_modules=[],
        target_parameters=[f"experts.p{i}" for i in range(len(shapes))],
        use_rslora=use_rslora,
    )
    model = get_peft_model(base_model, config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.1)
    state = get_peft_model_state_dict(model)
    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.merge_and_unload().named_parameters()
    }
    return base, meta, config.to_dict(), state, expected


@pytest.mark.parametrize("shapes", [[(4, 8, 8)] * 6, [(4, 8, 16), (4, 16, 8)]])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_rslora", [False, True])
def test_identity_merge_matches_peft(shapes, is_transposed, reverse, use_rslora):
    base, meta, config, state, expected = make_adapter(
        shapes, is_transposed, reverse, use_rslora
    )
    original_config = copy.deepcopy(config)
    original_names = list(dict(meta.named_parameters()))
    mapping = build_param_wrapper_map(meta, config, state)
    assert mapping == build_param_wrapper_map(meta, config, state)
    assert config == original_config
    assert list(dict(meta.named_parameters())) == original_names
    assert not any(isinstance(module, ParamWrapper) for module in meta.modules())
    assert set(mapping) == set(base)
    assert len({target.a_key for target in mapping.values()}) == len(shapes)
    if len(shapes) == 6:
        assert any(
            target.a_key.count(".base_layer") == 5 for target in mapping.values()
        )
    converters = [
        WeightConverter(
            source_patterns=f"experts.*.p{i}.weight",
            target_patterns=f"experts.p{i}",
            operations=[MergeModulelist(dim=0)],
        )
        for i in range(len(shapes))
    ]
    shard = {
        f"experts.{expert}.p{i}.weight": base[f"experts.p{i}"][expert]
        for i, shape in enumerate(shapes)
        for expert in range(shape[0])
    }
    for _ in range(2):
        fused, count, processed = _fuse_and_unfuse_with_merge(
            shard,
            converters,
            state,
            3.5,
            config,
            "cpu",
            expected_num_experts=4,
            param_wrapper_map=mapping,
        )
        assert count == len(shapes)
        assert set(fused) <= processed
        for name, tensor in base.items():
            merged, did_merge = _merge_tensor_with_lora(
                tensor,
                name,
                state,
                3.5,
                config,
                "cpu",
                param_wrapper_map=mapping,
            )
            assert did_merge
            torch.testing.assert_close(merged, expected[name], rtol=0, atol=1e-6)
            torch.testing.assert_close(fused[name], expected[name], rtol=0, atol=1e-6)
            assert not torch.equal(merged, tensor)


def test_identity_resolution_uses_renamings_and_packed_weight_metadata():
    base, meta, config, state, expected = make_adapter([(4, 8, 16), (4, 16, 8)])
    mapping = build_param_wrapper_map(meta, config, state)
    renamings = {r"^checkpoint": "experts"}
    key = "checkpoint.p0"
    assert _key_has_lora(key, (4, 8, 8), state, renamings, mapping)
    result, merged = _merge_tensor_with_lora(
        base["experts.p0"],
        key,
        state,
        3.5,
        config,
        "cpu",
        param_wrapper_map=mapping,
        weight_renamings=renamings,
    )
    assert merged
    torch.testing.assert_close(result, expected["experts.p0"], rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="Base parameter shape mismatch"):
        _find_param_wrapper_lora(state, key, (4, 8, 8), mapping, renamings)
    with pytest.raises(ValueError, match="Ambiguous ParamWrapper identity"):
        _param_wrapper_target("experts.p0", mapping, {"p0": "p1"})


@pytest.mark.parametrize("corruption", ["missing", "shape", "extra", "legacy"])
def test_invalid_adapter_fails_before_output_creation(
    tmp_path, monkeypatch, corruption
):
    import safetensors.torch

    _, meta, config, state, _ = make_adapter([(4, 8, 8)] * 2)
    key = next(iter(state))
    if corruption == "missing":
        del state[key]
    elif corruption == "shape":
        state[key] = state[key][:-1]
    elif corruption == "extra":
        state["base_model.model.unknown.lora_A.weight"] = torch.randn(2, 8)
    else:
        config["target_parameters"] = None
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    LoraConfig.from_peft_type(**config).save_pretrained(adapter)
    safetensors.torch.save_file(state, adapter / "adapter_model.safetensors")
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(
        "axolotl.cli.utils.lora_merge._build_meta_model", lambda *a, **k: meta
    )
    monkeypatch.setattr(
        "axolotl.cli.utils.lora_merge._get_conversion_info", lambda *a, **k: ({}, [])
    )
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="ParamWrapper|Unresolved adapter"):
        merge_lora_sharded_efficient(base, adapter, output)
    assert not output.exists()


def test_adapter_from_another_peft_release_still_reconstructs():
    _, meta, config, state, _ = make_adapter([(4, 8, 8)] * 3)
    reference = build_param_wrapper_map(meta, config, state)
    config["peft_version"] = "0.20.0"
    assert build_param_wrapper_map(meta, config, state) == reference


def test_missing_architecture_does_not_fall_back_to_shapes():
    _, _, config, state, _ = make_adapter([(4, 8, 8)] * 2)
    with pytest.raises(ValueError, match="Ambiguous ParamWrapper"):
        build_param_wrapper_map(None, config, state)
    with pytest.raises(ValueError, match="Ambiguous ParamWrapper"):
        _find_param_wrapper_lora(state, "experts.p0", (4, 8, 8))


def test_missing_architecture_still_merges_an_unambiguous_adapter():
    base, _, config, state, expected = make_adapter([(4, 8, 8)])
    assert build_param_wrapper_map(None, config, state) is None
    merged, did_merge = _merge_tensor_with_lora(
        base["experts.p0"], "experts.p0", state, 3.5, config, "cpu"
    )
    assert did_merge
    torch.testing.assert_close(merged, expected["experts.p0"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("fuse", [False, True])
def test_identity_map_is_used_by_checkpoint_merger(tmp_path, monkeypatch, fuse):
    import safetensors.torch

    base_weights, meta, config, state, expected = make_adapter([(4, 8, 8)] * 6)
    adapter = tmp_path / "adapter"
    LoraConfig.from_peft_type(**config).save_pretrained(adapter)
    safetensors.torch.save_file(state, adapter / "adapter_model.safetensors")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    converters = []
    checkpoint = base_weights
    if fuse:
        checkpoint = {
            f"experts.{expert}.p{i}.weight": base_weights[f"experts.p{i}"][
                expert
            ].clone()
            for i in range(6)
            for expert in range(4)
        }
        converters = [
            WeightConverter(
                source_patterns=f"experts.*.p{i}.weight",
                target_patterns=f"experts.p{i}",
                operations=[MergeModulelist(dim=0)],
            )
            for i in range(6)
        ]
    safetensors.torch.save_file(checkpoint, base / "model.safetensors")
    monkeypatch.setattr(
        "axolotl.cli.utils.lora_merge._build_meta_model", lambda *a, **k: meta
    )
    monkeypatch.setattr(
        "axolotl.cli.utils.lora_merge._get_conversion_info",
        lambda *a, **k: ({}, converters),
    )
    for output in (tmp_path / "merged1", tmp_path / "merged2"):
        merge_lora_sharded_efficient(base, adapter, output)
        actual = safetensors.torch.load_file(output / "model.safetensors")
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-6)


def make_moe_model():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=32,
        )
    )


def test_real_moe_checkpoint_merge_matches_peft(tmp_path):
    from transformers import Qwen3MoeForCausalLM

    torch.manual_seed(16)
    base = make_moe_model()
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=7,
            target_modules=["q_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        ),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.1)
    model.save_pretrained(tmp_path / "adapter")
    reference = model.merge_and_unload().eval()
    merge_lora_sharded_efficient(
        tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    )
    merged = Qwen3MoeForCausalLM.from_pretrained(tmp_path / "merged").eval()
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(merged.state_dict()[name], tensor, rtol=0, atol=1e-6)
    tokens = torch.arange(8).reshape(1, 8)
    with torch.no_grad():
        torch.testing.assert_close(
            merged(tokens).logits, reference(tokens).logits, rtol=1e-5, atol=1e-6
        )


def test_two_dimensional_parameters_use_the_identity_map():
    base, meta, config, state, expected = make_adapter([(8, 8)] * 2)
    mapping = build_param_wrapper_map(meta, config, state)
    for name, tensor in base.items():
        result, merged = _merge_tensor_with_lora(
            tensor, name, state, 3.5, config, "cpu", param_wrapper_map=mapping
        )
        assert merged
        torch.testing.assert_close(result, expected[name], rtol=0, atol=1e-6)


def test_weight_parameter_name_is_not_stripped_from_identity():
    base, meta, config, state, _ = make_adapter([(4, 8, 8)])
    mapping = build_param_wrapper_map(meta, config, state)
    target = mapping["experts.p0"]
    weight_map = {"experts.weight": target}
    a, b, _ = _find_param_wrapper_lora(
        state, "experts.weight", base["experts.p0"].shape, weight_map
    )
    assert a is state[target.a_key]
    assert b is state[target.b_key]


def test_quantized_training_order_matches_meta_reconstruction(monkeypatch):
    from peft.tuners.tuners_utils import BaseTuner
    from torch.nn.utils import parametrize

    from axolotl.monkeypatch import moe_quant

    model = ExpertModel([(4, 8, 8)] * 6)
    with torch.device("meta"):
        meta = ExpertModel([(4, 8, 8)] * 6)
    parameter_names = list(model.experts._parameters)
    for name in reversed(parameter_names):
        parametrize.register_parametrization(model.experts, name, torch.nn.Identity())
    assert list(model.experts.parametrizations) == list(reversed(parameter_names))

    with monkeypatch.context() as patch:
        patch.setattr(BaseTuner, "_inject_parameters", BaseTuner._inject_parameters)
        patch.setattr(
            BaseTuner,
            "_check_target_module_exists",
            BaseTuner.__dict__["_check_target_module_exists"],
        )
        patch.setattr(ParamWrapper, "_activate_lora", ParamWrapper._activate_lora)
        patch.setattr(
            moe_quant.patch_peft_target_parameters_matching,
            "_axolotl_patched",
            False,
            raising=False,
        )
        patch.setattr(
            moe_quant,
            "_moe_load_state",
            {
                **moe_quant._moe_load_state,
                "expert_param_order": {"experts": parameter_names},
            },
        )
        moe_quant.patch_peft_target_parameters_matching()
        config = LoraConfig(
            r=2,
            lora_alpha=7,
            target_modules=[],
            target_parameters=[f"experts.{name}" for name in parameter_names],
        )
        trained = get_peft_model(model, config)
        state = get_peft_model_state_dict(trained)
        mapping = build_param_wrapper_map(meta, config.to_dict(), state)
        for path, module in trained.named_modules():
            if isinstance(module, ParamWrapper):
                target = mapping[f"experts.{module.parameter_name}"]
                assert target.a_key == f"{path}.lora_A.weight"
                assert target.b_key == f"{path}.lora_B.weight"


def test_saved_full_weight_modules_do_not_block_reconstruction():
    torch.manual_seed(16)
    config = LoraConfig(
        r=2,
        lora_alpha=7,
        target_modules=["q_proj"],
        modules_to_save=["lm_head"],
        trainable_token_indices=[0, 1],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    state = get_peft_model_state_dict(get_peft_model(make_moe_model(), config))
    with torch.device("meta"):
        meta = make_moe_model()
    mapping = build_param_wrapper_map(meta, config.to_dict(), state)
    assert set(mapping) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }


def test_linear_only_adapter_needs_no_meta_model():
    config = LoraConfig(r=2, lora_alpha=7, target_modules=["q_proj", "k_proj"])
    state = get_peft_model_state_dict(get_peft_model(make_moe_model(), config))
    assert build_param_wrapper_map(None, config.to_dict(), state) is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.mlp.experts", "model.layers.0.mlp.experts"),
        ("model.layers.0.mlp.experts.base_layer", "model.layers.0.mlp.experts"),
        (
            "model.layers.0.mlp.experts.base_layer.base_layer",
            "model.layers.0.mlp.experts",
        ),
        ("model.base_layer.mlp.experts", "model.base_layer.mlp.experts"),
    ],
)
def test_strip_base_layers(name, expected):
    assert strip_base_layers(name) == expected

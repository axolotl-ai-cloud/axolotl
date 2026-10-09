"""Numerical and checkpoint tests for compact packed-expert LoRA."""

import copy
import json

import pytest
import torch
from peft import PeftModel, get_peft_model
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from axolotl.integrations.moe_sieve.peft import (
    MoeSieveLoraConfig,
    SelectiveExpertParamWrapper,
    register_selected_experts,
)
from axolotl.integrations.moe_sieve.plugin import MoeSievePlugin
from axolotl.integrations.moe_sieve.selection import (
    packed_experts,
    profile_routing,
    select_experts,
)
from axolotl.utils.dict import DictDefault


def tiny_model():
    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=8,
            num_experts_per_tok=2,
            _attn_implementation="eager",
        )
    )


def selection_for(model):
    return {
        name: {
            "num_experts": module.num_experts,
            "parameter_shapes": shapes,
            "selected_experts": [1, 5],
        }
        for name, (module, shapes) in packed_experts(model).items()
    }


def adapter_config(model):
    selection = selection_for(model)
    return MoeSieveLoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj"],
        target_parameters=[
            f"{name}.{key}"
            for name, spec in selection.items()
            for key in spec["parameter_shapes"]
        ],
        moe_sieve_selection=selection,
    )


def test_selection_and_profile_padding():
    assert select_experts([1, 9, 9, 0, 0, 3, 1, 0], 0.25) == [1, 2]
    with pytest.raises(ValueError, match="zero experts"):
        select_experts([1, 2], 0.25)
    model = tiny_model()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }
    selection = profile_routing(model, [batch])
    assert model.training
    for spec in selection.values():
        assert sum(spec["counts"]) == 6
        assert len(spec["selected_experts"]) == 2


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("selected_ids", [[1, 5], [5]])
def test_compact_delta_matches_reference_and_preserves_input_gradients(
    transposed, selected_ids
):
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_transposed = transposed
            self.weight = nn.Parameter(
                torch.randn(8, 3, 4) if transposed else torch.randn(8, 4, 3),
                requires_grad=False,
            )

        def forward(self, x):
            return torch.einsum(
                "bi,eio->ebo",
                x,
                self.weight if transposed else self.weight.transpose(-2, -1),
            )

    base = Experts()
    wrapper = SelectiveExpertParamWrapper(
        base,
        "default",
        parameter_name="weight",
        config=MoeSieveLoraConfig(),
        r=2,
        lora_alpha=4,
        selected_experts=selected_ids,
    )
    nn.init.normal_(wrapper.lora_B["default"].weight)
    assert wrapper.num_experts == 8
    count = len(selected_ids)
    assert sum(
        p.numel() for p in wrapper.parameters() if p.requires_grad
    ) == count * 2 * (3 + 4)
    x = torch.randn(2, 3, requires_grad=True)
    delta = wrapper.get_delta_weight("default")
    assert torch.count_nonzero(delta[[0, 2, 3, 4, 6, 7]]) == 0
    reference_delta = torch.zeros_like(base.weight)
    weight_a = wrapper.lora_A["default"].weight.reshape(count, 2, 3)
    weight_b = wrapper.lora_B["default"].weight.reshape(4, 2, count)
    for slot, expert in enumerate(selected_ids):
        update = weight_b[:, :, slot] @ weight_a[slot] * 2
        reference_delta[expert] = update.T if transposed else update
    torch.testing.assert_close(delta, reference_delta)
    expected_weight = base.weight + reference_delta
    expected = torch.einsum(
        "bi,eio->ebo",
        x,
        expected_weight if transposed else expected_weight.transpose(-2, -1),
    )
    torch.testing.assert_close(wrapper(x), expected)
    wrapper(x)[0].sum().backward()
    assert torch.count_nonzero(x.grad) > 0
    assert base.weight.grad is None
    assert torch.count_nonzero(wrapper.lora_B["default"].weight.grad) == 0


@pytest.mark.parametrize("experts_implementation", ["eager", "batched_mm"])
def test_training_save_reload_disable_and_merge(tmp_path, experts_implementation):
    torch.manual_seed(3)
    base = tiny_model()
    base.set_experts_implementation(experts_implementation)
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    original = copy.deepcopy(base)
    config = adapter_config(base)
    register_selected_experts(base, config)
    model = get_peft_model(base, config)
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, SelectiveExpertParamWrapper)
    ]
    assert len(wrappers) == 4
    tokens = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    for _ in range(2):
        model(input_ids=tokens, labels=tokens).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    expected = model(input_ids=tokens).logits
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["moe_sieve_selection"] == config.moe_sieve_selection
    reloaded_config = MoeSieveLoraConfig.from_pretrained(tmp_path)
    fresh = copy.deepcopy(original)
    register_selected_experts(fresh, reloaded_config)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path, config=reloaded_config)
    torch.testing.assert_close(reloaded(input_ids=tokens).logits, expected)
    with model.disable_adapter():
        torch.testing.assert_close(
            model(input_ids=tokens).logits, original(input_ids=tokens).logits
        )
    merged = reloaded.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(merged(input_ids=tokens).logits, expected)


@pytest.mark.parametrize(
    "resume",
    ["lora_model_dir", "resume_from_checkpoint", "auto_resume_from_checkpoints"],
)
def test_plugin_load_and_resume_without_profile(tmp_path, resume):
    model = tiny_model()
    selection_file = tmp_path / "selection.json"
    selection_file.write_text(
        json.dumps(
            {
                "version": 1,
                "base_model": "tiny",
                "revision": None,
                "fraction": 0.25,
                "selection": selection_for(model),
            }
        )
    )
    cfg = DictDefault(
        {
            "adapter": "moe_sieve",
            "base_model": "tiny",
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "lora_target_linear": True,
            "lora_target_parameters": ["gate.weight"],
            "output_dir": str(tmp_path),
            "moe_sieve": {"selection_file": str(selection_file)},
        }
    )
    trained, _ = MoeSievePlugin().load_adapter(model, cfg)
    trained.save_pretrained(tmp_path / "checkpoint-1")
    selection_file.unlink()
    cfg[resume] = (
        True
        if resume == "auto_resume_from_checkpoints"
        else str(tmp_path / "checkpoint-1")
    )
    reloaded, config = MoeSievePlugin().load_adapter(tiny_model(), cfg)
    assert config.moe_sieve_selection
    assert any(p.requires_grad for p in reloaded.parameters())


@pytest.mark.parametrize("mode", ["train", "inference", "merge"])
@pytest.mark.parametrize("resume", ["explicit", "auto"])
def test_resume_settings_do_not_override_merge_or_inference(tmp_path, mode, resume):
    base = tiny_model().eval()
    original = copy.deepcopy(base)
    config = adapter_config(base)
    register_selected_experts(base, config)
    adapted = get_peft_model(base, config)
    checkpoint = tmp_path / "checkpoint-1"
    snapshots = {}
    for directory, value in ((checkpoint, 0.01), (tmp_path, 0.05)):
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(value)
        adapted.save_pretrained(directory)
        snapshots[str(directory)] = {
            name: parameter.detach().clone()
            for name, parameter in adapted.named_parameters()
            if "lora_" in name
        }
    cfg = DictDefault(
        adapter="moe_sieve",
        output_dir=str(tmp_path),
        lora_model_dir=str(tmp_path),
        merge_lora=mode == "merge",
        resume_from_checkpoint=str(checkpoint) if resume == "explicit" else None,
        auto_resume_from_checkpoints=resume == "auto",
    )
    loaded, _ = MoeSievePlugin().load_adapter(
        original, cfg, inference=mode == "inference"
    )
    expected_path = str(checkpoint if mode == "train" else tmp_path)
    assert cfg.lora_model_dir == expected_path
    parameters = dict(loaded.named_parameters())
    for name, expected in snapshots[expected_path].items():
        torch.testing.assert_close(parameters[name], expected, atol=0, rtol=0)
    if mode == "merge":
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = adapted.eval()(input_ids=tokens).logits
            merged = loaded.eval().merge_and_unload(safe_merge=True)
            torch.testing.assert_close(merged(input_ids=tokens).logits, expected)


def test_profile_cleanup_on_unsupported_routing():
    model = tiny_model()
    model.model.layers[0].eval()
    modes = {module: module.training for module in model.modules()}
    with pytest.raises(ValueError, match="token count"):
        profile_routing(
            model,
            [{"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 3)}],
        )
    for module, training in modes.items():
        assert module.training == training
    assert all(
        not module._forward_pre_hooks for module, _ in packed_experts(model).values()
    )


@pytest.mark.parametrize(
    "key,value",
    [("load_in_4bit", True), ("tensor_parallel_size", 2), ("expert_parallel_size", 2)],
)
def test_unsupported_runtime_fails_early(key, value):
    with pytest.raises(ValueError, match=key):
        MoeSievePlugin().pre_model_load(
            DictDefault({"adapter": "moe_sieve", key: value})
        )


def test_stale_selection_rejected():
    model = tiny_model()
    config = adapter_config(model)
    first = next(iter(config.moe_sieve_selection.values()))
    first["selected_experts"] = [8]
    with pytest.raises(ValueError, match="Invalid selected"):
        register_selected_experts(model, config)


def test_merge_cli_uses_registered_wrapper(monkeypatch):
    from axolotl.cli import merge_lora

    calls = []
    monkeypatch.setattr(
        merge_lora, "_do_merge_lora_legacy", lambda **kwargs: calls.append(kwargs)
    )
    cfg = DictDefault({"adapter": "moe_sieve"})
    merge_lora.do_merge_lora(cfg=cfg)
    assert len(calls) == 1
    cfg.merge_method = "memory_efficient"
    with pytest.raises(ValueError, match="legacy"):
        merge_lora.do_merge_lora(cfg=cfg)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_calibration_command_writes_reproducible_profile(
    tmp_path, monkeypatch, mixed_precision
):
    from datasets import Dataset

    from axolotl.cli import config as config_module, utils as cli_utils
    from axolotl.common import datasets as datasets_module
    from axolotl.integrations.moe_sieve.profile import profile

    path = tmp_path / "selection.json"
    settings = {
        "base_model": "tiny",
        "adapter": "moe_sieve",
        "moe_sieve": {"selection_file": str(path), "calibration_samples": 2},
        "seed": 42,
    }
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [3, 2, 1], [2, 3, 4]],
            "attention_mask": [[1, 1, 1]] * 3,
        }
    )
    base = tiny_model()
    if mixed_precision:
        base.to(torch.bfloat16)
        base.get_input_embeddings().float()
        settings["torch_dtype"] = torch.bfloat16
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        config_module,
        "load_cfg",
        lambda *args, **kwargs: DictDefault(copy.deepcopy(settings)),
    )
    monkeypatch.setattr(
        cli_utils, "load_model_and_tokenizer", lambda **kwargs: (base, None, None)
    )
    monkeypatch.setattr(
        datasets_module,
        "load_datasets",
        lambda **kwargs: datasets_module.TrainDatasetMeta(dataset),
    )
    assert profile("config.yml") == str(path)
    first = json.loads(path.read_text())
    profile("config.yml")
    assert json.loads(path.read_text()) == first
    assert first["calibration_samples"] == 2
    assert len(first["calibration_sha256"]) == 64
    assert all(sum(spec["counts"]) == 12 for spec in first["selection"].values())


def test_seeded_initialization_matches_full_expert_slices():
    from peft import LoraConfig

    from axolotl.loaders.adapter import reinit_lora_from_seed

    base = tiny_model()
    full_base = copy.deepcopy(base)
    config = adapter_config(base)
    full_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        target_parameters=config.target_parameters,
    )
    register_selected_experts(base, config)
    selected = get_peft_model(base, config)
    full = get_peft_model(full_base, full_config)
    reinit_lora_from_seed(selected, 7)
    reinit_lora_from_seed(full, 7)
    for name, module in selected.named_modules():
        if isinstance(module, SelectiveExpertParamWrapper):
            full_a = full.get_submodule(name).lora_A["default"].weight.reshape(8, 2, -1)
            torch.testing.assert_close(
                module.lora_A["default"].weight, full_a[[1, 5]].flatten(0, 1)
            )


def local_experts(model, offset, count=4):
    for module, _ in packed_experts(model).values():
        total = module.num_experts
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.ndim == 3:
                setattr(
                    module,
                    name,
                    nn.Parameter(parameter[offset : offset + count].detach().clone()),
                )
        module.num_experts_global = total
        module.num_experts = count
        module.num_local_experts = count
        module.local_expert_offset = offset


@pytest.mark.parametrize("selected", [[1, 5], [0, 1], [7, 0]])
@pytest.mark.parametrize("offset", [0, 4])
def test_ep_compact_ownership_and_checkpoint(tmp_path, selected, offset):
    from axolotl.integrations.expert_parallel.shard import shard_expert_lora
    from axolotl.integrations.moe_sieve.distributed import local_adapter_checkpoint
    from axolotl.loaders.adapter import reinit_lora_from_seed
    from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
        _coordinates,
        _index,
        _layout,
        expert_ownership,
    )

    torch.manual_seed(42)
    base = tiny_model()
    original = copy.deepcopy(base)
    config = adapter_config(base)
    for spec in config.moe_sieve_selection.values():
        spec["selected_experts"] = selected
    register_selected_experts(base, config)
    full = get_peft_model(base, config)
    reinit_lora_from_seed(full, 42)
    for module in full.modules():
        if isinstance(module, SelectiveExpertParamWrapper):
            nn.init.normal_(module.lora_B["default"].weight)
    full.save_pretrained(tmp_path)
    local_experts(original, offset)
    local_config = MoeSieveLoraConfig.from_pretrained(tmp_path)
    register_selected_experts(original, local_config)
    with local_adapter_checkpoint(original, local_config, tmp_path) as directory:
        local = PeftModel.from_pretrained(
            original, directory, config=local_config, is_trainable=True
        )
    shard_expert_lora(local, 2)
    owners = expert_ownership(local)
    full_params = dict(full.named_parameters())
    for name, parameter in local.named_parameters():
        if name not in owners or "lora_" not in name:
            continue
        layout = _layout(parameter, owners[name])
        assert layout["shape"] == list(full_params[name].shape)
        expected = full_params[name][_index(_coordinates(layout))]
        torch.testing.assert_close(parameter, expected)
    full_modules = dict(full.named_modules())
    for name, wrapper in local.named_modules():
        if not isinstance(wrapper, SelectiveExpertParamWrapper):
            continue
        assert wrapper.selected_experts == tuple(
            i - offset for i in selected if offset <= i < offset + 4
        )
        delta = wrapper.get_delta_weight("default")
        torch.testing.assert_close(
            delta, full_modules[name].get_delta_weight("default")[offset : offset + 4]
        )
        delta.sum().backward()
        assert wrapper.lora_A["default"].weight.grad is not None
        assert wrapper.lora_B["default"].weight.grad is not None
    reinit_lora_from_seed(local, 42)
    for name, parameter in local.named_parameters():
        if name in owners and ".lora_A." in name:
            expected = full_params[name][
                _index(_coordinates(_layout(parameter, owners[name])))
            ]
            torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize("selected", [[1, 5], []])
def test_kernel_factor_expansion_preserves_compact_gradients(selected):
    from axolotl.integrations.moe_sieve.peft import SelectiveExpertParamWrapper

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 12, 16))
            self.num_experts = 8

    wrapper = SelectiveExpertParamWrapper(
        Experts(),
        "default",
        parameter_name="weight",
        config=MoeSieveLoraConfig(r=2, lora_alpha=4),
        r=2,
        lora_alpha=4,
        selected_experts=selected,
    )
    nn.init.normal_(wrapper.lora_B["default"].weight)
    a = wrapper.lora_A["default"].weight
    b = wrapper.lora_B["default"].weight
    expanded_a, expanded_b = wrapper.kernel_lora_factors(a, b)
    delta = (
        torch.einsum(
            "eri,ore->eoi", expanded_a.reshape(8, 2, 16), expanded_b.reshape(12, 2, 8)
        )
        * 2
    )
    reference = wrapper.get_delta_weight("default")
    torch.testing.assert_close(delta, reference)
    probe = torch.randn_like(delta)
    expected = torch.autograd.grad((reference * probe).sum(), (a, b))
    actual = torch.autograd.grad((delta * probe).sum(), (a, b))
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_fsdp_optimizer_keeps_distinct_empty_parameters(monkeypatch):
    from accelerate import Accelerator

    from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

    model = nn.Module()
    model.register_parameter("empty_a", nn.Parameter(torch.empty(0, 4)))
    model.register_parameter("empty_b", nn.Parameter(torch.empty(8, 0)))
    model.register_parameter("dense", nn.Parameter(torch.ones(4)))
    optimizer = torch.optim.AdamW(model.parameters())
    assert model.empty_a.data_ptr() == model.empty_b.data_ptr() == 0

    def prepare(_self, model, optimizer):
        old_pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
        group_pointers = [p.data_ptr() for p in optimizer.param_groups[0]["params"]]
        for name, parameter in list(model.named_parameters()):
            model.register_parameter(name, nn.Parameter(parameter.detach().clone()))
        mapping = {old_pointers[name]: p for name, p in model.named_parameters()}
        optimizer.param_groups[0]["params"] = [
            mapping[pointer] for pointer in group_pointers
        ]
        return model, optimizer

    monkeypatch.setattr(Accelerator, "_prepare_fsdp2", prepare)
    patch_accelerate_fsdp2()
    patched = Accelerator._prepare_fsdp2
    patch_accelerate_fsdp2()
    assert Accelerator._prepare_fsdp2 is patched
    patched(None, model, optimizer)
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for p in model.parameters()
    ]

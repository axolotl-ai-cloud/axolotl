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
    [("load_in_4bit", True), ("use_scattermoe", True), ("expert_parallel_size", 2)],
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


def test_calibration_command_writes_reproducible_profile(tmp_path, monkeypatch):
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
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        config_module, "load_cfg", lambda *args: DictDefault(copy.deepcopy(settings))
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

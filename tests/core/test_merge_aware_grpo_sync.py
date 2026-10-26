"""Merge-aware LoRA through the standard GRPO trainer's TRL vLLM weight sync."""

import types
from collections import defaultdict
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from trl import GRPOTrainer
from trl.generation.vllm_generation import VLLMGeneration

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
from axolotl.monkeypatch.lora_merge_aware import install_lora_merge_aware_linears
from axolotl.utils.lora_merge_precision import merged_lora_weight

SYNCS = 8


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 96, bias=False)
        self.v = nn.Linear(128, 96, bias=True)
        self.o = nn.Linear(96, 128, bias=False)


def _model():
    torch.manual_seed(0)
    base = _Toy()
    with torch.no_grad():
        # small weights next to the delta, where merge/unmerge rounding is most visible
        base.q.weight.mul_(0.05)
    model = get_peft_model(
        base.to(torch.bfloat16),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"]),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 2e-3)
    return model


def _step(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(torch.sign(torch.randn_like(param)) * 1e-4)


def _colocated_vllm(model, loaded):
    """A TRL VLLMGeneration in colocate mode whose engine records the loaded weights."""
    generation = object.__new__(VLLMGeneration)
    generation.model = model
    generation.mode = "colocate"
    generation.enable_sleep_mode = False
    generation.accelerator = types.SimpleNamespace(is_main_process=True)
    generation._dist = types.SimpleNamespace(
        is_fsdp=False, fsdp_version=None, gather_params=lambda params: nullcontext()
    )
    engine_model = types.SimpleNamespace(
        load_weights=lambda weights: loaded.update(
            (name, param.detach().clone()) for name, param in weights
        )
    )
    worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=engine_model)
    )
    generation.llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            model_executor=types.SimpleNamespace(driver_worker=worker)
        ),
        reset_prefix_cache=lambda: None,
    )
    return generation


def _trainer(model, generation):
    def stub_init(self, *args, **kwargs):
        self.args = types.SimpleNamespace(
            activation_offloading=False, vllm_merge_check_interval=1
        )
        self.optimizer = self.optimizer_cls_and_kwargs = None
        self.model = model
        self.accelerator = types.SimpleNamespace(is_main_process=True)
        self.state = types.SimpleNamespace(global_step=0)
        self._metrics = {"train": defaultdict(list)}
        self.vllm_generation = generation

    with patch.object(GRPOTrainer, "__init__", stub_init):
        return AxolotlGRPOTrainer()


def _base(model):
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n
    }


def test_merged_sync_serves_the_merge_aware_weights_and_keeps_the_base():
    model = _model()
    assert install_lora_merge_aware_linears(model) == (2, [])
    base = _base(model)
    loaded: dict[str, torch.Tensor] = {}
    trainer = _trainer(model, _colocated_vllm(model, loaded))
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    for step in range(1, SYNCS + 1):
        _step(model)
        trainer.state.global_step = step
        loaded.clear()
        trainer.vllm_generation.sync_weights()

        assert sorted(loaded) == ["o.weight", "q.weight", "v.bias", "v.weight"]
        for name in ("q", "v"):
            module = getattr(model.base_model.model, name)
            expected = merged_lora_weight(
                module.base_layer.weight,
                module.lora_A["default"].weight,
                module.lora_B["default"].weight,
                module.scaling["default"],
                torch.bfloat16,
            )
            assert torch.equal(loaded[f"{name}.weight"], expected), (step, name)
            # the weight vLLM serves reproduces the trainer's merge-aware forward exactly
            with torch.no_grad():
                bias = module.base_layer.bias
                assert torch.equal(
                    F.linear(x, loaded[f"{name}.weight"], bias), module(x)
                ), (step, name)
        assert torch.equal(loaded["o.weight"], base["base_model.model.o.weight"])
        for n, p in model.named_parameters():
            if n in base:
                assert torch.equal(p, base[n]), (step, n)
    # merge-aware layers are excluded from the rounding diagnostic
    assert "lora_merge/retained" not in trainer._metrics["train"]


def test_stock_trl_sync_drifts_the_same_base():
    """Guards that the setup exercises the in-place merge/unmerge rounding."""
    model = _model()
    install_lora_merge_aware_linears(model)
    base = _base(model)
    generation = _colocated_vllm(model, {})
    for _ in range(SYNCS):
        _step(model)
        VLLMGeneration.sync_weights(generation)
    q = "base_model.model.q.base_layer.weight"
    assert not torch.equal(dict(model.named_parameters())[q], base[q])

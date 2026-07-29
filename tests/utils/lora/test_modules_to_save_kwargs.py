"""Kwargs-only forward through modules wrapped by lora_modules_to_save.

PEFT < 0.20.0 required a positional first argument in
``ModulesToSaveWrapper.forward``, so any ``lora_modules_to_save`` entry the
model invokes with keyword arguments only (e.g. a VLM's
``vision_tower(pixel_values=...)``) crashed with a TypeError at the first
forward pass. Fixed upstream in huggingface/peft#3199 (peft 0.20.0).
"""

import torch
from peft import get_peft_model
from peft.utils import ModulesToSaveWrapper
from transformers import PretrainedConfig

from axolotl.loaders.adapter import _build_peft_lora_config
from axolotl.utils.dict import DictDefault

HIDDEN_SIZE = 16


class KwargsOnlyVisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_values)


class TinyVlmForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.vision_tower = KwargsOnlyVisionTower()
        self.q_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.v_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, pixel_values: torch.Tensor = None, **kwargs) -> torch.Tensor:
        hidden = self.vision_tower(pixel_values=pixel_values)
        return self.q_proj(hidden) + self.v_proj(hidden)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def build_peft_model():
    cfg = DictDefault(
        {
            "adapter": "lora",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_modules_to_save": ["vision_tower"],
        }
    )
    model = TinyVlmForCausalLM()
    lora_config = _build_peft_lora_config(model, cfg)
    return get_peft_model(model, lora_config)


class TestModulesToSaveKwargsForward:
    def test_kwargs_only_forward_through_wrapper(self):
        peft_model = build_peft_model()

        vision_tower = peft_model.base_model.model.vision_tower
        assert isinstance(vision_tower, ModulesToSaveWrapper)

        out = vision_tower(pixel_values=torch.randn(2, HIDDEN_SIZE))
        assert out.shape == (2, HIDDEN_SIZE)

    def test_full_forward_backward_trains_saved_module(self):
        peft_model = build_peft_model()

        out = peft_model(pixel_values=torch.randn(2, HIDDEN_SIZE))
        out.sum().backward()

        vision_tower = peft_model.base_model.model.vision_tower
        saved_params = list(vision_tower.modules_to_save["default"].parameters())
        assert saved_params
        assert all(p.requires_grad for p in saved_params)
        assert all(p.grad is not None for p in saved_params)

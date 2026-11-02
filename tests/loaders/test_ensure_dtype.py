"""ensure_dtype, the RL builder's FSDP dtype pass, can leave FP32 LoRA factors alone."""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from axolotl.loaders.utils import ensure_dtype
from axolotl.utils.lora_precision import is_lora_tensor_name, upcast_lora_parameters


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)


def _model():
    model = get_peft_model(
        _Toy().to(torch.bfloat16), LoraConfig(r=4, target_modules=["q"])
    )
    upcast_lora_parameters(model)
    model.base_model.model.norm.to(torch.float32)
    return model


def _dtypes(model):
    return {n: p.dtype for n, p in model.named_parameters()}


def test_keep_leaves_lora_factors_in_fp32():
    model = _model()
    ensure_dtype(model, dtype=torch.bfloat16, keep=is_lora_tensor_name)
    for name, dtype in _dtypes(model).items():
        expected = torch.float32 if "lora_" in name else torch.bfloat16
        assert dtype == expected, name


def test_keep_survives_casting_an_fp32_base_layer():
    model = get_peft_model(_Toy(), LoraConfig(r=4, target_modules=["q"]))
    ensure_dtype(model, dtype=torch.bfloat16, keep=is_lora_tensor_name)
    for name, dtype in _dtypes(model).items():
        expected = torch.float32 if "lora_" in name else torch.bfloat16
        assert dtype == expected, name


def test_without_keep_everything_is_cast():
    model = _model()
    ensure_dtype(model, dtype=torch.bfloat16)
    assert set(_dtypes(model).values()) == {torch.bfloat16}

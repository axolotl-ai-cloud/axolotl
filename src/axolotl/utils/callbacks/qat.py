from transformers import TrainerCallback
from torchao.quantization.qat.linear import FakeQuantizedLinear
import torch.nn as nn
from functools import partial


def toggle_fake_quant(mod: nn.Module, enable: bool):
    if isinstance(mod, FakeQuantizedLinear):
        if mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = enable
        if mod.weight_fake_quantizer is not None:
            mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    def __init__(self, fake_quant_after_n_steps: int):
        self.fake_quant_after_n_steps = fake_quant_after_n_steps

    def on_step_begin(self, args, state, control, model, **kwargs):
        if state.global_step == 0:
            model.apply(partial(toggle_fake_quant, enable=False))
        elif state.global_step == self.fake_quant_after_n_steps:
            model.apply(partial(toggle_fake_quant, enable=True))
from functools import partial

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.qat import FromIntXQuantizationAwareTrainingConfig
from torchao.quantization.qat.linear import FakeQuantizedLinear
from transformers import TrainerCallback


def toggle_fake_quant(mod: nn.Module, enable: bool):
    if isinstance(mod, FakeQuantizedLinear):
        if mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = enable
        if mod.weight_fake_quantizer is not None:
            mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    def __init__(self, fake_quant_after_n_steps: int | None = None):
        self.fake_quant_after_n_steps = fake_quant_after_n_steps

    def on_step_begin(self, args, state, control, model, **kwargs):
        if self.fake_quant_after_n_steps is not None:
            if state.global_step == 0:
                model.apply(partial(toggle_fake_quant, enable=False))
            elif state.global_step == self.fake_quant_after_n_steps:
                model.apply(partial(toggle_fake_quant, enable=True))

    def on_train_end(self, args, state, control, model, **kwargs):
        quantize_(model, FromIntXQuantizationAwareTrainingConfig())

from functools import partial

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.qat import FromIntXQuantizationAwareTrainingConfig
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from transformers import TrainerCallback

from axolotl.utils.quantization import get_ptq_config

from src.axolotl.utils.schemas.qat import QATConfig


def toggle_fake_quant(mod: nn.Module, enable: bool):
    if isinstance(mod, FakeQuantizedLinear) or isinstance(mod, FakeQuantizedEmbedding):
        if mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = enable
        if mod.weight_fake_quantizer is not None:
            mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def on_step_begin(self, args, state, control, model, **kwargs):
        if self.cfg.fake_quant_after_n_steps is not None:
            if state.global_step == 0:
                model.apply(partial(toggle_fake_quant, enable=False))
            elif state.global_step == self.cfg.fake_quant_after_n_steps:
                model.apply(partial(toggle_fake_quant, enable=True))

    def on_train_end(self, args, state, control, model, **kwargs):
        quantize_(model, FromIntXQuantizationAwareTrainingConfig())
        if self.cfg.save_quantized_model:
            ptq_config = get_ptq_config(
                weight_dtype=self.cfg.weight_dtype,
                activation_dtype=self.cfg.activation_dtype,
                group_size=self.cfg.group_size,
            )
            quantize_(model, ptq_config)

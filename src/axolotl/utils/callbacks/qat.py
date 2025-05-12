import logging
from functools import partial

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.qat import FromIntXQuantizationAwareTrainingConfig
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from transformers import TrainerCallback
from torchao.quantization.quant_api import _is_linear

from axolotl.utils.quantization import quantize_model_for_ptq

from axolotl.utils.schemas.quantization import QATConfig

LOG = logging.getLogger(__name__)


def toggle_fake_quant(mod: nn.Module, enable: bool):
    if isinstance(mod, FakeQuantizedLinear) or isinstance(mod, FakeQuantizedEmbedding):
        if isinstance(mod, FakeQuantizedLinear) and mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = enable
        mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def on_step_begin(self, args, state, control, model, **kwargs):
        if self.cfg.fake_quant_after_n_steps is not None:
            if state.global_step == 0:
                LOG.info(f"Disabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=False))
            elif state.global_step == self.cfg.fake_quant_after_n_steps:
                LOG.info(f"Enabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=True))
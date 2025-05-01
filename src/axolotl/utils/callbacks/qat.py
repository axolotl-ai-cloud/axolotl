from transformers import TrainerCallback
from torchao.quantization.qat.linear import FakeQuantizedLinear
from axolotl.utils.dict import DictDefault

class QATCallback(TrainerCallback):
    def __init__(self, cfg: DictDefault):
        self.fake_quant_after_n_steps = cfg.fake_quant_after_n_steps

    def on_step_begin(self, args, state, control, model, optimizer, lr_scheduler):
        if state.global_step == 0:
            model.apply(lambda mod: mod.disable_fake_quant() if isinstance(mod, FakeQuantizedLinear) else None)
        elif state.global_step == self.fake_quant_after_n_steps:
            model.apply(lambda mod: mod.enable_fake_quant() if isinstance(mod, FakeQuantizedLinear) else None)
        

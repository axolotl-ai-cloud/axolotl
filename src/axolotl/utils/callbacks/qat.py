"""QAT Callback for HF Causal Trainer"""

from functools import partial

from torch import nn
from transformers import TrainerCallback

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.quantization import QATConfig

LOG = get_logger(__name__)


def toggle_fake_quant(mod: nn.Module, enable: bool):
    """
    Toggle fake quantization for any fake quantized linear or embedding layers in the model.

    Args:
        mod: The module to toggle fake quantization for.
        enable: Whether to enable or disable fake quantization.
    """
    for attr in ("activation_fake_quantizer", "weight_fake_quantizer"):
        fake_quantizer = getattr(mod, attr, None)
        if fake_quantizer is not None and hasattr(fake_quantizer, "enabled"):
            fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    """
    Callback to toggle fake quantization for the model.
    """

    def __init__(self, cfg: QATConfig):
        self.cfg = cfg
        self.fake_quant_enabled: bool | None = None

    def on_step_begin(self, args, state, control, model, **kwargs):
        if self.cfg.fake_quant_after_n_steps is None:
            return

        # quantizers are constructed enabled, so a resume mid-warmup has to switch
        # them off; equality against the step would leave the warmup quantized
        enable = state.global_step >= self.cfg.fake_quant_after_n_steps
        if enable is self.fake_quant_enabled:
            return

        LOG.info(
            f"{'Enabling' if enable else 'Disabling'} fake quantization at step {state.global_step}"
        )
        model.apply(partial(toggle_fake_quant, enable=enable))
        self.fake_quant_enabled = enable

"""Definition of differential transformer plugin."""

import logging
from typing import List

from transformers import PreTrainedModel, TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.callbacks.differential import DifferentialAttentionMonitorCallback
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


class DifferentialTransformerPlugin(BasePlugin):
    """Plugin for differential transformer integration with Axolotl."""

    def __init__(self) -> None:
        """
        Constructor for differential transformers plugin. Calls `register_diff_attn`
        to register differential attention custom modeling implementation to `AutoConfig`
        and `AutoModel`.
        """
        from .modeling_diff_attn import register_diff_attn

        register_diff_attn()

    def get_input_args(self) -> str:
        """Returns module path to diff transformer plugin args for `axolotl` config."""
        return "axolotl.integrations.diff_transformer.args.DifferentialTransformerArgs"

    def add_callbacks_pre_trainer(
        self, cfg: DictDefault, model: PreTrainedModel
    ) -> List[TrainerCallback]:
        """
        Returns `DifferentialAttentionMonitorCallback` to be added to the list of
        callbacks for the `axolotl` trainer if wandb usage is enabled.

        Parameters:
            cfg: Dictionary mapping `axolotl` config keys to values.
            model: The loaded mfodel.

        Returns:
            A list (possibly) containing an instantiated `DifferentialAttentionMonitorCallback`.
        """
        callbacks = []
        if cfg.use_wandb:
            callbacks.append(
                DifferentialAttentionMonitorCallback(
                    log_every=cfg.diff_attn_log_every,
                    num_monitor_layers=cfg.diff_attn_num_monitor_layers,
                )
            )

        return callbacks

"""MoRA / ReMoRA plugin for Axolotl."""

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.loaders.adapter import _peft_supports_mora
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MoraPlugin(BasePlugin):
    """Plugin that exposes MoRA-specific config and validates runtime support."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.mora.MoraArgs"

    def pre_model_load(self, cfg: DictDefault):
        if cfg.adapter != "mora":
            return
        if not _peft_supports_mora():
            raise ImportError(
                "adapter: mora requires a PEFT build with MoRA support "
                "(LoraConfig(use_mora=..., mora_type=...)). "
                "Install the MoRA fork or another PEFT distribution that exposes "
                "those fields."
            )
        LOG.info("MoRA plugin enabled for adapter: mora")

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        if cfg.adapter == "mora" and getattr(cfg, "mora", None):
            LOG.debug(
                "Loaded MoRA model with mora_type=%s, relora=%s",
                cfg.mora.mora_type,
                cfg.mora.use_relora,
            )

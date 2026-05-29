"""MoRA / ReMoRA plugin for Axolotl."""

import inspect

from peft import LoraConfig, PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import AdapterCapabilities, BasePlugin
from axolotl.integrations.mora.args import MoraType
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _peft_supports_mora() -> bool:
    try:
        params = inspect.signature(LoraConfig).parameters
    except (TypeError, ValueError):
        return False
    return "use_mora" in params and "mora_type" in params


def _mora_type_peft_value(mora_type: MoraType | str | int) -> int:
    if isinstance(mora_type, MoraType):
        return mora_type.peft_value
    if mora_type == 1 or mora_type == MoraType.SHARING.value:
        return MoraType.SHARING.peft_value
    if mora_type == 6 or mora_type == MoraType.ROPE.value:
        return MoraType.ROPE.peft_value
    raise ValueError("mora_type must be one of `sharing`, `rope`, 1, or 6")


def _mora_type_label(mora_type: MoraType | str | int) -> str:
    if isinstance(mora_type, MoraType):
        return mora_type.value
    if mora_type == 1:
        return MoraType.SHARING.value
    if mora_type == 6:
        return MoraType.ROPE.value
    return str(mora_type)


class MoraPlugin(BasePlugin):
    """Plugin that exposes MoRA-specific config and validates runtime support."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.mora.MoraArgs"

    def get_adapter_capabilities(self) -> list[AdapterCapabilities]:
        return [AdapterCapabilities(name="mora", lora_like=True, relora=True)]

    def _validate_mora_config(self, cfg: DictDefault):
        mora_cfg = getattr(cfg, "mora", None)
        if mora_cfg is None:
            raise ValueError("adapter: mora requires a nested mora configuration block")
        if not getattr(mora_cfg, "use_mora", False):
            raise ValueError("mora.use_mora must be true when adapter: mora is set")
        if cfg.load_in_4bit or cfg.load_in_8bit:
            raise ValueError(
                "adapter: mora currently requires a full-precision base model. "
                "Use adapter: lora or qlora for quantized training."
            )
        if cfg.gptq:
            raise ValueError(
                "adapter: mora is not compatible with GPTQ quantized base models."
            )

    def get_lora_config_kwargs(self, cfg: DictDefault) -> dict:
        if cfg.adapter != "mora":
            return {}
        self._validate_mora_config(cfg)
        if not _peft_supports_mora():
            raise ImportError(
                "adapter: mora requires a PEFT build with MoRA support "
                "(LoraConfig(use_mora=..., mora_type=...)). "
                "Install the MoRA fork or another PEFT distribution that exposes "
                "those fields."
            )
        mora_cfg = cfg.mora
        return {
            "use_mora": mora_cfg.use_mora,
            "mora_type": _mora_type_peft_value(mora_cfg.mora_type),
        }

    def pre_model_load(self, cfg: DictDefault):
        if cfg.adapter != "mora":
            return
        LOG.info("MoRA plugin enabled for adapter: mora")

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        if cfg.adapter == "mora" and getattr(cfg, "mora", None):
            LOG.debug(
                "Loaded MoRA model with mora_type=%s, relora=%s",
                _mora_type_label(cfg.mora.mora_type),
                cfg.relora,
            )

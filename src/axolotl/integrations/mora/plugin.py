"""MoRA / ReMoRA plugin for Axolotl."""

import inspect

from peft import LoraConfig, PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import AdapterCapabilities, BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _peft_supports_mora() -> bool:
    try:
        params = inspect.signature(LoraConfig).parameters
    except (TypeError, ValueError):
        return False
    return "use_mora" in params and "mora_type" in params


class MoraPlugin(BasePlugin):
    """Plugin that exposes MoRA-specific config and validates runtime support."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.mora.MoraArgs"

    def get_adapter_capabilities(self) -> list[AdapterCapabilities]:
        return [AdapterCapabilities(name="mora", lora_like=True, relora=True)]

    def normalize_config_input(self, cfg: DictDefault):
        if cfg.get("adapter") != "mora":
            return
        mora_cfg = cfg.get("mora") or {}
        use_relora = bool(mora_cfg.get("use_relora"))
        use_relora_step = mora_cfg.get("use_relora_step")
        if use_relora_step is not None:
            cfg["relora"] = True
            if not cfg.get("jagged_restart_steps"):
                cfg["jagged_restart_steps"] = use_relora_step
            elif cfg.get("jagged_restart_steps") != use_relora_step:
                raise ValueError(
                    "mora.use_relora_step must match jagged_restart_steps when both are set"
                )
        elif use_relora:
            raise ValueError("mora.use_relora requires mora.use_relora_step")

    def validate_config(self, cfg: DictDefault):
        if cfg.adapter != "mora":
            return
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
            "mora_type": mora_cfg.mora_type,
        }

    def pre_model_load(self, cfg: DictDefault):
        if cfg.adapter != "mora":
            return
        LOG.info("MoRA plugin enabled for adapter: mora")

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        if cfg.adapter == "mora" and getattr(cfg, "mora", None):
            LOG.debug(
                "Loaded MoRA model with mora_type=%s, relora=%s",
                cfg.mora.mora_type,
                cfg.mora.use_relora,
            )

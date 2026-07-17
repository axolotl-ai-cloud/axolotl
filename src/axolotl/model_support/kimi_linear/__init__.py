"""Kimi-Linear model support.

The full modeling, configuration, and tokenization code ships in this
directory; the descriptor's pre-load hooks redirect transformers' remote-code
loading to these in-tree copies.
"""

from typing import TYPE_CHECKING

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.registry import register_model_support

if TYPE_CHECKING:
    from axolotl.utils.dict import DictDefault


@register_model_support
class KimiLinearSupport(ModelSupport):
    """Descriptor for Kimi-Linear (hybrid linear-attention remote-code model)."""

    model_types = ("kimi_linear",)

    def matches_cfg(self, cfg: "DictDefault") -> bool:
        return any(
            "kimi-linear" in (getattr(cfg, field, None) or "").lower()
            for field in ("base_model_config", "tokenizer_config")
        )

    def pre_config_load(self, cfg: "DictDefault") -> None:
        if "kimi-linear" in (getattr(cfg, "base_model_config", None) or "").lower():
            from .patch_kimi_linear import patch_kimi_config

            patch_kimi_config()

    def pre_tokenizer_load(self, cfg: "DictDefault") -> None:
        if "kimi-linear" in (getattr(cfg, "tokenizer_config", None) or "").lower():
            from .patch_kimi_linear import patch_kimi_tokenizer

            patch_kimi_tokenizer()

    def pre_model_load(self, cfg: "DictDefault") -> None:
        # catches checkpoints whose path lacks "kimi-linear"; patch is idempotent
        from .patch_kimi_linear import patch_kimi_model

        patch_kimi_model()

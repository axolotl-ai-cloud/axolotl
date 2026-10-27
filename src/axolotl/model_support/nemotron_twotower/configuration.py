"""Configuration for NVIDIA Nemotron-Labs-TwoTower (block-diffusion over Nemotron-H)."""

from __future__ import annotations

from huggingface_hub.dataclasses import strict
from transformers import AutoConfig
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from .names import TWOTOWER_ARCHITECTURE, TWOTOWER_MODEL_TYPE

__all__ = [
    "NemotronHTwoTowerConfig",
    "TWOTOWER_ARCHITECTURE",
    "TWOTOWER_MODEL_TYPE",
    "register_twotower_config",
]


@strict
class NemotronHTwoTowerConfig(NemotronHConfig):
    """Native Nemotron-H config plus the two-tower diffusion facts.

    The released checkpoint ships ``model_type: nemotron_h`` and a remote-code
    ``auto_map``; axolotl never executes that code. Loading goes through this
    subclass so the run resolves to its own ``model_type`` instead of hijacking
    plain Nemotron-H causal LMs. Both towers share this one config object, so a
    single ``set_experts_implementation`` reaches every MoE layer of both towers.

    ``mask_token_id`` defaults to 3, the id the reference
    ``generate_mask_diffusion`` uses as ``[MASK]`` (the tokenizer names it
    ``[INST]``). ``time_embed_frequency_size`` / ``time_embed_max_period`` are the
    reference ``TimestepEmbedder`` constants.
    """

    model_type = TWOTOWER_MODEL_TYPE

    mask_token_id: int = 3
    time_embed_frequency_size: int = 256
    time_embed_max_period: int = 1000

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        # The release labels itself ``nemotron_h``; claim it only when its
        # architectures say two-tower, so plain Nemotron-H configs still warn.
        config_dict, kwargs = super().get_config_dict(*args, **kwargs)
        if config_dict.get("model_type") == "nemotron_h" and TWOTOWER_ARCHITECTURE in (
            config_dict.get("architectures") or ()
        ):
            config_dict = {**config_dict, "model_type": TWOTOWER_MODEL_TYPE}
        return config_dict, kwargs

    def __post_init__(self, **kwargs):
        # The checkpoint's own ``model_type``/``auto_map`` describe the remote-code
        # class: drop them so this config (and anything saved from it) resolves to
        # the in-tree model and never points AutoModel at untrusted code.
        kwargs.pop("model_type", None)
        kwargs.pop("auto_map", None)
        super().__post_init__(**kwargs)
        self.__dict__.pop("model_type", None)
        self.__dict__.pop("auto_map", None)


def register_twotower_config() -> None:
    """Register the config with ``AutoConfig`` (idempotent)."""
    try:
        AutoConfig.register(TWOTOWER_MODEL_TYPE, NemotronHTwoTowerConfig)
    except ValueError:
        # already registered (re-import, or a second descriptor load)
        pass


register_twotower_config()

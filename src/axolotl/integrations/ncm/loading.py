"""Registers the NCM model types with transformers and axolotl's model loader."""

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig

from axolotl.model_support.base import (
    Capability,
    Experimental,
    ModelSupport,
    Unsupported,
)
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM
from axolotl.utils.logging import get_logger

from .configuration import NCMGeneratorConfig, NCMTokenizerConfig
from .modeling import NCMGeneratorModel, NCMTokenizerModel

LOG = get_logger(__name__)

AutoConfig.register(NCMTokenizerConfig.model_type, NCMTokenizerConfig)
AutoConfig.register(NCMGeneratorConfig.model_type, NCMGeneratorConfig)
AutoModel.register(NCMTokenizerConfig, NCMTokenizerModel)
AutoModel.register(NCMGeneratorConfig, NCMGeneratorModel)


_HUB_KWARGS = ("revision", "token", "cache_dir", "local_files_only")


def checkpoint_model_type(source: str, **hub_kwargs) -> str | None:
    """``model_type`` of the config at ``source``; None if it cannot be read."""
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(source, **hub_kwargs)
    except (OSError, ValueError):
        return None
    return config_dict.get("model_type")


class NCMAutoModel:
    """Loader shim. A generator config over a stage-1 checkpoint starts a fresh
    generator around the trained (frozen) tokenizer."""

    def __new__(cls, config, **kwargs):
        return cls.from_config(config, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        if isinstance(config, NCMGeneratorConfig):
            return NCMGeneratorModel(config)
        return NCMTokenizerModel(config)

    @classmethod
    def from_pretrained(cls, source, config=None, **kwargs):
        # Training keeps fp32 master weights under autocast, and Trainer/DDP places the model.
        hub = {k: kwargs[k] for k in _HUB_KWARGS if kwargs.get(k) is not None}
        if isinstance(config, NCMGeneratorConfig):
            if checkpoint_model_type(source, **hub) == NCMTokenizerConfig.model_type:
                model = NCMGeneratorModel(config)
                tokenizer = NCMTokenizerModel.from_pretrained(
                    source, dtype=torch.float32, **hub
                )
                model.vqvae.load_state_dict(tokenizer.state_dict())
                LOG.info("NCM: new generator over the stage-1 tokenizer at %s", source)
                return model
            return NCMGeneratorModel.from_pretrained(
                source, config=config, dtype=torch.float32, **hub
            )
        return NCMTokenizerModel.from_pretrained(
            source, config=config, dtype=torch.float32, **hub
        )


def _bind_tokenizer(context: ModelHookContext) -> None:
    """Fill token ids the model needs from the run's text tokenizer."""
    config, tokenizer = context.model_config, context.tokenizer
    if tokenizer is None or not isinstance(
        config, (NCMTokenizerConfig, NCMGeneratorConfig)
    ):
        return
    tok_cfg = (
        config.tokenizer_config if isinstance(config, NCMGeneratorConfig) else config
    )
    if len(tokenizer) > tok_cfg.vocab_size:
        fix = (
            "use the stage-1 run's text tokenizer"
            if isinstance(config, NCMGeneratorConfig)
            else "raise model_config.vocab_size"
        )
        raise ValueError(
            f"NCM tokenizer vocab_size={tok_cfg.vocab_size} is smaller than the text "
            f"tokenizer ({len(tokenizer)}); {fix}."
        )
    if tok_cfg.pad_token_id is None:
        tok_cfg.pad_token_id = tokenizer.pad_token_id
    if isinstance(config, NCMGeneratorConfig) and config.null_prefix_token_id is None:
        config.null_prefix_token_id = tokenizer.eos_token_id


def _register_cce_patch(_context: ModelHookContext) -> None:
    """Point cut_cross_entropy's registry at the NCM patch.

    Runs before the CCE plugin's pre_model_load, which resolves PATCH_FNS by
    string import, so the entry only has to exist by then.
    """
    try:
        from cut_cross_entropy.transformers.patch import PATCH_FNS
    except ImportError:
        return
    target = ("axolotl.integrations.ncm.cut_cross_entropy", "patch_ncm")
    for model_type in (NCMTokenizerConfig.model_type, NCMGeneratorConfig.model_type):
        PATCH_FNS.setdefault(model_type, target)


_CAPABILITIES: dict[str, Capability] = {
    name: Unsupported("NCM models are small in-tree transformers without these hooks.")
    for name in (
        "liger",
        "lora_kernels",
        "fused_attn_kernel",
        "context_parallel",
        "expert_kernels",
        "quantized_lora",
    )
}
_CAPABILITIES["cut_cross_entropy"] = Experimental(
    "Patched in-tree (cut_cross_entropy.py); needs bf16 or fp16 training."
)


@register_model_support
class NCMSupport(ModelSupport):
    """Next-Concept Modeling tokenizer (stage 1) and generator (stage 2)."""

    model_types = (NCMTokenizerConfig.model_type, NCMGeneratorConfig.model_type)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities=_CAPABILITIES,
        strategies=ModelStrategyOverrides(auto_model_cls=lambda: NCMAutoModel),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_MODEL_BUILD: (
                    _bind_tokenizer,
                    _register_cce_patch,
                )
            }
        ),
    )

"""Native descriptor for NVIDIA Nemotron-Labs-TwoTower (block diffusion, hybrid MoE).

The checkpoint (``nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16``) declares
``model_type: nemotron_h`` plus a remote-code ``auto_map``. Axolotl runs it
in-tree instead: a ``BEFORE_CONFIG_LOAD`` hook points ``cls_model_config`` at
:class:`NemotronHTwoTowerConfig` (``model_type: nemotron_h_twotower``), so plain
Nemotron-H runs keep their own support path, and the model is built from
transformers' native Nemotron-H modules (:mod:`.modeling`).

Run-config handling (``CONFIGURE_RUN``): only the denoiser tower is trainable, so
LoRA ``lora_target_modules`` / ``lora_target_parameters`` and PEFT trainable-token
module names are scoped to ``denoiser_tower`` / ``lm_head``; an
``AFTER_ADAPTER_LOAD`` check refuses any trainable context-tower parameter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from axolotl.model_support.base import ModelSupport, Supported, Unsupported
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelRegistrationOverrides,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import DIFFUSION_LM
from axolotl.utils.logging import get_logger

from .names import TWOTOWER_ARCHITECTURE, TWOTOWER_MODEL_TYPE

LOG = get_logger(__name__)

CONFIG_CLASS_PATH = (
    "axolotl.model_support.nemotron_twotower.configuration.NemotronHTwoTowerConfig"
)
DENOISER_PREFIX = "denoiser_tower"
CONTEXT_PREFIXES = ("context_tower", "context_lm_head")
EXPERT_PARAMETERS = ("up_proj", "down_proj")
_EMBEDDING_ALIASES = {
    "embed_tokens": f"{DENOISER_PREFIX}.embeddings",
    "embeddings": f"{DENOISER_PREFIX}.embeddings",
    "diffusion_head": "lm_head",
    "lm_head": "lm_head",
}


def _model_class():
    from .modeling import NemotronHTwoTowerForDiffusion

    class AutoTwoTowerModel:
        """Loader shim: never forwards ``trust_remote_code`` to the in-tree class."""

        def __new__(cls, config, **kwargs):
            return cls.from_config(config, **kwargs)

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            kwargs.pop("trust_remote_code", None)
            register_twotower_weight_conversions()
            return NemotronHTwoTowerForDiffusion.from_pretrained(model_source, **kwargs)

        @classmethod
        def from_config(cls, config, **kwargs):
            kwargs.pop("trust_remote_code", None)
            return NemotronHTwoTowerForDiffusion._from_config(config, **kwargs)

    return AutoTwoTowerModel


_ARCHITECTURES = {TWOTOWER_ARCHITECTURE, "NemotronHTwoTowerForDiffusion"}


def _local_config(source: str) -> dict:
    path = Path(source) / "config.json"
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _matches_cfg(cfg) -> bool:
    """The released repo (by name), or a local checkpoint whose config says two-tower
    (the release's architecture, or a checkpoint axolotl saved from this model)."""
    source = getattr(cfg, "base_model_config", None) or getattr(cfg, "base_model", None)
    if not isinstance(source, str):
        return False
    if "nemotron-labs-twotower" in source.lower():
        return True
    config = _local_config(source)
    architectures = {str(name) for name in config.get("architectures") or ()}
    return config.get("model_type") == TWOTOWER_MODEL_TYPE or bool(
        architectures & _ARCHITECTURES
    )


def _weight_conversions():
    """Per-expert checkpoint tensors -> transformers' fused ``[E, out, in]`` experts.

    Reuses the stock ``nemotron_h`` expert converters (``MergeModulelist`` over
    ``mixer.experts.*.{up,down}_proj.weight``); their patterns are unanchored, so
    they match under both ``context_tower.`` and ``denoiser_tower.``.
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter

    stock = get_checkpoint_conversion_mapping("nemotron_h") or []
    converters = [entry for entry in stock if isinstance(entry, WeightConverter)]
    if not converters:
        raise RuntimeError("transformers ships no nemotron_h expert converters")
    return {TWOTOWER_MODEL_TYPE: converters}


def register_twotower_weight_conversions() -> None:
    """Register the expert converters with transformers (idempotent)."""
    from axolotl.utils.weight_conversions import register_weight_conversions

    for key, entries in _weight_conversions().items():
        register_weight_conversions(key, entries)


def _before_config_load(context: ModelHookContext) -> None:
    cfg = context.cfg
    current = getattr(cfg, "cls_model_config", None)
    if current and current != CONFIG_CLASS_PATH:
        raise ValueError(
            f"Nemotron-Labs-TwoTower loads through {CONFIG_CLASS_PATH}; "
            f"cls_model_config={current!r} would bypass it."
        )
    cfg.cls_model_config = CONFIG_CLASS_PATH


def _moe_layers(model_config) -> list[int]:
    return [
        index
        for index, kind in enumerate(model_config.layers_block_type)
        if kind == "moe"
    ]


def scope_target_modules(target_modules) -> str | None:
    """Turn a module-name list into a regex limited to the denoiser tower.

    PEFT suffix-matches list entries, which would also wrap the frozen context
    tower's identically named projections. A string is already a full-match regex
    and is kept (the post-load check still refuses context-tower adapters).
    """
    if not target_modules:
        return None
    if isinstance(target_modules, str):
        return target_modules
    names = sorted({str(name) for name in target_modules})
    if "lm_head" in names or "context_lm_head" in names:
        raise ValueError(
            "TwoTower lora_target_modules cover denoiser-tower layers only; train the "
            "head's codebook rows with peft_trainable_token_indices (diffusion_head)"
        )
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(
                "TwoTower lora_target_modules entries must be bare module names "
                f"(got {name!r}); pass a single regex string for anything else"
            )
    alternatives = "|".join(re.escape(name) for name in names)
    return rf"{DENOISER_PREFIX}\..*\.(?:{alternatives})"


def scope_target_parameters(target_parameters, model_config) -> list[str]:
    """Expand expert-parameter targets to explicit denoiser-tower parameter names.

    PEFT ``target_parameters`` suffix-match without honouring ``exclude_modules``,
    so ``mixer.experts.up_proj`` alone would wrap both towers' experts.
    """
    if not target_parameters:
        return []
    if isinstance(target_parameters, str):
        target_parameters = [target_parameters]
    scoped: list[str] = []
    moe_layers = _moe_layers(model_config)
    for entry in target_parameters:
        entry = str(entry)
        if entry.startswith(f"{DENOISER_PREFIX}."):
            scoped.append(entry)
            continue
        module, _, parameter = entry.rpartition(".")
        if parameter not in EXPERT_PARAMETERS or not (
            module == "experts" or module.endswith(".experts")
        ):
            raise ValueError(
                "TwoTower lora_target_parameters supports the routed experts' "
                f"{EXPERT_PARAMETERS} (e.g. 'mixer.experts.up_proj'); got {entry!r}"
            )
        scoped.extend(
            f"{DENOISER_PREFIX}.layers.{index}.mixer.experts.{parameter}"
            for index in moe_layers
        )
    return list(dict.fromkeys(scoped))


def scope_trainable_token_indices(indices):
    """Map trainable-token module names onto the denoiser embedding / head."""
    if not isinstance(indices, Mapping):
        # A plain list targets get_input_embeddings(), the denoiser's table.
        return indices
    scoped = {}
    for name, ids in indices.items():
        target = _EMBEDDING_ALIASES.get(str(name), str(name))
        if target.startswith(CONTEXT_PREFIXES):
            raise ValueError(
                f"TwoTower trains only the denoiser tower; trainable tokens on {name!r} "
                "would target the frozen context tower"
            )
        if target != "lm_head" and not target.startswith(f"{DENOISER_PREFIX}."):
            raise ValueError(
                f"TwoTower trainable-token module {name!r} is not the denoiser embedding "
                "(embed_tokens) or head (diffusion_head / lm_head)"
            )
        if target in scoped:
            merged = list(dict.fromkeys([*scoped[target], *ids]))
            scoped[target] = merged
        else:
            scoped[target] = list(ids)
    return scoped


def _validate(context: ModelHookContext) -> None:
    cfg = context.cfg
    model_config = context.model_config
    if getattr(cfg, "trust_remote_code", False):
        LOG.warning(
            "Nemotron-Labs-TwoTower runs from axolotl's in-tree implementation; "
            "trust_remote_code is unused for it and can be set to false."
        )
    if getattr(cfg, "attn_implementation", None) is None:
        # Document boundaries come from position_ids; varlen packing metadata is
        # cheap, whereas the dense backend would build a [rows, T, T] mask.
        cfg.attn_implementation = "varlen"
    if getattr(cfg, "diffusion_lm", None) is not None and not getattr(
        cfg, "diffusion_decision", None
    ):
        # Only the decision trainer tells the model where each canvas starts.
        raise ValueError(
            "Nemotron-Labs-TwoTower currently trains through the diffusion_decision "
            "plugin only (plain diffusion_lm training/generation is not implemented)"
        )
    if getattr(cfg, "cut_cross_entropy", False):
        raise ValueError("Nemotron-Labs-TwoTower does not support cut_cross_entropy")
    if getattr(cfg, "load_in_4bit", False) or getattr(cfg, "load_in_8bit", False):
        raise ValueError("Nemotron-Labs-TwoTower quantized LoRA is not supported yet")
    if getattr(cfg, "lora_target_linear", False):
        raise ValueError(
            "Nemotron-Labs-TwoTower: lora_target_linear would include the frozen "
            "context tower; list lora_target_modules explicitly"
        )
    diffusion = getattr(cfg, "diffusion_lm", None)
    expected_mask = getattr(model_config, "mask_token_id", None)
    if diffusion is not None and expected_mask is not None:
        configured = (
            diffusion.get("mask_token_id")
            if isinstance(diffusion, Mapping)
            else getattr(diffusion, "mask_token_id", None)
        )
        if configured is None:
            if isinstance(diffusion, MutableMapping):
                diffusion["mask_token_id"] = expected_mask
            else:
                diffusion.mask_token_id = expected_mask
        elif int(configured) != int(expected_mask):
            raise ValueError(
                f"Nemotron-Labs-TwoTower was trained with mask_token_id={expected_mask}; "
                f"diffusion_lm.mask_token_id={configured} would feed it an unseen mask"
            )
    if getattr(cfg, "adapter", None):
        scoped_modules = scope_target_modules(getattr(cfg, "lora_target_modules", None))
        if scoped_modules is not None:
            cfg.lora_target_modules = scoped_modules
        if model_config is not None:
            scoped_parameters = scope_target_parameters(
                getattr(cfg, "lora_target_parameters", None), model_config
            )
            if scoped_parameters:
                cfg.lora_target_parameters = scoped_parameters
        indices = getattr(cfg, "peft_trainable_token_indices", None)
        if indices:
            cfg.peft_trainable_token_indices = scope_trainable_token_indices(indices)


def _check_context_frozen(context: ModelHookContext) -> None:
    model = context.model
    if model is None:
        return
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(f"{prefix}." in f".{name}" for prefix in CONTEXT_PREFIXES)
    ]
    if trainable:
        raise ValueError(
            "Nemotron-Labs-TwoTower: the context tower must stay frozen, but these "
            f"parameters are trainable: {trainable[:8]}"
        )


@register_model_support
class NemotronTwoTowerSupport(ModelSupport):
    """Two-tower block diffusion: frozen AR context tower + trainable denoiser."""

    model_types = (TWOTOWER_MODEL_TYPE,)
    profile = ModelProfile(
        family=DIFFUSION_LM,
        diffusion=DiffusionSpec(
            noise=DiffusionNoise.ABSORBING,
            layout=DiffusionLayout.FULL_SEQUENCE,
            logit_alignment=LogitAlignment.ALIGNED,
            first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
            self_conditioning=False,
            max_canvas=None,
            max_context=262144,
            eos_handling=EosHandling.INDEPENDENT,
            mask_token_policy=MaskTokenPolicy.MODEL,
            default_time_weighting=TimeWeighting.INV_T,
            objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
            time_floor=0.001,
            reduction_scope=ReductionScope.GLOBAL_WINDOW,
            generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        ),
        capabilities={
            "diffusion_varlen": Supported(
                "Documents are split from position_ids; varlen metadata is accepted."
            ),
            "cut_cross_entropy": Unsupported("The canvas head is projected directly."),
            "fused_attn_kernel": Unsupported("Attention is evaluated in-tree."),
            "lora_kernels": Unsupported("LoRA kernels patch single-tower attention."),
            "quantized_lora": Unsupported("Not validated for the two-tower model."),
            "fsdp": Unsupported("Not validated for the two-tower model."),
        },
        strategies=ModelStrategyOverrides(auto_model_cls=_model_class),
        registrations=ModelRegistrationOverrides(
            weight_conversions=_weight_conversions
        ),
        matchers=ModelMatchers(cfg=_matches_cfg),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (_before_config_load,),
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_check_context_frozen,),
            }
        ),
    )

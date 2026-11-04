"""Version-neutral access to transformers' activation checkpointing API, and readers for
the `activation_checkpointing` config block.

transformers 5.23 renamed ``gradient_checkpointing_enable``/``_disable`` and the
``gradient_checkpointing*`` training arguments to ``activation_checkpointing*``, keeping
the old names as deprecated aliases until v6. Hooks must target the name ``Trainer``
calls: on 5.23+ ``activation_checkpointing_enable`` hands off to an overridden
``gradient_checkpointing_enable``, so a model that overrides both recurses.
"""

from __future__ import annotations

from transformers import PreTrainedModel, TrainingArguments

from axolotl.utils.dict import DictDefault

HAS_ACTIVATION_CHECKPOINTING_API = hasattr(
    PreTrainedModel, "activation_checkpointing_enable"
)

if "activation_checkpointing" in TrainingArguments.__dataclass_fields__:
    CHECKPOINTING_ARG = "activation_checkpointing"
else:
    CHECKPOINTING_ARG = "gradient_checkpointing"
CHECKPOINTING_KWARGS_ARG = f"{CHECKPOINTING_ARG}_kwargs"


def checkpointing_enable_name(model) -> str:
    """Name of the enable method ``Trainer`` calls on ``model``."""
    if hasattr(model, "activation_checkpointing_enable"):
        return "activation_checkpointing_enable"
    return "gradient_checkpointing_enable"


def enable_activation_checkpointing(
    model, checkpointing_kwargs: dict | None = None, **kwargs
) -> None:
    checkpointing_kwargs = dict(checkpointing_kwargs or {})
    # a parameter of the enable method; torch would hand it to every layer's forward
    if "every_n_layers" in checkpointing_kwargs:
        kwargs.setdefault("every_n_layers", checkpointing_kwargs.pop("every_n_layers"))
    name = checkpointing_enable_name(model)
    getattr(model, name)(
        **{name.replace("_enable", "_kwargs"): checkpointing_kwargs or None}, **kwargs
    )


def disable_activation_checkpointing(model) -> None:
    if hasattr(model, "activation_checkpointing_disable"):
        model.activation_checkpointing_disable()
    else:
        model.gradient_checkpointing_disable()


def checkpointing_config(cfg) -> DictDefault:
    """The `activation_checkpointing` block of a validated config; disabled when unset."""
    if hasattr(cfg, "get"):
        block = cfg.get("activation_checkpointing")
    else:
        block = getattr(cfg, "activation_checkpointing", None)
    if block is True:
        block = {"enabled": True}
    elif not block:
        block = {"enabled": False}
    elif "enabled" not in block:
        # the dict form means on unless it says otherwise
        block = {**block, "enabled": True}
    return block if isinstance(block, DictDefault) else DictDefault(block)


def hf_checkpointing_enabled(cfg) -> bool:
    """Whether transformers' layer checkpointing (not FSDP's wrapper) is on."""
    block = checkpointing_config(cfg)
    return bool(block.enabled) and (block.implementation or "hf") == "hf"


def fsdp_checkpointing_enabled(cfg) -> bool:
    block = checkpointing_config(cfg)
    return bool(block.enabled) and block.implementation == "fsdp"


def offload_placement(cfg, target: str) -> str:
    """`none`, `cpu` or `disk` for an `activation_checkpointing.offload` target."""
    offload = checkpointing_config(cfg).get("offload") or {}
    placement = offload.get(target)
    if placement is True:
        return "cpu"
    return placement or "none"


def checkpointing_kwargs(cfg) -> dict:
    """`use_reentrant` plus the passthrough kwargs, as TrainingArguments takes them."""
    block = checkpointing_config(cfg)
    return {"use_reentrant": bool(block.reentrant), **(block.get("kwargs") or {})}

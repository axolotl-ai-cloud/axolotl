"""Version-neutral access to transformers' activation checkpointing API.

transformers 5.23 renamed ``gradient_checkpointing_enable``/``_disable`` and the
``gradient_checkpointing*`` training arguments to ``activation_checkpointing*``, keeping
the old names as deprecated aliases until v6. Hooks must target the name ``Trainer``
calls: on 5.23+ ``activation_checkpointing_enable`` hands off to an overridden
``gradient_checkpointing_enable``, so a model that overrides both recurses.
"""

from __future__ import annotations

from transformers import PreTrainedModel, TrainingArguments

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
    name = checkpointing_enable_name(model)
    getattr(model, name)(
        **{name.replace("_enable", "_kwargs"): checkpointing_kwargs}, **kwargs
    )


def disable_activation_checkpointing(model) -> None:
    if hasattr(model, "activation_checkpointing_disable"):
        model.activation_checkpointing_disable()
    else:
        model.gradient_checkpointing_disable()

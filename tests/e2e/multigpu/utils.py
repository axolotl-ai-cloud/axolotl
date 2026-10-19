"""Shared configuration for FSDP2 training checks."""

from axolotl.utils.dict import DictDefault


def multigpu_training_config(overrides: dict) -> DictDefault:
    return DictDefault({"seed": 42, **overrides})


def fsdp2_training_config(overrides: dict) -> DictDefault:
    return multigpu_training_config(
        {
            "max_steps": 40,
            "warmup_steps": 5,
            **overrides,
        }
    )

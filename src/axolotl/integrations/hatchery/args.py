# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pydantic config schema for the Hatchery integration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HatcheryConfig(BaseModel):
    """Nested config under `hatchery:` in the axolotl YAML.

    Only contains hatchery-specific settings. Standard training params
    (learning_rate, weight_decay, adam_beta1/2, max_grad_norm,
    gradient_accumulation_steps) are read from axolotl's top-level config.
    """

    # Backend & connection
    backend: Literal["tinker", "hatchery"] = "tinker"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    project_id: Optional[str] = None

    # LoRA config sent to remote
    lora_rank: int = Field(32, ge=1, le=256)
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True

    # Loss function
    loss_fn: Literal[
        "cross_entropy", "importance_sampling", "ppo", "cispo", "dro"
    ] = "cross_entropy"
    loss_fn_config: Optional[dict[str, Any]] = None

    # Pipelining: submit next batch before awaiting previous result
    pipeline: bool = True

    # Sampling params (for RL flows)
    max_sample_tokens: int = 256
    sample_temperature: float = 1.0
    num_samples: int = 4

    # Reward functions (for RL) — list of fully qualified names
    reward_funcs: Optional[list[str]] = None

    # Checkpointing
    save_steps: Optional[int] = None
    save_name_prefix: str = "checkpoint"

    # Timeout per future (seconds)
    future_timeout: float = 600.0


class HatcheryArgs(BaseModel):
    """Top-level mixin that adds the nested `hatchery:` field."""

    hatchery: Optional[HatcheryConfig] = None

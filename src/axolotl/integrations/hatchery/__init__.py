# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Hatchery/Tinker remote training integration for Axolotl.

Routes axolotl's preprocessed data to a remote training API (Tinker or
Hatchery) instead of running forward/backward locally. The remote
service handles model weights, LoRA adapters, and gradient updates.
"""

from .args import HatcheryArgs, HatcheryConfig
from .plugin import HatcheryPlugin

__all__ = ["HatcheryArgs", "HatcheryConfig", "HatcheryPlugin"]

# Usage:
#   plugins:
#     - axolotl.integrations.hatchery.HatcheryPlugin
#
#   hatchery:
#     backend: tinker  # or "hatchery"
#     lora_rank: 32
#     learning_rate: 1e-4
#     loss_fn: cross_entropy  # SFT
#     # loss_fn: ppo         # RL (auto-selects HatcheryRLTrainer)

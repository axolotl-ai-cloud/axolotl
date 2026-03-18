# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Plugin for NVIDIA NeMo Gym integration with Axolotl.

NeMo Gym provides RL training environments for LLMs with verification-based
reward signals. This plugin manages the NeMo Gym server lifecycle, loads
datasets in the NeMo Gym JSONL format, and creates reward functions that
call the NeMo Gym /verify endpoints.
"""

from .args import NemoGymArgs
from .plugin import NemoGymPlugin
from .rewards import reward_env, reward_nemo_gym_verify

__all__ = [
    "NemoGymArgs",
    "NemoGymPlugin",
    "reward_env",
    "reward_nemo_gym_verify",
]

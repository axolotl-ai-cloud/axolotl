"""
extra axolotl specific training args
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type

from transformers import TrainingArguments
from trl import RewardConfig
from trl.experimental.cpo import CPOConfig
from trl.experimental.kto import KTOConfig
from trl.experimental.orpo import ORPOConfig
from trl.experimental.prm import PRMConfig

from axolotl.integrations.config import merge_training_args

AxolotlTrainingMixins: Type = merge_training_args()


@dataclass
class AxolotlTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    """
    Training arguments for Causal trainer

    This code is duplicated due to HF TrainingArguments not setting output_dir with a
    default value so it can't be used as a mixin.
    """


@dataclass
class AxolotlORPOConfig(AxolotlTrainingMixins, ORPOConfig):
    """
    ORPO config for ORPO training
    """


@dataclass
class AxolotlKTOConfig(AxolotlTrainingMixins, KTOConfig):
    """
    KTO config for KTO training
    """


@dataclass
class AxolotlCPOConfig(AxolotlTrainingMixins, CPOConfig):
    """
    CPO config for CPO training
    """

    simpo_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "simpo gamma parameter"},
    )


@dataclass
class AxolotlRewardConfig(AxolotlTrainingMixins, RewardConfig):
    """
    Reward config for Reward training
    """


@dataclass
class AxolotlPRMConfig(AxolotlTrainingMixins, PRMConfig):
    """
    PRM config for PRM training
    """

# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plugin args for on-policy distillation (GKD). Each arg maps to an OPD axis:
A divergence (gkd_beta, gkd_divergence), B rollout (gkd_lmbda, gkd_seq_kd,
gkd_max_new_tokens, gkd_top_k), C teacher (gkd_teacher, gkd_teacher_init_kwargs).
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, model_validator


class GKDArgs(BaseModel):
    """Input args for on-policy (generalized) knowledge distillation."""

    gkd_trainer: bool | None = None
    gkd_teacher: str | None = None  # HF model id/path; must share the student's vocab
    gkd_teacher_init_kwargs: dict[str, Any] | None = None
    gkd_lmbda: float | None = 1.0  # fraction of steps trained on student rollouts
    gkd_seq_kd: bool | None = (
        False  # off-policy fraction distills teacher-generated seqs
    )
    gkd_max_new_tokens: int | None = 128
    gkd_top_k: int | None = 0
    gkd_beta: float | None = 1.0  # 0=forward-KL, 1=reverse-KL, between=JSD
    gkd_divergence: str | None = None
    gkd_temperature: float | None = 0.9

    @model_validator(mode="after")
    def _validate_gkd(self):
        if not self.gkd_trainer:
            return self
        if not self.gkd_teacher:
            raise ValueError("gkd_teacher is required when gkd_trainer is set.")
        if self.gkd_beta is not None and not 0.0 <= self.gkd_beta <= 1.0:
            raise ValueError("gkd_beta must be in [0, 1].")
        if self.gkd_lmbda is not None and not 0.0 <= self.gkd_lmbda <= 1.0:
            raise ValueError("gkd_lmbda must be in [0, 1].")
        if self.gkd_temperature is not None and self.gkd_temperature <= 0.0:
            raise ValueError("gkd_temperature must be > 0.")
        # On-policy rollouts need one prompt per unpacked sequence, with the prompt masked.
        if getattr(self, "sample_packing", None):
            raise ValueError(
                "gkd_trainer is incompatible with sample_packing; set sample_packing: false."
            )
        if getattr(self, "train_on_inputs", None):
            raise ValueError(
                "gkd_trainer requires train_on_inputs: false to separate prompt from completion."
            )
        return self


@dataclass
class GKDTrainingArgsMixin:
    """Training args consumed by ``AxolotlGKDTrainer`` off ``self.args``."""

    gkd_teacher: str | None = None
    gkd_teacher_init_kwargs: dict[str, Any] | None = None
    gkd_lmbda: float = 1.0
    gkd_seq_kd: bool = False
    gkd_max_new_tokens: int = 128
    gkd_top_k: int = 0
    gkd_beta: float = 1.0
    gkd_divergence: str | None = None
    gkd_temperature: float = 0.9

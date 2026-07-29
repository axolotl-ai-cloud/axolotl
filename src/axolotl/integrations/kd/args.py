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
Plugin args for KD support.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

KD_OFFLINE_DATASET_TYPE = "axolotl.integrations.kd.chat_template"


def _dataset_type(dataset) -> str:
    ds_type = (
        dataset.get("type")
        if isinstance(dataset, dict)
        else getattr(dataset, "type", None)
    )
    return str(ds_type) if isinstance(ds_type, str) else ""


class InferenceServerType(str, Enum):
    """
    Online inferences server types to handle different request args
    """

    vllm = "vllm"
    sglang = "sglang"


class KDArgs(BaseModel):
    """
    Input args for knowledge distillation.
    """

    kd_trainer: bool | None = None  # whether to use KD trainer
    kd_ce_alpha: float | None = (
        None  # loss coefficient for cross-entropy loss during KD
    )
    kd_alpha: float | None = None  # loss coefficient for KD loss
    kd_temperature: float | None = None  # temperature for sampling during KD
    kd_beta: float | None = 0.0  # beta coefficient for ratio of fwd and reverse KL
    kd_normalize_topk: bool | None = (
        None  # whether to normalize student logits during KD
    )
    kd_compiled_kernel: bool | None = None  # torch.compile the chunked KD loss kernel

    kd_prepared_targets_alignment: Literal["current", "legacy"] | None = Field(
        default="current",
        description=(
            "Which convention the dataset's baked target_* columns use, for datasets "
            "prepared ahead of time (typically loaded with skip_prepare_dataset). "
            "'current' means row j holds the teacher distribution over token j+1, what "
            "the loss expects and what axolotl > 0.18.0 prepares. 'legacy' means row j "
            "holds the distribution over token j, which is what axolotl <= 0.18.0 "
            "prepared; those rows are shifted into place at collation time so the "
            "dataset does not have to be rebuilt."
        ),
    )

    kd_online_server_base_url: str | None = None
    kd_online_topk: int | None = None
    kd_online_server: InferenceServerType | None = Field(
        default_factory=lambda: InferenceServerType.vllm
    )
    kd_online_timeout: int | None = 120
    kd_online_preflight: bool | None = Field(
        default=True,
        description=(
            "Probe the online teacher once at startup to check connectivity, the "
            "response contract and the server's logprob cap, so a teacher that cannot "
            "serve this config fails the run immediately instead of inside every batch."
        ),
    )
    kd_temperature_min: float | None = (
        None  # kd temperature scheduling during online kd
    )

    @model_validator(mode="after")
    def check_kd_args(self):
        if not self.kd_trainer:
            return self

        if self.kd_alpha is None:
            self.kd_alpha = 1.0
        if self.kd_ce_alpha is None:
            self.kd_ce_alpha = 0.0
        for name in ("kd_alpha", "kd_ce_alpha"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        if self.kd_alpha == 0.0 and self.kd_ce_alpha == 0.0:
            raise ValueError("kd_alpha and kd_ce_alpha cannot both be 0.0")

        if self.kd_temperature is None:
            self.kd_temperature = 1.0
        if self.kd_temperature <= 0.0:
            raise ValueError(
                f"kd_temperature must be positive, got {self.kd_temperature}"
            )
        if self.kd_beta is not None and not 0.0 <= self.kd_beta <= 1.0:
            raise ValueError(f"kd_beta must be between 0.0 and 1.0, got {self.kd_beta}")
        if self.kd_normalize_topk is None:
            self.kd_normalize_topk = True
        if self.kd_compiled_kernel is None:
            self.kd_compiled_kernel = True
        if self.kd_prepared_targets_alignment is None:
            self.kd_prepared_targets_alignment = "current"

        if self.kd_temperature_min is not None:
            if not self.kd_online_server_base_url:
                raise ValueError(
                    "kd_temperature_min schedules the online teacher temperature and "
                    "requires kd_online_server_base_url"
                )
            if not 0.0 < self.kd_temperature_min <= self.kd_temperature:
                raise ValueError(
                    "kd_temperature_min must be positive and <= kd_temperature "
                    f"({self.kd_temperature}), got {self.kd_temperature_min}"
                )

        if self.kd_online_server_base_url:
            if not self.kd_online_topk or self.kd_online_topk <= 0:
                raise ValueError(
                    "kd_online_topk must be a positive integer when using an online teacher"
                )
            if self.kd_prepared_targets_alignment == "legacy":
                raise ValueError(
                    "kd_prepared_targets_alignment describes baked target_* columns; an "
                    "online teacher produces its targets fresh, so 'legacy' has no meaning"
                )
            if any(
                _dataset_type(dataset).startswith(KD_OFFLINE_DATASET_TYPE)
                for dataset in (getattr(self, "datasets", None) or [])
            ):
                raise ValueError(
                    "kd_online_server_base_url cannot be combined with the offline KD "
                    f"dataset type ({KD_OFFLINE_DATASET_TYPE}); the offline logprobs "
                    "would be overwritten by the online teacher"
                )
        elif self.kd_online_topk is not None:
            raise ValueError(
                "kd_online_topk requires kd_online_server_base_url to be set"
            )

        return self


@dataclass
class KDTrainingArgsMixin:
    """
    Additional args for KD training.
    """

    kd_ce_alpha: float | None = (
        None  # loss coefficient for cross-entropy loss during KD
    )
    kd_alpha: float | None = None  # loss coefficient for KD loss
    kd_temperature: float | None = None  # temperature for sampling during KD
    kd_beta: float | None = None  # beta coefficient for ratio of fwd and reverse KL
    kd_normalize_topk: bool | None = (
        None  # whether to normalize student logits during KD
    )
    kd_compiled_kernel: bool | None = None  # torch.compile the chunked KD loss kernel

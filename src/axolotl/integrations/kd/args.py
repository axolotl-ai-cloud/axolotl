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
from enum import Enum

from pydantic import BaseModel


class InferenceServerType(str, Enum):
    vllm = "vllm"
    sglang = "sglang"


class KDArgs(BaseModel):
    """
    Input args for knowledge distillation.
    """

    kd_trainer: float | None = None  # whether to use KD trainer
    kd_ce_alpha: float | None = (
        None  # loss coefficient for cross-entropy loss during KD
    )
    kd_alpha: float | None = None  # loss coefficient for KD loss
    kd_temperature: float | None = None  # temperature for sampling during KD

    # TODO online kd
    kd_online_server_base_url: str | None = None
    kd_online_topk: int | None = None
    kd_online_server: InferenceServerType | None = "vllm"

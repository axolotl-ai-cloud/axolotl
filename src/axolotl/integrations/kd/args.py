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
from typing import Optional

from pydantic import BaseModel


class KDArgs(BaseModel):
    """
    Input args for knowledge distillation.
    """

    kd_trainer: Optional[bool] = None  # whether to use KD trainer
    kd_ce_alpha: Optional[float] = (
        None  # loss coefficient for cross-entropy loss during KD
    )
    kd_alpha: Optional[float] = None  # loss coefficient for KD loss
    kd_temperature: Optional[float] = None  # temperature for sampling during KD
    kd_zscore_base_temp: Optional[float] = None  # base temperature for zscore scaling
    kd_top_k_before_softmax: Optional[bool] = (
        None  # whether to sample top k before softmax during KD
    )

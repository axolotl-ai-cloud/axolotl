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
Module for handling Cut Cross Entropy input arguments.
"""
import logging
from typing import Optional

from pydantic import BaseModel, model_validator

LOG = logging.getLogger("axolotl.integrations.cut_cross_entropy.args")


class CutCrossEntropyArgs(BaseModel):
    """
    Input args for Cut Cross Entropy.
    """

    cut_cross_entropy: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_dtype_is_half(cls, data):
        if data.get("cut_cross_entropy") and not (data.get("bf16") or data.get("fp16")):
            raise ValueError(
                "Cut Cross Entropy requires fp16/bf16 training for backward pass. "
                "Please set `bf16` or `fp16` to `True`."
            )

        return data

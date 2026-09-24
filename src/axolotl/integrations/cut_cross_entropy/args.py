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

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CutCrossEntropyArgs(BaseModel):
    """
    Input args for Cut Cross Entropy.
    """

    cut_cross_entropy: Optional[bool] = True
    cut_cross_entropy_accum_c_fp32: Optional[bool] = Field(
        default=False,
        json_schema_extra={
            "description": "Accumulate the classifier (lm_head) gradient in fp32 for better numerical stability at the cost of a full fp32 copy of the gradient."
        },
    )
    cut_cross_entropy_c_grad_chunk_size: Optional[int | Literal["auto"]] = Field(
        default=None,
        json_schema_extra={
            "description": "Bound the fp32 classifier-gradient accumulator to this many vocabulary rows (a positive multiple of 128) to reduce peak backward memory, or 'auto' to let CCE pick a size that balances GPU occupancy against a 1 GiB scratch cap. Requires cut_cross_entropy_accum_c_fp32 and Triton >= 3.2."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_dtype_is_half(cls, data):
        if data.get("cut_cross_entropy") and not (data.get("bf16") or data.get("fp16")):
            raise ValueError(
                "Cut Cross Entropy requires fp16/bf16 training for backward pass. "
                "Please set `bf16` or `fp16` to `True`."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_chunked_cross_entropy_not_set(cls, data):
        if data.get("chunked_cross_entropy"):
            raise ValueError(
                "Cut Cross Entropy does not support chunked cross entropy. "
                "Please set `chunked_cross_entropy` to `False` or disable Cut Cross Entropy."
            )
        return data

    @model_validator(mode="after")
    def check_c_grad_chunk_size(self):
        chunk = self.cut_cross_entropy_c_grad_chunk_size
        if chunk is None or chunk == 0:
            return self
        if chunk != "auto" and (chunk < 0 or chunk % 128):
            raise ValueError(
                "`cut_cross_entropy_c_grad_chunk_size` must be 'auto' or a positive multiple of 128."
            )
        if not self.cut_cross_entropy_accum_c_fp32:
            raise ValueError(
                "`cut_cross_entropy_c_grad_chunk_size` requires `cut_cross_entropy_accum_c_fp32: true`."
            )
        return self

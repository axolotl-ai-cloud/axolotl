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
Module for handling LIGER input arguments.
"""
import logging
from typing import Optional

from pydantic import BaseModel, model_validator

LOG = logging.getLogger("axolotl.integrations.liger.args")


class LigerArgs(BaseModel):
    """
    Input args for LIGER.
    """

    liger_rope: Optional[bool] = None
    liger_rms_norm: Optional[bool] = None
    liger_layer_norm: Optional[bool] = None
    liger_swiglu: Optional[bool] = None
    liger_glu_activation: Optional[bool] = None
    liger_cross_entropy: Optional[bool] = None
    liger_fused_linear_cross_entropy: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_deprecated_swiglu(cls, data):
        if data.get("liger_swiglu") is not None:
            if data.get("liger_glu_activation") is not None:
                raise ValueError(
                    "You cannot have both `liger_swiglu` and `liger_glu_activation` set."
                )

            LOG.warning(
                "The 'liger_swiglu' argument is deprecated and will be removed in a future release. "
                "Please use 'liger_glu_activation' instead."
            )
            data["liger_glu_activation"] = data.pop("liger_swiglu")
        return data

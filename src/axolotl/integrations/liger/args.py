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
from typing import Optional

from pydantic import BaseModel


class LigerArgs(BaseModel):
    """
    Input args for LIGER.
    """

    liger_rope: Optional[bool] = None
    liger_rms_norm: Optional[bool] = None
    liger_swiglu: Optional[bool] = None
    liger_cross_entropy: Optional[bool] = None
    liger_fused_linear_cross_entropy: Optional[bool] = None

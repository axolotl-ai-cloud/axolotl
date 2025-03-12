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
Module for handling Spectrum input arguments.
"""
from typing import Optional

from pydantic import BaseModel, model_validator


class SpectrumArgs(BaseModel):
    """
    Input args for Spectrum.
    """

    spectrum_top_fraction: Optional[float] = 0.5
    spectrum_model_name: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_use_orig_params(cls, data):
        if (
            data.get("fsdp")
            and data.get("fsdp_config")
            and not data["fsdp_config"].get("use_orig_params")
            and data.get("plugins")
            and any("SpectrumPlugin" in plugin for plugin in data["plugins"])
        ):
            # would otherwise raise
            # ValueError: Must flatten tensors with uniform `requires_grad` when `use_orig_params=False`
            raise ValueError(
                "FSDP + SpectrumPlugin cannot be used together when `use_orig_params=False` is set"
            )
        return data

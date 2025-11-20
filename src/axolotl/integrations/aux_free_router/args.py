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
Plugin args for the Aux-Loss-Free MoE router integration.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AuxFreeRouterArgs(BaseModel):
    """
    Input args for Aux-Loss-Free MoE routing.
    """

    moe_balance_type: Literal["gshard", "noaux_tc"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "MoE load balancing strategy: 'gshard' for auxiliary loss, "
            "'noaux_tc' for aux-loss-free bias updates affecting top-k selection only. "
            "Defaults to model's native behavior when unset."
        },
    )
    moe_update_rate: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Per-step bias update rate (gamma). Recommended: 0.005–0.05. "
            "If unset, plugin default is 0.01."
        },
    )
    moe_update_momentum: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "EMA momentum for expert load smoothing (0–1). "
            "If unset, plugin default is 0.9."
        },
    )
    moe_bias_cap: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Absolute clamp for expert bias magnitude. "
            "If unset, plugin default is 2.0."
        },
    )
    moe_afb_warmup_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of initial steps to delay aux-free bias updates, "
            "allowing routing to stabilize. If unset, plugin default is 0."
        },
    )
    moe_bias_sync_group: Literal["world", "ep"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Reduction group for expert load counts: 'world' (DP) or "
            "'ep' (expert-parallel group if available). Defaults to 'world' when unset."
        },
    )


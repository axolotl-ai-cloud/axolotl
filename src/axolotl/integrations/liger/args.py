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

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class LigerArgs(BaseModel):
    """
    Input args for LIGER.
    """

    liger_rope: bool | None = None
    liger_rms_norm: bool | None = None
    liger_layer_norm: bool | None = None
    liger_swiglu: bool | None = None
    liger_glu_activation: bool | None = None
    liger_cross_entropy: bool | None = None
    liger_fused_linear_cross_entropy: bool | None = None
    liger_use_token_scaling: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Enables use_token_scaling in fused_linear_cross_entropy. "
                "When True, each token's loss is multiplied by its predicted probability (detached from gradients)."
            )
        },
    )

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

    @model_validator(mode="before")
    @classmethod
    def check_tiled_mlp_conflict(cls, data):
        if (
            data.get("liger_glu_activation") is True
            and data.get("tiled_mlp") is True
            and not data.get("tiled_mlp_use_original_mlp")
        ):
            raise ValueError(
                "You cannot have both `liger_glu_activation` and `tiled_mlp` set without `tiled_mlp_use_original_mlp: true`."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_liger_rms_norm_tensor_parallel(cls, data):
        if data.get("liger_rms_norm") and data.get("tensor_parallel_size", 1) > 1:
            raise ValueError(
                "`liger_rms_norm` is incompatible with tensor parallelism, "
                "see https://github.com/linkedin/Liger-Kernel/issues/826"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_liger_use_token_scaling_flce(cls, data):
        if data.get("liger_use_token_scaling") and not data.get(
            "liger_fused_linear_cross_entropy"
        ):
            raise ValueError(
                "`liger_use_token_scaling: true` requires `liger_fused_linear_cross_entropy` enabled."
            )

        return data

    @model_validator(mode="after")
    def check_tensor_parallel_size_liger_fused_linear_cross_entropy(self):
        # TODO @SalmanMohammadi this is a larger fix - investigate
        if self.tensor_parallel_size > 1 and self.liger_fused_linear_cross_entropy:
            raise ValueError("Tensor parallelism is not compatible with liger losses.")
        return self

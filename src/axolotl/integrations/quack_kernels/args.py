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
Input args for the QuACK kernels plugin.
"""

from pydantic import BaseModel, Field, model_validator


class QuackKernelsArgs(BaseModel):
    """
    Input args for QuACK (CuTe-DSL) kernels.
    """

    quack_mlp_kernel: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Enables QuACK's fused gated-MLP (SwiGLU / GeGLU) kernel, which fuses "
                "the up-projection GEMM with the gated activation. Requires a "
                "Hopper/Blackwell/RTX50 GPU (SM90+) and the `quack-kernels` package; "
                "patched MLPs fall back to the original forward when ineligible."
            )
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_quack_mlp_conflicts(cls, data):
        # Runs after base-class `mode="before"` validators (base classes precede plugin
        # args in the merged MRO), so an auto-enabled lora_mlp_kernel is reflected here.
        if not data.get("quack_mlp_kernel"):
            return data

        # All of these also rewrite the MLP forward; only one MLP kernel path may win.
        conflicts = []
        if data.get("liger_glu_activation") or data.get("liger_swiglu"):
            conflicts.append("liger_glu_activation")
        if data.get("lora_mlp_kernel"):
            conflicts.append("lora_mlp_kernel")
        if data.get("tiled_mlp") and not data.get("tiled_mlp_use_original_mlp"):
            conflicts.append("tiled_mlp")

        if conflicts:
            raise ValueError(
                "`quack_mlp_kernel` patches the MLP forward and cannot be combined with: "
                f"{', '.join(conflicts)}. Enable only one MLP kernel path."
            )
        return data

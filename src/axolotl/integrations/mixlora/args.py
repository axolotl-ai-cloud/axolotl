# Copyright 2025 Axolotl AI. All rights reserved.
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
Module for handling MixLoRA input arguments.
"""

from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from axolotl.integrations.mixlora.constants import (
    MIXLORA_DEFAULTS,
    MIXLORA_FFN_MODULE_NAMES,
)


class MixLoraArgs(BaseModel):
    """
    Input args for MixLoRA, MoE-style LoRA finetuning of dense models.
    """

    DEFAULT_NUM_EXPERTS: ClassVar[int] = int(MIXLORA_DEFAULTS["mixlora_num_experts"])
    DEFAULT_TOP_K: ClassVar[int] = int(MIXLORA_DEFAULTS["mixlora_top_k"])
    DEFAULT_ROUTER_AUX_LOSS_COEF: ClassVar[float] = MIXLORA_DEFAULTS[
        "mixlora_router_aux_loss_coef"
    ]
    DEFAULT_ROUTER_INIT_RANGE: ClassVar[float] = MIXLORA_DEFAULTS[
        "mixlora_router_init_range"
    ]
    DEFAULT_JITTER_NOISE: ClassVar[float] = MIXLORA_DEFAULTS["mixlora_jitter_noise"]

    mixlora_num_experts: int | None = Field(
        default=DEFAULT_NUM_EXPERTS,
        ge=1,
        json_schema_extra={
            "description": "Number of LoRA experts per FFN layer for MixLoRA (default 8)"
        },
    )
    mixlora_top_k: int | None = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        json_schema_extra={
            "description": "Number of experts to route each token to (default 2)"
        },
    )
    mixlora_router_aux_loss_coef: float | None = Field(
        default=DEFAULT_ROUTER_AUX_LOSS_COEF,
        ge=0.0,
        json_schema_extra={
            "description": "Coefficient for the auxiliary load balance loss (default 0.01)"
        },
    )
    mixlora_router_init_range: float | None = Field(
        default=DEFAULT_ROUTER_INIT_RANGE,
        gt=0.0,
        json_schema_extra={
            "description": "Initialization range for router weights (default 0.02)"
        },
    )
    mixlora_jitter_noise: float | None = Field(
        default=DEFAULT_JITTER_NOISE,
        ge=0.0,
        json_schema_extra={
            "description": "Noise added to router inputs during training for exploration (default 0.0)"
        },
    )
    mixlora_expert_lora_r: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Separate LoRA rank for MixLoRA experts. Defaults to lora_r if not set."
        },
    )
    mixlora_expert_lora_alpha: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Separate LoRA alpha for MixLoRA experts. Defaults to lora_alpha if not set."
        },
    )
    mixlora_expert_lora_dropout: float | None = Field(
        default=None,
        ge=0.0,
        json_schema_extra={
            "description": "Separate LoRA dropout for MixLoRA experts. Defaults to lora_dropout if not set."
        },
    )

    @model_validator(mode="after")
    def validate_mixlora_top_k(self):
        num_experts = (
            self.mixlora_num_experts
            if self.mixlora_num_experts is not None
            else self.DEFAULT_NUM_EXPERTS
        )
        top_k = (
            self.mixlora_top_k if self.mixlora_top_k is not None else self.DEFAULT_TOP_K
        )

        if top_k > num_experts:
            raise ValueError(
                f"mixlora_top_k ({top_k}) must be <= "
                f"mixlora_num_experts ({num_experts})"
            )
        return self

    @model_validator(mode="after")
    def validate_mixlora_adapter(self):
        # these fields only exist once this model is merged into the root config
        if getattr(self, "adapter", None) != "mixlora":
            return self

        if not getattr(self, "lora_r", None):
            raise ValueError("lora_r is required when using the mixlora adapter")
        if not getattr(self, "lora_alpha", None):
            raise ValueError("lora_alpha is required when using the mixlora adapter")
        if getattr(self, "flash_attn_fuse_qkv", False):
            raise ValueError("flash_attn_fuse_qkv is not supported with MixLoRA")
        if getattr(self, "flash_attn_fuse_mlp", False):
            raise ValueError("flash_attn_fuse_mlp is not supported with MixLoRA")
        if getattr(self, "rl", None) is not None:
            raise ValueError(
                "MixLoRA is not compatible with RL training (DPO/KTO/GRPO). "
                "Use adapter: lora or qlora for RL fine-tuning."
            )

        if getattr(self, "lora_target_linear", None):
            raise ValueError(
                "MixLoRA is incompatible with lora_target_linear=true because FFN "
                "gate_proj/up_proj/down_proj are patched by MixLoRA. "
                "Set explicit attention-only lora_target_modules instead."
            )

        target_modules = getattr(self, "lora_target_modules", None) or []
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        overlap = sorted(set(MIXLORA_FFN_MODULE_NAMES).intersection(target_modules))
        if overlap:
            raise ValueError(
                "MixLoRA cannot be combined with LoRA targets on FFN modules. "
                f"Remove overlapping lora_target_modules entries: {overlap}"
            )

        return self

    @model_validator(mode="before")
    @classmethod
    def set_mixlora_gradient_checkpointing_kwargs(cls, data):
        # MixLoRA reads _aux_loss from the module after the forward pass; reentrant
        # checkpointing recomputes forward under no_grad, dropping the autograd graph.
        if not isinstance(data, dict):
            return data
        if data.get("adapter") == "mixlora" and data.get("gradient_checkpointing"):
            kwargs = dict(data.get("gradient_checkpointing_kwargs") or {})
            if kwargs.get("use_reentrant") is not False:
                kwargs["use_reentrant"] = False
            data["gradient_checkpointing_kwargs"] = kwargs
        return data

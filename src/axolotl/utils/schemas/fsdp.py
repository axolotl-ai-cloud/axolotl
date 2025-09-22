"""
FSDP Configuration Schema
"""

from typing import Literal

from pydantic import BaseModel, Field


class FSDPConfig(BaseModel):
    """
    FSDP Configuration Schema
    """

    # Core FSDP settings
    activation_checkpointing: bool | None = Field(
        default=None,
        description="Enable activation checkpointing to reduce memory usage during forward passes",
    )
    offload_params: bool | None = Field(
        default=None,
        description="Offload parameters to CPU to reduce GPU memory usage",
    )
    sync_module_states: bool | None = Field(
        default=None,
        description="Synchronize module states across all processes",
    )
    cpu_ram_efficient_loading: bool | None = Field(
        default=None,
        description="Enable CPU RAM efficient loading to reduce memory usage during model loading",
    )
    use_orig_params: bool | None = Field(
        default=None,
        description="Use original parameters instead of flattened parameters",
    )

    # State dict configuration
    state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = Field(
        default=None,
        description="Type of state dict to use for saving/loading checkpoints",
    )
    final_state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = Field(
        default=None,
        description="Final state dict type to use after training completion",
    )

    # Wrapping policy configuration
    auto_wrap_policy: Literal["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"] | None = (
        Field(
            default=None,
            description="Policy for automatically wrapping modules with FSDP",
        )
    )
    transformer_layer_cls_to_wrap: str | None = Field(
        default=None,
        description="Class name of transformer layers to wrap (e.g., 'LlamaDecoderLayer')",
    )

    # Memory and performance settings
    reshard_after_forward: bool | None = Field(
        default=None,
        description="Reshard parameters after forward pass to save memory",
    )
    mixed_precision_policy: str | None = Field(
        default=None,
        description="Mixed precision policy for FSDP (e.g., 'fp16', 'bf16')",
    )

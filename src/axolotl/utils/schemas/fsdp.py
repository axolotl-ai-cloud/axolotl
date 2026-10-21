"""
FSDP Configuration Schema
"""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class FSDPConfig(BaseModel):
    """
    FSDP Configuration Schema
    """

    fsdp_version: int | None = Field(
        validation_alias=AliasChoices("fsdp_version", "version"),
        default=None,
        json_schema_extra={"description": "FSDP version"},
    )
    activation_checkpointing: bool | None = Field(
        default=None,
        description="Enable activation checkpointing to reduce memory usage during forward passes",
    )
    offload_params: bool | None = Field(
        default=None,
        description="Offload parameters to CPU to reduce GPU memory usage",
    )
    cpu_ram_efficient_loading: bool | None = Field(
        default=None,
        description="Enable CPU RAM efficient loading to reduce memory usage during model loading",
    )
    cpu_offload_pin_memory: bool | None = Field(
        default=None,
        description="Disabling this enables swap memory usage for resource-constrained setups when offload_params is enabled.",
    )
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
    adapter_save_strategy: Literal["gather", "sharded"] | None = Field(
        default=None,
        description="How FSDP2 writes PEFT adapters (unset: 'gather'), for checkpoint-N saves and the end-of-training export to output_dir. 'gather' assembles every adapter tensor on rank 0 before writing adapter_model.safetensors. 'sharded' has every rank write the slices it holds to <dir>/.adapter_shards/, with no cross-rank tensor transfer when every rank sees the output directory, then rank 0 consolidates them into the same adapter_model.safetensors. Checkpoints consolidate on a background thread while training continues: one consolidation runs at a time, and a save that finds one running and one queued waits. Until a checkpoint is consolidated, its on_save callbacks (checkpoint mirror, publishers) and save_total_limit rotation are held back, and training end waits for it; a resume from an unconsolidated checkpoint finishes the consolidation first. The final export waits for outstanding checkpoint consolidations, then consolidates its own shards before training returns. Requires an adapter, fsdp_version 2 with state_dict_type FULL_STATE_DICT (with or without expert_parallel_size), tensor_parallel_size 1, no rl, load_best_model_at_end false, and hub_strategy end when hub_model_id is set; config validation rejects anything else. Single-process runs, multiple adapters, and quantized adapter tensors fall back to 'gather' at save time.",
    )

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

    min_num_params: int | None = Field(
        default=None,
        ge=1,
        description="Minimum parameter count a module must have to be wrapped under the SIZE_BASED_WRAP policy",
    )

    reshard_after_forward: bool | None = Field(
        default=None,
        description="Reshard parameters after forward pass to save memory",
    )
    mixed_precision_policy: str | None = Field(
        default=None,
        description="Mixed precision policy for FSDP (e.g., 'fp16', 'bf16')",
    )

    @model_validator(mode="after")
    def validate_size_based_wrap(self):
        if self.auto_wrap_policy == "SIZE_BASED_WRAP" and self.min_num_params is None:
            raise ValueError(
                "fsdp_config.min_num_params is required when auto_wrap_policy is SIZE_BASED_WRAP"
            )
        return self

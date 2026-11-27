# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the Expert-Parallel plugin."""

from importlib.util import find_spec
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ExpertParallelArgs(BaseModel):
    """Input args for the expert_parallel plugin. See the integration README."""

    expert_parallel_size: int = 1
    """Number of EP ranks. 1 = disabled (default), > 1 = enabled."""

    expert_parallel_backend: Literal["auto", "deep_ep", "torch", "optmoe"] = "auto"
    """Token dispatch backend. ``deep_ep``: DeepEP fused kernels (NVLink/RDMA). ``torch``: plain
    ``all_to_all_single`` over the EP process group (any NCCL/gloo setup). ``optmoe``: the
    optmoe package over NVLink symmetric memory on one node, in the mode
    ``expert_parallel_optmoe_mode`` picks (fused persistent kernels, or peer dispatch/combine
    around the configured local experts kernel). ``auto``: ``deep_ep`` when importable, else
    ``torch``; it never selects ``optmoe``."""

    expert_parallel_optmoe_mode: Literal["auto", "fused", "dispatch"] = "auto"
    """How the ``optmoe`` backend runs. ``fused``: one persistent kernel per direction fusing
    dispatch, SwiGLU experts and combine (Hopper/Blackwell, full fine-tuning of plain SwiGLU
    experts only; the local experts implementation is not used). ``dispatch``: optmoe's peer
    dispatch and combine feed whichever local experts kernel the config names
    (``use_sonicmoe``, ``use_scattermoe``, ``experts_implementation``), so LoRA, quantized
    experts, biases, other activations and clamped SwiGLU all work. ``auto``: ``fused`` when the
    config and model support it, otherwise ``dispatch``, otherwise
    ``expert_parallel_fallback_on_unsupported``."""

    expert_parallel_optmoe_compact: bool = False
    """Dispatch mode only: hand the local experts kernel just the routed receive rows
    instead of the fixed ``ep_size * tokens_per_rank`` layout. Costs one device-to-host
    count per MoE layer (``optmoe::peer_count_to_host``, saved under activation
    checkpointing like the torch backend's split counts); smaller receive buffers and
    less row-count-sized work in the local kernel."""

    expert_parallel_optmoe_bounded: bool = False
    """With ``expert_parallel_optmoe_compact``: receive into a bound from each layer's
    high-water routed count (filled to SonicMoE's own row bucket) and read the count only
    after the local kernel was issued, so the dispatch never stalls the stream on it.
    Outputs and gradients are bitwise those of the compact rows; a count above the bound
    re-runs that layer exactly."""

    expert_parallel_optmoe_precision: Literal[
        "auto",
        "bf16",
        "mxfp8",
        "fp8",
        "fp8_bwd",
        "nvfp4_lora",
        "bf16_lora",
        "mxfp8_lora",
        "fp8_lora",
    ] = "auto"
    """Compute precision of the ``optmoe`` backend's fused mode. ``auto``: ``bf16`` on SM90,
    ``mxfp8`` on SM100, and ``nvfp4_lora`` on SM100 when the experts load as packed NVFP4 and a
    LoRA adapter targets them. ``fp8`` is blockwise E4M3 forward and backward (SM90);
    ``fp8_bwd`` is BF16 forward with blockwise FP8 backward. ``nvfp4_lora`` keeps the packed
    NVFP4 experts frozen and trains a PEFT ``lora_target_parameters`` adapter on them (SM100;
    ``adapter: lora`` with ``use_sonicmoe: true``, ``nvfp4_merge_aware: false`` and
    ``lora_dropout: 0``). With an expert LoRA adapter on a BF16 base, ``auto``, ``bf16`` and
    ``mxfp8`` select ``bf16_lora`` / ``mxfp8_lora`` (see ``expert_parallel_optmoe_fused_lora``);
    ``fp8_lora`` is not implemented yet. Master weights stay in their own dtype. The dispatch
    mode moves BF16 rows and leaves compute precision to the local kernel."""

    expert_parallel_optmoe_fused_lora: bool = True
    """Fused mode with a LoRA adapter on BF16 experts: train it on the fused kernels
    (``bf16_lora`` / ``mxfp8_lora`` on SM100: the frozen base merged with the adapter once per
    optimizer step, adapter gradients from rank-r GEMMs on the backward's pools). Needs
    ``adapter: lora`` on ``experts.gate_up_proj`` and ``experts.down_proj``
    (``lora_target_parameters``), ``use_sonicmoe: true``, ``lora_dropout: 0`` and no DoRA.
    ``false`` keeps today's behavior: an adapter runs the dispatch mode with the local
    kernel."""

    expert_parallel_optmoe_lora_merged_cache: Literal["layer", "all"] = "layer"
    """Fused LoRA precisions (``bf16_lora``, ``mxfp8_lora``): when the merged BF16 expert
    weights are built (exported as ``OPTMOE_LORA_MERGED_CACHE``). ``layer`` merges one layer
    right before its forward and again before its backward when evicted, keeping the last two
    layers resident; ``all`` merges every layer once per optimizer step and holds the merges
    across the step (one BF16 copy of the routed experts more memory)."""

    expert_parallel_optmoe_workspace_bound: Literal["auto", "max"] = "auto"
    """Fused mode on SM100: size the backward's per-rank row pools from the rows each layer's
    forward actually routed to this rank (exported as ``OPTMOE_WORKSPACE_BOUND``). ``auto``
    starts at 1.25 x tokens x top-k and grows on demand, so no tuning is needed; ``max``
    restores the worst-case sizing (every token of every rank routed here)."""
    expert_parallel_optmoe_stored_context: Literal[
        "auto", "recompute", "always", "off"
    ] = "auto"
    """Fused mode on SM100: let the activation-checkpoint recompute of a MoE layer store its
    gate/up pre-activations so that layer's backward skips recomputing them (exported as
    ``OPTMOE_STORED_CONTEXT``). ``auto``: only where the backward is GEMM-bound
    (hidden x moe intermediate >= 4096 x 1536) and the buffer (sized from the routed-row
    high-water mark, about 1-2 GB per rank) fits in free memory with headroom; one log line
    says which. ``recompute`` forces it, ``always`` stores in every forward, ``off`` never.
    It only acts when the fused op is recomputed, i.e. activation checkpointing without
    ``selective`` (which saves the fused op's output instead)."""

    expert_parallel_optmoe_nvfp4_merged_cache: Literal["layer", "all"] = "layer"
    """``nvfp4_lora`` precision: when the merged NVFP4 expert weights (frozen base plus the
    current adapter) are requantized (exported as ``OPTMOE_NVFP4_MERGED_CACHE``). ``layer``
    builds one layer's forms right before its forward and again before its backward and keeps
    only the last two layers' forms resident; ``all`` builds every layer's forms once per
    optimizer step and holds them across the step (more memory, fewer requant launches)."""

    expert_parallel_optmoe_nvfp4_headroom_log2: int = Field(default=0, ge=0, le=4)
    """``nvfp4_lora`` precision: headroom of the merged NVFP4 weights over the base expert's
    largest magnitude, as a power of two (exported as ``OPTMOE_NVFP4_LORA_HEADROOM_LOG2``).
    ``0`` keeps the checkpoint's per-tensor scale, so a merged weight clamps at the base's
    largest magnitude; ``k`` raises that clamp ``2**k`` times at the cost of resolution for
    the smallest blocks."""

    expert_parallel_num_nvl_bytes: int = 256 << 20

    expert_parallel_num_rdma_bytes: int = 0

    expert_parallel_token_capacity: int | None = None
    """Max tokens routed to any single expert per forward; the lowest-weight excess (token,expert)
    assignments are dropped. DeepEP's intranode combine deadlocks once one expert is overloaded, and
    GLM-style routers concentrate more with depth — set this (e.g. 1024) for them. ``None`` = no cap."""

    expert_parallel_fallback_on_unsupported: bool = True

    expert_parallel_save_dispatch: bool = False
    """``torch`` backend under activation checkpointing (``activation_checkpointing`` with
    either implementation): save the dispatch/combine all-to-all outputs so
    backward does not re-issue the collectives, at the cost of holding the received rows per
    layer (24 GiB on Qwen3-30B-A3B at 32k packed on 2 GPUs). Recomputing them cost 1.6 s/step on
    a PCIe-only pair and was a tie on an NVLink pair, so the default recomputes; enable this on
    bandwidth-starved links with memory to spare. The routing ``topk`` and host-side split
    counts are always saved."""

    expert_parallel_dispatch_chunks: int = Field(default=1, ge=1)
    """``torch`` backend only: split each MoE forward's tokens into this many chunks and pipeline
    them, so chunk ``i+1``'s dispatch all-to-all overlaps chunk ``i``'s expert GEMMs. ``1`` keeps
    the unchunked path. Overlaps the forward only. Each chunk adds collective and launch overhead,
    so this can be slower than ``1``; benchmark before raising it past ``2``."""

    @model_validator(mode="after")
    def _validate(self):
        if self.expert_parallel_size < 1:
            raise ValueError(
                f"expert_parallel_size must be >= 1 (got {self.expert_parallel_size!r}). "
                f"Use 1 to disable EP."
            )

        if (
            self.expert_parallel_token_capacity is not None
            and self.expert_parallel_token_capacity < 1
        ):
            raise ValueError(
                f"expert_parallel_token_capacity must be >= 1 when set "
                f"(got {self.expert_parallel_token_capacity!r}). Use None to disable the cap."
            )

        if (
            self.expert_parallel_size > 1
            and self.expert_parallel_backend == "deep_ep"
            and find_spec("deep_ep") is None
        ):
            outcome = (
                "expert parallelism will be skipped at model load and the experts "
                "sharded like any other weight"
                if self.expert_parallel_fallback_on_unsupported
                else "model loading will fail"
            )
            LOG.warning(
                f"expert_parallel_backend: deep_ep, but `deep_ep` is not installed here: "
                f"{outcome}. Use an image with DeepEP (the -blackwell cloud image) or "
                "set expert_parallel_backend: torch (or auto)."
            )

        if (
            self.expert_parallel_optmoe_bounded
            and not self.expert_parallel_optmoe_compact
        ):
            raise ValueError(
                "expert_parallel_optmoe_bounded needs expert_parallel_optmoe_compact: true"
            )

        if self.expert_parallel_size > 1 and self.expert_parallel_num_rdma_bytes != 0:
            LOG.warning(
                "expert_parallel_num_rdma_bytes != 0 — RDMA path requires "
                "Hopper + IBGDA-capable InfiniBand. Will fail on Ampere/intranode."
            )

        return self

    @model_validator(mode="after")
    def _validate_topology(self):
        # only the merged input config carries the fsdp/mesh fields; standalone args skip
        if (
            self.expert_parallel_size <= 1
            or "fsdp_config" not in type(self).model_fields
        ):
            return self
        validate_expert_parallel_topology(self)
        return self


def validate_expert_parallel_topology(cfg) -> None:
    """Reject EP layouts the sharding, clipping and save paths do not handle."""
    ep_size = getattr(cfg, "expert_parallel_size", 1) or 1
    if ep_size <= 1:
        return

    fsdp_config = getattr(cfg, "fsdp_config", None)
    fsdp_version = getattr(cfg, "fsdp_version", None) or (
        getattr(fsdp_config, "fsdp_version", None) if fsdp_config else None
    )
    if not fsdp_config or fsdp_version != 2:
        raise ValueError(
            f"expert_parallel_size ({ep_size}) > 1 requires FSDP2: set fsdp_version: 2 "
            "and an fsdp_config block. Under DDP the expert LoRA sync, gradient clipping, "
            "gradient-accumulation scaling and checkpoint save are all wrong for sharded experts."
        )

    for key in ("state_dict_type", "final_state_dict_type"):
        value = getattr(fsdp_config, key, None)
        if key == "state_dict_type" and value is None:
            raise ValueError(
                f"expert_parallel_size ({ep_size}) > 1 requires fsdp_config.state_dict_type: "
                "FULL_STATE_DICT. accelerate defaults FSDP2 to SHARDED_STATE_DICT, whose "
                "checkpoints keep only EP group 0's experts (every group's expert shard has the "
                "same name, shape and offset, so DCP deduplicates them)."
            )
        if value is not None and value != "FULL_STATE_DICT":
            raise ValueError(
                f"expert_parallel_size ({ep_size}) > 1 requires fsdp_config.{key}: "
                f"FULL_STATE_DICT, got {value!r}. {value} checkpoints keep only EP group 0's "
                "experts; the full state dict gathers every EP group's experts before rank 0 writes."
            )

    optimizer = getattr(cfg, "optimizer", None)
    if optimizer in ("adamw_torch_4bit", "ao_adamw_4bit", "ao_adamw_fp8"):
        raise ValueError(
            f"expert_parallel_size ({ep_size}) > 1 does not support optimizer: {optimizer}. "
            "EP full checkpoints keep only torchao 8-bit optimizer state quantized; 4-bit and "
            "FP8 state would resume dequantized at several times the memory. Use "
            "adamw_torch_8bit or an unquantized optimizer."
        )

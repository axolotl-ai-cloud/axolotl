# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Pydantic config schema for the Arctic Platform SFT integration.

The nested ``arctic_sft:`` block only carries the *connection / server*
settings that map onto ``ArcticSFTClientConfig``. Standard training knobs
(``learning_rate``, ``weight_decay``, ``adam_beta1/2``, ``max_grad_norm``,
``gradient_accumulation_steps``, ``sequence_len``, ``attn_implementation`` …)
are read from axolotl's top-level config so they don't need to be duplicated
here.

Usage (colocated vs Ray, pitfalls, top-level knobs the plugin forwards):
``axolotl/integrations/arctic_platform/README.md``. Server client fields:
arctic-platform ``docs/sft.md``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

_ONPREM_PROTOCOLS = frozenset({"http", "ray"})
_REMOTE_PROTOCOLS = frozenset({"http", "cortex"})


class ArcticSFTConfig(BaseModel):
    """Nested config under ``arctic_sft:`` in the axolotl YAML."""

    backend: Literal["onprem", "remote"] = Field(
        default="onprem",
        json_schema_extra={
            "description": (
                "Deployment target. onprem uses protocol http|ray; remote uses "
                "http|cortex. This integration only constructs onprem; "
                "backend=remote validates here and is rejected at client-build "
                "until it is wired."
            )
        },
    )
    protocol: Literal["http", "ray", "cortex"] = Field(
        default="http",
        json_schema_extra={
            "description": (
                "Transport on that backend. onprem: http|ray. remote: "
                "http|cortex. Mapped onto OnPremConfig.protocol."
            )
        },
    )
    host: str = Field(
        default="localhost",
        json_schema_extra={"description": "AP client host. Default localhost."},
    )
    port: int = Field(
        default=8000,
        json_schema_extra={"description": "AP client port. Default 8000."},
    )
    training_gpus: int = Field(
        ...,
        ge=1,
        json_schema_extra={
            "description": "Training GPUs on the server (>= 1)."
        },
    )
    launch_local_server: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Spawn a local HTTP server from the client. The Axolotl "
                "process stays CPU-only."
            )
        },
    )
    server_cuda_visible_devices: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "CUDA_VISIBLE_DEVICES for a locally launched server subprocess "
                "(e.g. '0,1'). Required when the axolotl process runs with "
                "CUDA_VISIBLE_DEVICES= (empty) and launch_local_server=true."
            )
        },
    )
    model_name: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Override the model id sent to the server. If unset, uses "
                "top-level base_model."
            )
        },
    )
    loss_fn: Literal["sft", "sft_ce"] = Field(
        default="sft",
        json_schema_extra={
            "description": (
                "Server loss. Both are causal-LM CE on labels != -100. sft: HF "
                "outputs.loss (fused CE, per-shard token-mean); bit-exact vs "
                "native Axolotl SFT. sft_ce: explicit fp32 CE on logits "
                "(labels not passed into HF). Mean is global across DP when "
                "the worker injects global_num_tokens; needed when ranks see "
                "unequal valid token counts. logits_optimization applies only "
                "to sft_ce."
            )
        },
    )
    logits_optimization: Literal["none", "compute", "memory"] = Field(
        default="none",
        json_schema_extra={
            "description": (
                "Memory strategy for the sft_ce vocab projection / "
                "cross-entropy (ignored by sft, which uses HF fused CE). "
                "none: full [B, S, V] logits + fp32 CE. compute: full logits "
                "once, softmax follow-up chunked under the peak-mem budget. "
                "memory: hidden states tiled through the LM head; full logits "
                "are never manifested (extra forward replay in backward)."
            )
        },
    )
    logits_optimization_peak_mem_size_in_gib: int = Field(
        default=4,
        json_schema_extra={
            "description": (
                "Tile/chunk budget in GiB for logits_optimization compute and "
                "memory."
            )
        },
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Server-side checkpoint dir (required by the server for "
                "training jobs). Defaults from output_dir if unset."
            )
        },
    )
    ds_config: Optional[dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "DeepSpeed config passed through to the Arctic client as-is. "
                "If unset, the plugin builds it from top-level axolotl knobs. "
                "Optimizer and LR schedule go here."
            )
        },
    )
    ds_worker_config: Optional[dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "DeepSpeed worker config passed through to the Arctic client "
                "as-is. If unset, the plugin builds it from top-level axolotl "
                "knobs."
            )
        },
    )
    startup_timeout: float = Field(
        default=600.0,
        json_schema_extra={
            "description": "Seconds to wait for the local or remote server to start."
        },
    )
    job_ready_timeout: float = Field(
        default=1800.0,
        json_schema_extra={
            "description": "Seconds to wait for the training job to become ready."
        },
    )
    request_timeout: float = Field(
        default=1800.0,
        json_schema_extra={
            "description": "Seconds to wait for a single server request."
        },
    )
    gradient_checkpointing: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Nested override for ds_worker_config when top-level "
                "gradient_checkpointing is unset. Attention comes from "
                "top-level attn_implementation only."
            )
        },
    )
    export_hf: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "After each remote DeepSpeed save, also write an HF-format "
                "directory under {checkpoint}/hf/ (rank-0 conversion via "
                "zero_to_fp32 / save_pretrained)."
            )
        },
    )
    sampling_gpus: int = Field(
        default=0,
        json_schema_extra={
            "description": (
                "vLLM sampling GPUs for generate_samples. 0 is training-only."
            )
        },
    )
    colocate: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Share GPUs between training and the vLLM sampling job."
            )
        },
    )
    vllm_config: Optional[dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "description": "Forwarded to the sampling job when sampling_gpus > 0."
        },
    )
    sampling_job_id: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": "Reattach to an existing server sampling job."
        },
    )
    training_job_id: Optional[int] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Reconnect to an existing server training job instead of "
                "creating one."
            )
        },
    )

    @model_validator(mode="after")
    def _backend_protocol_pair(self) -> ArcticSFTConfig:
        allowed = _ONPREM_PROTOCOLS if self.backend == "onprem" else _REMOTE_PROTOCOLS
        if self.protocol not in allowed:
            raise ValueError(
                f"arctic_sft: backend={self.backend!r} does not support "
                f"protocol={self.protocol!r} (allowed: {sorted(allowed)})"
            )
        return self


class ArcticSFTArgs(BaseModel):
    """Top-level mixin that adds the nested ``arctic_sft:`` field."""

    arctic_sft: Optional[ArcticSFTConfig] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Nested arctic_sft: connection / server settings for the "
                "Arctic Platform SFT plugin."
            )
        },
    )

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

    # AP layout: ``backend`` is the deployment, ``protocol`` is the
    # transport on that backend.
    #   onprem -> http | ray
    #   remote -> http | cortex
    # ArcticSFTClient still only constructs onprem (maps ``protocol`` onto
    # ``comm_protocol``). ``backend: remote`` validates here; the plugin
    # rejects it at client-build until the SFT facade accepts remote.
    backend: Literal["onprem", "remote"] = "onprem"
    protocol: Literal["http", "ray", "cortex"] = "http"

    # Connection
    host: str = "localhost"
    port: int = 8000

    # Server GPU topology
    training_gpus: int = Field(..., ge=1, description="Training GPUs on the server (>= 1).")

    # Local server colocation (client stays CPU-only)
    launch_local_server: bool = False
    server_cuda_visible_devices: Optional[str] = Field(
        None,
        description=(
            "CUDA_VISIBLE_DEVICES for a locally launched server subprocess "
            "(e.g. '0,1'). Required when the axolotl process runs with "
            "CUDA_VISIBLE_DEVICES= (empty) and launch_local_server=true."
        ),
    )

    # Overrides the axolotl `base_model` sent to the server when set.
    model_name: Optional[str] = None

    # Loss selection on the server: HF outputs.loss ("sft") or explicit CE ("sft_ce").
    loss_fn: Literal["sft", "sft_ce"] = "sft"

    # Memory strategy for the sft_ce vocab projection / cross-entropy (ignored by
    # the "sft" loss, which uses HF's fused CE). Mirrors the RL knob:
    #   none    -> full [B, S, V] logits + fp32 CE (classic sft_ce).
    #   compute -> full logits once, softmax follow-up chunked under the peak-mem
    #              budget so full-size intermediates are never materialized.
    #   memory  -> hidden states tiled through the LM head; the full logits are
    #              never manifested (extra forward replay in backward).
    logits_optimization: Literal["none", "compute", "memory"] = "none"
    logits_optimization_peak_mem_size_in_gib: int = 4

    # Server-side checkpoint dir (required by the server for training jobs).
    checkpoint_path: Optional[str] = None

    # Escape hatches forwarded verbatim to ArcticSFTClientConfig. When unset the
    # plugin synthesizes sensible defaults from the top-level axolotl config.
    # Optimizer + LR schedule are folded into ``ds_config`` (DeepSpeed config-json),
    # matching current arctic-platform (no separate high-level training_config).
    ds_config: Optional[dict[str, Any]] = None
    ds_worker_config: Optional[dict[str, Any]] = None

    # Timeouts (seconds)
    startup_timeout: float = 600.0
    job_ready_timeout: float = 1800.0
    request_timeout: float = 1800.0

    # Nested override for ds_worker_config when top-level gradient_checkpointing
    # is unset. Attention comes from top-level ``attn_implementation`` only.
    gradient_checkpointing: bool = False

    # After each remote DS save, also write an HF-format directory under
    # ``{checkpoint}/hf/`` (rank-0 conversion via zero_to_fp32 / save_pretrained).
    export_hf: bool = False

    # Sampling job for generate_samples (mirrors ArcticRLClientConfig defaults).
    sampling_gpus: int = 0
    colocate: bool = False
    vllm_config: Optional[dict[str, Any]] = None
    sampling_job_id: Optional[int] = None

    # Reconnect to an existing server training job instead of creating one.
    training_job_id: Optional[int] = None

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

    arctic_sft: Optional[ArcticSFTConfig] = None

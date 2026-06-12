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
Input args for the TorchSpec speculator-training plugin.

These map onto TorchSpec's ``Config`` sections (model/dataset/training/inference/
mooncake) but are flattened into a single ``speculator:`` block in axolotl style.
``base_model``, ``sequence_len``, ``learning_rate`` etc. are pulled from the
top-level axolotl config; this block only carries the speculative-decoding knobs.
"""

from typing import Literal

from pydantic import BaseModel, model_validator

EngineType = Literal["sgl", "vllm", "hf"]


class TorchSpecArgs(BaseModel):
    """The ``speculator:`` config block for training EAGLE-3 draft models."""

    # --- draft model ---
    # Path to a draft-model config JSON. When None, TorchSpec auto-generates a
    # reduced-layer EAGLE-3 config from the target model.
    draft_model_config: str | None = None
    ttt_length: int = 7  # speculative depth (train-time-test rollout length)
    # Per-position TTT loss weights; length must equal ttt_length when supplied.
    ploss_weights: list[float] | None = None
    # Target hidden-state layers fed to the draft model (None = TorchSpec default).
    aux_hidden_states_layers: list[int] | None = None
    last_hidden_states_prenorm: bool | None = None

    # --- inference engine (produces target hidden states) ---
    inference_engine: EngineType = "sgl"
    inference_num_gpus: int | None = None
    inference_num_gpus_per_engine: int = 1
    inference_num_gpus_per_node: int = 8
    inference_batch_size: int = 8
    inference_tp_size: int = 1
    mem_fraction_static: float | None = None

    # --- training workers (FSDP) ---
    training_num_gpus: int | None = None  # gpus per node for the training group
    training_num_nodes: int = 1
    fsdp_strategy: str = "REPLICATE"
    attention_backend: str = "flex_attention"
    draft_accumulation_steps: int = 1
    max_concurrent_batches: int = 1
    train_with_decode: bool = False

    # --- mooncake hidden-state transfer ---
    mooncake_protocol: Literal["tcp", "rdma"] = "tcp"
    mooncake_global_segment_size: str = "16GB"
    mooncake_local_buffer_size: str = "4GB"
    mooncake_device_name: str | None = None  # RDMA NIC, e.g. "mlx5_0"
    mooncake_master_server_address: str | None = None
    mooncake_metadata_server: str | None = None

    # --- dataset ---
    # Reuse axolotl's dataset loading to standardize `datasets:` into a
    # conversations JSONL that TorchSpec tokenizes. When False, pass the first
    # dataset path through to TorchSpec untouched (it must already be a
    # conversations-format file/HF id).
    prepare_dataset: bool = True

    # --- output ---
    output_dir: str | None = None  # defaults to axolotl output_dir
    cache_dir: str | None = None
    prompt_key: str = "conversations"
    chat_template: str | None = None  # override the auto-mapped TorchSpec template

    @model_validator(mode="after")
    def _check(self):
        if self.inference_num_gpus is not None and self.inference_num_gpus < 1:
            raise ValueError("speculator.inference_num_gpus must be >= 1")
        if self.training_num_gpus is not None and self.training_num_gpus < 1:
            raise ValueError("speculator.training_num_gpus must be >= 1")
        if self.mooncake_protocol == "rdma" and not self.mooncake_device_name:
            raise ValueError(
                "speculator.mooncake_protocol='rdma' requires "
                "speculator.mooncake_device_name (the RDMA NIC, e.g. 'mlx5_0')"
            )
        if (
            self.ploss_weights is not None
            and len(self.ploss_weights) != self.ttt_length
        ):
            raise ValueError(
                f"speculator.ploss_weights length ({len(self.ploss_weights)}) "
                f"must equal ttt_length ({self.ttt_length})"
            )
        return self


class TorchSpecArgsMixin(BaseModel):
    """Mixin exposing the ``speculator`` block on the axolotl input config."""

    speculator: TorchSpecArgs | None = None

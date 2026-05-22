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

"""Pydantic argument model for the ProTrain plugin."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Canonical plugin id strict allow-list; runtime gate and validator both import from here.
# Only the dotted class form loads via the integration loader; bare module form is rejected.
_PROTRAIN_PLUGIN_KEYS = frozenset(
    {
        "axolotl.integrations.protrain.ProTrainPlugin",
    }
)


# Optimizer names the chunk manager's AdamW-shaped adapters drive correctly.
# ``adamw_apex_fused`` routes to ``chunk/optim.py:GpuFusedAdamAdapter`` which already
# tries ``apex.optimizers.FusedAdam`` first and falls back to ``torch.optim.AdamW``.
_SUPPORTED_OPTIMIZERS: frozenset[str] = frozenset(
    {
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_apex_fused",
        "adamw_8bit",
        "adamw_bnb_8bit",
        "paged_adamw_8bit",
    }
)


def _has_protrain_plugin(plugins) -> bool:
    """Return True iff the iterable contains an explicit ProTrain plugin id."""
    if not isinstance(plugins, (list, tuple, set, frozenset)):
        return False
    return any(isinstance(p, str) and p in _PROTRAIN_PLUGIN_KEYS for p in plugins)


# Re-exported for plugin.py so the allow-list lives in one place.
__all__ = ["ProTrainArgs", "_has_protrain_plugin", "_PROTRAIN_PLUGIN_KEYS"]


class ProTrainArgs(BaseModel):
    """Input args for the ProTrain plugin (opt-in: plugins entry + protrain_auto_memory=True)."""

    protrain_auto_memory: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Master enable flag for ProTrain automatic memory management. "
                "When True, the plugin's post_model_load hook wraps the model "
                "with the hierarchical chunk manager + interleaved block manager, "
                "and post_trainer_create installs the ProTrain optimizer on the "
                "trainer. Requires "
                "``plugins: [axolotl.integrations.protrain.ProTrainPlugin]``. "
                "Mutually exclusive with DeepSpeed, FSDP, gradient_checkpointing, "
                "and TP/CP/SP > 1 (see `_reject_incompatible_features`). "
                "Composes with bitsandbytes ``load_in_8bit`` / ``load_in_4bit`` "
                "(M2/M3 validated; ``Params4bit`` / ``Int8Params`` survive the "
                "chunk gather/offload path because ``quant_state`` lives as a "
                "Python attribute on the param and ``chunk/manager.py`` rebinds "
                "``param.data`` without touching python attrs)."
            )
        },
    )

    protrain_auto_mode: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Auto-select the multi-GPU mode (A/B/C) based on measured fit "
                "and CPU-RAM-per-rank. When True (the default) the wrapper "
                "ignores the mode-picking intent of ``protrain_force_all_persistent`` "
                "and ``protrain_zero3_shard`` and picks one of: "
                "(A) GPU-resident / DDP-friendly (force_all_persistent=True), "
                "when the searcher can place ``n_persist == N_chunk`` under the "
                "capacity budget; "
                "(B) replicated CPU-offload (zero3_shard=False), when the model "
                "needs offload and per-rank CPU RAM can hold the full "
                "non-persistent chunk set; "
                "(C) ZeRO-3 sharded CPU-offload (zero3_shard=True), when the "
                "model needs offload but per-rank CPU RAM is too tight for "
                "replication. Set this to False to bypass the auto-selector and "
                "honour ``protrain_force_all_persistent`` + ``protrain_zero3_shard`` "
                "as explicit overrides — useful for reproducing specific "
                "benchmark configurations or for heterogeneous-CPU setups where "
                "the node-RAM/world-size heuristic is wrong. See DESIGN.md "
                "§Multi-GPU for the measured throughput ordering that motivates "
                "this default."
            )
        },
    )

    protrain_force_all_persistent: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Explicit override for the GPU-resident mode. "
                "When ``protrain_auto_mode`` is True (default) this flag is "
                "IGNORED — the plugin auto-selects based on workload fit. When "
                "``protrain_auto_mode`` is False, True here bypasses the "
                "4-knob searcher and forces every chunk to stay GPU-resident "
                "(n_persist = N_chunk, n_swap = 0, n_checkpoint = N_block). "
                "Set ``protrain_auto_mode: false`` alongside to make this "
                "effective — otherwise the auto-selector may override it."
            )
        },
    )

    protrain_capacity_bytes: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Override the GPU memory budget (bytes) the searcher respects. "
                "When None, defaults to ``gpu_memory_bytes - 2 GiB`` headroom "
                "for the CUDA context + allocator reserve."
            )
        },
    )

    protrain_cpu_capacity_bytes: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Per-rank pinned CPU RAM budget (bytes) the searcher uses as a "
                "HARD feasibility filter. Configs whose estimated per-rank "
                "non-persistent chunk footprint exceeds this are dropped before "
                "runtime evaluation. When None, the wrapper auto-derives "
                "``psutil.virtual_memory().available // gpu_count - 2 GiB`` "
                "(disabled with a warning if psutil isn't installed)."
            )
        },
    )

    protrain_cache_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Override the profiler-cache directory. When None, the cache "
                "lives under the standard XDG cache root."
            )
        },
    )

    # Debugging escape hatches — bypass the searcher. Production runs leave these None.
    protrain_n_persist_override: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Debug override: force the number of persistent chunks. "
                "Bypasses the exhaustive searcher when set alongside the other "
                "three overrides."
            )
        },
    )
    protrain_n_buffer_override: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Debug override for n_buffer."},
    )
    protrain_n_swap_override: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Debug override for n_swap."},
    )
    protrain_n_checkpoint_override: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Debug override for n_checkpoint."},
    )
    protrain_n_offload_override: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Debug override for n_offload (Option B). When set, forces the "
                "given count of OFFLOAD-mode blocks (saved-tensors-hooks for "
                "params, no recompute). Only meaningful with "
                "``protrain_force_all_persistent: false`` and a layout that has "
                "non-persistent chunks; ignored otherwise."
            )
        },
    )

    protrain_zero3_shard: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Explicit override for the ZeRO-3 sharded-offload mode. "
                "When ``protrain_auto_mode`` is True (default) this flag is "
                "IGNORED by the mode-selector — the plugin auto-picks A/B/C "
                "based on workload fit + CPU-RAM-per-rank. When "
                "``protrain_auto_mode`` is False, None preserves the pre-auto "
                "behaviour (auto-enable at world_size>1 unless DDP is on top), "
                "True forces sharding on (subject to world_size>1), False "
                "disables sharding. M7 benchmark (DESIGN.md §Multi-GPU) shows "
                "sharded throughput lands around 0.70x single-rank on PCIe "
                "Gen3 4x RTX 3090 — only pick this when CPU RAM is truly the "
                "binding constraint."
            )
        },
    )

    protrain_force_replicated_cpu_offload: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Explicit override for the Mode B (replicated CPU-offload) "
                "mode. Parity knob with ``protrain_force_all_persistent`` "
                "(Mode A) and ``protrain_zero3_shard`` (Mode C). When "
                "``protrain_auto_mode`` is True (default) this flag is "
                "IGNORED — the plugin auto-picks A/B/C based on workload fit "
                "+ CPU-RAM-per-rank. When ``protrain_auto_mode`` is False, "
                "True forces the wrapper into Mode B by pinning "
                "``force_all_persistent=False`` AND ``zero3_shard=False`` "
                "(non-persistent chunks live on CPU, replicated across "
                "ranks). Set ``protrain_auto_mode: false`` alongside to make "
                "this effective. Mutually exclusive with "
                "``protrain_force_all_persistent: true`` and "
                "``protrain_zero3_shard: true`` (the model validator rejects "
                "two or more force flags set simultaneously). Useful for "
                "reproducing benchmark configurations where the auto-picker "
                "would otherwise pick Mode C; in particular this is the only "
                "way today to explicitly select Mode B in isolation."
            )
        },
    )

    # Optimizer-state checkpoint/resume.

    protrain_save_optimizer_state: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Opt-in: persist ProTrain optimizer state (Adam momentums + "
                "step counters) alongside HF Trainer checkpoints. Default "
                "False — resumed runs cold-start every momentum buffer, "
                "which matches today's behavior. When True, a TrainerCallback "
                "writes per-chunk shard files under "
                "``{checkpoint_dir}/protrain_optim/`` after each save; "
                "``Trainer._load_optimizer_and_scheduler`` is wrapped to load "
                "from the same path on resume. Supported configurations: "
                "single-rank non-ZeRO (Phase 1), multi-rank DDP-replicated "
                "(Phase 2 Mode-B, rank-0-only writes to ``chunk_<N>.pt``), "
                "and multi-rank ZeRO-3 sharded (Phase 2 Mode-C, every rank "
                "writes its own ``chunk_<N>_rank_<R>.pt``). Saves are gated "
                "by ``protrain_optim_save_max_bytes`` to avoid silently "
                "writing 84 GB blobs for 7B full-FT; in multi-rank runs "
                "rank-0's gate decision is broadcast so all ranks save or "
                "none do."
            )
        },
    )

    protrain_optim_save_max_bytes: int | None = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=0,
        json_schema_extra={
            "description": (
                "Soft cap (bytes) on the estimated optimizer-state save "
                "size. Default 2 GiB — small enough that LoRA always passes, "
                "7B full-FT (~84 GB) never silently passes. The estimate "
                "walks the inner adapters' state dicts (``_gpu_optim._optim`` "
                "and every ``_cpu_optim._optims[*]``) and sums each Adam "
                "state tensor's bytes — matching what gets pickled to disk. "
                "Walking the user-facing param_groups would undercount: "
                "ChunkManager.materialize_offload replaces offloaded "
                "params' ``.data`` with empty placeholders, so "
                "``p.numel()`` returns 0 for offloaded chunks between "
                "training steps. When the estimate exceeds this cap, the "
                "save callback emits a WARN naming the estimate and skips "
                "writing. Set explicitly higher to opt in to large saves."
            )
        },
    )

    protrain_save_optim_verify_replicated: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Mode-B (DDP-replicated) only: if True, on the FIRST save "
                "of each run every rank hashes its inner optimizer state "
                "and ``all_gather_object``-s the hashes; the save aborts "
                "with ``RuntimeError`` if the hashes don't match. Default "
                "False because DDP determinism makes a divergence very "
                "unlikely in practice and the check costs one full state "
                "hash + an all_gather. Subsequent saves skip the check "
                "(per-save would be expensive). Has no effect on "
                "single-rank or ZeRO-3 sharded runs."
            )
        },
    )

    protrain_allow_online_reshard: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Mode-C (ZeRO-3 sharded) only: if True, allow the load "
                "path to automatically reshard a saved Mode-C checkpoint "
                "from its saved world_size to the current run's "
                "world_size. Default False — a world_size mismatch hard-"
                "errors and points the user at the offline reshard tool "
                "(``python -m scripts.protrain.reshard_optim``). The opt-"
                "in is off by default because (a) resharding mutates "
                "files in (or under) the checkpoint dir before loading, "
                "(b) silent automatic resharding could mask "
                "configuration drift the user actually wanted to know "
                "about. When True, on world_size mismatch rank-0 invokes "
                "the same reshard logic as the offline tool against a "
                "temp dir (``<saved-protrain_optim>/.reshard_to_N<W>/``), "
                "all ranks barrier, then load from the temp dir using "
                "the existing same-world-size load path. Cleanup runs "
                "on successful load; failures leave the temp dir for "
                "post-mortem. Mode-B replicated saves do not need this "
                "knob — they already tolerate world_size drift natively "
                "(CHECKPOINT_DESIGN_PHASE2.md §4.1 Option B). The reshard "
                "logic is the offline tool's: see "
                "``src/axolotl/integrations/protrain/api/reshard.py``."
            )
        },
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _require_plugin_registration(cls, data):
        """protrain_auto_memory=True requires the plugin in plugins:."""
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        # Let Pydantic emit its standard field-type error for malformed plugins.
        if not isinstance(plugins, (list, tuple, set, frozenset)):
            return data
        if not _has_protrain_plugin(plugins):
            raise ValueError(
                "`protrain_auto_memory: true` requires the ProTrain plugin to be "
                "listed in `plugins:`. Add "
                "`- axolotl.integrations.protrain.ProTrainPlugin` to the "
                "`plugins` list."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_incompatible_features(cls, data):
        """Mutex with deepspeed/fsdp/gradient_checkpointing/TP/CP/SP that conflict with ProTrain."""
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not _has_protrain_plugin(plugins):
            return data
        if data.get("deepspeed"):
            raise ValueError(
                "ProTrain + DeepSpeed cannot be used together: both manage "
                "per-rank model-state placement. Remove `deepspeed:` or disable "
                "`protrain_auto_memory`."
            )
        if data.get("fsdp") or data.get("fsdp_config"):
            raise ValueError(
                "ProTrain + FSDP cannot be used together: both manage "
                "per-rank model-state placement. Remove `fsdp:` / `fsdp_config:` "
                "or disable `protrain_auto_memory`."
            )
        if data.get("gradient_checkpointing"):
            raise ValueError(
                "ProTrain is incompatible with gradient_checkpointing=true "
                "(ProTrain installs its own activation checkpointing per the M3 "
                "block manager; HuggingFace's gradient_checkpointing on top "
                "would double-checkpoint the forward pass). Set "
                "gradient_checkpointing=false or remove the ProTrain plugin."
            )
        tp_size = data.get("tensor_parallel_size")
        if tp_size is not None:
            try:
                tp_size_int = int(tp_size)
            except (TypeError, ValueError):
                # Non-numeric value (e.g., "auto"); let Pydantic surface the type error.
                tp_size_int = None
            if tp_size_int is not None and tp_size_int > 1:
                raise ValueError(
                    "ProTrain is incompatible with tensor_parallel_size > 1 "
                    "(scope-excluded per plan.md — the chunk layout does not shard "
                    "across TP ranks in this milestone). Set tensor_parallel_size=1 "
                    "or remove the ProTrain plugin."
                )
        cp_size = data.get("context_parallel_size")
        if cp_size is not None:
            try:
                cp_size_int = int(cp_size)
            except (TypeError, ValueError):
                cp_size_int = None
            if cp_size_int is not None and cp_size_int > 1:
                raise ValueError(
                    "ProTrain is incompatible with context_parallel_size > 1 "
                    "(scope-excluded per plan.md — single-3090 target). Set "
                    "context_parallel_size=1 or remove the ProTrain plugin."
                )
        sp_degree = data.get("sequence_parallel_degree")
        if sp_degree is not None:
            try:
                sp_degree_int = int(sp_degree)
            except (TypeError, ValueError):
                sp_degree_int = None
            if sp_degree_int is not None and sp_degree_int > 1:
                raise ValueError(
                    "ProTrain is incompatible with sequence_parallel_degree > 1 "
                    "(scope-excluded per plan.md — single-3090 target). Set "
                    "sequence_parallel_degree=1 or remove the ProTrain plugin."
                )
        # bnb 8-bit / 4-bit composes with ProTrain in both Mode A and offload paths.
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_unsupported_optimizer(cls, data):
        """Reject ``cfg.optimizer`` values that ProTrain's adapters cannot drive."""
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not _has_protrain_plugin(plugins):
            return data
        optimizer = data.get("optimizer")
        if optimizer is None:
            return data
        # Tolerate enum + str values.
        optimizer_str = getattr(optimizer, "value", optimizer)
        normalized = str(optimizer_str).strip().lower()
        if normalized not in _SUPPORTED_OPTIMIZERS:
            supported = ", ".join(sorted(_SUPPORTED_OPTIMIZERS))
            raise ValueError(
                f"ProTrain currently supports AdamW family optimizers only "
                f"(got `{optimizer_str}`). Lion, Adafactor, GaLore, Sophia, "
                f"Muon, and torchao optimizers require optimizer-specific "
                f"chunk-manager adapters that have not been implemented. See "
                f"src/axolotl/integrations/protrain/chunk/optim.py for the "
                f"supported adapter list. Supported optimizers: "
                f"{supported}. Set `optimizer: adamw_torch` (or another "
                f"supported value above) or remove the ProTrain plugin."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_multiple_force_modes(cls, data):
        """Reject more than one ``protrain_force_*`` mode flag set at once.

        Forces are explicit overrides. Two of them on at the same time would
        contradict each other (e.g. ``force_all_persistent=true`` together
        with ``force_replicated_cpu_offload=true`` would tell the wrapper
        to keep everything on GPU AND offload everything to CPU). Reject at
        config-load time with an actionable error rather than silently
        having one flag win inside the mode-selector.
        """
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not _has_protrain_plugin(plugins):
            return data
        # Only count flags that are explicitly truthy (None / False / unset = inactive).
        set_flags = [
            name
            for name in (
                "protrain_force_all_persistent",
                "protrain_force_replicated_cpu_offload",
                "protrain_zero3_shard",
            )
            if bool(data.get(name))
        ]
        if len(set_flags) > 1:
            joined = ", ".join(f"`{f}: true`" for f in set_flags)
            raise ValueError(
                f"ProTrain mode-force flags are mutually exclusive but multiple "
                f"are set: {joined}. The three force flags correspond to "
                f"different multi-GPU modes — "
                f"`protrain_force_all_persistent` (Mode A, GPU-resident), "
                f"`protrain_force_replicated_cpu_offload` (Mode B, replicated "
                f"CPU offload), and `protrain_zero3_shard` (Mode C, sharded "
                f"CPU offload). Pick at most one, or set "
                f"`protrain_auto_mode: true` to let the searcher pick."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _require_model_or_adapter(cls, data):
        """Basic sanity: a training run needs a base model (adapter is optional)."""
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
        if not _has_protrain_plugin(plugins):
            return data
        if not (data.get("base_model") or data.get("model_name_or_path")):
            raise ValueError(
                "`protrain_auto_memory: true` requires a `base_model` (or "
                "`model_name_or_path`) to be configured."
            )
        return data

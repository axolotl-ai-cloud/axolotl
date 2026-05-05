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

"""Pydantic argument model for the ProTrain plugin (M5, DESIGN.md §Plugin Integration).

Merged into the top-level Axolotl config schema at validation time via the
``plugins:`` entry in the user YAML. Mirrors the shape of
``axolotl.integrations.liger.LigerArgs`` / ``axolotl.integrations.spectrum.SpectrumArgs``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Canonical plugin identifier strings that activate the ProTrain validators.
#
# THIS IS THE SINGLE SOURCE OF TRUTH for the strict allow-list used at
# Pydantic config-validation time. ``axolotl.integrations.protrain.plugin
# ::_is_plugin_active`` (the runtime activation gate) imports
# :func:`_has_protrain_plugin` / :data:`_PROTRAIN_PLUGIN_KEYS` from this
# module so both sites agree on what counts as "the ProTrain plugin is
# registered". If you add a new accepted form, add it here — do not
# fork the list in ``plugin.py``.
#
# Only `axolotl.integrations.protrain.ProTrainPlugin` is accepted — that's
# the form used by tests, the example config
# (examples/protrain/3090-7b-lora.yml), and the docstrings in this file,
# and it's the only form that actually loads (the integration loader
# rsplits on '.' for module/class). The bare module form
# `axolotl.integrations.protrain` is intentionally REJECTED: it would
# silently bypass plugin registration entirely (the loader can't resolve
# a class from it), so accepting it here would let
# `protrain_auto_memory: true` pass validation while the runtime hooks
# never install. Users who type the bare module form get the same
# "missing plugin" ValueError as users who omit `plugins:` altogether,
# pointing them at the correct class form.
#
# The runtime gate ``plugin._is_plugin_active`` historically accepted
# additional fully-qualified spellings (e.g. ``...plugin.ProTrainPlugin``)
# under a case-insensitive normalize — those forms are not produced by
# the documented user-facing config and are NOT part of this allow-list.
# Unifying on the strict set here is intentional: the runtime gate
# should never fire for an id the config validator would have rejected.
_PROTRAIN_PLUGIN_KEYS = frozenset(
    {
        "axolotl.integrations.protrain.ProTrainPlugin",
    }
)


def _has_protrain_plugin(plugins) -> bool:
    """Return True iff the iterable contains an explicit ProTrain plugin id.

    Uses exact-match against ``_PROTRAIN_PLUGIN_KEYS`` rather than a
    substring check so that unrelated plugin names containing the
    substring ``"protrain"`` (or future plugins under a different module
    path) cannot accidentally activate the ProTrain validators.

    This helper is the single source of truth for "is the ProTrain
    plugin registered in ``plugins:``": both the Pydantic validators in
    this module AND ``plugin._is_plugin_active`` (the runtime activation
    gate) call it so config validation and runtime activation cannot
    drift apart on which ids count as registered.

    Tolerates malformed ``plugins`` values: a non-iterable scalar (None,
    int, bool, dict, etc.) returns False rather than raising
    ``TypeError`` from ``any(... for p in plugins)``, and non-string
    entries inside the iterable are skipped via the ``isinstance(p, str)``
    guard. This keeps config-validation failures actionable — the user
    sees the schema-level type error on ``plugins`` itself rather than
    a confusing crash from this helper.
    """
    if not isinstance(plugins, (list, tuple, set, frozenset)):
        return False
    return any(isinstance(p, str) and p in _PROTRAIN_PLUGIN_KEYS for p in plugins)


# Re-exported so ``plugin.py`` (and any future call site that needs the
# strict ProTrain-plugin allow-list) can import a single canonical name
# rather than copy-pasting the set. Keeping this in ``__all__`` also
# documents the public-to-the-package contract: this constant + helper
# are the answer to "which strings count as the ProTrain plugin id".
__all__ = ["ProTrainArgs", "_PROTRAIN_PLUGIN_KEYS", "_has_protrain_plugin"]


class ProTrainArgs(BaseModel):
    """Input args for the ProTrain plugin.

    The plugin is opt-in at two levels: (1) the YAML must list
    ``axolotl.integrations.protrain.ProTrainPlugin`` in ``plugins:``,
    and (2) ``protrain_auto_memory`` must be True. The second gate lets
    users add the plugin import for args-schema registration without
    actually rewiring the training path (useful for validation /
    documentation).
    """

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
                "TP/CP/SP > 1, and load_in_8bit/load_in_4bit (see "
                "`_reject_incompatible_features`)."
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

    # Debugging escape hatches — bypass the searcher. Intended for
    # reproducibility experiments and bug-hunting; production runs should
    # leave these None and let the cost model pick.
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

    # ------------------------------------------------------------------
    # Optimizer-state checkpoint/resume (CHECKPOINT_DESIGN.md Phase 1,
    # CHECKPOINT_DESIGN_PHASE2.md Modes B + C)
    # ------------------------------------------------------------------

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
        """``protrain_auto_memory=True`` requires the plugin in ``plugins:``.

        Clone of the enable-guard pattern used by Liger / Spectrum: the
        plugin being present in ``plugins:`` is what causes its args
        model to be merged in, but a user could set the YAML flag without
        the plugin import — this validator surfaces that misconfiguration
        as a clear ValueError instead of a silently-ignored flag.
        """
        if not isinstance(data, dict):
            return data
        if not data.get("protrain_auto_memory"):
            return data
        plugins = data.get("plugins") or []
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
        """Mutex with features that conflict with ProTrain's runtime.

        ProTrain owns per-rank memory policy (chunk placement, activation
        checkpointing, optimizer-state hosting). Several Axolotl features
        either duplicate that policy or operate on representations the
        chunk manager cannot see:

        * ``deepspeed`` / ``fsdp`` / ``fsdp_config`` — alternative
          per-rank model-state managers; running either alongside
          ProTrain double-manages params, grads, and optim state.
        * ``gradient_checkpointing: true`` — ProTrain's M3 block manager
          installs its own CKPT hooks from ``n_checkpoint``; adding
          HuggingFace's ckpt wrapper on top double-checkpoints forwards
          (recomputes twice, doubles activation traffic).
        * ``tensor_parallel_size`` / ``context_parallel_size`` /
          ``sequence_parallel_degree`` > 1 — scope-excluded per plan.md
          (M6 single-3090 focus); the chunk layout does not shard
          correctly across TP/CP ranks in this milestone.
        * ``load_in_8bit`` / ``load_in_4bit`` — bnb weight quantization
          wraps ``nn.Linear.weight`` in a non-owning proxy. The chunk
          manager reads unquantized storage for gather / offload and
          cannot reason about the 8-bit / 4-bit packed buffers.

        Each rejection surfaces at config-load time rather than as a
        silent mis-training run.
        """
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
                # Non-numeric value (e.g., "auto") — let Pydantic surface
                # the type error from its own field validators.
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
        if data.get("load_in_8bit"):
            raise ValueError(
                "ProTrain is incompatible with load_in_8bit=true (bitsandbytes "
                "8-bit quantization wraps nn.Linear.weight in a non-owning proxy; "
                "the chunk manager operates on unquantized storage for gather / "
                "offload). Set load_in_8bit=false or remove the ProTrain plugin."
            )
        if data.get("load_in_4bit"):
            raise ValueError(
                "ProTrain is incompatible with load_in_4bit=true (bitsandbytes "
                "4-bit quantization wraps nn.Linear.weight in a non-owning proxy; "
                "the chunk manager operates on unquantized storage for gather / "
                "offload). Set load_in_4bit=false or remove the ProTrain plugin."
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

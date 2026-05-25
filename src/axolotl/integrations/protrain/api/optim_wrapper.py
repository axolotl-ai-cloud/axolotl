"""Public optimizer-wrapper for the ProTrain runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from axolotl.integrations.protrain.chunk import (
    CpuFusedAdamAdapter,
    GpuAdamW8bitAdapter,
    GpuFusedAdamAdapter,
)
from axolotl.integrations.protrain.sentinels import (
    _PROTRAIN_PERSISTENT_HUGE_PARAM_WITHIN_SHARD_VERSION,
    _PROTRAIN_PERSISTENT_ROUND_ROBIN_PARTITION_VERSION,
)
from axolotl.integrations.protrain.types import ChunkId, WrappedModel
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

    from axolotl.integrations.protrain.chunk import ChunkManager

LOG = get_logger(__name__)

# Sentinels (definitions in sentinels.py) re-exported here so external bench
# scripts that grep this file by sentinel name still find a match:
# _PROTRAIN_PERSISTENT_ROUND_ROBIN_PARTITION_VERSION,
# _PROTRAIN_PERSISTENT_HUGE_PARAM_WITHIN_SHARD_VERSION.

# Default huge-param threshold: 512 MiB. Override via cfg.protrain_persistent_huge_param_threshold_bytes.
_DEFAULT_HUGE_PARAM_THRESHOLD_BYTES = 512 * 1024 * 1024


class _ProTrainOptimizer(torch.optim.Optimizer):
    """Optimizer facade over the ProTrain GPU/CPU adapter pair."""

    def __init__(
        self,
        gpu_optim: GpuFusedAdamAdapter | GpuAdamW8bitAdapter | None,
        cpu_optim: CpuFusedAdamAdapter | None,
        params: list["nn.Parameter"],
        defaults: dict[str, Any],
        chunk_manager: Any,
        *,
        persistent_params_full: list["nn.Parameter"] | None = None,
        persistent_owner_rank: list[int] | None = None,
        persistent_world_size: int = 1,
        persistent_huge_originals: list["nn.Parameter"] | None = None,
        persistent_huge_shards: list["nn.Parameter"] | None = None,
    ) -> None:
        """Wire the GPU/CPU adapter pair into a Trainer-compatible Optimizer facade."""
        # Pass full param list so schedulers iterate over the real set.
        if not params:
            raise ValueError(
                "_ProTrainOptimizer: model has no tunable parameters; "
                "nothing to optimize."
            )
        super().__init__(params, defaults)
        self._gpu_optim = gpu_optim
        self._cpu_optim = cpu_optim
        self._chunk_manager = chunk_manager
        # Round-robin partition state for post-step persistent param sync.
        self._persistent_params_full: list["nn.Parameter"] = list(
            persistent_params_full or []
        )
        self._persistent_owner_rank: list[int] = list(persistent_owner_rank or [])
        self._persistent_world_size: int = int(persistent_world_size)
        # Within-param shard fallback metadata for huge persistent params (v51).
        # _persistent_huge_originals[i] is the full-shape source param; the
        # corresponding _persistent_huge_shards[i] is this rank's narrow view.
        self._persistent_huge_originals: list["nn.Parameter"] = list(
            persistent_huge_originals or []
        )
        self._persistent_huge_shards: list["nn.Parameter"] = list(
            persistent_huge_shards or []
        )

    # ---- step / zero_grad ----------------------------------------------

    def step(self, closure: Any = None) -> Any:
        """Drive both adapters then block on in-flight CPU futures."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Forward LR-scheduler mutations to inner optim param_groups (separate dicts).
        self._forward_hyperparams_to_inner_optims()

        # Sharded path: sweep orphan non-persistent chunks (lm_head/embed) that no block-backward hook reached.
        cm = self._chunk_manager
        if getattr(cm, "zero3_shard", False):
            non_persist = getattr(cm, "_non_persistent_ids", None)
            if non_persist:
                for cid in list(non_persist):
                    cm.reduce_grads_and_offload(cid)

        # Within-shard fallback: route grad of each huge original onto the
        # rank-local shard view BEFORE the inner step. The shard is a
        # narrow into orig.data, so narrowing orig.grad the same way gives
        # the shard the matching slice of the gradient.
        self._route_huge_grads_to_shards()

        if self._gpu_optim is not None:
            self._gpu_optim.step()
        # Broadcast each owner's persistent-param update to peers before next forward.
        self._sync_persistent_params_after_step()
        # Drain all CPU Adam futures enqueued by grad hooks + orphan sweep.
        self._chunk_manager.wait_cpu_optim_all()
        # Per-step boundary: sync prefetch/swap/offload streams + flush deferred offloads.
        scheduler = getattr(self._chunk_manager, "_scheduler_ref", None)
        if scheduler is not None:
            scheduler.drain()
        return loss

    def _route_huge_grads_to_shards(self) -> None:
        """Narrow each huge original param's grad onto this rank's shard view.

        ``GpuFusedAdamAdapter`` (and the test fake) steps each shard
        nn.Parameter directly; the shard's ``data`` is a narrow into
        ``orig.data`` along dim-0, and autograd populates ``orig.grad``
        (the shard is not in the autograd graph). Narrowing ``orig.grad``
        the matching way hands each shard the correct slice of the
        gradient just before .step() runs.
        """
        if not self._persistent_huge_originals:
            return
        rank = int(getattr(self._chunk_manager, "rank", 0) or 0)
        world = int(self._persistent_world_size)
        if world <= 0:
            return
        for orig, shard in zip(
            self._persistent_huge_originals,
            self._persistent_huge_shards,
            strict=True,
        ):
            if orig.grad is None:
                shard.grad = None
                continue
            shard_size = orig.shape[0] // world
            shard.grad = orig.grad.narrow(0, rank * shard_size, shard_size)

    def _sync_persistent_params_after_step(self) -> None:
        """All-reduce(SUM)-with-zeros to broadcast each owner's post-step update.

        Each rank only steps its owned persistent params (round-robin
        partition). Non-owned params hold their pre-step value. Zeroing
        non-owned param.data and then summing across ranks yields the
        owner's post-step value everywhere. param.data is the only
        write; ``state[param]`` is untouched (Adam keys by tensor id).

        Huge-param within-shard fallback (v51): each rank already updated
        its own dim-0 slice of ``orig.data`` (the shard is a narrow view
        into the original storage). An ``all_gather_into_tensor`` over
        the per-rank shards reconstructs the full ``orig.data`` on every
        rank.
        """
        if self._persistent_world_size <= 1:
            return
        if not self._persistent_params_full and not self._persistent_huge_originals:
            return
        import torch.distributed as dist
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        if not (dist.is_available() and dist.is_initialized()):
            return

        rank = int(dist.get_rank())

        # Small-param round-robin sync.
        if self._persistent_params_full:
            # Zero non-owned param.data BEFORE the collective so the SUM lands the owner's value.
            for i, param in enumerate(self._persistent_params_full):
                if self._persistent_owner_rank[i] != rank:
                    param.data.zero_()

            # Bucket by (dtype, device) so a single collective covers each homogeneous group.
            buckets: dict[tuple[Any, Any], list["nn.Parameter"]] = {}
            for param in self._persistent_params_full:
                key = (param.data.dtype, param.data.device)
                buckets.setdefault(key, []).append(param)

            for params in buckets.values():
                if len(params) == 1:
                    dist.all_reduce(params[0].data, op=dist.ReduceOp.SUM)
                    continue
                tensors = [p.data for p in params]
                flat = _flatten_dense_tensors(tensors)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                for orig, synced in zip(
                    tensors, _unflatten_dense_tensors(flat, tensors), strict=True
                ):
                    orig.copy_(synced)

        # Huge-param within-shard sync: each rank's shard is a narrow into
        # orig.data, so the gather lands directly in-place across all ranks.
        if self._persistent_huge_originals:
            for orig, shard in zip(
                self._persistent_huge_originals,
                self._persistent_huge_shards,
                strict=True,
            ):
                # Source tensor must be contiguous; shard_view.contiguous()
                # is a no-op for the narrow we created (dim-0 slice of a
                # contiguous parent), but gloo's collective requires it.
                src = shard.data.contiguous()
                dst = orig.data
                if not dst.is_contiguous():
                    raise RuntimeError(
                        "protrain: huge persistent param data is not "
                        "contiguous; within-shard all_gather requires "
                        "a contiguous destination storage."
                    )
                dist.all_gather_into_tensor(dst, src)
                # If we had to materialize a contiguous copy, the shard's
                # data view now points at the correct slice of dst again
                # (still a narrow into orig.data); no rewrite needed.

    # ---- LR-scheduler hyperparam forwarding -----------------------------

    # weight_decay excluded: inner has two groups (decay/no-decay) and the facade has one.
    _FORWARDED_HYPERPARAM_KEYS = ("lr", "betas", "eps")

    def _forward_hyperparams_to_inner_optims(self) -> None:
        """Copy facade ``param_groups[0]`` hyperparams to each inner optim."""
        if not self.param_groups:
            return
        src = self.param_groups[0]

        def _push(inner_optim) -> None:
            if inner_optim is None:
                return
            for inner_group in inner_optim.param_groups:
                for key in self._FORWARDED_HYPERPARAM_KEYS:
                    if key in src and key in inner_group:
                        inner_group[key] = src[key]

        if self._gpu_optim is not None:
            _push(getattr(self._gpu_optim, "_optim", None))
        if self._cpu_optim is not None:
            for inner in getattr(self._cpu_optim, "_optims", {}).values():
                _push(inner)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """Zero gradients on every adapter and any unrouted param-group entries."""
        if self._gpu_optim is not None:
            self._gpu_optim.zero_grad(set_to_none=set_to_none)
        if self._cpu_optim is not None:
            self._cpu_optim.zero_grad(set_to_none=set_to_none)
        # Also zero any param grads that weren't routed through either
        # adapter (e.g. buffers that slipped through the chunk layout) so
        # the next iteration starts clean.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    # ---- checkpointing: torch-side no-ops, real save/load lives in the
    # ProTrain checkpoint callback (M5/M6) -------------------------------
    #
    # ``protrain_optimizer_wrapper`` is exported in the public API and
    # ``create_optimizer`` returns the raw wrapper before
    # ``post_trainer_create`` would have a chance to monkey-patch the
    # instance. HF Trainer (when ``save_only_model`` is False) and
    # Accelerate (at ``prepare`` time, unconditionally) both call
    # ``state_dict`` / ``load_state_dict`` on the optimizer; raising
    # ``NotImplementedError`` here would crash any out-of-trainer
    # consumer (model_wrapper.py profiling, tests). The adapters own
    # their own state and persist it through the dedicated ProTrain
    # checkpoint hook, so torch-side state is safely empty.

    #: Sentinel key marking a state_dict produced by this class as a
    #: hollow shell (CHECKPOINT_DESIGN.md §1.7 Option P). Lets
    #: ``load_state_dict`` distinguish the safe round-trip case
    #: (Accelerate ``prepare`` walk OR user ``torch.save(state_dict()) →
    #: torch.load → load_state_dict``) from a payload from a different
    #: optimizer that was incorrectly fed to this wrapper. The marker
    #: is a plain bool so Accelerate's ``move_to_device`` / ``.to(...)``
    #: walks ignore it.
    _PROTRAIN_HOLLOW_MARKER_KEY = "_protrain_hollow_state_dict"

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return an empty torch-side optimizer state.

        Real ProTrain optimizer state (per-shard moments held inside the
        CPU/GPU FusedAdam adapters) is saved by the dedicated checkpoint
        callback (see ``api/checkpoint.py``), NOT through this method.
        We still preserve HF's ``{"state": ..., "param_groups": ...}``
        shape so Accelerate's ``move_to_device(state_dict, ...)`` +
        ``load_state_dict`` round trip at ``prepare`` time does not
        crash. A ``_protrain_hollow_state_dict: True`` marker is added
        so ``load_state_dict`` can recognise the round trip and silently
        no-op (instead of raising on payloads it can't actually
        consume).

        IMPORTANT: this method does NOT serialise adapter moments. A
        naive ``torch.save(optim.state_dict())`` / ``torch.load`` /
        ``optim.load_state_dict(...)`` round trip will discard
        per-parameter moments — the saved blob is the hollow shell.
        Use the ProTrain checkpoint flow
        (``_save_protrain_optim_dir`` / ``_load_protrain_optim_dir``,
        wired via the ``post_trainer_create`` hook) for real persistence.
        """
        next_param_idx = 0
        param_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            n_params = len(group["params"])
            param_groups.append(
                {k: v for k, v in group.items() if k != "params"}
                | {"params": list(range(next_param_idx, next_param_idx + n_params))}
            )
            next_param_idx += n_params
        return {
            self._PROTRAIN_HOLLOW_MARKER_KEY: True,
            "state": {},
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Round-trip the hollow shell or fail loudly on foreign payloads.

        Accepts the hollow shell produced by ``state_dict()`` (Accelerate
        ``prepare`` round-trip OR a user ``torch.save / torch.load``
        sequence over the same wrapper) — that path silently no-ops,
        consistent with the wrapper's documented contract that real
        persistence flows through the dedicated ProTrain checkpoint
        hook (``api/checkpoint.py``).

        Any OTHER payload — e.g. a state_dict produced by a different
        optimizer, or a real torch state_dict the user thinks should
        restore — raises ``NotImplementedError`` with a pointer at
        the dedicated checkpoint hook. Replacing the previous silent
        no-op stops the footgun where users assumed
        ``optim.load_state_dict(saved_blob)`` would restore moments.
        """
        if not isinstance(state_dict, dict):
            raise NotImplementedError(
                "_ProTrainOptimizer.load_state_dict requires a dict; got "
                f"{type(state_dict).__name__}. Use the ProTrain checkpoint "
                "hook (api/checkpoint.py::_load_protrain_optim_dir) for real "
                "optimizer-state restore."
            )
        if state_dict.get(
            self._PROTRAIN_HOLLOW_MARKER_KEY
        ) is True and not state_dict.get("state"):
            # Hollow-shell round-trip (Accelerate prepare); silent no-op.
            return None
        raise NotImplementedError(
            "_ProTrainOptimizer.load_state_dict cannot restore an arbitrary "
            "torch optimizer state. The wrapper's public state_dict is a "
            "hollow shell by design (CHECKPOINT_DESIGN.md §1.7 Option P) — "
            "real per-shard FusedAdam moments are persisted via the "
            "dedicated ProTrain checkpoint flow. Load via "
            "api/checkpoint.py::_load_protrain_optim_dir (wired through "
            "Trainer._load_optimizer_and_scheduler in post_trainer_create) "
            "instead of torch.save / torch.load over state_dict."
        )

    # Private snapshot/restore for phase-2 rollback — bypasses the hollow public shell.
    def _protrain_snapshot_inner_state(self) -> dict[str, Any]:
        """Snapshot the REAL inner adapter state (not the hollow public shell)."""
        gpu_state: dict[str, Any] | None = None
        if self._gpu_optim is not None:
            inner = self._gpu_optim._optim
            if inner is not None:
                gpu_state = inner.state_dict()
        cpu_state_per_chunk: dict[ChunkId, dict[str, Any]] = {}
        if self._cpu_optim is not None:
            for cid, inner in self._cpu_optim._optims.items():
                cpu_state_per_chunk[cid] = inner.state_dict()
        return {"gpu": gpu_state, "cpu_per_chunk": cpu_state_per_chunk}

    def _protrain_restore_inner_state(self, snapshot: dict[str, Any]) -> None:
        """Restore inner-adapter state previously captured by the snapshot helper."""
        gpu_state = snapshot.get("gpu")
        if (
            gpu_state is not None
            and self._gpu_optim is not None
            and self._gpu_optim._optim is not None
        ):
            self._gpu_optim._optim.load_state_dict(gpu_state)
        cpu_state_per_chunk = snapshot.get("cpu_per_chunk") or {}
        if self._cpu_optim is not None and cpu_state_per_chunk:
            for cid, inner in self._cpu_optim._optims.items():
                inner_state = cpu_state_per_chunk.get(cid)
                if inner_state is not None:
                    inner.load_state_dict(inner_state)


def _collect_no_decay_param_ids(module: "nn.Module") -> set[int]:
    """Return ``id(p)`` for params HF Trainer puts in the no-decay group (norm + bias)."""
    from torch import nn

    no_decay: set[int] = set()
    for mod_name, mod in module.named_modules():
        is_norm_module = (
            isinstance(mod, nn.LayerNorm) or "norm" in type(mod).__name__.lower()
        )
        for param_name, param in mod.named_parameters(recurse=False):
            full_name = f"{mod_name}.{param_name}" if mod_name else param_name
            if is_norm_module or full_name.lower().endswith("bias"):
                no_decay.add(id(param))
    return no_decay


def _collect_sharded_no_decay_shard_param_ids(
    chunk_manager: "ChunkManager",
    cpu_params_per_chunk: "dict[ChunkId, list[nn.Parameter]]",
    no_decay_orig_param_ids: set[int],
) -> set[int]:
    """Map original-param no-decay set onto sharded shard_param ids (correctness-conservative)."""
    if not no_decay_orig_param_ids:
        return set()
    chunk_shards = getattr(chunk_manager, "_chunk_shards", None)
    if not chunk_shards:
        return set()
    cpu_slots_by_cid = getattr(chunk_manager, "_cpu_slots", {}) or {}
    no_decay_shard_ids: set[int] = set()
    for cid, _params in cpu_params_per_chunk.items():
        shard_state = chunk_shards.get(cid)
        if shard_state is None or not shard_state.regions:
            continue
        slots = cpu_slots_by_cid.get(cid, [])
        if not slots:
            continue
        # Pre-resolve each slot to (start, end, is_no_decay) once.
        slot_extents: list[tuple[int, int, bool]] = []
        for slot in slots:
            param = chunk_manager._params_by_id.get(slot.param_id)
            if param is None:
                continue
            start = int(slot.byte_offset)
            end = start + int(slot.numel) * int(slot.element_size)
            slot_extents.append((start, end, id(param) in no_decay_orig_param_ids))
        for region in shard_state.regions:
            r_start = int(region.chunk_offset)
            r_end = r_start + int(region.region_bytes)
            region_has_no_decay = False
            for s_start, s_end, slot_no_decay in slot_extents:
                if not slot_no_decay:
                    continue
                # Intersection check.
                if s_start < r_end and s_end > r_start:
                    region_has_no_decay = True
                    break
            if region_has_no_decay:
                no_decay_shard_ids.add(id(region.shard_param))
    return no_decay_shard_ids


def _split_optim_param_groups(
    inner: torch.optim.Optimizer | None,
    no_decay_param_ids: set[int],
) -> None:
    """Split each of ``inner.param_groups`` into a decay/no-decay pair in place."""
    if inner is None or not no_decay_param_ids:
        return
    new_groups: list[dict[str, Any]] = []
    changed = False
    for group in inner.param_groups:
        params = list(group["params"])
        decay_params = [p for p in params if id(p) not in no_decay_param_ids]
        no_decay_params = [p for p in params if id(p) in no_decay_param_ids]
        if not no_decay_params:
            new_groups.append(group)
            continue
        if not decay_params:
            if group.get("weight_decay", 0.0) != 0.0:
                group["weight_decay"] = 0.0
                changed = True
            new_groups.append(group)
            continue
        # Mixed: split into two groups; only weight_decay differs.
        decay_group = {**group, "params": decay_params}
        no_decay_group = {**group, "params": no_decay_params, "weight_decay": 0.0}
        new_groups.append(decay_group)
        new_groups.append(no_decay_group)
        changed = True
    if not changed:
        return
    # Safe direct replacement: optimizer state keys on id(param), not group index.
    inner.param_groups = new_groups


# Optimizer-name strings routing persistent chunks through GpuAdamW8bitAdapter.
_BNB_8BIT_OPTIMIZERS: frozenset[str] = frozenset(
    {"adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit"}
)
_BNB_8BIT_PAGED_OPTIMIZERS: frozenset[str] = frozenset({"paged_adamw_8bit"})


def _normalize_optimizer_name(name: str | None) -> str | None:
    """Lower-case + strip whitespace, unwrapping ``OptimizerNames`` enums via ``.value``."""
    if name is None:
        return None
    return str(getattr(name, "value", name)).strip().lower()


def protrain_optimizer_wrapper(
    wrapped: WrappedModel,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    optimizer_name: str | None = None,
    huge_param_threshold_bytes: int = _DEFAULT_HUGE_PARAM_THRESHOLD_BYTES,
) -> torch.optim.Optimizer:
    """Rebuild the GPU/CPU FusedAdam adapters at user-specified hyperparams."""
    chunk_manager = cast("ChunkManager", wrapped.chunk_manager)
    layout = chunk_manager.layout
    persistent_ids = set(chunk_manager._persistent_ids)

    # Membership-test against _persistent_ids (non-block pinning makes the set non-contiguous).
    # Resolve params via _params_by_id (pre-block-wrap) since named_parameters acquires a .block. infix.
    persistent_params: list["nn.Parameter"] = []
    cpu_params_per_chunk: dict[ChunkId, list["nn.Parameter"]] = {}

    for cid, chunk_param_ids in enumerate(layout.chunks):
        # Fail fast on unresolvable pids; silent drop yielded partial freezing.
        missing_ids = [
            pid for pid in chunk_param_ids if pid not in chunk_manager._params_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"chunk cid={cid} references param ids {missing_ids} that are "
                "not registered in ChunkManager._params_by_id; cannot build "
                "per-chunk optimizer (would silently skip these params). "
                "Known pids: "
                f"{sorted(chunk_manager._params_by_id.keys())[:8]}"
                f"{'...' if len(chunk_manager._params_by_id) > 8 else ''}"
            )
        chunk_params = [chunk_manager._params_by_id[pid] for pid in chunk_param_ids]
        if cid in persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[ChunkId(cid)] = chunk_params

    # bnb 8-bit Adam is CUDA-only; non-persistent CPU shards keep the 32-bit DeepSpeedCPUAdam path.
    normalized_optim_name = _normalize_optimizer_name(optimizer_name)
    use_bnb_8bit = normalized_optim_name in _BNB_8BIT_OPTIMIZERS
    use_paged_8bit = normalized_optim_name in _BNB_8BIT_PAGED_OPTIMIZERS

    # Round-robin partition the persistent set across ranks so each rank only owns 1/W of state.
    persistent_world_size = int(getattr(chunk_manager, "world_size", 1) or 1)
    persistent_rank = int(getattr(chunk_manager, "rank", 0) or 0)
    persistent_params_full_all: list["nn.Parameter"] = list(persistent_params)

    # Within-param shard fallback for huge persistent params (v51).
    # Round-robin pins each whole nn.Parameter to one rank; for a single huge
    # param (e.g., Llama-3-70B lm_head ~2.6 GB optim state) that defeats the
    # memory-balance goal. Slice such params dim-0 into world_size shards.
    persistent_huge_originals: list["nn.Parameter"] = []
    persistent_huge_shards: list["nn.Parameter"] = []
    if persistent_world_size > 1:
        # fp32 master state estimate per param: 4 bytes/elem * Adam carries m+v (2x).
        # We compare against raw element-bytes (numel * 4) to keep the cfg
        # threshold name "...bytes" intuitive.
        small_params: list["nn.Parameter"] = []
        huge_params: list["nn.Parameter"] = []
        for p in persistent_params_full_all:
            if int(p.numel()) * 4 > int(huge_param_threshold_bytes):
                if int(p.shape[0]) % persistent_world_size != 0:
                    raise RuntimeError(
                        "protrain: persistent param of shape "
                        f"{tuple(p.shape)} exceeds huge-param threshold "
                        f"({huge_param_threshold_bytes} bytes) but dim-0 "
                        f"size {int(p.shape[0])} is not divisible by "
                        f"world_size={persistent_world_size}. Pad-and-mask "
                        "fallback is not implemented; either reduce "
                        "protrain_persistent_huge_param_threshold_bytes to "
                        "keep this param on the round-robin path, or use a "
                        f"world_size that divides {int(p.shape[0])}."
                    )
                huge_params.append(p)
            else:
                small_params.append(p)
        from torch import nn as _nn  # local import keeps top-only-TYPE_CHECKING clean

        for p in huge_params:
            shard_size = int(p.shape[0]) // persistent_world_size
            shard_view = p.data.narrow(0, persistent_rank * shard_size, shard_size)
            shard_param = _nn.Parameter(shard_view, requires_grad=p.requires_grad)
            persistent_huge_originals.append(p)
            persistent_huge_shards.append(shard_param)

        persistent_params_full = small_params
        owned_persistent_params: list["nn.Parameter"] = small_params[
            persistent_rank::persistent_world_size
        ]
        # Inner optim owns small-owned shards PLUS the rank's huge-param shard views.
        owned_persistent_params = list(owned_persistent_params) + list(
            persistent_huge_shards
        )
    else:
        persistent_params_full = persistent_params_full_all
        owned_persistent_params = persistent_params_full_all

    persistent_owner_rank: list[int] = [
        i % persistent_world_size for i in range(len(persistent_params_full))
    ]

    gpu_optim: GpuFusedAdamAdapter | GpuAdamW8bitAdapter | None = None
    cpu_optim: CpuFusedAdamAdapter | None = None
    if owned_persistent_params:
        if use_bnb_8bit:
            LOG.info(
                "protrain_optimizer_wrapper: routing %d/%d persistent params "
                "through bnb %s (optimizer_name=%s, rank=%d, world=%d)",
                len(owned_persistent_params),
                len(persistent_params_full),
                "PagedAdamW8bit" if use_paged_8bit else "AdamW8bit",
                optimizer_name,
                persistent_rank,
                persistent_world_size,
            )
            gpu_optim = GpuAdamW8bitAdapter(
                params=owned_persistent_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                paged=use_paged_8bit,
            )
        else:
            gpu_optim = GpuFusedAdamAdapter(
                params=owned_persistent_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )

    # M7: for sharded non-persistent chunks the CPU Adam updates each
    # :class:`_DtypeRegion`'s flat shard_param (one per rank slice per
    # dtype region) rather than the user-facing per-param list.
    # Homogeneous-dtype chunks have exactly one region and behave
    # identically to the pre-followup path; mixed-dtype chunks expose
    # one shard_param per region.
    cpu_params_per_chunk_for_optim: dict[ChunkId, list["nn.Parameter"]] = {}
    for cid, chunk_params in cpu_params_per_chunk.items():
        shard_state = chunk_manager._chunk_shards.get(cid)
        if shard_state is not None and shard_state.regions:
            cpu_params_per_chunk_for_optim[cid] = [
                r.shard_param for r in shard_state.regions
            ]
        else:
            cpu_params_per_chunk_for_optim[cid] = chunk_params

    if use_bnb_8bit and any(
        params for params in cpu_params_per_chunk_for_optim.values()
    ):
        # bnb 8-bit Adam requires CUDA tensors; non-persistent chunks
        # live on CPU. We keep the
        # 32-bit CpuFusedAdamAdapter on those chunks so training stays
        # correct (and the user still gets the persistent-chunk 8-bit
        # win from above). Surface this once, loudly, so users
        # configuring `adamw_8bit` aren't surprised by the partial
        # adoption.
        n_cpu_chunks = sum(
            1 for params in cpu_params_per_chunk_for_optim.values() if params
        )
        LOG.warning(
            "protrain_optimizer_wrapper: optimizer_name=%s requested 8-bit "
            "AdamW, but %d non-persistent chunk(s) live on CPU and bnb's "
            "8-bit Adam kernels are CUDA-only. Those chunks will keep "
            "using 32-bit DeepSpeedCPUAdam (still correct, but the "
            "optimizer-state memory win applies only to the persistent "
            "set). To get end-to-end 8-bit, configure ProTrain to force "
            "all chunks persistent (Mode A): set "
            "``protrain_auto_mode: false`` AND "
            "``protrain_force_all_persistent: true`` together — "
            "``protrain_force_all_persistent`` is ignored while "
            "``protrain_auto_mode`` is on (the auto-mode selector picks "
            "the mode itself based on capacity), so disabling auto-mode "
            "first is required for the Mode-A override to take effect.",
            optimizer_name,
            n_cpu_chunks,
        )

    if any(params for params in cpu_params_per_chunk_for_optim.values()):
        try:
            cpu_optim = CpuFusedAdamAdapter(
                params_per_chunk=cpu_params_per_chunk_for_optim,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        except Exception as err:
            # Only ImportError + CUDAMismatchException are caught; others propagate.
            is_cuda_mismatch = type(err).__name__ == "CUDAMismatchException"
            if not isinstance(err, ImportError) and not is_cuda_mismatch:
                raise
            # Stringify err before logging to avoid traceback->frame-locals leak in test log capture.
            err_kind = type(err).__name__
            err_str = str(err)
            base_msg = (
                "protrain_optimizer_wrapper: CPU FusedAdam unavailable "
                "(%s: %s). Non-persistent chunks will NOT receive "
                "optimizer steps — only persistent chunks (the GPU "
                "optimizer) update. Training is incorrect in this "
                "state for any model whose non-persistent params "
                "matter for convergence."
            )
            if is_cuda_mismatch:
                LOG.error(
                    base_msg + " Detected DeepSpeed CUDAMismatchException — "
                    "system CUDA does not match torch's CUDA wheel. "
                    "Workaround: set env DS_SKIP_CUDA_CHECK=1 (CPU Adam "
                    "JIT-compiles correctly despite the mismatch on "
                    "most rigs).",
                    err_kind,
                    err_str,
                )
            else:
                LOG.error(
                    base_msg + " Install DeepSpeed (or fix its dependencies) to "
                    "enable async CPU Adam.",
                    err_kind,
                    err_str,
                )
            raise RuntimeError(
                "CpuFusedAdamAdapter is required whenever ProTrain has "
                "non-persistent chunks (cpu_params_per_chunk_for_optim "
                "is non-empty); without it those offloaded params receive "
                "computed gradients but never an optimizer step, silently "
                "corrupting training. Fix the DeepSpeed install (e.g., set "
                "DS_SKIP_CUDA_CHECK=1 if this is a CUDA-toolkit / "
                "torch-wheel mismatch) or switch to an all-persistent "
                "config so no CPU optimizer is needed."
            ) from err

    # Preserve HF Trainer's bias/norm no-decay split per inner optim's param_groups.
    no_decay_param_ids = _collect_no_decay_param_ids(wrapped.module)
    if no_decay_param_ids:
        if gpu_optim is not None:
            _split_optim_param_groups(gpu_optim.underlying, no_decay_param_ids)
        if cpu_optim is not None:
            sharded_no_decay_ids = _collect_sharded_no_decay_shard_param_ids(
                chunk_manager,
                cpu_params_per_chunk,
                no_decay_param_ids,
            )
            # Union covers replicated (orig param ids) and sharded (shard_param ids); disjoint per inner.
            cpu_no_decay_ids = no_decay_param_ids | sharded_no_decay_ids
            inner_optims = getattr(cpu_optim, "_optims", {}) or {}
            for inner in inner_optims.values():
                _split_optim_param_groups(inner, cpu_no_decay_ids)

    # Shutdown-before-swap: previous CpuFusedAdamAdapter owns thread pool + DeepSpeed C state.
    _old_cpu_optim = getattr(chunk_manager, "cpu_optim", None)
    if _old_cpu_optim is not None and _old_cpu_optim is not cpu_optim:
        _old_cpu_optim.shutdown()
    chunk_manager.cpu_optim = cpu_optim
    chunk_manager.gpu_optim = cast("GpuFusedAdamAdapter | None", gpu_optim)

    # Build the flat param list for the Optimizer base class.
    all_params: list["nn.Parameter"] = list(persistent_params)
    for params in cpu_params_per_chunk.values():
        all_params.extend(params)
    # Dedupe while preserving order — shared weights may appear twice.
    seen: set[int] = set()
    unique_params: list["nn.Parameter"] = []
    for p in all_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique_params.append(p)

    defaults: dict[str, Any] = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return _ProTrainOptimizer(
        gpu_optim=gpu_optim,
        cpu_optim=cpu_optim,
        params=unique_params,
        defaults=defaults,
        chunk_manager=chunk_manager,
        persistent_params_full=persistent_params_full,
        persistent_owner_rank=persistent_owner_rank,
        persistent_world_size=persistent_world_size,
        persistent_huge_originals=persistent_huge_originals,
        persistent_huge_shards=persistent_huge_shards,
    )


__all__ = ["protrain_optimizer_wrapper"]

"""Phase-2 selective checkpointing: CPU-offload the SAC-saved tensors.

Replaces torch's SAC caching/cached dispatch modes with variants that move
MUST_SAVE outputs to pinned CPU memory on a side stream during forward and
stream them back during backward with one-region-lookahead prefetch. Saved
attention outputs then cost ~zero GPU memory.

Mutation caveat: an offloaded tensor is a snapshot at pack time; in-place
mutation of the original after the save cannot be detected (torch's version
guard only covers non-offloaded leaves). The known mutating case — PEFT's
in-place adapter add on matmul outputs — is rejected at config validation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import (
    SAC_IGNORED_OPS,
    CheckpointPolicy,
    SelectiveCheckpointContext,
    _policy_from_bool,
    _VersionWrapper,
)

try:
    # torch <= 2.12
    from torch.utils.checkpoint import _maybe_detach
except ImportError:
    # torch >= 2.13 replaced _maybe_detach with _detach_helper, dropping the
    # float/alias-info gate to detach any tensor unconditionally
    from torch.utils.checkpoint import _detach_helper

    def _maybe_detach(x, any_ret_has_alias_info):
        del any_ret_has_alias_info
        return _detach_helper(x)


from axolotl.monkeypatch.selective_checkpointing import (
    SacPolicyState,
    build_sac_policy,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_SAVE_POLICIES = (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)

_BufferKey = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.layout]


@dataclass
class _OffloadRef:
    region_id: int
    device: torch.device
    size: torch.Size
    stride: tuple[int, ...]
    storage_offset: int
    buffer_key: _BufferKey
    cpu_tensor: torch.Tensor | None = None
    gpu_tensor: torch.Tensor | None = None
    restore_event: torch.cuda.Event | None = None


@dataclass
class SacOffloadStats:
    offloaded_tensors: int = 0
    offloaded_bytes: int = 0
    restored_tensors: int = 0
    logged: bool = field(default=False, repr=False)


class SacOffloadEngine:
    """Pinned-pool, side-stream D2H/H2D engine for SAC-saved tensors."""

    def __init__(
        self,
        min_offload_bytes: int = 1 << 20,
        max_fwd_stash_size: int = 2,
        max_cpu_buffer_pool_size: int = 128,
    ) -> None:
        self.min_offload_bytes = min_offload_bytes
        self.max_fwd_stash_size = max_fwd_stash_size
        self.max_cpu_buffer_pool_size = max_cpu_buffer_pool_size
        self.stats = SacOffloadStats()

        self.s0 = torch.cuda.current_stream() if torch.cuda.is_available() else None
        self.s1 = torch.cuda.Stream() if torch.cuda.is_available() else None

        self._pack_seq = 0
        self._fwd_stash: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._regions: dict[int, list[_OffloadRef]] = defaultdict(list)
        self._cpu_pool: dict[_BufferKey, list[torch.Tensor]] = {}
        self._cpu_pool_size = 0
        self._pending_cpu_buffers: list[
            tuple[_BufferKey, torch.Tensor, torch.cuda.Event]
        ] = []

    @property
    def compute_stream(self):
        return torch.cuda.current_stream()

    def should_offload(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.device.type == "cuda"
            and tensor.element_size() * tensor.nelement() >= self.min_offload_bytes
            and not isinstance(tensor, torch.nn.Parameter)
        )

    def _buffer_key(self, tensor: torch.Tensor) -> _BufferKey:
        return (
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.layout,
        )

    def _pool_cpu_buffer(self, key: _BufferKey, tensor: torch.Tensor) -> None:
        if self._cpu_pool_size >= self.max_cpu_buffer_pool_size:
            return
        self._cpu_pool.setdefault(key, []).append(tensor)
        self._cpu_pool_size += 1

    def _reap_pending_cpu_buffers(self) -> None:
        pending = self._pending_cpu_buffers
        self._pending_cpu_buffers = []
        for key, tensor, event in pending:
            if event.query():
                self._pool_cpu_buffer(key, tensor)
            else:
                self._pending_cpu_buffers.append((key, tensor, event))

    def _empty_pinned_like(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, _BufferKey]:
        self._reap_pending_cpu_buffers()
        key = self._buffer_key(tensor)
        pool = self._cpu_pool.get(key)
        if pool:
            self._cpu_pool_size -= 1
            return pool.pop(), key
        return (
            torch.empty_strided(
                tuple(tensor.size()),
                tuple(tensor.stride()),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
                pin_memory=True,
            ),
            key,
        )

    def _reap_fwd_stash(self, new_seq: int) -> None:
        compute = self.compute_stream
        for seq in list(self._fwd_stash):
            if seq > new_seq - self.max_fwd_stash_size:
                continue
            _, event = self._fwd_stash.pop(seq)
            compute.wait_event(event)

    def pack(self, tensor: torch.Tensor, region_id: int) -> _OffloadRef:
        self._pack_seq += 1
        self._reap_fwd_stash(self._pack_seq)

        compute = self.compute_stream
        self.s1.wait_stream(compute)
        with torch.cuda.stream(self.s1):
            cpu_tensor, key = self._empty_pinned_like(tensor)
            cpu_tensor.copy_(tensor, non_blocking=True)
        event = self.s1.record_event()
        self._fwd_stash[self._pack_seq] = (tensor, event)

        ref = _OffloadRef(
            region_id=region_id,
            device=tensor.device,
            size=tensor.size(),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            buffer_key=key,
            cpu_tensor=cpu_tensor,
        )
        self._regions[region_id].append(ref)
        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += tensor.element_size() * tensor.nelement()
        return ref

    def _start_restore(self, ref: _OffloadRef) -> None:
        if ref.gpu_tensor is not None or ref.cpu_tensor is None:
            return
        with torch.cuda.stream(self.s1):
            gpu_tensor = ref.cpu_tensor.to(ref.device, non_blocking=True)
            if (
                gpu_tensor.storage_offset() != ref.storage_offset
                or gpu_tensor.stride() != ref.stride
            ):
                gpu_tensor = torch.as_strided(
                    gpu_tensor, ref.size, ref.stride, ref.storage_offset
                )
        ref.gpu_tensor = gpu_tensor
        ref.restore_event = self.s1.record_event()

    def prefetch_region(self, region_id: int) -> None:
        for ref in self._regions.get(region_id, []):
            self._start_restore(ref)

    def restore(self, ref: _OffloadRef) -> torch.Tensor:
        if ref.gpu_tensor is None:
            self._start_restore(ref)
        compute = self.compute_stream
        compute.wait_event(ref.restore_event)
        gpu_tensor = ref.gpu_tensor
        gpu_tensor.record_stream(compute)

        self._pending_cpu_buffers.append(
            (ref.buffer_key, ref.cpu_tensor, ref.restore_event)
        )
        ref.cpu_tensor = None
        ref.gpu_tensor = None

        region = self._regions.get(ref.region_id)
        if region is not None:
            try:
                region.remove(ref)
            except ValueError:
                pass
            if not region:
                self._regions.pop(ref.region_id, None)

        self.stats.restored_tensors += 1
        return gpu_tensor

    def log_once(self) -> None:
        if self.stats.logged or not self.stats.offloaded_tensors:
            return
        self.stats.logged = True
        LOG.info(
            "selective_checkpointing offload: "
            f"{self.stats.offloaded_tensors} tensors "
            f"({self.stats.offloaded_bytes / 2**30:.2f} GiB) offloaded to CPU "
            "in the first forward"
        )


class _OffloadCachingMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage, engine, region_id):
        self.policy_fn = policy_fn
        self.storage = storage
        self.engine = engine
        self.region_id = region_id

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        out = func(*args, **kwargs)

        if isinstance(func, torch._ops.HigherOrderOperator):
            any_ret_has_alias_info = False
        else:
            any_ret_has_alias_info = any(
                ret.alias_info is not None for ret in func._schema.returns
            )

        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        if policy in _SAVE_POLICIES:

            def pack_leaf(leaf):
                detached = _maybe_detach(leaf, any_ret_has_alias_info)
                if torch.is_tensor(detached) and self.engine.should_offload(detached):
                    return self.engine.pack(detached, self.region_id)
                return _VersionWrapper(detached)

            self.storage[func].append(tree_map(pack_leaf, out))
        return out


class _OffloadCachedMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage, engine, region_id):
        self.policy_fn = policy_fn
        self.storage = storage
        self.engine = engine
        self.region_id = region_id

    def __enter__(self):
        # recompute of this region is starting: bring its tensors back and
        # prefetch the next region backward will need
        self.engine.prefetch_region(self.region_id)
        if self.region_id > 0:
            self.engine.prefetch_region(self.region_id - 1)
        self.engine.log_once()
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        if policy in _SAVE_POLICIES:
            entries = self.storage.get(func)
            if not entries:
                raise RuntimeError(
                    f"selective_checkpointing offload: {func} encountered during "
                    "recompute but nothing was saved for it (or an extra backward "
                    "was attempted)"
                )

            def unpack_leaf(leaf):
                if isinstance(leaf, _OffloadRef):
                    return self.engine.restore(leaf)
                if isinstance(leaf, _VersionWrapper):
                    return leaf.get_val(False)
                return leaf

            return tree_map(unpack_leaf, entries.pop(0))
        return func(*args, **kwargs)


def build_sac_offload_context_fn(
    save: list[str] | None = None,
    save_sliding_window: bool = False,
    state: SacPolicyState | None = None,
    engine: SacOffloadEngine | None = None,
    recompute_layer_types: list[str] | None = None,
) -> Callable:
    """Return a ``context_fn`` whose MUST_SAVE tensors are offloaded to CPU."""
    state = state or SacPolicyState()
    engine = engine or SacOffloadEngine()
    policy_fn = build_sac_policy(
        save, state, save_sliding_window, recompute_layer_types
    )

    def context_fn():
        region_id = state.regions_seen
        state.regions_seen += 1
        storage: dict[Any, list[Any]] = defaultdict(list)
        return (
            _OffloadCachingMode(policy_fn, storage, engine, region_id),
            _OffloadCachedMode(policy_fn, storage, engine, region_id),
        )

    return context_fn

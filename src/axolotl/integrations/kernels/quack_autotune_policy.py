# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Coarsen quack's autotune key for grouped GEMMs, or skip quack's sweep altogether.

quack keys its winners on every tensor's exact shape, so each new routed row count of a grouped
(``cu_seqlens_m``/``cu_seqlens_k``) GEMM starts a full sweep. ``quack_autotune_key_buckets: pow2``
rounds the variable-length dims of the key up to a power of two, so one sweep serves a bucket. A
winner reused from a bucket goes through quack's validity pruning for the actual call first; a
winner it rejects sends that call to a sweep of its own exact key. ``shape`` drops those dims from
the key entirely, so one sweep per kernel and weight shape serves every token count.

A winner seeded from the cache file is re-tuned at its first lookup when the candidates this run
sweeps (after ``autotune_profiles``) differ from the ones it was measured against, or, for a
bucket, when it is not among this call's candidates.

``quack_autotune: false`` never benchmarks: each key gets one config from a fixed rule (by kernel,
output width N and reduction size K), measured on B200, or the first valid candidate when the rule's
config is not one (other GPUs).
"""

from typing import Any

import torch

from axolotl.integrations.kernels.autotune_cache import (
    _kernel_name,
    _note_quack_calls,
    _TimingCache,
    quack_candidates_sha256,
)
from axolotl.integrations.kernels.autotune_profiles import (
    _quack_fields,
    noop_quack_gpu_warmup,
)
from axolotl.integrations.kernels.sonicmoe_autotune import sonicmoe_autotuners
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

KEY_BUCKETS = ("pow2", "shape", "none")

_VARLEN_M_ROWS = (
    "A",
    "out",
    "C",
    "preact_out",
    "postact_out",
    "PreAct",
    "dx_out",
    "colvec_scale",
    "A_idx",
)
_NOTED_ATTRS = (
    "_axolotl_call_args",
    "_axolotl_profile_hit",
    "_axolotl_skip_bench",
    "_axolotl_candidate_count",
    "_axolotl_candidate_sha256",
)
_CAPTURING: dict[type, type] = {}


def _pow2_count(n: int) -> int:
    return n if n <= 1 else 1 << (n - 1).bit_length()


def _any_count(n: int) -> int:
    return -1


def _shape_slot_matches(slot: Any, shape: torch.Size) -> bool:
    if isinstance(slot, str):
        return slot == str(shape)
    if isinstance(slot, (tuple, list)):
        return tuple(slot) == tuple(shape)
    return False


def _format_shape(slot: Any, dims: list[int]) -> Any:
    if isinstance(slot, str):
        return str(torch.Size(dims))
    return type(slot)(dims)


def _bucketed_dims(
    name: str, shape: torch.Size, varlen_m: bool, varlen_k: bool, mode: str = "pow2"
):
    dims = list(shape)
    count = _any_count if mode == "shape" else _pow2_count
    if varlen_m and name in _VARLEN_M_ROWS and len(dims) in (1, 2):
        dims[0] = count(dims[0])
    if varlen_k:
        if name == "A" and len(dims) == 2:
            dims[1] = count(dims[1])
        elif name == "B" and len(dims) == 2:
            dims[0] = count(dims[0])
        elif name == "A_idx" and len(dims) == 1:
            dims[0] = count(dims[0])
    return dims


def bucket_key(
    tuner: Any, raw: tuple, args: dict[str, Any], marker: str, mode: str = "pow2"
) -> tuple | None:
    """``raw`` with its variable-length dims bucketed, or None for a call that is not grouped.

    Mirrors how quack's ``Autotuner.__call__`` builds the key: the declared key args present in
    the call, then a group of entries (shape first) per tensor argument in call order. A key whose
    layout does not line up with that is left alone.
    """
    names = getattr(tuner, "arg_names", None) or []
    call = {k: v for k, v in args.items() if k in names}
    varlen_m = isinstance(call.get("cu_seqlens_m"), torch.Tensor)
    varlen_k = isinstance(call.get("cu_seqlens_k"), torch.Tensor)
    if not (varlen_m or varlen_k):
        return None
    if any(isinstance(call.get(n), torch.Tensor) for n in ("SFA", "SFB")):
        return None
    tensors = [(n, v) for n, v in call.items() if isinstance(v, torch.Tensor)]
    scalars = sum(1 for k in getattr(tuner, "keys", None) or () if k in call)
    rest = len(raw) - scalars
    if not tensors or rest <= 0 or rest % len(tensors):
        return None
    per = rest // len(tensors)
    key = list(raw)
    for i, (name, tensor) in enumerate(tensors):
        slot = scalars + i * per
        if not _shape_slot_matches(key[slot], tensor.shape):
            return None
        dims = _bucketed_dims(name, tensor.shape, varlen_m, varlen_k, mode)
        if dims != list(tensor.shape):
            key[slot] = _format_shape(key[slot], dims)
    return tuple(key) + (marker,)


def _rule(tile_m, tile_n, cluster_m, cluster_n, persistent, gather) -> dict[str, Any]:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "is_dynamic_persistent": persistent,
        "use_tma_gather": gather,
        "pingpong": False,
        "swap_ab": False,
    }


def heuristic_fields(kernel: str, args: dict[str, Any]) -> dict[str, Any]:
    """The ``GemmConfig`` fields the no-sweep mode picks for a call."""
    gather = args.get("A_idx") is not None
    if kernel == "gemm_gated_tuned":
        return _rule(256, 512, 2, 1, True, gather)
    if kernel == "gemm_dgated_tuned":
        return _rule(256, 256, 2, 1, True, gather)
    if kernel == "gemm_tuned":
        out, b = args.get("out"), args.get("B")
        n = out.shape[-1] if isinstance(out, torch.Tensor) else None
        if n is None and isinstance(b, torch.Tensor):
            n = b.shape[-1]
        k = b.shape[-2] if isinstance(b, torch.Tensor) and b.ndim >= 2 else None
        if n is not None and n <= 128:
            varlen_k = args.get("cu_seqlens_k") is not None
            return _rule(128, 64, 2 if varlen_k else 1, 1, True, gather)
        if k is not None and k <= 128:
            return _rule(128, 256, 1, 2, False, gather)
    return _rule(256, 256, 2, 1, True, gather)


def choose_heuristic_config(tuner: Any, candidates: list, args: dict[str, Any]) -> Any:
    kernel = _kernel_name(tuner)
    wanted = heuristic_fields(kernel, args)
    for config in candidates:
        have = _quack_fields(config)
        if all(have.get(k) == v for k, v in wanted.items()):
            return config
    warned = tuner.__dict__.setdefault("_axolotl_heuristic_warned", set())
    if kernel not in warned:
        warned.add(kernel)
        LOG.info(
            "quack_autotune: false: %s has no valid candidate with %s for %s; using the "
            "first valid one (%s)",
            kernel,
            wanted,
            getattr(tuner, "_axolotl_call_args", None),
            _quack_fields(candidates[0]),
        )
    return candidates[0]


class QuackKeyPolicy:
    """Decides, per quack call, which cache key stores the winner and whether to sweep it.

    ``_TimingCache`` consults it from inside ``Autotuner.__call__`` only: the cache membership test
    resolves the raw key to its storage key (``lookup``) and the store and lookup that follow in
    the same call reuse that resolution (``storage_for``).
    """

    def __init__(self, tuner: Any, buckets: str = "pow2", tune: bool = True):
        self.tuner = tuner
        self.buckets = buckets
        self.tune = tune
        self._current: tuple[tuple, tuple] | None = None

    @property
    def marker(self) -> str:
        return f"axolotl_bucket={self.buckets}"

    def in_call(self) -> bool:
        return bool(getattr(self.tuner, "_axolotl_in_call", False))

    def end_call(self) -> None:
        self._current = None

    def _args(self) -> dict[str, Any]:
        kwargs = getattr(self.tuner, "_axolotl_call_kwargs", None) or {}
        return {**(getattr(self.tuner, "nargs", None) or {}), **kwargs}

    def storage_for(self, key: tuple) -> tuple:
        current = self._current
        if current is not None and self.in_call() and key == current[0]:
            return current[1]
        return key

    def stored(self, cache: _TimingCache, key: tuple) -> None:
        current = self._current
        if current is not None and self.in_call() and key == current[1]:
            cache.origins[key] = current[0]

    def _candidates(self) -> list | None:
        """This call's candidates (quack's pruning plus any profiles), without side effects."""
        tuner = self.tuner
        kwargs = dict(getattr(tuner, "_axolotl_call_kwargs", None) or {})
        saved = {name: getattr(tuner, name, None) for name in _NOTED_ATTRS}
        try:
            return list(tuner.prune_configs(kwargs))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.debug("quack bucket check for %s failed: %s", _kernel_name(tuner), exc)
            return None
        finally:
            for name, value in saved.items():
                setattr(tuner, name, value)

    def _seed_is_stale(
        self, cache: _TimingCache, key: tuple, candidates: list | None
    ) -> bool:
        """Whether the seeded ``key`` was swept over other candidates than this call's.

        Checked once per seeded key; a stale one stops counting as seeded so it is re-tuned.
        """
        want = cache.unverified.pop(key, None)
        if want is None or candidates is None:
            return False
        if quack_candidates_sha256(candidates) == want:
            return False
        cache.seeded.discard(key)
        LOG.info(
            "quack %s: the seeded winner for %s was measured over other candidates than "
            "this run's %d (autotune_profiles changed); re-tuning it",
            _kernel_name(self.tuner),
            getattr(self.tuner, "_axolotl_call_args", None) or key,
            len(candidates),
        )
        return True

    def _resolve(self, cache: _TimingCache, raw: tuple) -> tuple[tuple, bool]:
        if dict.__contains__(cache, raw):
            if raw in cache.unverified and self._seed_is_stale(
                cache, raw, self._candidates()
            ):
                return raw, False
            return raw, True
        if self.buckets == "none":
            return raw, False
        bucket = bucket_key(self.tuner, raw, self._args(), self.marker, self.buckets)
        if bucket is None:
            return raw, False
        if not dict.__contains__(cache, bucket):
            return bucket, False
        if cache.origins.get(bucket) == raw:
            return bucket, True
        winner = dict.__getitem__(cache, bucket)
        known = cache.aliases.get(raw)
        if known is not None and known[0] == bucket and known[1] is winner:
            return bucket, True
        candidates = self._candidates()
        if bucket in cache.unverified and self._seed_is_stale(
            cache, bucket, candidates
        ):
            return bucket, False
        if candidates is not None and any(c == winner for c in candidates):
            cache.aliases[raw] = (bucket, winner)
            return bucket, True
        call = getattr(self.tuner, "_axolotl_call_args", None) or raw
        if bucket in cache.seeded:
            cache.seeded.discard(bucket)
            LOG.info(
                "quack %s: the seeded %s bucket winner is not among this run's candidates "
                "for %s; re-tuning the bucket",
                _kernel_name(self.tuner),
                self.buckets,
                call,
            )
            return bucket, False
        cache.exact_fallbacks.add(raw)
        LOG.info(
            "quack %s: the %s bucket winner tuned in this run is not among the candidates "
            "for %s; tuning its exact key",
            _kernel_name(self.tuner),
            self.buckets,
            call,
        )
        return raw, False

    def lookup(self, cache: _TimingCache, raw: tuple) -> bool:
        storage, found = self._resolve(cache, raw)
        self._current = (raw, storage)
        if found or self.tune:
            return found
        tuner = self.tuner
        kwargs = dict(getattr(tuner, "_axolotl_call_kwargs", None) or {})
        candidates = list(tuner.prune_configs(kwargs))
        if not candidates:
            return False
        profiled = bool(getattr(tuner, "_axolotl_profile_hit", False))
        cache[raw] = choose_heuristic_config(tuner, candidates, self._args())
        if not profiled:
            cache.heuristic.add(storage)
        return True


def _capture_call_kwargs(tuner: Any) -> None:
    """Swap in a subclass whose ``__call__`` keeps the call's kwargs on the autotuner.

    quack's key lists tensor shapes without names and its membership test sees only the key, so
    naming the key's tensors needs the keyword arguments, which quack does not keep.
    """
    cls = type(tuner)
    if getattr(cls, "_axolotl_captures_kwargs", False):
        return
    sub = _CAPTURING.get(cls)
    if sub is None:

        def __call__(self, *args, **kwargs):
            self._axolotl_call_kwargs = kwargs
            self._axolotl_in_call = True
            try:
                return cls.__call__(self, *args, **kwargs)
            finally:
                self._axolotl_in_call = False
                self._axolotl_call_kwargs = None
                policy = getattr(self, "_axolotl_key_policy", None)
                if policy is not None:
                    policy.end_call()

        sub = type(
            cls.__name__,
            (cls,),
            {
                "__call__": __call__,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "_axolotl_captures_kwargs": True,
            },
        )
        _CAPTURING[cls] = sub
    tuner.__class__ = sub


def _wrap_disk_cache(tuner: Any, policy: QuackKeyPolicy) -> None:
    orig = getattr(tuner, "check_disk_cache", None)
    if orig is None or getattr(tuner, "_axolotl_disk_key_wrapped", False):
        return

    def check_disk_cache(tuning_key, configs, bench_fn):
        return orig(policy.storage_for(tuning_key), configs, bench_fn)

    tuner.check_disk_cache = check_disk_cache
    tuner._axolotl_disk_key_wrapped = True


def install_quack_key_policy(buckets: str = "pow2", tune: bool = True) -> int:
    """Apply key bucketing and/or the no-sweep mode to every loaded quack autotuner.

    Idempotent; quack modules imported since the last call are picked up. Returns the number of
    autotuners newly covered.
    """
    if buckets not in KEY_BUCKETS:
        raise ValueError(f"quack_autotune_key_buckets must be one of {KEY_BUCKETS}")
    installed = 0
    names = []
    for _name, _module, tuner, is_quack in sonicmoe_autotuners():
        if not is_quack or not hasattr(tuner, "prune_configs"):
            continue
        noop_quack_gpu_warmup(tuner)
        policy = getattr(tuner, "_axolotl_key_policy", None)
        if policy is not None:
            policy.buckets, policy.tune = buckets, tune
            continue
        # exact keys need a policy only to check seeded winners' candidate fingerprints
        cache = tuner.cache
        if buckets == "none" and tune and not getattr(cache, "unverified", None):
            continue
        if not isinstance(tuner.cache, _TimingCache):
            tuner.cache = _TimingCache(tuner, tuner.cache)
        _note_quack_calls(tuner)
        _capture_call_kwargs(tuner)
        policy = QuackKeyPolicy(tuner, buckets, tune)
        tuner._axolotl_key_policy = policy
        _wrap_disk_cache(tuner, policy)
        installed += 1
        names.append(_kernel_name(tuner))
    if installed:
        LOG.info(
            "quack autotune (key buckets: %s, %s) on %d kernel(s): %s",
            buckets,
            "sweep" if tune else "rule-picked configs, no sweep",
            installed,
            ", ".join(sorted(set(names))),
        )
    return installed

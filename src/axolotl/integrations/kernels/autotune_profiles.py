# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Pin or narrow Triton autotune search spaces from the config.

A measured cache (``autotune_cache_path``) only helps for keys it has seen; a kernel whose
key carries the routed-row count still benchmarks every unseen bucket, one rank at a time
while its expert-parallel peers wait. A profile sidesteps the benchmark: Triton's
``Autotuner.run`` asks ``prune_configs`` for the candidates of an unseen key and caches
the only candidate without timing it, so a profile that resolves to one config never
benchmarks, and one that resolves to a few runs a short search.

```yaml
autotune_profiles:
  - kernel: _scatter2scatter_lora
    gpu: B200                      # optional substring of the GPU name
    match: { N: 5120, K: [2560, 5120] }   # optional, on the kernel's declared key args
    configs:
      - { BLOCK_M: 128, BLOCK_N: 64, BLOCK_K: 64, num_stages: 4, num_warps: 4 }
```
"""

import fnmatch
import json
import sys
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch

from axolotl.integrations.kernels.autotune_cache import (
    _autotuners,
    _kernel_name,
    note_call_args,
    note_candidates,
)
from axolotl.integrations.kernels.autotune_collector import _config_to_dict
from axolotl.integrations.kernels.sonicmoe_autotune import _plain, sonicmoe_autotuners
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_CONFIG_META_KEYS = ("num_warps", "num_stages", "num_ctas")


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(torch.cuda.current_device()).name


def _matches(match: dict[str, Any] | None, args: dict[str, Any]) -> bool:
    for name, wanted in (match or {}).items():
        if name not in args:
            return False
        allowed = wanted if isinstance(wanted, (list, tuple, set)) else (wanted,)
        if args[name] not in allowed:
            return False
    return True


_CONFIG_DEFAULTS = {"num_warps": 4, "num_stages": 3, "num_ctas": 1}


def _resolve_config(autotuner: Any, wanted: dict[str, Any]) -> Any:
    normalized = {
        **_CONFIG_DEFAULTS,
        **{k: v for k, v in wanted.items() if v is not None},
    }
    for config in getattr(autotuner, "configs", None) or []:
        have = {
            **_CONFIG_DEFAULTS,
            **{k: v for k, v in _config_to_dict(config).items() if v is not None},
        }
        if have == normalized:
            return config
    from triton import Config

    kwargs = {k: v for k, v in normalized.items() if k not in _CONFIG_META_KEYS}
    return Config(
        kwargs,
        num_warps=normalized.get("num_warps", 4),
        num_stages=normalized.get("num_stages", 3),
        num_ctas=normalized.get("num_ctas", 1),
    )


def _install(autotuner: Any, profiles: list[dict[str, Any]]) -> bool:
    """Wrap ``prune_configs`` so a matching profile decides the candidates for unseen keys."""
    if getattr(autotuner, "_axolotl_profiles", None) is not None:
        autotuner._axolotl_profiles = profiles
        return False
    orig_prune = autotuner.prune_configs
    autotuner._axolotl_profiles = profiles

    def prune_configs(kwargs):
        args = {**(getattr(autotuner, "nargs", None) or {}), **kwargs}
        for profile in autotuner._axolotl_profiles:
            if _matches(profile.get("match"), args):
                autotuner._axolotl_profile_hit = True
                return [_resolve_config(autotuner, c) for c in profile["configs"]]
        autotuner._axolotl_profile_hit = False
        return orig_prune(kwargs)

    autotuner.prune_configs = prune_configs
    return True


def _quack_fields(config: Any) -> dict[str, Any]:
    kwargs = getattr(config, "kwargs", None) or {}
    inner = kwargs.get("config")
    if is_dataclass(inner) and not isinstance(inner, type):
        return {f.name: _plain(getattr(inner, f.name)) for f in fields(inner)}
    return {k: _plain(v) for k, v in kwargs.items()}


def _resolve_quack_configs(autotuner: Any, wanted: dict[str, Any]) -> list:
    """Every candidate whose fields include ``wanted``; else one built from the first candidate.

    Building goes through the autotuner's own config classes, so a vendored quack (whose
    ``GemmConfig`` is a different class from the installed one) gets its own type.
    """
    missing = object()
    hits = [
        c
        for c in autotuner.configs
        if all(_quack_fields(c).get(k, missing) == v for k, v in wanted.items())
    ]
    if hits:
        return hits
    template = autotuner.configs[0]
    inner = template.kwargs.get("config")
    if is_dataclass(inner) and not isinstance(inner, type):
        try:
            built = {**template.kwargs, "config": replace(inner, **wanted)}
        except TypeError as exc:
            raise ValueError(
                f"autotune profile config {wanted} for {_kernel_name(autotuner)}: {exc}"
            ) from exc
    else:
        built = {**template.kwargs, **wanted}
    return [type(template)(**built)]


def _shape(value: Any) -> tuple | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(d) for d in shape)


def _same(value: Any, wanted: Any) -> bool:
    if value == wanted:
        return True
    try:
        value = _plain(value)
    except TypeError:
        pass
    return str(value) == str(wanted)


def _quack_matches(match: dict[str, Any] | None, args: dict[str, Any]) -> bool:
    """``arg``: a scalar argument (a list = any of); ``arg.shape``: a tensor's full shape
    (``null`` = any size); ``arg.dimN``: one dimension (negative N counts from the end; a list
    = any of). A scalar or dimension can also take a range such as ``{ge: 262144}``."""
    for name, wanted in (match or {}).items():
        arg, _, attr = name.partition(".")
        if arg not in args:
            return False
        value = args[arg]
        if attr == "shape":
            shape = _shape(value)
            if shape is None or len(shape) != len(wanted):
                return False
            if any(
                w is not None and w != d for w, d in zip(wanted, shape, strict=True)
            ):
                return False
            continue
        if attr:
            shape = _shape(value)
            dim = int(attr[3:])
            if shape is None or not -len(shape) <= dim < len(shape):
                return False
            value = shape[dim]
        if isinstance(wanted, dict):
            if not _in_range(value, wanted):
                return False
            continue
        allowed = wanted if isinstance(wanted, (list, tuple, set)) else (wanted,)
        if not any(_same(value, a) for a in allowed):
            return False
    return True


_RANGE_OPS = {
    "ge": lambda v, b: v >= b,
    "gt": lambda v, b: v > b,
    "le": lambda v, b: v <= b,
    "lt": lambda v, b: v < b,
}


def _in_range(value: Any, bounds: dict[str, Any]) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return all(_RANGE_OPS[op](value, bound) for op, bound in bounds.items())


def _validate_quack_match(profile: dict[str, Any]) -> None:
    for name, wanted in (profile.get("match") or {}).items():
        if isinstance(wanted, dict) and (not wanted or set(wanted) - set(_RANGE_OPS)):
            raise ValueError(
                f"quack autotune profile match {name!r}: a range takes ge, gt, le or lt"
            )
        _, _, attr = name.partition(".")
        if not attr or attr == "shape":
            continue
        try:
            int(attr[3:])
            ok = attr.startswith("dim")
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f"quack autotune profile match key {name!r}: use 'arg', "
                "'arg.shape' or 'arg.dimN'"
            )


def _alternatives(name: str, wanted: Any) -> list:
    if name == "cluster":
        pairs = list(wanted)
        if pairs and all(isinstance(v, int) for v in pairs):
            return [pairs]
        return [list(pair) for pair in pairs]
    return list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]


def _field_allowed(config_fields: dict[str, Any], name: str, wanted: Any) -> bool:
    if name == "cluster":
        value: Any = [config_fields.get("cluster_m"), config_fields.get("cluster_n")]
    else:
        value = config_fields.get(name)
    return value in _alternatives(name, wanted)


def config_allowed(config_fields: dict[str, Any], allow: dict[str, Any]) -> bool:
    """Whether a config's ``GemmConfig`` fields pass an ``allow`` filter.

    Each field lists its allowed values (``cluster`` takes ``[cluster_m, cluster_n]`` pairs);
    each ``where`` clause ``{if: {...}, then: {...}}`` requires its ``then`` fields only for
    configs whose fields match all of ``if``.
    """
    for name, wanted in allow.items():
        if name == "where":
            continue
        if not _field_allowed(config_fields, name, wanted):
            return False
    for clause in allow.get("where") or []:
        when = clause.get("if") or {}
        if all(_field_allowed(config_fields, n, w) for n, w in when.items()):
            if not all(
                _field_allowed(config_fields, n, w)
                for n, w in (clause.get("then") or {}).items()
            ):
                return False
    return True


def _validate_allow(autotuner: Any, profile: dict[str, Any]) -> None:
    allow = profile.get("allow")
    if allow is None:
        return
    known = set(_quack_fields(autotuner.configs[0])) | {"cluster"}
    names = [n for n in allow if n != "where"]
    for clause in allow.get("where") or []:
        if set(clause) - {"if", "then"} or not clause.get("then"):
            raise ValueError(
                f"autotune profile for {profile['kernel']}: each allow.where clause is "
                "{if: {field: values}, then: {field: values}}"
            )
        names += list(clause.get("if") or {}) + list(clause["then"])
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"autotune profile for {profile['kernel']}: unknown allow field(s) {unknown}; "
            f"known: {sorted(known)}"
        )


def _skip_gpu_warmup(*_args, **_kwargs) -> None:
    return None


def noop_quack_gpu_warmup(autotuner: Any) -> bool:
    """Drop the ~200 ms GPU warm-up matmul quack runs before every new key's sweep.

    Patched on the autotuner's defining module, whose ``benchmark`` closure looks it up as a
    global; a quack without ``_gpu_warmup`` is left alone.
    """
    module = sys.modules.get(getattr(type(autotuner), "__module__", ""))
    current = getattr(module, "_gpu_warmup", None)
    if current is None or current is _skip_gpu_warmup:
        return False
    module._gpu_warmup = _skip_gpu_warmup  # type: ignore[union-attr]
    return True


def _install_quack(autotuner: Any, profiles: list[dict[str, Any]]) -> bool:
    """Restrict a quack autotuner's candidates per call.

    ``configs`` replaces the candidate list (a single survivor is not benchmarked: quack
    benchmarks even one pruned candidate, so ``_bench`` is short-circuited for it). ``allow``
    filters whatever quack's own validity pruning (or ``configs``) left.
    """
    resolved = []
    for profile in profiles:
        _validate_quack_match(profile)
        _validate_allow(autotuner, profile)
        selected: list | None = None
        if profile.get("configs"):
            selected = []
            for wanted in profile["configs"]:
                for config in _resolve_quack_configs(autotuner, wanted):
                    if config not in selected:
                        selected.append(config)
        resolved.append((profile, selected))
    noop_quack_gpu_warmup(autotuner)
    first = getattr(autotuner, "_axolotl_quack_profiles", None) is None
    autotuner._axolotl_quack_profiles = resolved
    if not first:
        return False
    orig_prune = autotuner.prune_configs
    orig_bench = autotuner._bench
    autotuner._axolotl_skip_bench = False
    autotuner._axolotl_allow_warned = set()

    def _filter(profile, configs):
        allow = profile.get("allow")
        if not allow:
            return configs
        kept = [c for c in configs if config_allowed(_quack_fields(c), allow)]
        if kept:
            return kept
        call = json.dumps(getattr(autotuner, "_axolotl_call_args", None), default=str)
        if call not in autotuner._axolotl_allow_warned:
            autotuner._axolotl_allow_warned.add(call)
            LOG.warning(
                "autotune profile allow filter for %s removes every valid config for %s; "
                "searching the %d valid ones",
                profile["kernel"],
                call,
                len(configs),
            )
        return configs

    def _decide(kwargs):
        args = {**(getattr(autotuner, "nargs", None) or {}), **kwargs}
        for profile, selected in autotuner._axolotl_quack_profiles:
            if not _quack_matches(profile.get("match"), args):
                continue
            if selected is None:
                return _filter(profile, list(orig_prune(kwargs))), False
            configs = list(selected)
            prune = getattr(autotuner, "early_config_prune", None)
            if prune is not None:
                configs = list(prune(configs, autotuner.nargs, **kwargs))
            if not configs:
                LOG.warning(
                    "autotune profile for %s: no profiled config is valid for this call; "
                    "trying the next matching profile",
                    profile["kernel"],
                )
                continue
            return _filter(profile, configs), True
        return list(orig_prune(kwargs)), False

    def prune_configs(kwargs):
        note_call_args(autotuner, kwargs)
        configs, pinned = _decide(kwargs)
        autotuner._axolotl_profile_hit = pinned
        autotuner._axolotl_skip_bench = pinned and len(configs) == 1
        note_candidates(autotuner, configs)
        return configs

    def _bench(*args, config, **meta):
        if autotuner._axolotl_skip_bench:
            return [0.0, 0.0, 0.0]
        return orig_bench(*args, config=config, **meta)

    autotuner.prune_configs = prune_configs
    autotuner._bench = _bench
    return True


def _profiles_for(kname: str, modname: str, active: list, backend: str) -> list:
    return [
        p
        for p in active
        if p.get("backend", "triton") == backend
        and (
            fnmatch.fnmatchcase(kname, p["kernel"])
            or fnmatch.fnmatchcase(f"{modname}.{kname}", p["kernel"])
        )
    ]


def apply_autotune_profiles(profiles: list[dict[str, Any]], quiet: bool = False) -> int:
    """Attach ``profiles`` to every loaded autotuner they name. Returns the autotuner count.

    Safe to call repeatedly: kernels imported since the last call get their profiles, and
    only those are logged.
    """
    gpu = _gpu_name()
    active = []
    for profile in profiles:
        wanted_gpu = profile.get("gpu")
        if wanted_gpu and gpu and wanted_gpu.lower() not in gpu.lower():
            continue
        backend = profile.get("backend", "triton")
        if backend not in ("triton", "quack"):
            raise ValueError(
                f"autotune profile for {profile.get('kernel')!r}: backend must be "
                "'triton' or 'quack'"
            )
        if profile.get("allow") is not None and backend != "quack":
            raise ValueError(
                f"autotune profile for {profile.get('kernel')!r}: allow needs backend: quack"
            )
        if not profile.get("configs") and not profile.get("allow"):
            raise ValueError(
                f"autotune profile for {profile.get('kernel')!r} has no configs"
            )
        active.append(profile)
    if not active:
        return 0

    installed = 0
    new: list[str] = []
    targets = [(m, t, False) for m, _module, t in _autotuners()]
    targets += [(m, t, q) for m, _module, t, q in sonicmoe_autotuners()]
    for modname, autotuner, is_quack in targets:
        kname = _kernel_name(autotuner)
        mine = _profiles_for(kname, modname, active, "quack" if is_quack else "triton")
        if not mine:
            continue
        if (_install_quack if is_quack else _install)(autotuner, mine):
            new.append(kname)
        installed += 1
    if new:
        LOG.info(
            "Autotune profiles attached to %d kernel(s): %s",
            len(new),
            ", ".join(sorted(set(new))),
        )
    elif not installed and not quiet:
        LOG.warning(
            "autotune_profiles matched no loaded kernel (%s); kernels import lazily, "
            "profiles are applied again at train start and during the first steps",
            ", ".join(p["kernel"] for p in active),
        )
    return installed

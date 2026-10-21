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
from typing import Any

import torch

from axolotl.integrations.kernels.autotune_cache import _autotuners, _kernel_name
from axolotl.integrations.kernels.autotune_collector import _config_to_dict
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


def _install(autotuner: Any, profiles: list[dict[str, Any]]) -> None:
    """Wrap ``prune_configs`` so a matching profile decides the candidates for unseen keys."""
    if getattr(autotuner, "_axolotl_profiles", None) is not None:
        autotuner._axolotl_profiles = profiles
        return
    orig_prune = autotuner.prune_configs
    autotuner._axolotl_profiles = profiles

    def prune_configs(kwargs):
        args = {**(getattr(autotuner, "nargs", None) or {}), **kwargs}
        for profile in autotuner._axolotl_profiles:
            if _matches(profile.get("match"), args):
                return [_resolve_config(autotuner, c) for c in profile["configs"]]
        return orig_prune(kwargs)

    autotuner.prune_configs = prune_configs


def apply_autotune_profiles(profiles: list[dict[str, Any]]) -> int:
    """Attach ``profiles`` to every loaded autotuner they name. Returns the autotuner count."""
    gpu = _gpu_name()
    active = []
    for profile in profiles:
        wanted_gpu = profile.get("gpu")
        if wanted_gpu and gpu and wanted_gpu.lower() not in gpu.lower():
            continue
        if not profile.get("configs"):
            raise ValueError(
                f"autotune profile for {profile.get('kernel')!r} has no configs"
            )
        active.append(profile)
    if not active:
        return 0

    installed = 0
    for modname, _module, autotuner in _autotuners():
        kname = _kernel_name(autotuner)
        mine = [
            p
            for p in active
            if fnmatch.fnmatchcase(kname, p["kernel"])
            or fnmatch.fnmatchcase(f"{modname}.{kname}", p["kernel"])
        ]
        if mine:
            _install(autotuner, mine)
            installed += 1
    if installed:
        LOG.info(
            "Autotune profiles attached to %d kernel(s): %s",
            installed,
            ", ".join(sorted({p["kernel"] for p in active})),
        )
    else:
        LOG.warning(
            "autotune_profiles matched no loaded kernel (%s); kernels import lazily, "
            "profiles are applied again at train start",
            ", ".join(p["kernel"] for p in active),
        )
    return installed

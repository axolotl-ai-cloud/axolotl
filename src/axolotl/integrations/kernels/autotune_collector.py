"""Collect Triton autotune results from scattermoe-lora kernels.

This module reads the ``.cache`` attribute from Triton ``@triton.autotune``
decorated kernel objects and returns structured dicts describing the selected
configurations.  It has **no** telemetry dependency — callers decide what to
do with the data.
"""

import sys
from types import ModuleType
from typing import Any

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# (human-readable name, attribute on the lora_ops module)
_KERNEL_REGISTRY: list[tuple[str, str]] = [
    ("scatter2scatter_lora_fwd", "_scatter2scatter_lora"),
    ("scatter2scatter_lora_dX", "_scatter2scatter_lora_dX"),
    ("group_bwd_lora", "_group_bwd_lora"),
    ("group_bwd_lora_fused", "_group_bwd_lora_fused"),
]

# The autotune key declared on every kernel: key=["M_BUCKET", "N", "K"].
# M_BUCKET is the seqlen-bucketed M (see _bucket_m in lora_ops.py) so cache
# entries don't churn with every distinct M.
_KEY_NAMES: list[str] = ["M_BUCKET", "N", "K"]


def _parse_key_tuple(key_tuple: tuple) -> dict[str, Any]:
    """Turn the autotune cache key tuple into a labelled dict.

    Triton builds the cache key from the values of the declared ``key``
    args (``M_BUCKET``, ``N``, ``K``) followed by dtype signature elements.
    We label the first three and store the rest under ``_extra``.
    """
    result: dict[str, Any] = {}
    for i, name in enumerate(_KEY_NAMES):
        if i < len(key_tuple):
            result[name] = key_tuple[i]
    if len(key_tuple) > len(_KEY_NAMES):
        result["_extra"] = [str(v) for v in key_tuple[len(_KEY_NAMES) :]]
    return result


def _find_lora_ops_module() -> ModuleType | None:
    """Locate the *runtime* ``lora_ops`` module in ``sys.modules``.

    Normally there is a single canonical instance (the axolotl import path); a duplicate can
    appear if the same file is imported under a second module name (its kernel objects would
    carry separate ``.cache`` dicts). ``register_scattermoe_experts`` calls
    ``_ensure_single_lora_ops`` to collapse such duplicates, but this lookup stays defensive:
    it returns the first ``sys.modules`` entry whose name contains ``lora_ops`` and that
    exposes the ``_scatter2scatter_lora`` kernel.
    """
    for name, module in list(sys.modules.items()):
        if (
            module is not None
            and "lora_ops" in name
            and hasattr(module, "_scatter2scatter_lora")
        ):
            return module
    return None


# Substrings identifying the kernel-bearing modules to scan in sys.modules. Covers both
# the HF-`kernels` hash-suffixed scattermoe copies and the normally-imported dsv4 kernels.
_KERNEL_MODULE_HINTS: tuple[str, ...] = (
    "lora_ops",
    "scattermoe_lora.kernels",
    "libs.dsv4.attention",
    "libs.dsv4.attention_csa",
    "libs.dsv4.attention_gather",
    "libs.dsv4.rope",
    "libs.dsv4.mhc",
    "libs.dsv4.gated_pool",
    "libs.dsv4.indexer",
    ".dsv4.",
)


def _is_autotuner(obj: Any) -> bool:
    """A Triton ``Autotuner`` exposes a ``.cache`` dict and the wrapped ``.base_fn``/``.fn``."""
    return isinstance(getattr(obj, "cache", None), dict) and (
        getattr(obj, "base_fn", None) is not None
        or getattr(obj, "fn", None) is not None
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    out = dict(getattr(config, "kwargs", {}) or {})
    out["num_warps"] = getattr(config, "num_warps", None)
    out["num_stages"] = getattr(config, "num_stages", None)
    if getattr(config, "num_ctas", None) is not None:
        out["num_ctas"] = config.num_ctas
    return out


def _label_key(autotuner: Any, key_tuple: tuple) -> dict[str, Any]:
    """Label the cache key tuple by the kernel's own declared autotune ``key`` arg names
    (e.g. ``["M_BUCKET","N","K"]`` or ``["S","T","H"]``); extras (dtype specialization) go
    under ``_extra``."""
    names = list(getattr(autotuner, "keys", None) or _KEY_NAMES)
    result: dict[str, Any] = {}
    for i, name in enumerate(names):
        if i < len(key_tuple):
            result[name] = key_tuple[i]
    if len(key_tuple) > len(names):
        result["_extra"] = [str(v) for v in key_tuple[len(names) :]]
    return result


def collect_autotune_configs() -> list[dict[str, Any]]:
    """Read autotune caches from ALL dsv4 + scattermoe ``@triton.autotune`` kernels.

    Returns a (possibly empty) list of dicts, each containing:

    * ``kernel`` – kernel function name
    * ``module`` – short module name it lives in
    * ``key``    – dict of the autotune-key args (problem shapes), labeled by the kernel's
      own declared ``key`` names
    * ``config`` – selected tile sizes, ``num_warps``, ``num_stages`` (and ``num_ctas``)

    Scans ``sys.modules`` for both the HF-``kernels`` hash-suffixed scattermoe copies (whose
    caches are the populated runtime ones) and the normally-imported dsv4 kernel modules.
    """
    results: list[dict[str, Any]] = []
    # Dedup by the FULL module path so two distinct module instances of the same kernel
    # (e.g. a duplicate import) stay visible as separate entries; that duplication is exactly
    # what telemetry should surface (a single Autotuner.cache is a dict and can't hold a key
    # twice, so any same-(kernel,key) duplicate means >1 module instance).
    seen: set[tuple[str, str, tuple]] = set()

    for modname, module in list(sys.modules.items()):
        if module is None or not any(h in modname for h in _KERNEL_MODULE_HINTS):
            continue
        for attr in dir(module):
            obj = getattr(module, attr, None)
            if not _is_autotuner(obj):
                continue
            cache = obj.cache  # type: ignore[union-attr]
            if not cache:
                continue
            base = getattr(obj, "base_fn", None) or getattr(obj, "fn", None)
            kname = getattr(base, "__name__", attr)
            for key_tuple, config in cache.items():
                dedup = (modname, kname, tuple(key_tuple))
                if dedup in seen:
                    continue
                seen.add(dedup)
                results.append(
                    {
                        "kernel": kname,
                        "module": modname.rsplit(".", 1)[-1],
                        "module_path": modname,
                        "key": _label_key(obj, key_tuple),
                        "config": _config_to_dict(config),
                    }
                )

    return results

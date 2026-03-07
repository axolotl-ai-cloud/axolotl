"""Collect Triton autotune results from scattermoe-lora kernels.

This module reads the ``.cache`` attribute from Triton ``@triton.autotune``
decorated kernel objects and returns structured dicts describing the selected
configurations.  It has **no** telemetry dependency — callers decide what to
do with the data.
"""

import logging
from typing import Any

LOG = logging.getLogger(__name__)

# (human-readable name, attribute on the lora_ops module)
_KERNEL_REGISTRY: list[tuple[str, str]] = [
    ("scatter2scatter_lora_fwd", "_scatter2scatter_lora"),
    ("scatter2scatter_lora_dX", "_scatter2scatter_lora_dX"),
    ("group_bwd_lora", "_group_bwd_lora"),
    ("group_bwd_lora_fused", "_group_bwd_lora_fused"),
]

# The autotune key declared on every kernel: key=["M", "N", "K"]
_KEY_NAMES: list[str] = ["M", "N", "K"]


def _parse_key_tuple(key_tuple: tuple) -> dict[str, Any]:
    """Turn the autotune cache key tuple into a labelled dict.

    Triton builds the cache key from the values of the declared ``key``
    args (``M``, ``N``, ``K``) followed by dtype signature elements.
    We label the first three and store the rest under ``_extra``.
    """
    result: dict[str, Any] = {}
    for i, name in enumerate(_KEY_NAMES):
        if i < len(key_tuple):
            result[name] = key_tuple[i]
    if len(key_tuple) > len(_KEY_NAMES):
        result["_extra"] = [str(v) for v in key_tuple[len(_KEY_NAMES) :]]
    return result


def collect_autotune_configs() -> list[dict[str, Any]]:
    """Read autotune caches from the four scattermoe-lora kernels.

    Returns a (possibly empty) list of dicts, each containing:

    * ``kernel`` – human-readable kernel name
    * ``key``    – dict with the ``M``/``N``/``K`` problem dimensions
    * ``config`` – dict with the selected tile sizes, ``num_warps``,
      and ``num_stages``

    Returns ``[]`` if the kernel module cannot be imported (e.g. Triton
    is not installed) or if no autotune cache entries exist yet.
    """
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (  # noqa: F401
            lora_ops,
        )
    except ImportError:
        LOG.debug(
            "scattermoe_lora.kernels.lora_ops not importable; "
            "skipping autotune collection"
        )
        return []

    results: list[dict[str, Any]] = []

    for friendly_name, attr_name in _KERNEL_REGISTRY:
        kernel_fn = getattr(lora_ops, attr_name, None)
        if kernel_fn is None:
            continue

        cache = getattr(kernel_fn, "cache", None)
        if not cache:
            continue

        for key_tuple, config in cache.items():
            config_dict = dict(config.kwargs)
            config_dict["num_warps"] = config.num_warps
            config_dict["num_stages"] = config.num_stages
            if getattr(config, "num_ctas", None) is not None:
                config_dict["num_ctas"] = config.num_ctas

            results.append(
                {
                    "kernel": friendly_name,
                    "key": _parse_key_tuple(key_tuple),
                    "config": config_dict,
                }
            )

    return results

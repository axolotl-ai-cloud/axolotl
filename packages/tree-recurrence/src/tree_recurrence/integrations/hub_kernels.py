# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run transformers' short causal convs on the causal-conv1d hub kernel.

transformers binds a modeling module's ``causal_conv1d_fn`` and
``causal_conv1d_update`` when the module imports: to the causal-conv1d package, or
to a PyTorch reference without it; hub kernels apply only under ``use_kernels``.
``use_hub_causal_conv1d`` points the modules at ``kernels-community/causal-conv1d``,
which has the package's API, beneath any tree hook on them. Needs ``kernels``.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from tree_recurrence._hooks import hook_of
from tree_recurrence.registry import mixer_family

LOG = logging.getLogger(__name__)

HUB_REPO = "kernels-community/causal-conv1d"
HUB_VERSION = 2
NAMES = ("causal_conv1d_fn", "causal_conv1d_update")


@functools.cache
def hub_causal_conv1d() -> Any:
    """The hub kernel, when the ``causal_conv1d`` package is not installed."""
    if importlib.util.find_spec("causal_conv1d") is not None:
        LOG.info("causal conv: using the installed causal_conv1d package")
        return None
    try:
        from kernels import get_kernel

        kernel = get_kernel(HUB_REPO, version=HUB_VERSION)
    except Exception as error:
        LOG.warning(
            f"causal conv: {HUB_REPO} v{HUB_VERSION} is unavailable ({error}); "
            "transformers keeps its PyTorch reference conv"
        )
        return None
    LOG.info(f"causal conv: using the {HUB_REPO} v{HUB_VERSION} hub kernel")
    return kernel


def _keyword_filtered(kernel_fn: Callable[..., Any]) -> Callable[..., Any]:
    params = set(inspect.signature(kernel_fn).parameters)

    @functools.wraps(kernel_fn)
    def call(*args, **kwargs):
        return kernel_fn(*args, **{k: v for k, v in kwargs.items() if k in params})

    return call


def use_hub_causal_conv1d(target: torch.nn.Module | ModuleType) -> bool:
    """Bind the conv functions of ``target`` (a modeling module, or a model's
    recurrent mixers' modules) to the hub kernel. Returns whether any were bound."""
    if isinstance(target, ModuleType):
        modules = [target]
    else:
        modules = list(
            {
                sys.modules[type(m).__module__]
                for m in target.modules()
                if mixer_family(type(m).__name__) is not None
            }
        )
    names = [
        (module, name)
        for module in modules
        for name in NAMES
        if callable(getattr(module, name, None))
    ]
    kernel = hub_causal_conv1d() if names else None
    if kernel is None:
        return False
    for module, name in names:
        function = _keyword_filtered(getattr(kernel, name))
        hook = hook_of(module, name)
        if hook is not None:
            hook.original = function
        else:
            setattr(module, name, function)
        LOG.info(f"Bound {module.__name__}.{name} to {HUB_REPO}")
    return True

"""Run transformers' short causal convs on the causal-conv1d hub kernel.

transformers binds the ``causal_conv1d_fn`` / ``causal_conv1d_update`` of its
linear-attention and Mamba models (Qwen3.5, Qwen3-Next, Mamba2, Nemotron-H, ...)
when the modeling module is imported: to the ``causal_conv1d`` package, or to a
pure-PyTorch reference without it. Its hub mapping only applies under
``use_kernels``. Without the package, these models therefore run the conv eagerly,
and refuse ``seq_idx`` (packed sequences) outright. This points them at
``kernels-community/causal-conv1d``, which has the package's API.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

HUB_REPO = "kernels-community/causal-conv1d"
HUB_VERSION = 2


@functools.cache
def hub_causal_conv1d():
    """The hub kernel, when the ``causal_conv1d`` package is not installed."""
    if importlib.util.find_spec("causal_conv1d") is not None:
        LOG.info("causal conv: using the installed causal_conv1d package")
        return None
    try:
        from kernels import get_kernel

        kernel = get_kernel(HUB_REPO, version=HUB_VERSION)
    except Exception as err:  # noqa: BLE001
        LOG.warning(
            f"causal conv: {HUB_REPO} v{HUB_VERSION} is unavailable ({err}); "
            "transformers keeps its PyTorch reference conv"
        )
        return None
    LOG.info(f"causal conv: using the {HUB_REPO} v{HUB_VERSION} hub kernel")
    return kernel


def _bind(kernel_fn):
    params = set(inspect.signature(kernel_fn).parameters)

    @functools.wraps(kernel_fn)
    def call(*args, **kwargs):
        return kernel_fn(*args, **{k: v for k, v in kwargs.items() if k in params})

    return call


def patch_causal_conv1d(model_type: str | None) -> bool:
    """Rebind ``model_type``'s transformers conv functions to the hub kernel."""
    if not model_type:
        return False
    try:
        module = importlib.import_module(
            f"transformers.models.{model_type}.modeling_{model_type}"
        )
    except ImportError:
        return False
    names = [
        name
        for name in ("causal_conv1d_fn", "causal_conv1d_update")
        if callable(getattr(module, name, None))
    ]
    if not names:
        return False
    kernel = hub_causal_conv1d()
    if kernel is None:
        return False
    for name in names:
        setattr(module, name, _bind(getattr(kernel, name)))
    LOG.info(f"Bound {model_type}'s {', '.join(names)} to {HUB_REPO}")
    return True

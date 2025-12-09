"""Monkey patch to allow context parallelism with FlashAttention in HF Trainer."""

from __future__ import annotations

import importlib
import inspect

from transformers import Trainer

from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

GUARD_PATTERN = 'if model.config._attn_implementation != "sdpa":'
PATCHED_GUARD = 'if (attn_impl := (getattr(model.config, "_attn_implementation", None) or getattr(model.model.config, "_attn_implementation", None))) and attn_impl not in ("sdpa", "flash_attention_2"):'


def patch_prepare_context_parallel_inputs() -> None:
    """Relax the SDPA-only guard when running context parallelism with FlashAttention."""
    if getattr(Trainer, "_axolotl_prepare_context_parallel_inputs_patched", False):
        LOG.debug("Trainer._prepare_context_parallel_inputs already patched")
        return

    try:
        original_source = inspect.getsource(Trainer._prepare_context_parallel_inputs)
    except OSError as exc:  # pragma: no cover - occurs when source is unavailable
        LOG.warning("Unable to patch Trainer._prepare_context_parallel_inputs: %s", exc)
        return

    if GUARD_PATTERN not in original_source:
        LOG.warning(
            "Expected guard not found in Trainer._prepare_context_parallel_inputs; \n"
            "skipping FlashAttention context parallelism patch"
        )
        return

    patched_source = original_source.replace(GUARD_PATTERN, PATCHED_GUARD)
    patched_source, _ = detab_code(patched_source)
    patched_source = patched_source.replace(
        "def _prepare_context_parallel_inputs(",
        "def axolotl_prepare_context_parallel_inputs(",
        1,
    )

    module_name = Trainer.__module__
    module = importlib.import_module(module_name)

    # import symbols referenced in the method so exec can succeed
    items_to_import = []
    for item in dir(module):
        if item in patched_source:
            items_to_import.append(item)

    # Use a separate namespace to capture the exec'd function
    namespace = {}
    exec(f"from {module_name} import ({', '.join(items_to_import)})", namespace)
    exec(patched_source, namespace)

    # Explicitly get the function from the namespace
    axolotl_prepare_context_parallel_inputs = namespace[
        "axolotl_prepare_context_parallel_inputs"
    ]
    Trainer._original_prepare_context_parallel_inputs = (
        Trainer._prepare_context_parallel_inputs
    )
    Trainer._prepare_context_parallel_inputs = axolotl_prepare_context_parallel_inputs
    Trainer._axolotl_prepare_context_parallel_inputs_source = patched_source
    Trainer._axolotl_prepare_context_parallel_inputs_patched = True
    LOG.debug(
        "Patched Trainer._prepare_context_parallel_inputs for FlashAttention + CP"
    )

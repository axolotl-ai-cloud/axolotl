"""Transparently upgrade FA2 to FA4 when available on SM90+ hardware."""

import importlib.util

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_flash_attn_4():
    """Patch _lazy_imports to redirect FA2 imports to FA4 if available on supported hardware."""
    if not importlib.util.find_spec("flash_attn.cute"):
        return

    if not torch.cuda.is_available():
        return

    major, _ = torch.cuda.get_device_capability()
    # Matches flash_attn/cute/interface.py: arch / 10 in [9, 10, 11]
    if major not in (9, 10, 11):
        return

    import transformers.modeling_flash_attention_utils as fa_utils

    if getattr(fa_utils._lazy_imports, "_axolotl_patched", False):
        return

    def _patched_lazy_imports(
        implementation, attention_wrapper=None, allow_all_kernels=False
    ):
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        return (
            flash_attn_func,
            flash_attn_varlen_func,
            fa_utils._pad_input,
            fa_utils._unpad_input,
        )

    _patched_lazy_imports._axolotl_patched = True
    fa_utils._lazy_imports = _patched_lazy_imports
    LOG.info("Flash Attention 4 enabled")

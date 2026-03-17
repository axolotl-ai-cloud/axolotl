"""Transparently upgrade FA2 to FA4 when available on SM90+ hardware."""

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _get_head_dims(model_config):
    """Extract (head_dim, head_dim_v) from a model config.

    Handles composite models (e.g. Qwen3.5 VL) via text_config and
    MLA models (DeepSeek/Kimi) that have separate Q/V head dimensions.
    """
    cfg = model_config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config

    # MLA models: Q head_dim = qk_nope + qk_rope, V head_dim = v_head_dim
    if hasattr(cfg, "qk_nope_head_dim") and hasattr(cfg, "qk_rope_head_dim"):
        head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        head_dim_v = getattr(cfg, "v_head_dim", head_dim)
        return head_dim, head_dim_v

    # Standard models
    if hasattr(cfg, "head_dim"):
        return cfg.head_dim, cfg.head_dim
    if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        return head_dim, head_dim

    return None, None


def patch_flash_attn_4(model_config=None):
    """Patch _lazy_imports to redirect FA2 imports to FA4 if available on supported hardware."""
    if not torch.cuda.is_available():
        return

    major, _ = torch.cuda.get_device_capability()
    # Matches flash_attn/cute/interface.py: arch / 10 in [9, 10, 11]
    if major not in (9, 10, 11):
        return

    try:
        from flash_attn.cute import (  # noqa: F401
            flash_attn_func,
            flash_attn_varlen_func,
        )
    except ImportError:
        LOG.info(
            "Flash Attention 4 is available for your GPU and offers faster training speeds. "
            "To enable: pip install flash-attn-4"
        )
        return

    # Validate head dimensions against FA4's own constraints
    head_dim = None
    if model_config is not None:
        head_dim, head_dim_v = _get_head_dims(model_config)
        if head_dim is not None:
            try:
                from flash_attn.cute.interface import _validate_head_dims
            except ImportError:
                LOG.warning(
                    "Could not import _validate_head_dims from flash_attn.cute.interface, "
                    "unable to verify head dimension compatibility, falling back to FA2"
                )
                return

            # alignment = 16 // element_size; bf16/fp16 = 2 bytes -> alignment = 8
            alignment = 8
            try:
                _validate_head_dims(head_dim, head_dim_v, major, alignment)
            except AssertionError as exc:
                LOG.warning(
                    "Model head dimensions not supported by FA4, "
                    "falling back to FA2: %s",
                    exc,
                )
                return

    import transformers.modeling_flash_attention_utils as fa_utils

    if getattr(fa_utils._lazy_imports, "_axolotl_patched", False):
        return

    def _patched_lazy_imports(
        implementation, attention_wrapper=None, allow_all_kernels=False
    ):
        return (
            flash_attn_func,
            flash_attn_varlen_func,
            fa_utils._pad_input,
            fa_utils._unpad_input,
        )

    _patched_lazy_imports._axolotl_patched = True
    fa_utils._lazy_imports = _patched_lazy_imports
    LOG.info(
        "Flash Attention 4 enabled (head_dim=%s)",
        head_dim if model_config else "unknown",
    )

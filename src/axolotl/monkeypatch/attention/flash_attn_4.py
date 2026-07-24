"""Route flash attention through native FA4 when it is available on SM90+ hardware."""

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# quack < 0.6.0 (built for cutlass 4.6.0.dev0) crashes the FA4 backward on stable 4.6.0.
FA4_MIN_QUACK_VERSION = "0.6.0"


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


def _quack_supported():
    """Return (ok, installed_version). ``ok`` is False only when quack-kernels is installed
    and older than ``FA4_MIN_QUACK_VERSION``; an absent/unreadable quack is treated as ok."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version("quack-kernels")
    except PackageNotFoundError:
        return True, None
    except Exception:  # pylint: disable=broad-except
        return True, None

    from packaging.version import Version

    return Version(installed) >= Version(FA4_MIN_QUACK_VERSION), installed


def _warn_stale_quack(installed):
    LOG.warning(
        "Flash Attention 4 needs quack-kernels>=%s (found %s): the FA4 backward raises "
        "cudaErrorIllegalInstruction on nvidia-cutlass-dsl 4.6.0 with older quack. "
        "Upgrade with: pip install 'quack-kernels>=%s'.",
        FA4_MIN_QUACK_VERSION,
        installed,
        FA4_MIN_QUACK_VERSION,
    )


def fa4_usable(model_config=None):
    """Whether native FA4 can serve attention for this model on this GPU.

    Checks GPU arch (SM90/100/110), ``flash_attn.cute`` import, FA4 head-dim limits, and the
    quack-kernels floor. Warns (with the fix) when head dims or quack are the blocker.
    """
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    # Matches flash_attn/cute/interface.py: arch / 10 in [9, 10, 11]
    if major not in (9, 10, 11):
        return False

    try:
        from flash_attn.cute import (  # noqa: F401
            flash_attn_func,
            flash_attn_varlen_func,
        )
    except ImportError:
        LOG.info(
            "Flash Attention 4 is available for your GPU and offers faster training. "
            "To enable: pip install --pre flash-attn-4"
        )
        return False

    if model_config is not None:
        head_dim, head_dim_v = _get_head_dims(model_config)
        if head_dim is not None:
            try:
                from flash_attn.cute.interface import _validate_head_dims
            except ImportError:
                LOG.warning(
                    "Could not import _validate_head_dims from flash_attn.cute.interface; "
                    "cannot verify FA4 head-dim compatibility, keeping the requested backend."
                )
                return False

            # alignment = 16 // element_size; bf16/fp16 = 2 bytes -> 8
            try:
                _validate_head_dims(head_dim, head_dim_v, major, 8)
            except AssertionError as exc:
                LOG.warning(
                    "Model head dimensions not supported by FA4, keeping the requested "
                    "backend: %s",
                    exc,
                )
                return False

    ok, installed = _quack_supported()
    if not ok:
        _warn_stale_quack(installed)
        return False

    return True


def configure_fa4():
    """Prepare the process to run native FA4.

    Silences the harmless first-compile ``AuxData`` warning and, for an explicitly requested
    ``flash_attention_4``, surfaces the stale-quack warning (the auto-upgrade path checks
    quack in ``fa4_usable`` before reaching here).
    """
    import warnings

    # FA4's unannotated AuxData triggers a harmless CuTe-DSL warning on first compile.
    warnings.filterwarnings(
        "ignore",
        message=r".*aux_data.*JitArgument.*",
        category=UserWarning,
    )

    ok, installed = _quack_supported()
    if not ok:
        _warn_stale_quack(installed)

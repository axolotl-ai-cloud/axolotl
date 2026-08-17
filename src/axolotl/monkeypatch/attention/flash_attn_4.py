"""Route flash attention through native FA4 when it is available on SM90+ hardware."""

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# quack < 0.6.0 (built for cutlass 4.6.0.dev0) crashes the FA4 backward on stable 4.6.0.
FA4_MIN_QUACK_VERSION = "0.6.0"

# FA4's Blackwell backward still elects around its own bulk copies; from cutlass-dsl 4.6.0
# cute.copy elects internally too, and the nested election hangs the kernel.
FA4_SM100_ELECT_CUTLASS_VERSION = "4.6.0"
FA4_SM100_FIX_INSTALL = (
    "pip install --no-deps --force-reinstall 'git+https://github.com/dongxiao92/"
    "flash-attention@fix/sm100-elect-one-and-div-alignment#subdirectory=flash_attn/cute'"
)


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


def _fa4_elects_around_bulk_copy():
    """Whether the installed FA4 still wraps its Blackwell stats bulk copies in ``elect_one``."""
    import importlib.util
    import re
    from pathlib import Path

    try:
        spec = importlib.util.find_spec("flash_attn.cute.flash_bwd_sm100")
        source = Path(spec.origin).read_text(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        return False

    return re.search(r"elect_one\(\):\s*\n\s*copy_stats\(", source) is not None


def _sm100_backward_supported():
    """Return (ok, installed_version) for nvidia-cutlass-dsl. ``ok`` is False only when the
    installed cutlass-dsl elects inside ``cute.copy`` and FA4 still elects around it."""
    try:
        from importlib.metadata import version

        installed = version("nvidia-cutlass-dsl")
    except Exception:  # pylint: disable=broad-except
        return True, None

    from packaging.version import Version

    if Version(installed) < Version(FA4_SM100_ELECT_CUTLASS_VERSION):
        return True, installed

    return not _fa4_elects_around_bulk_copy(), installed


def _warn_sm100_elect_hang(installed):
    LOG.warning(
        "Flash Attention 4's backward hangs on Blackwell with nvidia-cutlass-dsl>=%s "
        "(found %s): FA4 elects around the bulk copies that cute.copy now elects internally. "
        "Install the fix (Dao-AILab/flash-attention#2689) with: %s",
        FA4_SM100_ELECT_CUTLASS_VERSION,
        installed,
        FA4_SM100_FIX_INSTALL,
    )


def fa4_usable(model_config=None):
    """Whether native FA4 can serve attention for this model on this GPU.

    Checks GPU arch (SM90/100/110), ``flash_attn.cute`` import, FA4 head-dim limits, the
    quack-kernels floor, and the Blackwell nested-election hang. Warns (with the fix) when
    head dims, quack, or the Blackwell backward are the blocker.
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

    if major in (10, 11):
        ok, cutlass_version = _sm100_backward_supported()
        if not ok:
            _warn_sm100_elect_hang(cutlass_version)
            return False

    return True


def configure_fa4():
    """Prepare the process to run native FA4.

    Silences the harmless first-compile ``AuxData`` warning and, for an explicitly requested
    ``flash_attention_4``, surfaces the stale-quack and Blackwell nested-election warnings
    (the auto-upgrade path checks both in ``fa4_usable`` before reaching here).
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

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11):
        ok, cutlass_version = _sm100_backward_supported()
        if not ok:
            _warn_sm100_elect_hang(cutlass_version)

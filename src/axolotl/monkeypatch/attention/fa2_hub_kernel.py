"""Pin the ``kernels-community/flash-attn2`` hub kernel to a working major version."""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# transformers 5.16 resolves `flash_attention_2` to hub kernel v3, whose stable-ABI CUDA
# builds fail the backward pass for every GQA/MQA model; v2 is the newest major with
# working CUDA builds on torch 2.11-2.13.
# https://github.com/huggingface/kernels-community/issues/1085
FA2_HUB_KERNEL_VERSION = 2
_BROKEN_FA2_HUB_KERNEL_VERSION = 3


def patch_fa2_hub_kernel_version() -> None:
    """Resolve ``flash_attention_2`` to hub kernel v2 instead of the broken v3."""
    from transformers.integrations import hub_kernels

    mapping = getattr(hub_kernels, "_FLASH_ATTN_KERNEL_VERSION_MAPPING", None)
    # Leave anything but the known-bad v3 alone so an upstream bump isn't clobbered.
    if mapping is None or mapping.get(2) != _BROKEN_FA2_HUB_KERNEL_VERSION:
        return

    mapping[2] = FA2_HUB_KERNEL_VERSION
    LOG.info(
        "Pinned the flash-attn2 hub kernel to v%d; v%d's stable-ABI build fails the "
        "backward pass for GQA/MQA models.",
        FA2_HUB_KERNEL_VERSION,
        _BROKEN_FA2_HUB_KERNEL_VERSION,
    )


def is_fa2_hub_kernel_available() -> bool:
    """Whether the pinned flash-attn2 hub kernel has a build for this torch."""
    from transformers.utils import is_kernels_available

    if not is_kernels_available():
        return False

    from kernels import get_kernel
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK

    try:
        get_kernel(
            FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"],
            version=FA2_HUB_KERNEL_VERSION,
        )
    except Exception:
        return False
    return True

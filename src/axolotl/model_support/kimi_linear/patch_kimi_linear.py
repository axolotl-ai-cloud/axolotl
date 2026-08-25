"""Redirect Kimi-Linear's remote-code loading to the in-tree training copies."""

from axolotl.model_support.remote_code import redirect_dynamic_modules
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

KIMI_PATCH_PACKAGE = "axolotl.model_support.kimi_linear"
KIMI_MODULES = ("configuration_kimi", "modeling_kimi", "tokenization_kimi")


def patch_kimi():
    """Apply all Kimi patches. Must be called BEFORE loading config/tokenizer/model."""
    redirect_dynamic_modules(KIMI_PATCH_PACKAGE, KIMI_MODULES)
    LOG.info("Kimi patches applied successfully!")


patch_kimi_config = patch_kimi
patch_kimi_tokenizer = patch_kimi
patch_kimi_model = patch_kimi

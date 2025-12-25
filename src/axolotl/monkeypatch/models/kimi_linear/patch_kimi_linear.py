import importlib.resources
import importlib.util
import sys
from pathlib import Path

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

KIMI_PATCH_PACKAGE = "axolotl.monkeypatch.models.kimi_linear"


def get_patch_file_path(package_dot_path: str, filename: str) -> Path:
    """
    Gets the absolute path to a patch file using importlib.resources.files.
    """
    try:
        return importlib.resources.files(package_dot_path) / filename
    except ModuleNotFoundError:
        return None


def _load_local_module(module_name: str, filename: str):
    """Helper to load a local module if not already loaded."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    patch_path = get_patch_file_path(KIMI_PATCH_PACKAGE, filename)
    if patch_path and patch_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, patch_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None


def _patch_get_class_in_module():
    """
    Core patch function that hijacks Transformers' dynamic module loading.
    """
    from transformers.dynamic_module_utils import get_class_in_module

    if hasattr(get_class_in_module, "_axolotl_patched"):
        return

    original_get_class_in_module = get_class_in_module

    # Mapping of module path patterns to (module_name, filename)
    KIMI_MODULE_MAP = {
        "configuration_kimi": ("configuration_kimi", "configuration_kimi.py"),
        "modeling_kimi": ("modeling_kimi", "modeling_kimi.py"),
        "tokenization_kimi": ("tokenization_kimi", "tokenization_kimi.py"),
    }

    def patched_get_class_in_module(class_name, module_path, **kwargs):
        """Patched version that returns our local modules instead of remote ones."""
        for pattern, (module_name, filename) in KIMI_MODULE_MAP.items():
            if pattern in module_path:
                module = _load_local_module(module_name, filename)
                if module:
                    return getattr(module, class_name)
                break  # Pattern matched but file not found, fall through

        return original_get_class_in_module(class_name, module_path, **kwargs)

    import transformers.dynamic_module_utils

    transformers.dynamic_module_utils.get_class_in_module = patched_get_class_in_module
    patched_get_class_in_module._axolotl_patched = True


def patch_kimi():
    """
    Apply all Kimi patches.
    Must be called BEFORE loading config/tokenizer/model.
    """
    _patch_get_class_in_module()
    LOG.info("Kimi patches applied successfully!")


# Keep these for backward compatibility if needed
patch_kimi_config = patch_kimi
patch_kimi_tokenizer = patch_kimi
patch_kimi_model = patch_kimi

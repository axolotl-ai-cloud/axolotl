import importlib.resources
import importlib.util
from pathlib import Path

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_patch_file_path(package_dot_path: str, filename: str) -> Path:
    """
    Gets the absolute path to a patch file using importlib.resources.files.
    This is the modern and preferred way.

    Args:
        package_dot_path (str): The package path in dot notation
                                (e.g., "axolotl.monkeypatch.models.kimi_linear")
        filename (str): The name of the file within that package.

    Returns:
        A pathlib.Path object with the absolute path to the file.
    """
    try:
        # importlib.resources.files() returns a Traversable object
        # that can be joined with / or .joinpath()
        return importlib.resources.files(package_dot_path) / filename
    except ModuleNotFoundError:
        # Handle cases where the path might be wrong
        return None


def _patch_get_class_in_module():
    """
    Core patch function that hijacks Transformers' dynamic module loading.
    This is shared between tokenizer and model patching.
    """
    from transformers.dynamic_module_utils import get_class_in_module

    KIMI_PATCH_PACKAGE = "axolotl.monkeypatch.models.kimi_linear"

    # Check if already patched to avoid double-patching
    if hasattr(get_class_in_module, "_axolotl_patched"):
        return

    # Store original function
    original_get_class_in_module = get_class_in_module

    def patched_get_class_in_module(class_name, module_path, **kwargs):
        """Patched version that returns our local modules instead of remote ones."""
        # Check if this is a Kimi model module
        if "modeling_kimi" in module_path:
            # Load our local modeling_kimi.py instead
            patch_path = get_patch_file_path(KIMI_PATCH_PACKAGE, "modeling_kimi.py")
            if patch_path and patch_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "modeling_kimi", patch_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, class_name)

        if "tokenization_kimi" in module_path:
            # Load our local tokenization_kimi.py instead
            patch_path = get_patch_file_path(KIMI_PATCH_PACKAGE, "tokenization_kimi.py")
            if patch_path and patch_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "tokenization_kimi", patch_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, class_name)

        # For all other modules, use the original function with all kwargs
        return original_get_class_in_module(class_name, module_path, **kwargs)

    # Apply the monkey patch
    import transformers.dynamic_module_utils

    transformers.dynamic_module_utils.get_class_in_module = patched_get_class_in_module
    # Mark as patched to avoid double-patching
    patched_get_class_in_module._axolotl_patched = True


def _patch_resolve_trust_remote_code():
    """
    Patch resolve_trust_remote_code to handle Kimi model auto_map.
    This helps Transformers find our local modules instead of remote ones.
    """
    from transformers.dynamic_module_utils import resolve_trust_remote_code

    # Check if already patched to avoid double-patching
    if hasattr(resolve_trust_remote_code, "_axolotl_patched"):
        return

    original_resolve_trust_remote_code = resolve_trust_remote_code

    def patched_resolve_trust_remote_code(repo_id, model_id, *args, **kwargs):
        """Patched version to handle Kimi model auto_map."""
        # Check if this is a Kimi model
        if "kimi" in repo_id.lower() or "kimi" in model_id.lower():
            # Get the original result
            result = original_resolve_trust_remote_code(
                repo_id, model_id, *args, **kwargs
            )

            # If it contains auto_map for Kimi, replace with our local files
            if hasattr(result, "get") and result.get("auto_map"):
                auto_map = result["auto_map"].copy()
                # Replace remote modules with our local ones
                for key in auto_map:
                    if "modeling_kimi" in auto_map[key]:
                        auto_map[key] = "modeling_kimi"
                    if "tokenization_kimi" in auto_map[key]:
                        auto_map[key] = "tokenization_kimi"
                result["auto_map"] = auto_map
                result["trust_remote_code"] = True

            return result

        return original_resolve_trust_remote_code(repo_id, model_id, *args, **kwargs)

    import transformers.dynamic_module_utils

    transformers.dynamic_module_utils.resolve_trust_remote_code = (
        patched_resolve_trust_remote_code
    )
    patched_resolve_trust_remote_code._axolotl_patched = True


def patch_kimi_tokenizer():
    """
    Apply Kimi tokenizer patches.
    This must be called BEFORE tokenizer loading to intercept remote code.
    """
    _patch_get_class_in_module()
    _patch_resolve_trust_remote_code()
    LOG.info("Kimi tokenizer patches applied successfully!")


def patch_kimi_model():
    """
    Apply Kimi model patches.
    This is called during model loading.
    Note: The core interception is already done by patch_kimi_tokenizer,
    but we keep this for any model-specific patches that might be needed.
    """
    _patch_get_class_in_module()
    _patch_resolve_trust_remote_code()
    LOG.info("Kimi model patches applied successfully!")

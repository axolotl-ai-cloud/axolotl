import importlib.resources
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path


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


def patch_kimi_model():
    """
    Apply Kimi model patches by hijacking Transformers' dynamic module loading.
    This intercepts the remote code loading and replaces it with our local patches.
    """
    from transformers.dynamic_module_utils import get_class_in_module

    KIMI_PATCH_PACKAGE = "axolotl.monkeypatch.models.kimi_linear"

    # Store original function
    original_get_class_in_module = get_class_in_module

    def patched_get_class_in_module(class_name, module_path, **kwargs):
        """Patched version that returns our local modules instead of remote ones."""
        # Check if this is a Kimi model module
        if "modeling_kimi" in module_path:
            print("Intercepting remote Kimi modeling module, using local patch instead")
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
            print(
                "Intercepting remote Kimi tokenizer module, using local patch instead"
            )
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

    # Also patch the resolve_trust_remote_code to handle auto_map
    from transformers.dynamic_module_utils import resolve_trust_remote_code

    original_resolve_trust_remote_code = resolve_trust_remote_code

    def patched_resolve_trust_remote_code(repo_id, model_id, *args, **kwargs):
        """Patched version to handle Kimi model auto_map."""
        # Check if this is a Kimi model
        if "kimi" in repo_id.lower() or "kimi" in model_id.lower():
            print(f"Resolving trust remote code for Kimi model: {repo_id}")
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

    transformers.dynamic_module_utils.resolve_trust_remote_code = (
        patched_resolve_trust_remote_code
    )

    print("Kimi model patches applied successfully!")


# The context manager code from before remains the same
@contextmanager
def patch_hf_imports():
    """
    A context manager to temporarily inject custom modules into sys.modules.

    Args:
        patch_map (dict): A dictionary mapping the target module name
                          (e.g., "modeling_falcon") to the local path of the
                          custom Python file.
    """

    KIMI_PATCH_PACKAGE = "axolotl.monkeypatch.models.kimi_linear"

    patches_to_apply = {
        "modeling_kimi": "modeling_kimi.py",
        "tokenization_kimi": "tokenization_kimi.py",
    }

    patch_map = {}
    for target_module, filename in patches_to_apply.items():
        patch_path = get_patch_file_path(KIMI_PATCH_PACKAGE, filename)
        if patch_path and patch_path.exists():
            print(f"Found patch for '{target_module}' at '{patch_path}'")
            patch_map[target_module] = patch_path
        else:
            raise FileNotFoundError(
                f"Could not find the patch file '{filename}' "
                f"in package '{KIMI_PATCH_PACKAGE}'"
            )

    original_modules = {}
    injected_modules = []

    for target_module_name, patch_file_path in patch_map.items():
        if not Path(patch_file_path).exists():
            print(f"Warning: Patch file not found at {patch_file_path}. Skipping.")
            continue

        # If the original module is already loaded, save it for restoration
        if target_module_name in sys.modules:
            original_modules[target_module_name] = sys.modules[target_module_name]

        # Use importlib to load our custom file as a module
        spec = importlib.util.spec_from_file_location(
            target_module_name, patch_file_path
        )
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # Inject it into sys.modules
        sys.modules[target_module_name] = custom_module
        injected_modules.append(target_module_name)

    try:
        # Yield control back to the 'with' block
        yield
    finally:
        # Cleanup: restore original modules or remove injected ones
        for module_name in injected_modules:
            if module_name in original_modules:
                # Restore the original module if it existed
                sys.modules[module_name] = original_modules[module_name]
            else:
                # Otherwise, just remove our injected module
                del sys.modules[module_name]

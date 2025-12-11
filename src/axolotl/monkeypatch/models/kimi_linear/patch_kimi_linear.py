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

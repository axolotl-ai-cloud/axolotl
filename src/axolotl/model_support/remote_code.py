"""Serve a model's remote-code classes from an in-tree package."""

import importlib
from collections.abc import Collection
from typing import Any

_REDIRECTS: dict[str, str] = {}


def owned_stem(stems: Collection[str], module_path) -> str | None:
    """Exact-stem match: a sibling remote module such as ``modeling_kimi_vl``
    must fall through to the original loader."""
    tail = str(module_path).replace("\\", "/").rsplit("/", 1)[-1]
    stem = tail.removesuffix(".py").rsplit(".", 1)[-1]
    return stem if stem in stems else None


def redirect_dynamic_modules(package: str, stems: Collection[str]) -> None:
    """Resolve ``stems`` from ``package`` whenever transformers loads remote code."""
    _REDIRECTS.update(dict.fromkeys(stems, package))
    _patch_get_class_in_module()


def _patch_get_class_in_module() -> None:
    import transformers.dynamic_module_utils as dynamic_module_utils

    original = dynamic_module_utils.get_class_in_module
    if getattr(original, "_axolotl_patched", False):
        return

    def patched_get_class_in_module(class_name, module_path, **kwargs):
        stem = owned_stem(_REDIRECTS, module_path)
        if stem is None:
            return original(class_name, module_path, **kwargs)
        return getattr(
            importlib.import_module(f"{_REDIRECTS[stem]}.{stem}"), class_name
        )

    marked: Any = patched_get_class_in_module
    marked._axolotl_patched = True
    dynamic_module_utils.get_class_in_module = marked

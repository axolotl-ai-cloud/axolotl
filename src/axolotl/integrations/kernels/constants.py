"""Diagnostic helpers for MoE kernel integrations (kernel dispatch itself
is architecture-agnostic via the ExpertsInterface)."""

import importlib

# Models where MoE is embedded in the decoder layer (no separate SparseMoeBlock).
EXPERTS_ONLY_BLOCK = {
    "gemma4_text": "Gemma4TextExperts",
}


def resolve_experts_class(model_type: str):
    """Resolve the Experts class for a known model type, or ``None``."""
    entry = EXPERTS_ONLY_BLOCK.get(model_type)
    if entry is None:
        return None

    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        if model_type.endswith("_text"):
            parent_type = model_type.removesuffix("_text")
            module_path = f"transformers.models.{parent_type}.modeling_{parent_type}"
            module = importlib.import_module(module_path)
        else:
            raise

    cls = getattr(module, entry, None)
    if cls is None:
        raise ValueError(f"Could not find class '{entry}' in '{module_path}'")
    return cls


def is_experts_only_model(model_type: str) -> bool:
    return model_type in EXPERTS_ONLY_BLOCK

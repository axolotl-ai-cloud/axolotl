"""Per-architecture model support descriptors.

Each supported architecture gets one directory here containing a
`ModelSupport` descriptor (declarative capabilities + lifecycle hooks) and any
model-specific code it needs (processing strategy, patches). Features query
the registry instead of hardcoding ``model_type`` checks.
"""

from .base import (
    Capability,
    Experimental,
    ModelSupport,
    Supported,
    Unsupported,
    check_capability,
)
from .registry import (
    get_model_support,
    get_model_support_for_cfg,
    get_model_support_for_processor,
    register_model_support,
)

__all__ = [
    "Capability",
    "Experimental",
    "ModelSupport",
    "Supported",
    "Unsupported",
    "check_capability",
    "get_model_support",
    "get_model_support_for_cfg",
    "get_model_support_for_processor",
    "register_model_support",
]

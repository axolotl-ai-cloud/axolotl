"""Print the uv commands required to install Unsloth without altering Torch."""

try:
    import torch
except ImportError as error:
    raise ImportError("Install torch via `pip install torch`") from error

from packaging.version import Version as V

TORCH_MIN = V("2.6.0")
UNSLOTH_BASE = (
    "uv pip install --system --no-deps unsloth-zoo==2025.9.12"
    ' && uv pip install --system --no-deps "unsloth[huggingface]==2025.9.9"'
)

version = V(torch.__version__)
if version < TORCH_MIN:
    raise RuntimeError(
        f"Torch {version} detected, but Unsloth requires >= {TORCH_MIN}. "
        "Upgrade your torch install and re-run this helper."
    )

print(UNSLOTH_BASE)

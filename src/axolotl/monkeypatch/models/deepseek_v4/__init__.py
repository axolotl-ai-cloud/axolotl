"""Training-enablement patches for DeepSeek V4."""

from axolotl.monkeypatch.models.deepseek_v4.modeling import (
    freeze_deepseek_v4_indexer,
    patch_deepseek_v4_supports_flex,
)

__all__ = [
    "freeze_deepseek_v4_indexer",
    "patch_deepseek_v4_supports_flex",
]

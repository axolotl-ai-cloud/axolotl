"""Flip ``accepts_loss_kwargs`` to True on Gemma 4 (Unified) ForConditionalGeneration.

They inherit ``accepts_loss_kwargs = False`` from PaliGemma (whose loss filtered
logits/labels by attention_mask). Gemma 4's loss is the stock ``ForCausalLMLoss``
with no such filtering, so the flag wrongly makes the Trainer withhold
``num_items_in_batch`` and mis-normalize the loss under gradient accumulation.
Install before ``Trainer.__init__`` reads the flag.
"""

from __future__ import annotations

import importlib

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_TARGETS = (
    ("transformers.models.gemma4.modeling_gemma4", "Gemma4ForConditionalGeneration"),
    (
        "transformers.models.gemma4_unified.modeling_gemma4_unified",
        "Gemma4UnifiedForConditionalGeneration",
    ),
)


def patch_gemma4_accepts_loss_kwargs() -> None:
    """Set ``accepts_loss_kwargs=True`` on Gemma 4 (Unified) ForConditionalGeneration."""
    for module_path, cls_name in _TARGETS:
        try:
            cls = getattr(importlib.import_module(module_path), cls_name)
        except (ImportError, AttributeError):
            continue
        if getattr(cls, "accepts_loss_kwargs", None) is True:
            continue
        cls.accepts_loss_kwargs = True
        LOG.info(
            "Set %s.accepts_loss_kwargs=True so the Trainer forwards "
            "num_items_in_batch (correct gradient-accumulation loss normalization).",
            cls_name,
        )

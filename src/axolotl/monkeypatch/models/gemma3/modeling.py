"""Monkeypatch for gemma3 conditional generation forward to fix high loss"""


def patch_gemma3_conditional_generation_forward():
    # Remove when https://github.com/huggingface/transformers/pull/37208 merged

    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForConditionalGeneration,
    )

    setattr(Gemma3ForConditionalGeneration, "accepts_loss_kwargs", False)

    def unpatch():
        delattr(Gemma3ForConditionalGeneration, "accepts_loss_kwargs")

    return unpatch

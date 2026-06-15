"""DeepSeek V4 training patches.

HF transformers' DeepSeek V4 ships with ``_supports_flash_attn = False`` and
``_supports_sdpa = False`` because the attention pattern (sinks +
sliding-window KV concatenated with a compressed KV pool + padded 4D mask)
can't be expressed in FA2/SDPA. The flex-attention backend, however, can:
``transformers.integrations.flex_attention.flex_attention_forward`` already
applies attention sinks via post-softmax LSE renormalization and accepts a
4D additive mask as ``score_mask``. Flipping ``_supports_flex_attn`` lets
axolotl train V4 with ``flex_attention: true``.

The sparse-attention :class:`DeepseekV4Indexer` returns ``topk.indices``
(``LongTensor``, non-differentiable). Its learnable projections therefore
receive no gradient from the LM loss, which trips DDP's unused-parameter
check and wastes autograd bookkeeping under FSDP. Freezing the indexer
parameters after model build keeps the pretrained top-k ranking intact and
sidesteps both problems.
"""

from __future__ import annotations

from transformers import PreTrainedModel

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_SUPPORTS_FLEX_APPLIED = False


def patch_deepseek_v4_supports_flex() -> bool:
    """Flip ``DeepseekV4PreTrainedModel._supports_flex_attn`` on.

    Idempotent. Returns ``True`` if the patch was installed (or already
    installed), ``False`` if transformers doesn't ship V4 yet — in which
    case nothing is done and the caller can proceed unaffected.
    """
    global _SUPPORTS_FLEX_APPLIED
    if _SUPPORTS_FLEX_APPLIED:
        return True

    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4
    except ImportError:
        LOG.debug(
            "deepseek_v4: transformers.models.deepseek_v4 not importable, "
            "skipping flex-support patch. Fine for non-V4 training."
        )
        return False

    cls = modeling_deepseek_v4.DeepseekV4PreTrainedModel
    cls._supports_flex_attn = True
    _SUPPORTS_FLEX_APPLIED = True
    LOG.info("deepseek_v4: enabled flex attention support on %s", cls.__name__)
    return True


def freeze_deepseek_v4_indexer(model: PreTrainedModel) -> int:
    """Set ``requires_grad=False`` on every :class:`DeepseekV4Indexer` param.

    Called from ``apply_post_model_build_patches`` so it runs before any
    PEFT wrapping inspects which params are trainable. Matching by class
    name rather than ``isinstance`` avoids importing the transformers V4
    module at file-load time.

    Returns the number of indexer modules frozen.
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV4Indexer":
            continue
        for param in module.parameters():
            param.requires_grad = False
        count += 1
    if count:
        LOG.info("deepseek_v4: froze %d indexer module(s)", count)
    return count

"""Flash attention monkey patch for mistral model"""

from functools import partial

import transformers

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_mistral_cross_entropy():
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    LOG.info("patching with flash_attn.losses.cross_entropy")
    transformers.models.mistral.modeling_mistral.CrossEntropyLoss = partial(
        CrossEntropyLoss, inplace_backward=True
    )

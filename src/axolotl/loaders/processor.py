"""Processor loading functionality for multi-modal models"""

import transformers
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerBase,
)

from axolotl.telemetry.errors import send_errors
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@send_errors
def load_processor(cfg: DictDefault, tokenizer: PreTrainedTokenizerBase):
    processor_cls = AutoProcessor
    if cfg.processor_type:
        processor_cls = getattr(transformers, cfg.processor_type)

    if cfg.tokenizer_use_mistral_common:

        def _patch_mistralcommontokenizer():
            """
            Transformers v5 stops reading the sub-processor.

            We need to patch this, so both processors use this.
            """
            import transformers.tokenization_mistral_common as tokenization_mistral_common

            from axolotl.utils.mistral import HFMistralTokenizer

            tokenization_mistral_common.MistralCommonBackend = HFMistralTokenizer

        _patch_mistralcommontokenizer()

        from transformers import VoxtralProcessor

        if processor_cls == VoxtralProcessor:
            return VoxtralProcessor.from_pretrained(
                cfg.processor_config,
            )

        from axolotl.utils.mistral import Mistral3Processor

        return Mistral3Processor(
            tokenizer=tokenizer,
        )

    processor = processor_cls.from_pretrained(
        cfg.processor_config,
        trust_remote_code=cfg.trust_remote_code or False,
        tokenizer=tokenizer,
    )

    # Attempt to load image size from processor if available
    if (
        cfg.image_size is None
        and hasattr(processor, "size")
        and any(dim in processor.size for dim in ["width", "height"])
    ):
        im_width = None
        im_height = None
        if "width" in processor.size:
            im_width = processor.size["width"]
        if "height" in processor.size:
            im_height = processor.size["height"]

        # If both width and height are set, use a tuple
        if im_width is not None and im_height is not None:
            cfg.image_size = (im_width, im_height)
        # If only width is set, use as integer
        elif im_width is not None:
            cfg.image_size = im_width
        # If only height is set, use as integer
        elif im_height is not None:
            cfg.image_size = im_height

        LOG.debug(f"Loaded image size: {cfg.image_size} from processor")

    return processor

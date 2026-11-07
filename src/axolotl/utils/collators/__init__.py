"""Shared axolotl collators for multipacking, multimodal."""

from axolotl.utils import make_lazy_getattr

from .batching import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    PretrainingBatchSamplerDataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)

__all__ = [
    "DataCollatorForSeq2Seq",
    "BatchSamplerDataCollatorForSeq2Seq",
    "V2BatchSamplerDataCollatorForSeq2Seq",
    "PretrainingBatchSamplerDataCollatorForSeq2Seq",
    "AxolotlDPODataCollatorWithPadding",
]

# lazy so DataLoader workers unpickling a batching collator skip trl (and its peft chain)
_LAZY_IMPORTS = {"AxolotlDPODataCollatorWithPadding": ".dpo"}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())

"""
collators for Mamba
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from axolotl.utils.collators.core import IGNORE_INDEX


@dataclass
class MambaDataCollator:
    """
    Collator for State Space Models (Mamba)
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [torch.LongTensor(instance[key]) for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

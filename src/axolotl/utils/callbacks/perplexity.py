"""callback to calculate perplexity as an evaluation metric."""

from typing import Dict, List, Optional

import torch
from torch import Tensor
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.utils.distributed import is_main_process


class Perplexity:
    """
    Calculate perplexity as defined in https://huggingface.co/docs/transformers/en/perplexity.
    This is a custom variant that doesn't re-tokenize the input or re-load the model.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        stride: int = 512,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.tokenizer = tokenizer
        self.name = "perplexity"

    def _feature_names(self) -> List[str]:
        return ["references"]

    def compute(
        self,
        model: PreTrainedModel,
        references: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute perplexity in a fixed length sliding window across the sequence.
        """
        assert references is not None, "Missing parameter: references"

        model.eval()

        references_tokenized = self.tokenizer(
            references, return_tensors="pt", padding=True, truncation=True
        )
        input_ids: Tensor = references_tokenized["input_ids"]  # type: ignore
        input_ids = input_ids.to(model.device)

        sequence_length = input_ids.size(1)

        losses = []
        prev_end_loc = 0
        for begin_loc in tqdm(
            range(0, sequence_length, self.stride), disable=not is_main_process()
        ):
            end_loc = min(begin_loc + self.max_seq_len, sequence_length)
            trg_len = end_loc - prev_end_loc
            input_ids_slice = input_ids[:, begin_loc:end_loc]
            labels_slice = input_ids_slice.clone()
            labels_slice[:, :-trg_len] = -100

            with torch.no_grad():
                outputs: CausalLMOutput = model(
                    input_ids=input_ids_slice, labels=labels_slice
                )

            losses.append(outputs.loss)

            prev_end_loc = end_loc
            if end_loc == sequence_length:
                break

        perplexity = torch.exp(torch.stack(losses).mean()).item()

        return {
            "score": perplexity,
        }

"""callback to calculate perplexity as an evaluation metric."""

from typing import Dict, List

import torch
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

from axolotl.utils.distributed import is_main_process
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class Perplexity:
    """
    Calculate perplexity as defined in https://huggingface.co/docs/transformers/en/perplexity.
    This is a custom variant that doesn't re-tokenize the input or re-load the model.
    """

    def __init__(self) -> None:
        self.name = "perplexity"

    def _feature_names(self) -> List[str]:
        return ["references"]

    def compute(
        self,
        model: PreTrainedModel,
        eval_dataloader=None,
        **kwargs,
    ) -> Dict[str, float]:
        model.eval()

        scores = []
        for batch in tqdm(eval_dataloader, disable=not is_main_process()):
            if "position_ids" not in batch:
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)

                # Drop any that are just -100
                valid_rows_mask = (labels != -100).any(dim=1)
                input_ids = input_ids[valid_rows_mask]
                labels = labels[valid_rows_mask]

                if input_ids.shape[0] == 0:
                    continue

                with torch.no_grad():
                    model_loss = model(
                        input_ids=input_ids,
                        labels=labels,
                    ).loss

                scores.append(torch.exp(model_loss).item())
            else:
                # do_causal_lm_eval + sample_packing already gives ValidationError. Extra protection.
                LOG.debug(
                    "Packed evaluation samples not supported with perplexity metric"
                )
                return {"score": float("nan")}

        if len(scores) == 0:
            LOG.debug("No valid tokens found for perplexity metric")
            return {"score": float("nan")}

        return {"score": sum(scores) / len(scores)}

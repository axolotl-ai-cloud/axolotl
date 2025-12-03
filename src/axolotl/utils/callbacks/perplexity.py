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
        return []

    def compute(
        self,
        model: PreTrainedModel,
        eval_dataloader=None,
    ) -> Dict[str, float]:
        model.eval()

        total_loss_sum = 0.0
        total_token_count = 0
        for batch in tqdm(eval_dataloader, disable=not is_main_process()):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = (
                batch["attention_mask"].to(model.device)
                if "attention_mask" in batch
                else None
            )
            position_ids = (
                batch["position_ids"].to(model.device)
                if "position_ids" in batch
                else None
            )

            # Drop any rows that are just -100
            valid_rows_mask = (labels != -100).any(dim=1)
            input_ids = input_ids[valid_rows_mask]
            if input_ids.shape[0] == 0:  # Skip batch if no rows are left
                continue
            labels = labels[valid_rows_mask]
            if attention_mask is not None:
                attention_mask = attention_mask[valid_rows_mask]
            if position_ids is not None:
                position_ids = position_ids[valid_rows_mask]

            with torch.no_grad():
                model_loss = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                ).loss

            batch_token_count = (labels != -100).sum().item()
            total_loss_sum += model_loss.item() * batch_token_count
            total_token_count += batch_token_count

        if total_token_count == 0:
            LOG.debug("No valid tokens found for perplexity metric")
            return {"score": float("nan")}

        return {
            "score": torch.exp(torch.tensor(total_loss_sum / total_token_count)).item()
        }

"""
This module defines the BatchEval class, which handles batch evaluation of a model.
"""

from typing import Dict

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.utils.dict import DictDefault

LOG = get_logger(__name__)


def collate_fn(
    batch, pad_value: int = 0, padding_direction: str = "left"
) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]

    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=pad_value)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    if padding_direction.lower() == "right":
        padded_inputs = padded_inputs.flip([1])
        padded_labels = padded_labels.flip([1])
        padded_attention_masks = padded_attention_masks.flip([1])

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": padded_attention_masks,
    }


class BatchEval:
    """Handles batch evaluation of a model."""

    def __init__(
        self,
        cfg: DictDefault,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: IterableDataset,
    ) -> None:
        # pylint: disable=R0801
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def validate_and_warn(self) -> None:
        # TODO: finish validation

        if self.cfg.micro_batch_size is None:
            raise ValueError("micro_batch_size is required for batch_eval")

        if self.cfg.pretraining_dataset is not None:
            self.cfg.pretraining_dataset = None
            LOG.warning("batch_eval is forcing pretraining_dataset=None")

    def run(self) -> None:
        """Run batch evaluation and return average loss and perplexity."""

        # TODO: Do we need to make any changes to the dataset format for evaluation?
        # TODO: Lay rails for other validation metrics
        # TODO: Do we need to do something with the scores aside from logging to stdout?

        if self.cfg.seed is not None:
            transformers.enable_full_determinism(seed=self.cfg.seed)

        derived_micro_batch_size = (
            self.cfg.micro_batch_size if self.cfg.micro_batch_size is not None else 1
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=derived_micro_batch_size,
            collate_fn=lambda batch: collate_fn(
                batch,
                pad_value=self.tokenizer.pad_token_id,
                padding_direction=self.tokenizer.padding_side,
            ),
        )

        # Prepare model & dataset for distributed eval
        dataloader, model = self.accelerator.prepare(dataloader, self.model)
        model.eval()

        if self.accelerator.is_local_main_process:
            LOG.info(
                "Running batch evaluation on %i samples with derived_micro_batch_size of %i",
                len(self.dataset),
                derived_micro_batch_size,
            )

        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                with self.accelerator.split_between_processes(batch) as batch:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    outputs = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss_reduced = self.accelerator.gather(outputs.loss)
                    if self.accelerator.is_local_main_process:
                        total_loss += (
                            torch.mean(loss_reduced) / self.accelerator.num_processes
                        )

        # Only main process computes average loss and perplexity
        if self.accelerator.is_local_main_process:
            avg_loss = total_loss / len(dataloader)
            perplexity = torch.exp(avg_loss)
            LOG.info(
                "Batch evaluation completed. Average loss: %f, Perplexity: %f",
                avg_loss,
                perplexity,
            )

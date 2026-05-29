"""
Synthetic dataset generator for benchmarking and testing.

Generates datasets with configurable sequence length, dataset size, and token ID ranges.
Useful for benchmarking memory usage and speed by sequence length, and for validating
weighted dataset mixes.

YAML configuration example:

    datasets:
      - path: synthetic
        type: _synthetic
        length: 1000
        sequence_length: 2048
        min_input_id: 100
        max_input_id: 32000
        seed: 42
"""

from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset

from axolotl.prompt_tokenizers import DatasetWrappingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class SyntheticDatasetStrategy(DatasetWrappingStrategy):
    """Strategy that generates synthetic tokenized data, ignoring the source dataset."""

    def __init__(
        self,
        sequence_length: int = 2048,
        length: int = 1000,
        min_input_id: int = 100,
        max_input_id: int = 32000,
        seed: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.length = length
        self.min_input_id = min_input_id
        self.max_input_id = max_input_id
        self.seed = seed

    def wrap_dataset(
        self,
        dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        **kwargs,
    ) -> Dataset:
        LOG.info(
            f"Generating synthetic dataset: {self.length} samples, "
            f"sequence_length={self.sequence_length}, "
            f"input_id_range=[{self.min_input_id}, {self.max_input_id})"
        )

        rng = np.random.default_rng(self.seed)
        input_ids = rng.integers(
            low=self.min_input_id,
            high=self.max_input_id,
            size=(self.length, self.sequence_length),
        ).tolist()

        attention_mask = [[1] * self.sequence_length] * self.length
        # labels == input_ids means we train on all tokens
        labels = [row[:] for row in input_ids]

        return Dataset.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    ds_cfg = ds_cfg or {}

    sequence_length = ds_cfg.get("sequence_length", cfg.sequence_len)
    length = ds_cfg.get("length", 1000)
    min_input_id = ds_cfg.get("min_input_id", 100)
    max_input_id = ds_cfg.get("max_input_id", tokenizer.vocab_size)
    seed = ds_cfg.get("seed", None)

    return SyntheticDatasetStrategy(
        sequence_length=sequence_length,
        length=length,
        min_input_id=min_input_id,
        max_input_id=max_input_id,
        seed=seed,
    )

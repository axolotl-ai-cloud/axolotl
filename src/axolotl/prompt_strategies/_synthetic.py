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

To generate chat-shaped label masking, add these dataset options:

        min_turns: 2
        max_turns: 8
        min_turn_length: 32
        input_fraction: 0.2

Every sample fills sequence_length tokens. Each turn is an input/output pair.
Turn counts are sampled uniformly from min_turns to the smaller of max_turns and
the number of min_turn_length pairs that fit in the row.
The sequence is divided as evenly as possible among turns. An input_fraction of 0.2
masks approximately 20% of each turn and labels the remaining 80%. Lengths are
rounded and clamped to keep at least one input and one output token per turn,
and each pair has at least min_turn_length tokens (default 32). The sequence
must fit min_turns pairs. No chat special tokens are inserted. Defaults are
min_turns: 1, max_turns: 1, and input_fraction: 0 (all tokens labeled). For multiple
turns, max_turns must exceed min_turns and an omitted input_fraction becomes 0.25.
An explicit input_fraction: 0 always labels all tokens.
The default single-turn, unmasked mode does not enforce min_turn_length.
"""

from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset

from axolotl.prompt_tokenizers import DatasetWrappingStrategy
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.datasets import SyntheticDataset

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
        min_turns: int = 1,
        max_turns: int = 1,
        input_fraction: float | None = None,
        min_turn_length: int = 32,
    ):
        chat_config = SyntheticDataset(
            sequence_length=sequence_length,
            min_turns=min_turns,
            max_turns=max_turns,
            min_turn_length=min_turn_length,
            **(
                {"input_fraction": input_fraction} if input_fraction is not None else {}
            ),
        )
        self.sequence_length = sequence_length
        self.length = length
        self.min_input_id = min_input_id
        self.max_input_id = max_input_id
        self.seed = seed
        self.min_turns = chat_config.min_turns
        self.max_turns = chat_config.max_turns
        self.input_fraction = chat_config.input_fraction
        self.min_turn_length = max(
            chat_config.min_turn_length, 2 if self.input_fraction > 0 else 1
        )

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
        labels = [row[:] for row in input_ids]
        if self.input_fraction > 0:
            for row in labels:
                max_turns = min(self.max_turns, len(row) // self.min_turn_length)
                num_turns = int(rng.integers(self.min_turns, max_turns + 1))
                turn_length, remainder = divmod(len(row), num_turns)
                offset = 0
                for turn in range(num_turns):
                    length = turn_length + (turn < remainder)
                    input_length = max(
                        1, min(length - 1, round(length * self.input_fraction))
                    )
                    row[offset : offset + input_length] = [-100] * input_length
                    offset += length

        return Dataset.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    ds_cfg = ds_cfg or {}

    sequence_length = ds_cfg.get("sequence_length")
    if sequence_length is None:
        sequence_length = cfg.sequence_len
    length = ds_cfg.get("length", 1000)
    min_input_id = ds_cfg.get("min_input_id", 100)
    max_input_id = ds_cfg.get("max_input_id")
    if max_input_id is None:
        max_input_id = tokenizer.vocab_size
    seed = ds_cfg.get("seed", None)

    return SyntheticDatasetStrategy(
        sequence_length=sequence_length,
        length=length,
        min_input_id=min_input_id,
        max_input_id=max_input_id,
        seed=seed,
        min_turns=ds_cfg.get("min_turns", 1),
        max_turns=ds_cfg.get("max_turns", 1),
        input_fraction=ds_cfg.get("input_fraction"),
        min_turn_length=ds_cfg.get("min_turn_length", 32),
    )

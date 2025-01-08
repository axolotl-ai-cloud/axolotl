"""Dataset loading utilities."""

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, Union

from datasets import Dataset

from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs
from axolotl.utils.data import prepare_dataset
from axolotl.utils.data.rl import load_prepare_dpo_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_processor, load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels

LOG = logging.getLogger(__name__)


@dataclass
class TrainDatasetMeta:
    """Dataclass with fields for training and validation datasets and metadata."""

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


def sample_dataset(dataset: Dataset, num_samples: int):
    """
    Randomly sample `num_samples` samples from `dataset`.

    Args:
        dataset: Dataset.
        num_samples: Number of samples to return.

    Returns:
        Random sample (with replacement) of examples in `dataset`.
    """
    return dataset.select(
        [random.randrange(0, len(dataset) - 1) for _ in range(num_samples)]  # nosec
    )


def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: Union[PreprocessCliArgs, TrainerCliArgs],
) -> TrainDatasetMeta:
    """
    Loads one or more training or evaluation datasets, calling
    `axolotl.utils.data.prepare_dataset`. Optionally, logs out debug information.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Command-specific CLI arguments.

    Returns:
        Dataclass with fields for training and evaluation datasets and the computed
        `total_num_steps`.
    """
    tokenizer = load_tokenizer(cfg)
    processor = load_processor(cfg, tokenizer=tokenizer) if cfg.processor_type else None

    train_dataset, eval_dataset, total_num_steps, prompters = prepare_dataset(
        cfg,
        tokenizer,
        processor=processor,
    )

    if (
        cli_args.debug
        or cfg.debug
        or cli_args.debug_text_only
        or int(cli_args.debug_num_examples) > 0
    ):
        LOG.info("check_dataset_labels...")

        train_samples = sample_dataset(train_dataset, cli_args.debug_num_examples)
        check_dataset_labels(
            train_samples,
            tokenizer,
            num_examples=cli_args.debug_num_examples,
            text_only=cli_args.debug_text_only,
        )

        LOG.info("printing prompters...")
        for prompter in prompters:
            LOG.info(prompter)

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )


def load_rl_datasets(
    *,
    cfg: DictDefault,
    cli_args: Union[
        PreprocessCliArgs, TrainerCliArgs
    ],  # pylint: disable=unused-argument
) -> TrainDatasetMeta:
    """
    Loads one or more training or evaluation datasets for RL training, calling
    `axolotl.utils.data.rl.load_prepare_dpo_datasets`. Optionally, logs out debug
    information.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Command-specific CLI arguments.

    Returns:
        Dataclass with fields for training and evaluation datasets and the computed
        `total_num_steps`.
    """
    train_dataset, eval_dataset = load_prepare_dpo_datasets(cfg)
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )

    if cli_args.debug or cfg.debug:
        LOG.info("check_dataset_labels...")

        tokenizer = load_tokenizer(cfg)
        train_samples = sample_dataset(train_dataset, cli_args.debug_num_examples)
        check_dataset_labels(
            train_samples,
            tokenizer,
            num_examples=cli_args.debug_num_examples,
            text_only=cli_args.debug_text_only,
            rl_mode=True,
        )

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )

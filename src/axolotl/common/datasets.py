"""Dataset loading utilities."""

import math
import random
from dataclasses import dataclass

from datasets import Dataset

import axolotl.monkeypatch.data.batch_dataset_fetcher  # noqa: F401
from axolotl.cli.args import PreprocessCliArgs, TrainerCliArgs
from axolotl.loaders import load_processor, load_tokenizer
from axolotl.telemetry.errors import send_errors
from axolotl.utils.data import prepare_datasets, prepare_preference_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RLType
from axolotl.utils.tokenization import check_dataset_labels

LOG = get_logger(__name__)


@dataclass
class TrainDatasetMeta:
    """Dataclass with fields for training and validation datasets and metadata."""

    train_dataset: Dataset
    eval_dataset: Dataset | None = None
    total_num_steps: int | None = None


def sample_dataset(dataset: Dataset, num_samples: int) -> Dataset:
    """Randomly sample `num_samples` samples with replacement from `dataset`."""
    return dataset.select(
        [random.randrange(0, len(dataset) - 1) for _ in range(num_samples)]  # nosec
    )


@send_errors
def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: PreprocessCliArgs | TrainerCliArgs | None = None,
    debug: bool = False,
) -> TrainDatasetMeta:
    """Loads one or more training or evaluation datasets, calling
    `axolotl.utils.data.prepare_datasets`. Optionally, logs out debug information.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Command-specific CLI arguments.
        debug: Whether to print out tokenization of sample. This is duplicated in
            `cfg` and `cli_args`, but is kept due to use in our Colab notebooks.

    Returns:
        Dataclass with fields for training and evaluation datasets and the computed
            `total_num_steps`.
    """
    tokenizer = load_tokenizer(cfg)
    processor = load_processor(cfg, tokenizer=tokenizer) if cfg.processor_type else None

    train_dataset, eval_dataset, total_num_steps, prompters = prepare_datasets(
        cfg,
        tokenizer,
        processor=processor,
    )

    if (
        cfg.debug
        or getattr(cli_args, "debug", False)
        or getattr(cli_args, "debug_text_only", False)
        or getattr(cli_args, "debug_num_examples", 0) > 0
        or debug
    ):
        LOG.info("check_dataset_labels...")

        num_examples = cli_args.debug_num_examples if cli_args else 1
        text_only = cli_args.debug_text_only if cli_args else False
        try:
            train_samples = sample_dataset(train_dataset, num_examples)
            check_dataset_labels(
                train_samples,
                tokenizer,
                num_examples=num_examples,
                text_only=text_only,
            )
        except AttributeError:
            # can't sample iterable datasets
            pass

        LOG.info("printing prompters...")
        for prompter in prompters:
            LOG.info(prompter)

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )


@send_errors
def load_preference_datasets(
    *, cfg: DictDefault, cli_args: PreprocessCliArgs | TrainerCliArgs | None = None
) -> TrainDatasetMeta:
    """Loads one or more training or evaluation datasets for RL training using paired
    preference data, calling `axolotl.utils.data.rl.prepare_preference_datasets`.
    Optionally, logs out debug information.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Command-specific CLI arguments.

    Returns:
        Dataclass with fields for training and evaluation datasets and the computed
        `total_num_steps`.
    """
    tokenizer = load_tokenizer(cfg)
    train_dataset, eval_dataset = prepare_preference_datasets(cfg, tokenizer)

    total_num_steps: int | None = None
    if cfg.rl is not RLType.GRPO:
        total_num_steps = int(
            math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
        )

    if ((cli_args and cli_args.debug) or cfg.debug) and cfg.rl != RLType.ORPO:
        LOG.info("check_dataset_labels...")

        num_examples = cli_args.debug_num_examples if cli_args else 1
        text_only = cli_args.debug_text_only if cli_args else False

        tokenizer = load_tokenizer(cfg)
        train_samples = sample_dataset(train_dataset, num_examples)
        check_dataset_labels(
            dataset=train_samples,
            tokenizer=tokenizer,
            num_examples=num_examples,
            text_only=text_only,
            rl_mode=True,
        )

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )

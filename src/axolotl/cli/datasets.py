"""Dataset loading utilities."""
import logging
import math
import random

from axolotl.common.cli import TrainerCliArgs
from axolotl.train import TrainDatasetMeta
from axolotl.utils.data import prepare_dataset
from axolotl.utils.data.rl import load_prepare_dpo_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_processor, load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels

LOG = logging.getLogger(__name__)


def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
) -> TrainDatasetMeta:
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
        check_dataset_labels(
            train_dataset.select(
                [
                    random.randrange(0, len(train_dataset) - 1)  # nosec
                    for _ in range(cli_args.debug_num_examples)
                ]
            ),
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
    cli_args: TrainerCliArgs,  # pylint: disable=unused-argument
) -> TrainDatasetMeta:
    train_dataset, eval_dataset = load_prepare_dpo_datasets(cfg)
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )

    if cli_args.debug or cfg.debug:
        LOG.info("check_dataset_labels...")

        tokenizer = load_tokenizer(cfg)
        check_dataset_labels(
            train_dataset.select(
                [
                    random.randrange(0, len(train_dataset) - 1)  # nosec
                    for _ in range(cli_args.debug_num_examples)
                ]
            ),
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

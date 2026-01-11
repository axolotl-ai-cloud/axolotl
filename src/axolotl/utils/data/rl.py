"""Data handling specific to RL trainers."""

import inspect
from functools import partial
from typing import Any, Callable, Literal

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from axolotl.loaders import load_tokenizer
from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.orpo import load as load_orpo
from axolotl.utils.data.lock import FileLockLoader
from axolotl.utils.data.shared import (
    create_train_validation_split,
    datasets_with_name_generator,
    generate_dataset_hash_from_config,
    load_dataset_with_config,
    load_preprocessed_dataset,
    merge_datasets,
    save_preprocessed_dataset,
    try_load_from_hub,
)
from axolotl.utils.data.utils import (
    deduplicate_and_log_datasets,
    retry_on_request_exceptions,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RLType

LOG = get_logger(__name__)


@retry_on_request_exceptions(max_retries=3, delay=5)
def prepare_preference_datasets(
    cfg: DictDefault, tokenizer: PreTrainedTokenizer
) -> tuple[Dataset, Dataset | None]:
    """Load and prepare preference datasets for RL training.

    Loads training and evaluation datasets, handling preprocessing, caching, and
    deduplication as configured. Uses FileLock for distributed coordination.

    Args:
        cfg: Configuration object containing dataset and training settings.
        tokenizer: Tokenizer to use for processing text.

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset may be None
            if no evaluation dataset is configured.
    """

    def _load_datasets():
        # Load training dataset
        train_dataset = _load_or_create_dataset_split(cfg, tokenizer, split="train")

        # Load or create evaluation dataset
        eval_dataset: Dataset | None = None
        if cfg.test_datasets:
            eval_dataset = _load_or_create_dataset_split(cfg, tokenizer, split="test")
        elif cfg.val_set_size:
            # Create validation split from training data
            train_dataset, eval_dataset = create_train_validation_split(
                train_dataset, cfg, cfg.val_set_size
            )

        return train_dataset, eval_dataset

    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        train_dataset, eval_dataset = loader.load(_load_datasets)
    finally:
        loader.cleanup()

    # Apply deduplication if configured
    if cfg.dataset_exact_deduplication:
        train_dataset, eval_dataset = deduplicate_and_log_datasets(
            dataset=train_dataset, other_dataset=eval_dataset
        )

    return train_dataset, eval_dataset


def _map_dataset(
    cfg: DictDefault,
    dataset: Dataset | DatasetDict,
    ds_transform_fn: Callable[..., Any],
    tokenizer: Any | None = None,
    **map_kwargs: Any,
) -> Dataset:
    """Apply transformation function to dataset.

    Args:
        cfg: Configuration object.
        dataset: Dataset to transform.
        ds_transform_fn: Transformation function to apply.
        tokenizer: Optional tokenizer for transformation.
        **map_kwargs: Additional arguments for dataset mapping.

    Returns:
        Transformed dataset.
    """
    sig = inspect.signature(ds_transform_fn)
    if "tokenizer" in sig.parameters:
        if not tokenizer:
            tokenizer = load_tokenizer(cfg)
        ds_transform_fn = partial(ds_transform_fn, tokenizer=tokenizer)

    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    dataset = dataset.map(
        ds_transform_fn,
        num_proc=cfg.dataset_num_proc,
        load_from_cache_file=not cfg.is_preprocess,
        desc="Mapping RL Dataset",
        **map_kwargs,
    )

    return dataset


def _drop_long_sequences(
    sample: dict[str, Any], rl: RLType, tokenizer: Any, sequence_len: int
) -> bool:
    """Filter out samples that exceed maximum sequence length.

    Args:
        sample: Dataset sample to check.
        rl: Reinforcement learning type.
        tokenizer: Tokenizer for length calculation.
        sequence_len: Maximum allowed sequence length.

    Returns:
        True if sample should be kept, False if it should be dropped.

    Raises:
        ValueError: If required keys are missing or RL type is unknown.
    """
    if rl in {RLType.DPO, RLType.IPO, RLType.ORPO, RLType.SIMPO}:
        if not (
            sample.get("prompt") and sample.get("chosen") and sample.get("rejected")
        ):
            raise ValueError(
                "Prompt, chosen and rejected keys are required for DPO/ORPO datasets"
            )

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        len_chosen = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        len_rejected = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])

        return (len_prompt + len_chosen) <= sequence_len and (
            len_prompt + len_rejected
        ) <= sequence_len

    if rl is RLType.KTO:
        if not (sample.get("prompt") and sample.get("completion")):
            raise ValueError("Prompt and completion keys are required for KTO datasets")

        prompt = sample["prompt"]
        completion = sample["completion"]

        len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        len_completion = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )

        return (len_prompt + len_completion) <= sequence_len

    if rl in {RLType.GRPO, RLType.GDPO}:
        return True

    raise ValueError("Unknown RL type")


def _load_split(cfg: DictDefault, split: Literal["train", "test"]) -> Dataset:
    """Load and process dataset split for RL training.

    Args:
        cfg: Configuration object containing dataset settings.
        split: Dataset split to load ("train" or "test").

    Returns:
        Combined and processed dataset for the specified split.
    """
    datasets_configs = cfg.datasets if split == "train" else cfg.test_datasets
    split_datasets: list[Dataset | DatasetDict] = []

    for dataset_config in datasets_with_name_generator(datasets_configs):
        dataset: Dataset | DatasetDict = load_dataset_with_config(
            dataset_config, cfg.hf_use_auth_token, streaming=False
        )
        split_datasets.append(dataset)

    tokenizer = load_tokenizer(cfg)

    for i, dataset in enumerate(split_datasets):
        _type = datasets_configs[i]["type"]
        if _type:
            if isinstance(_type, DictDefault):
                _type = "user_defined.default"
            if cfg.rl is RLType.ORPO:
                ds_transform_fn = load_orpo(_type, cfg, dataset_idx=i)
            elif cfg.rl is RLType.KTO:
                ds_transform_fn = load_kto(_type, cfg, dataset_idx=i)
            else:
                ds_transform_fn = load_dpo(_type, cfg, dataset_idx=i)

            map_kwargs: dict[str, Any] = {}
            if isinstance(ds_transform_fn, tuple):
                ds_transform_fn, map_kwargs = ds_transform_fn
            split_datasets[i] = _map_dataset(
                cfg, dataset, ds_transform_fn, tokenizer, **map_kwargs
            )
        else:
            # If no `type` is provided, assume the dataset is already in the expected format with
            # "prompt", "chosen", and "rejected" already preprocessed
            split_datasets[i] = dataset

        if not cfg.skip_prepare_dataset:
            drop_long = partial(
                _drop_long_sequences,
                rl=cfg.rl,
                tokenizer=tokenizer,
                sequence_len=cfg.sequence_len,
            )

            prior_len = len(split_datasets[i])
            split_datasets[i] = split_datasets[i].filter(
                drop_long,
                num_proc=cfg.dataset_num_proc,
                load_from_cache_file=not cfg.is_preprocess,
                desc="Dropping Long Sequences",
            )
            dropped = prior_len - len(split_datasets[i])
            if dropped:
                LOG.warning(f"Dropped {dropped} long samples from dataset index {i}")

    # Merge datasets
    dataset = merge_datasets(split_datasets, cfg)

    if not cfg.skip_prepare_dataset:
        # Save preprocessed dataset
        dataset_hash = generate_dataset_hash_from_config(
            cfg, datasets_configs, tokenizer.name_or_path
        )
        save_preprocessed_dataset(cfg, dataset, dataset_hash, split)

    return dataset


def _load_or_create_dataset_split(
    cfg: DictDefault, tokenizer: PreTrainedTokenizer, split: Literal["train", "test"]
) -> Dataset:
    """Load preprocessed dataset or create new one for given split.

    Args:
        cfg: Configuration object.
        tokenizer: Tokenizer to use for processing text.
        split: Dataset split to load.

    Returns:
        Tuple of (dataset, is_preprocessed).
    """
    # Select correct dataset configuration based on split
    datasets_config = cfg.datasets if split == "train" else cfg.test_datasets

    # Generate dataset hash for caching
    dataset_hash = generate_dataset_hash_from_config(
        cfg, datasets_config, tokenizer.name_or_path
    )

    # Try loading from hub if push_dataset_to_hub is configured
    dataset = None
    if cfg.push_dataset_to_hub:
        dataset = try_load_from_hub(cfg, dataset_hash, split)

    # Attempt to load preprocessed dataset
    if dataset is None:
        dataset = load_preprocessed_dataset(cfg, dataset_hash)

    # Otherwise, load it
    if dataset is None:
        dataset = _load_split(cfg, split=split)

    return dataset

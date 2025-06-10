"""Data handling specific to SFT."""

import functools
import tempfile
from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    load_dataset,
)
from transformers import PreTrainedTokenizer, ProcessorMixin

from axolotl.prompters import Prompter
from axolotl.utils.data.lock import FileLockLoader
from axolotl.utils.data.pretraining import wrap_pretraining_dataset
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
    drop_long_seq_in_dataset,
    retry_on_request_exceptions,
)
from axolotl.utils.data.wrappers import get_dataset_wrapper
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_local_main_process
from axolotl.utils.logging import get_logger
from axolotl.utils.trainer import (
    calculate_total_num_steps,
    process_datasets_for_packing,
)

LOG = get_logger(__name__)


@retry_on_request_exceptions(max_retries=3, delay=5)
def prepare_datasets(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None = None,
    preprocess_iterable: bool = False,
) -> tuple[IterableDataset | Dataset, Dataset | None, int, list[Prompter | None]]:
    """Prepare training and evaluation datasets based on configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        tokenizer: Tokenizer to use for processing text.
        processor: Optional processor for multimodal datasets.
        preprocess_iterable: Whether to use iterable preprocessing.

    Returns:
        Tuple of (train_dataset, eval_dataset, total_steps, prompters).
    """
    if cfg.pretraining_dataset:
        return _prepare_pretraining_dataset(
            cfg, tokenizer, processor, preprocess_iterable
        )
    return _prepare_standard_dataset(cfg, tokenizer, processor, preprocess_iterable)


def _prepare_standard_dataset(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None,
    preprocess_iterable: bool,
) -> tuple[Dataset, Dataset | None, int, list[Prompter | None]]:
    """Prepare standard (non-pretraining) datasets."""

    def _load_datasets():
        # Always load training dataset
        train_dataset, eval_dataset, prompters = _load_and_prepare_datasets(
            tokenizer,
            cfg,
            split="train",
            processor=processor,
            preprocess_iterable=preprocess_iterable,
        )

        # Overwrite eval_dataset if test data exists
        if cfg.test_datasets:
            _, eval_dataset, _ = _load_and_prepare_datasets(
                tokenizer,
                cfg,
                split="test",
                processor=processor,
                preprocess_iterable=preprocess_iterable,
            )

        return train_dataset, eval_dataset, prompters

    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        train_dataset, eval_dataset, prompters = loader.load(_load_datasets)
    finally:
        loader.cleanup()

    # Validate sample packing configuration for evaluation
    if eval_dataset and cfg.sample_packing and cfg.eval_sample_packing is not False:
        total_eval_steps = calculate_total_num_steps(cfg, eval_dataset, update=False)
        if total_eval_steps == 0:
            raise ValueError(
                "eval dataset split is too small for sample_packing. "
                "You should set `eval_sample_packing: False` in your config."
            )

    # Calculate total number of training steps
    if cfg.max_steps:
        total_num_steps = min(
            calculate_total_num_steps(cfg, train_dataset), cfg.max_steps
        )
    else:
        total_num_steps = calculate_total_num_steps(cfg, train_dataset)
    LOG.info(f"Maximum number of steps set at {total_num_steps}")
    return train_dataset, eval_dataset, total_num_steps, prompters


def _prepare_pretraining_dataset(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None,
    preprocess_iterable: bool,
) -> tuple[IterableDataset, Dataset | None, int, list[Prompter | None]]:
    """
    Prepare dataset for pretraining mode.

    Note: Pre-training datasets are streamed from the HuggingFace Hub.
    """
    # Extract pretraining dataset configuration
    pretraining_config = _extract_pretraining_config(cfg)

    # Load streaming dataset for training
    train_dataset = _load_pretraining_dataset(pretraining_config, cfg, tokenizer)

    # Load evaluation dataset if specified
    eval_dataset = None
    if cfg.test_datasets:
        _, eval_dataset, _ = _load_and_prepare_datasets(
            tokenizer,
            cfg,
            split="test",
            processor=processor,
            preprocess_iterable=preprocess_iterable,
        )

    if cfg.dataset_exact_deduplication:
        LOG.info("Deduplication not available for pretrained datasets")

    # For pretraining, we return max_steps directly from config
    return train_dataset, eval_dataset, cfg.max_steps, []


def _extract_pretraining_config(cfg: DictDefault) -> DictDefault:
    """Extract pretraining configuration from the main config."""
    if isinstance(cfg.pretraining_dataset, list) and isinstance(
        cfg.pretraining_dataset[0], dict
    ):
        config = cfg.pretraining_dataset[0]
        return DictDefault(
            {
                "path": config["path"],
                "name": config["name"],
                "skip": config["skip"],
                "split": config.get("split", "train"),
                "data_files": config.get("data_files"),
                "type": config.get("type", "pretrain"),
            }
        )
    # Simple string path case
    return DictDefault(
        {
            "path": cfg.pretraining_dataset,
            "name": None,
            "skip": 0,
            "split": "train",
            "data_files": None,
            "type": "pretrain",
        }
    )


def _load_pretraining_dataset(
    pretraining_config: DictDefault, cfg: DictDefault, tokenizer: PreTrainedTokenizer
) -> IterableDataset:
    """Load and prepare a streaming dataset for pretraining."""
    # Create dataset wrapper partial function
    dataset_wrapper_partial = functools.partial(
        get_dataset_wrapper,
        dataset_config=pretraining_config,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset_base_type=pretraining_config["type"],
    )

    # Load the actual dataset
    if (
        cfg.accelerator_config
        and cfg.accelerator_config.dispatch_batches
        and not is_local_main_process()
    ):
        iter_dataset = _create_placeholder_dataset()
    else:
        iter_dataset = load_dataset(
            pretraining_config["path"],
            streaming=True,
            split=pretraining_config["split"],
            name=pretraining_config["name"],
            data_files=pretraining_config["data_files"],
        )

    # Apply skip if specified
    if pretraining_config["skip"]:
        LOG.info(f"Skipping {pretraining_config['skip']} samples from the dataset")
        iter_dataset = iter_dataset.skip(pretraining_config["skip"])

    # Wrap the dataset for pretraining
    train_dataset = wrap_pretraining_dataset(
        iter_dataset,
        tokenizer,
        cfg,
        dataset_wrapper_partial,
        max_tokens=cfg.sequence_len,
        batch_size=cfg.micro_batch_size,
        seed=cfg.seed,
        buffer_size=cfg.pretrain_multipack_buffer_size or 10_000,
    )

    # Format for PyTorch
    return train_dataset.with_format("torch")


def _create_placeholder_dataset() -> IterableDataset:
    """Create a minimal placeholder dataset for non-main processes."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("text\n")
        f.write("lorem ipsum dolor sit amet\n")
        f.seek(0)
        return load_dataset("csv", data_files=f.name, split="train", streaming=True)


def _load_tokenized_prepared_datasets(
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    split: Literal["train", "test"] = "train",
    processor: ProcessorMixin | None = None,
    preprocess_iterable: bool = False,
) -> tuple[Dataset | DatasetDict, list[Prompter | None]]:
    """Load or create tokenized and prepared datasets for training or testing.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.
        preprocess_iterable: Whether to use iterable preprocessing.

    Returns:
        Tuple of (dataset, prompters list).
    """
    # Select correct dataset configuration based on split
    datasets_configs = cfg.datasets if split == "train" else cfg.test_datasets

    # Generate dataset hash for caching
    dataset_hash = generate_dataset_hash_from_config(
        cfg, datasets_configs, tokenizer.name_or_path
    )

    # Try loading from hub if push_dataset_to_hub is configured
    dataset = None
    if cfg.push_dataset_to_hub:
        dataset = try_load_from_hub(cfg, dataset_hash, split)

    # If not found on hub, try loading from disk
    if dataset is None:
        dataset = load_preprocessed_dataset(cfg, dataset_hash)

    # If not found on disk or skipping prepared dataset, load and process raw datasets
    prompters: list[Prompter | None] = []
    if dataset is None:
        dataset, prompters = _load_raw_datasets(
            cfg,
            datasets_configs,
            tokenizer,
            split,
            processor,
            preprocess_iterable,
        )

    return dataset, prompters


def _load_raw_datasets(
    cfg: DictDefault,
    datasets_configs: list,
    tokenizer: PreTrainedTokenizer,
    split: str,
    processor: ProcessorMixin | None = None,
    preprocess_iterable: bool = False,
) -> tuple[Dataset, list[Prompter | None]]:
    """Load, process, merge, and save raw datasets."""
    LOG.info("Loading raw datasets...", main_process_only=False)
    if not cfg.is_preprocess:
        LOG.warning(
            "Processing datasets during training can lead to VRAM instability. Please "
            "pre-process your dataset using `axolotl preprocess path/to/config.yml`."
        )

    # Load and process individual datasets
    datasets = []
    prompters = []
    for dataset_config in datasets_with_name_generator(datasets_configs):
        dataset_wrapper, dataset_prompter = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=tokenizer,
            split=split,
            seed=cfg.seed,
            processor=processor,
            preprocess_iterable=preprocess_iterable,
        )
        datasets.append(dataset_wrapper)
        prompters.append(dataset_prompter)

    # Merge datasets
    dataset = merge_datasets(datasets, cfg)

    if not cfg.skip_prepare_dataset:
        dataset = drop_long_seq_in_dataset(dataset, cfg)
        if cfg.sample_packing:
            dataset, _ = process_datasets_for_packing(cfg, dataset, None)

        # Save the prepared dataset
        dataset_hash = generate_dataset_hash_from_config(
            cfg, datasets_configs, tokenizer.name_or_path
        )
        save_preprocessed_dataset(cfg, dataset, dataset_hash, split)

    return dataset, prompters


def _load_and_process_single_dataset(
    dataset_config: DictDefault,
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    split: str,
    seed: int,
    processor: ProcessorMixin | None = None,
    preprocess_iterable: bool = False,
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Load and process a single dataset based on the passed config."""
    # Load the dataset
    dataset = load_dataset_with_config(
        dataset_config, cfg.hf_use_auth_token, streaming=preprocess_iterable
    )

    # Parse dataset type
    d_base_type, d_prompt_style = _parse_dataset_type(dataset_config.type)

    # Select the appropriate split
    if isinstance(dataset, DatasetDict):
        if dataset_config.split and dataset_config.split in dataset:
            dataset = dataset[dataset_config.split]
        elif split in dataset:
            dataset = dataset[split]
        else:
            raise ValueError(
                f"no {split} split found for dataset {dataset_config.path}, you may "
                "specify a split with 'split: ...'"
            )

    # Apply sharding if configured
    if dataset_config.shards:
        shards_idx = dataset_config.get("shards_idx", 0)
        dataset = dataset.shuffle(seed=seed).shard(
            num_shards=dataset_config.shards, index=shards_idx
        )

    # Apply dataset wrapper
    dataset_wrapper, dataset_prompter = get_dataset_wrapper(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset_base_type=d_base_type,
        dataset=dataset,
        dataset_prompt_style=d_prompt_style,
        processor=processor,
    )

    return dataset_wrapper, dataset_prompter


def _parse_dataset_type(d_type: str) -> tuple[str | None, str | None]:
    """Parse the dataset type string into base type and prompt style."""
    if not isinstance(d_type, str):
        return None, None

    d_type_split = d_type.split(":")
    d_base_type = d_type_split[0]
    d_prompt_style = d_type_split[1] if len(d_type_split) > 1 else None

    return d_base_type, d_prompt_style


def _handle_train_dataset_split(
    dataset: Dataset, cfg: DictDefault
) -> tuple[Dataset, Dataset | None]:
    """Handle processing for train split, including validation set creation."""
    val_set_size = (
        int(cfg.val_set_size) if cfg.val_set_size > 1 else float(cfg.val_set_size)
    )

    if val_set_size:
        # Create train/validation split
        train_dataset, eval_dataset = create_train_validation_split(
            dataset, cfg, val_set_size
        )
        return train_dataset, eval_dataset

    # No validation split - apply deduplication if needed and return as train dataset
    if cfg.dataset_exact_deduplication:
        train_dataset, _ = deduplicate_and_log_datasets(dataset=dataset)
    else:
        train_dataset = dataset

    return train_dataset, None


def _handle_test_dataset_split(
    dataset: Dataset, cfg: DictDefault
) -> tuple[None, Dataset | None]:
    """Handle processing for test split."""
    if cfg.dataset_exact_deduplication:
        eval_dataset, _ = deduplicate_and_log_datasets(dataset=dataset)
    else:
        eval_dataset = dataset

    return None, eval_dataset


def _apply_dataset_sharding(dataset: Dataset, cfg: DictDefault) -> Dataset:
    """Apply dataset sharding if configured.

    Args:
        dataset: Dataset to shard.
        cfg: Configuration object containing shard settings.

    Returns:
        Sharded dataset or original dataset if no sharding configured.
    """
    if cfg.dataset_shard_num and cfg.dataset_shard_idx is not None:
        LOG.info(
            f"Using index #{cfg.dataset_shard_idx} of {cfg.dataset_shard_num} shards"
        )
        dataset = dataset.shard(
            num_shards=cfg.dataset_shard_num,
            index=cfg.dataset_shard_idx,
        )
    return dataset


def _load_and_prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    split: Literal["train", "test"] = "train",
    processor: ProcessorMixin | None = None,
    preprocess_iterable: bool = False,
) -> tuple[Dataset | None, Dataset | None, list[Prompter | None]]:
    """Load and prepare datasets with optional validation split and sharding.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.
        preprocess_iterable: Whether to use iterable preprocessing.

    Returns:
        Tuple of (train_dataset, eval_dataset, prompters).
    """
    # Load the base dataset
    dataset, prompters = _load_tokenized_prepared_datasets(
        tokenizer,
        cfg,
        split=split,
        processor=processor,
        preprocess_iterable=preprocess_iterable,
    )

    # Apply dataset sharding if configured using shared function
    dataset = _apply_dataset_sharding(dataset, cfg)

    # Apply deduplication and create train / validation splits based on the split type
    if split == "train":
        train_dataset, eval_dataset = _handle_train_dataset_split(dataset, cfg)
    else:
        train_dataset, eval_dataset = _handle_test_dataset_split(dataset, cfg)

    return train_dataset, eval_dataset, prompters

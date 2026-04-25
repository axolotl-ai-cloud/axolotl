"""Data handling specific to SFT."""

import functools
import os
import tempfile
from typing import Literal

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)
from transformers import PreTrainedTokenizer, ProcessorMixin

from axolotl.prompters import Prompter
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
from axolotl.utils.data.streaming import wrap_streaming_dataset
from axolotl.utils.data.utils import (
    deduplicate_and_log_datasets,
    handle_long_seq_in_dataset,
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
) -> tuple[IterableDataset | Dataset, Dataset | None, int, list[Prompter | None]]:
    """Prepare training and evaluation datasets based on configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        tokenizer: Tokenizer to use for processing text.
        processor: Optional processor for multimodal datasets.

    Returns:
        Tuple of (train_dataset, eval_dataset, total_steps, prompters).
    """
    if cfg.streaming or cfg.pretraining_dataset:
        return _prepare_streaming_dataset(cfg, tokenizer, processor)
    return _prepare_standard_dataset(cfg, tokenizer, processor)


def _prepare_standard_dataset(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None,
) -> tuple[Dataset, Dataset | None, int, list[Prompter | None]]:
    """Prepare standard (non-pretraining) datasets."""

    def _load_datasets():
        # Always load training dataset
        train_dataset, eval_dataset, prompters = _load_and_prepare_datasets(
            tokenizer,
            cfg,
            split="train",
            processor=processor,
        )

        # Overwrite eval_dataset if test data exists
        if cfg.test_datasets:
            _, eval_dataset, _ = _load_and_prepare_datasets(
                tokenizer,
                cfg,
                split="test",
                processor=processor,
            )

        return train_dataset, eval_dataset, prompters

    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        train_dataset, eval_dataset, prompters = loader.load(_load_datasets)
    finally:
        loader.cleanup()

    if os.environ.get("AXOLOTL_IS_PREPROCESS") == "1":
        return train_dataset, eval_dataset, -1, prompters

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


def _prepare_streaming_dataset(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None,
) -> tuple[IterableDataset, Dataset | None, int, list[Prompter | None]]:
    """
    Prepare dataset for streaming mode.

    Note: Streaming datasets are loaded incrementally from the source.
    """
    if cfg.pretraining_dataset:
        dataset_config = _extract_pretraining_config(cfg)
        train_dataset = _load_streaming_dataset(
            dataset_config, cfg, tokenizer, processor=processor
        )
    elif cfg.sample_packing:
        # TODO(djsaunde): Implement for multiple datasets
        dataset_config = DictDefault(cfg.datasets[0])

        # Ensure we have a split set - default to 'train' if not specified
        if not hasattr(dataset_config, "split") or not dataset_config.split:
            dataset_config.split = "train"
        train_dataset = _load_streaming_dataset(
            dataset_config, cfg, tokenizer, processor=processor
        )
    else:
        # Use legacy loading function for non-packed streaming datasets
        train_dataset, eval_dataset, prompters = _load_and_prepare_datasets(
            tokenizer,
            cfg,
            split="train",
            processor=processor,
            streaming=True,
        )

        # Return early for non-packed streaming datasets
        total_num_steps = cfg.max_steps if cfg.max_steps else -1
        return train_dataset, eval_dataset, total_num_steps, prompters

    # Load evaluation dataset if specified
    eval_dataset = None
    if cfg.test_datasets:
        test_dicts = [t if isinstance(t, dict) else dict(t) for t in cfg.test_datasets]
        is_mm_cpt_eval = any(
            t.get("type") == "multimodal_pretrain" or bool(t.get("multimodal"))
            for t in test_dicts
        )
        if is_mm_cpt_eval:
            # Modality homogeneity is enforced by check_multimodal_cpt at config
            # parse time; every entry here is guaranteed to be MM.
            eval_streams = [
                _load_streaming_dataset(
                    _pretraining_config_from_entry(entry),
                    cfg,
                    tokenizer,
                    processor=processor,
                    is_eval=True,
                )
                for entry in test_dicts
            ]
            eval_dataset = (
                eval_streams[0]
                if len(eval_streams) == 1
                else concatenate_datasets(eval_streams)
            )
        else:
            _, eval_dataset, _ = _load_and_prepare_datasets(
                tokenizer,
                cfg,
                split="test",
                processor=processor,
                streaming=False,
            )

    # For streaming, we return max_steps directly from config or -1 if not set
    total_num_steps = cfg.max_steps if cfg.max_steps else -1
    return train_dataset, eval_dataset, total_num_steps, []


def _pretraining_config_from_entry(entry: dict) -> DictDefault:
    return DictDefault(
        {
            "path": entry["path"],
            "name": entry.get("name"),
            "skip": entry.get("skip"),
            "split": entry.get("split", "train"),
            "data_files": entry.get("data_files"),
            "ds_type": entry.get("ds_type"),
            "type": entry.get("type", "pretrain"),
            "text_column": entry.get("text_column", "text"),
            "multimodal": entry.get("multimodal"),
            "image_column": entry.get("image_column", "images"),
            "image_base_dir": entry.get("image_base_dir"),
            "image_token": entry.get("image_token"),
            "trust_remote_code": entry.get("trust_remote_code", False),
        }
    )


def _extract_pretraining_config(cfg: DictDefault) -> DictDefault:
    """Extract pretraining configuration from the main config."""
    if isinstance(cfg.pretraining_dataset, list) and isinstance(
        cfg.pretraining_dataset[0], dict
    ):
        return _pretraining_config_from_entry(cfg.pretraining_dataset[0])
    # Simple string path case
    return DictDefault(
        {
            "path": cfg.pretraining_dataset,
            "name": None,
            "skip": 0,
            "split": "train",
            "data_files": None,
            "ds_type": None,
            "type": "pretrain",
            "text_column": "text",
            "multimodal": None,
            "image_column": "images",
            "image_base_dir": None,
            "image_token": None,  # nosec
            "trust_remote_code": False,
        }
    )


def _load_streaming_dataset(
    pretraining_config: DictDefault,
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin | None = None,
    is_eval: bool = False,
) -> IterableDataset:
    """Load and prepare a streaming dataset for pretraining."""
    # Create dataset wrapper partial function
    dataset_wrapper_partial = functools.partial(
        get_dataset_wrapper,
        dataset_config=pretraining_config,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset_base_type=pretraining_config["type"],
        processor=processor,
    )

    # Load the actual dataset
    if (
        cfg.accelerator_config
        and cfg.accelerator_config.dispatch_batches
        and not is_local_main_process()
    ):
        iter_dataset = _create_placeholder_dataset(pretraining_config)
    else:
        ds_type = pretraining_config.get("ds_type")
        if ds_type:
            # ds_type names the loader (e.g. 'json'); path is the data_files glob.
            iter_dataset = load_dataset(
                ds_type,
                streaming=True,
                split=pretraining_config["split"],
                name=pretraining_config["name"],
                data_files=(
                    pretraining_config["data_files"] or pretraining_config["path"]
                ),
                trust_remote_code=pretraining_config.get("trust_remote_code", False),
            )
        else:
            iter_dataset = load_dataset(
                pretraining_config["path"],
                streaming=True,
                split=pretraining_config["split"],
                name=pretraining_config["name"],
                data_files=pretraining_config["data_files"],
                trust_remote_code=pretraining_config.get("trust_remote_code", False),
            )

    # Apply skip if specified
    if pretraining_config["skip"]:
        LOG.info(f"Skipping {pretraining_config['skip']} samples from the dataset")
        iter_dataset = iter_dataset.skip(pretraining_config["skip"])

    # Wrap the dataset for pretraining
    train_dataset = wrap_streaming_dataset(
        iter_dataset,
        tokenizer,
        cfg,
        dataset_wrapper_partial,
        processor=processor,
        pretraining_config=pretraining_config,
        is_eval=is_eval,
    )

    # Format for PyTorch
    return train_dataset.with_format("torch")


def _create_placeholder_dataset(
    pretraining_config: DictDefault | None = None,
) -> IterableDataset:
    """Create a minimal placeholder dataset for non-main processes."""
    text_column = "text"
    image_column: str | None = None
    if pretraining_config is not None:
        text_column = pretraining_config.get("text_column") or "text"
        is_mm = pretraining_config.get("type") == "multimodal_pretrain" or bool(
            pretraining_config.get("multimodal")
        )
        if is_mm:
            image_column = pretraining_config.get("image_column") or "images"

    if image_column is None:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            f.write(f"{text_column}\n")
            f.write("lorem ipsum dolor sit amet\n")
            f.seek(0)
            return load_dataset("csv", data_files=f.name, split="train", streaming=True)

    def _gen():
        yield {text_column: "lorem ipsum dolor sit amet", image_column: []}

    return IterableDataset.from_generator(_gen)


def _load_tokenized_prepared_datasets(
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    split: Literal["train", "test"] = "train",
    processor: ProcessorMixin | None = None,
    streaming: bool = False,
) -> tuple[Dataset | DatasetDict, list[Prompter | None]]:
    """Load or create tokenized and prepared datasets for training or testing.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.
        streaming: Whether to use iterable preprocessing.

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
            streaming,
        )

    return dataset, prompters


def _load_raw_datasets(
    cfg: DictDefault,
    datasets_configs: list,
    tokenizer: PreTrainedTokenizer,
    split: str,
    processor: ProcessorMixin | None = None,
    streaming: bool = False,
) -> tuple[Dataset, list[Prompter | None]]:
    """Load, process, merge, and save raw datasets."""
    LOG.info("Loading raw datasets...", main_process_only=False)
    if not cfg.is_preprocess and not cfg.skip_prepare_dataset:
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
            streaming=streaming,
        )
        datasets.append(dataset_wrapper)
        prompters.append(dataset_prompter)

    # Merge datasets
    dataset = merge_datasets(datasets, cfg)

    if not cfg.skip_prepare_dataset and not streaming:
        if split == "test" and cfg.eval_sequence_len:
            dataset = handle_long_seq_in_dataset(dataset, cfg.eval_sequence_len, cfg)
        else:
            dataset = handle_long_seq_in_dataset(dataset, cfg.sequence_len, cfg)
        if (split == "train" and cfg.sample_packing) or (
            split == "test" and cfg.eval_sample_packing
        ):
            dataset, _ = process_datasets_for_packing(cfg, dataset, None)

        # Deduplicate before saving so the saved dataset is already de-duplicated
        if cfg.dataset_exact_deduplication:
            dataset, _ = deduplicate_and_log_datasets(dataset=dataset)

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
    streaming: bool = False,
) -> tuple[Dataset | IterableDataset, Prompter | None]:
    """Load and process a single dataset based on the passed config."""
    # For synthetic datasets, create a minimal placeholder instead of loading from path
    if dataset_config.type == "_synthetic":
        dataset = Dataset.from_dict({"text": [""]})
    else:
        # Load the dataset
        dataset = load_dataset_with_config(
            dataset_config, cfg.hf_use_auth_token, streaming=streaming
        )

    # Parse dataset type
    d_base_type, d_prompt_style = _parse_dataset_type(dataset_config.type)

    # Select the appropriate split
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
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

    # No validation split - deduplication already applied during preprocessing
    return dataset, None


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
    streaming: bool = False,
) -> tuple[Dataset | None, Dataset | None, list[Prompter | None]]:
    """Load and prepare datasets with optional validation split and sharding.

    Args:
        tokenizer: Tokenizer for processing text.
        cfg: Configuration object.
        split: Dataset split to load ('train' or 'test').
        processor: Optional processor for multimodal datasets.
        streaming: Whether to use iterable preprocessing.

    Returns:
        Tuple of (train_dataset, eval_dataset, prompters).
    """
    # Load the base dataset
    dataset, prompters = _load_tokenized_prepared_datasets(
        tokenizer,
        cfg,
        split=split,
        processor=processor,
        streaming=streaming,
    )

    # Apply dataset sharding if configured using shared function
    dataset = _apply_dataset_sharding(dataset, cfg)

    # Apply deduplication and create train / validation splits based on the split type
    if split == "train":
        train_dataset, eval_dataset = _handle_train_dataset_split(dataset, cfg)
    else:
        # Deduplication already applied during preprocessing
        train_dataset, eval_dataset = None, dataset

    return train_dataset, eval_dataset, prompters

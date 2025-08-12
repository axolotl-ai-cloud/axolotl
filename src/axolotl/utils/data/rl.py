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
        num_proc=cfg.dataset_processes,
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

        # Truncate first, then drop if still invalid (although truncate should handle it)
        handling = sample.get("sequence_len_overflow_handling", "drop")
        if handling == "truncate":
            # If both sequences fit, return sample unchanged
            if (len_prompt + len_chosen) <= sequence_len and (
                len_prompt + len_rejected
            ) <= sequence_len:
                result = sample
            else:
                # Calculate maximum response length that can fit with the prompt
                max_response_len = sequence_len - len_prompt

                if max_response_len <= 0:
                    # Prompt itself exceeds sequence length. Cannot truncate responses to fix it.
                    # Keep sample shape for map(), but log a warning. A subsequent filter will drop it.
                    LOG.warning(
                        "Prompt length (%s) exceeds sequence length (%s) for DPO-like sample; will be dropped post-truncation",
                        len_prompt,
                        sequence_len,
                    )
                    result = sample

                else:
                    # Truncate the chosen and rejected responses if needed
                    if len_chosen > max_response_len:
                        chosen_tokens = tokenizer(chosen, add_special_tokens=False)[
                            "input_ids"
                        ][:max_response_len]
                        sample["chosen"] = tokenizer.decode(
                            chosen_tokens, skip_special_tokens=True
                        )

                    if len_rejected > max_response_len:
                        rejected_tokens = tokenizer(rejected, add_special_tokens=False)[
                            "input_ids"
                        ][:max_response_len]
                        sample["rejected"] = tokenizer.decode(
                            rejected_tokens, skip_special_tokens=True
                        )
                    result = sample
        else:  # handling == "drop"
            result = (len_prompt + len_chosen) <= sequence_len and (
                len_prompt + len_rejected
            ) <= sequence_len

    elif rl == RLType.KTO:
        if not (sample.get("prompt") and sample.get("completion")):
            raise ValueError("Prompt and completion keys are required for KTO datasets")

        prompt = sample["prompt"]
        completion = sample["completion"]

        len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        len_completion = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )

        # Truncate first
        handling = sample.get("sequence_len_overflow_handling", "drop")
        if handling == "truncate":
            # If sequence fits, return sample unchanged
            if (len_prompt + len_completion) <= sequence_len:
                result = sample
            else:
                # Calculate maximum completion length
                max_completion_len = sequence_len - len_prompt

                if max_completion_len <= 0:
                    # Prompt itself exceeds sequence length. Cannot truncate completion to fix it.
                    LOG.warning(
                        "Prompt length (%s) exceeds sequence length (%s) for KTO sample; will be dropped post-truncation",
                        len_prompt,
                        sequence_len,
                    )
                    result = sample
                else:
                    # Truncate the completion if needed
                    if len_completion > max_completion_len:
                        completion_tokens = tokenizer(
                            completion, add_special_tokens=False
                        )["input_ids"][:max_completion_len]
                        sample["completion"] = tokenizer.decode(
                            completion_tokens, skip_special_tokens=True
                        )
                    result = sample
        else:  # handling == "drop"
            result = (len_prompt + len_completion) <= sequence_len

    elif rl == RLType.GRPO:
        # For GRPO always keep
        result = True
    else:
        raise ValueError("Unknown RL type")

    return result


def load_prepare_preference_datasets(cfg):
    def _is_rl_seq_within_sequence_len(sample, rl, tokenizer, sequence_len):
        """
        Boolean predicate to check whether a preference-learning sample fits within sequence_len.
        Used with dataset.filter() after truncation to drop unsalvageable samples.
        """
        if rl in (RLType.DPO, RLType.IPO, RLType.ORPO, RLType.SIMPO):
            if not (
                sample.get("prompt")
                and sample.get("chosen")
                and sample.get("rejected")
            ):
                return False
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]
            len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            len_chosen = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
            len_rejected = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
            return (len_prompt + len_chosen) <= sequence_len and (
                len_prompt + len_rejected
            ) <= sequence_len
        if rl == RLType.KTO:
            if not (sample.get("prompt") and sample.get("completion")):
                return False
            prompt = sample["prompt"]
            completion = sample["completion"]
            len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            len_completion = len(
                tokenizer(completion, add_special_tokens=False)["input_ids"]
            )
            return (len_prompt + len_completion) <= sequence_len
        if rl == RLType.GRPO:
            # GRPO does not enforce this check here
            return True
        return False
    def load_split(dataset_cfgs, _cfg):
        split_datasets: List[Any] = []
        use_auth_token = _cfg.hf_use_auth_token
        for config_dataset in datasets_w_name_generator(dataset_cfgs):
            ds: Union[Dataset, DatasetDict] = load_dataset_w_config(
                config_dataset, use_auth_token, streaming=False
            )
            split_datasets.append(ds)
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

                map_kwargs = {}
                if isinstance(ds_transform_fn, tuple):
                    ds_transform_fn, map_kwargs = ds_transform_fn
                split_datasets[i] = map_dataset(
                    cfg, data_set, ds_transform_fn, tokenizer, **map_kwargs
                )
            elif _cfg.rl is RLType.KTO:
                ds_transform_fn = load_kto(_type, _cfg, dataset_idx=i)
                map_kwargs = {}
                if isinstance(ds_transform_fn, tuple):
                    ds_transform_fn, map_kwargs = ds_transform_fn
                split_datasets[i] = map_dataset(
                    cfg, data_set, ds_transform_fn, tokenizer, **map_kwargs
                )
            else:
                # If no `type` is provided, assume the dataset is already in the expected format with
                # "prompt", "chosen" and "rejected" already preprocessed
                split_datasets[i] = data_set

            if not cfg.skip_prepare_dataset:
                # Determine handling mode
                # Support legacy alias "excess_token_handling" for compatibility
                handling = cfg.get(
                    "sequence_len_overflow_handling",
                    cfg.get("excess_token_handling", "drop"),
                )

                drop_long = partial(
                    drop_long_rl_seq,
                    rl=_cfg.rl,
                    tokenizer=tokenizer,
                    sequence_len=cfg.sequence_len,
                    handling=handling,  # Pass the handling mode
                )

                prior_len = len(split_datasets[i])

                # Use map for truncate mode and filter for drop mode
                if handling == "truncate":
                    split_datasets[i] = split_datasets[i].map(
                        drop_long,  # Function now returns modified sample or original
                        num_proc=cfg.dataset_processes,
                        load_from_cache_file=not cfg.is_preprocess,
                        desc="Truncating Long Sequences",
                    )
                    # After truncation, drop any samples that still exceed sequence_len (e.g., prompt alone too long)
                    split_datasets[i] = split_datasets[i].filter(
                        partial(
                            _is_rl_seq_within_sequence_len,
                            rl=_cfg.rl,
                            tokenizer=tokenizer,
                            sequence_len=cfg.sequence_len,
                        ),
                        num_proc=cfg.dataset_processes,
                        load_from_cache_file=not cfg.is_preprocess,
                        desc="Dropping Oversize Samples After Truncation",
                    )
                    LOG.info(
                        f"Processed dataset index {i} with truncation handling for sequence length {cfg.sequence_len}"
                    )
                else:  # handling == "drop"
                    split_datasets[i] = split_datasets[i].filter(
                        drop_long,  # Function now returns boolean
                        num_proc=cfg.dataset_processes,
                        load_from_cache_file=not cfg.is_preprocess,
                        desc="Dropping Long Sequences",
                    )
                    dropped = prior_len - len(split_datasets[i])
                    if dropped:
                        LOG.warning(
                            f"Dropped {dropped} long samples from dataset index {i}"
                        )

        combined_datasets = concatenate_datasets(split_datasets)
        combined_datasets = combined_datasets.shuffle(seed=cfg.seed or 42)

        return combined_datasets

    with zero_first(is_main_process()):
        train_is_preprocessed = False
        eval_is_preprocessed = False
        if train_dataset := _load_preprocessed_ds(cfg, cfg.datasets):
            train_is_preprocessed = True
        else:
            train_dataset = load_split(cfg.datasets, cfg)

        eval_dataset = None
        if cfg.test_datasets:
            if eval_dataset := _load_preprocessed_ds(cfg, cfg.test_datasets):
                eval_is_preprocessed = True
            else:
                eval_dataset = load_split(cfg.test_datasets, cfg)
        if not eval_dataset:
            if cfg.val_set_size:
                seed = cfg.seed if cfg.seed is not None else 42

                # ensure we end up with the same fingerprint by doing rank0 first and being able to cache
                to_hash_train = (
                    train_dataset._fingerprint  # pylint: disable=protected-access
                    + "|"
                    + str(cfg.val_set_size)
                    + "|"
                    + "train"
                    + "|"
                    + str(seed)
                )
                to_hash_test = (
                    train_dataset._fingerprint  # pylint: disable=protected-access
                    + "|"
                    + str(cfg.val_set_size)
                    + "|"
                    + "test"
                    + "|"
                    + str(seed)
                )
                train_fingerprint = md5(to_hash_train)
                test_fingerprint = md5(to_hash_test)
                ds_w_test_split = train_dataset.train_test_split(
                    test_size=cfg.val_set_size,
                    seed=seed,
                    shuffle=False,
                    train_new_fingerprint=train_fingerprint,
                    test_new_fingerprint=test_fingerprint,
                )
                eval_dataset = ds_w_test_split["test"]
                train_dataset = ds_w_test_split["train"]

        if not train_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.datasets, train_dataset)
        if eval_dataset and not eval_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.test_datasets, eval_dataset)

    if cfg.dataset_exact_deduplication:
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=train_dataset, eval_dataset=eval_dataset
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
                num_proc=cfg.dataset_processes,
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


# pylint: disable=duplicate-code
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

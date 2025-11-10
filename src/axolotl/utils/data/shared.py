"""Dataset loading shared utils."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.data.utils import deduplicate_and_log_datasets, md5
from axolotl.utils.datasets import get_default_process_count
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from adlfs import AzureBlobFileSystem
    from gcsfs import GCSFileSystem
    from ocifs import OCIFileSystem
    from s3fs import S3FileSystem

LOG = get_logger(__name__)

EXTENSIONS_TO_DATASET_TYPES = {
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
    ".txt": "text",
}


def get_dataset_type(dataset_config: DictDefault) -> str:
    """Get the dataset type from the path if it's not specified."""
    if dataset_config.ds_type:
        return dataset_config.ds_type

    for extension, dataset_type in EXTENSIONS_TO_DATASET_TYPES.items():
        if extension in dataset_config.path:
            return dataset_type

    return "json"


def datasets_with_name_generator(
    dataset_configs: list[DictDefault],
) -> Generator[DictDefault, None, None]:
    """Yields expanded dataset configurations based on multiple names or preprocessing
    shards.

    When a dataset config has a list of names, it yields separate configs for each
    name. When a dataset config specifies preprocessing shards, it yields configs for
    each shard.

    Args:
        dataset_configs: List of dataset configuration objects.

    Yields:
        Individual dataset configurations, expanded as needed for names or shards.
    """
    for config in dataset_configs:
        if config.name and isinstance(config.name, list):
            for name in config.name:
                yield DictDefault({**config, "name": name})
        elif config.preprocess_shards and not config.shards:
            for shard_idx in range(config.preprocess_shards):
                yield DictDefault(
                    {
                        **config,
                        "shards": config.preprocess_shards,
                        "shards_idx": shard_idx,
                    }
                )
        else:
            yield config


def load_dataset_with_config(
    dataset_config: DictDefault, use_auth_token: bool, streaming=False
) -> Dataset | IterableDataset:
    """Load a dataset from a config. Handles datasets that are stored locally, in the
    HuggingFace Hub, in a remote filesystem (S3, GCS, Azure, OCI), a URL, or
    `data_files`.

    Args:
        dataset_config: Single dataset config.
        use_auth_token: Whether to use HF auth token.
        streaming: Whether to stream the dataset.

    Returns:
        Loaded dataset.
    """
    # Set up common kwargs for dataset loading
    load_dataset_kwargs = {
        "split": dataset_config.split if dataset_config.split else None,
        "name": dataset_config.name,
        "streaming": streaming,
        "trust_remote_code": dataset_config.trust_remote_code,
    }

    # First check if it's a local path
    if Path(dataset_config.path).exists():
        return _load_from_local_path(dataset_config, load_dataset_kwargs)

    # Check if it's a HuggingFace dataset
    is_hub_dataset = _check_if_hub_dataset(dataset_config, use_auth_token)

    # Check if it's a cloud storage path and get appropriate filesystem
    remote_fs, storage_options = _get_remote_filesystem(dataset_config.path)
    is_cloud_dataset = False
    if remote_fs:
        try:
            is_cloud_dataset = remote_fs.exists(dataset_config.path)
        except (FileNotFoundError, ConnectionError):
            pass

    # Load from appropriate source
    if is_hub_dataset:
        return _load_from_hub(dataset_config, use_auth_token, load_dataset_kwargs)
    if is_cloud_dataset:
        return _load_from_cloud(
            dataset_config, remote_fs, storage_options, load_dataset_kwargs
        )
    if dataset_config.path.startswith("https://"):
        return _load_from_url(dataset_config, load_dataset_kwargs)
    if dataset_config.data_files:
        return _load_from_data_files(dataset_config, load_dataset_kwargs)

    raise ValueError(
        f"The dataset could not be loaded. This could be due to a misconfigured dataset path "
        f"({dataset_config.path}). Try double-check your path / name / data_files. "
        f"This is not caused by the dataset type."
    )


def _check_if_hub_dataset(dataset_config: DictDefault, use_auth_token: bool) -> bool:
    """Check if a dataset exists on the HuggingFace Hub."""
    try:
        snapshot_download(
            repo_id=dataset_config.path,
            repo_type="dataset",
            token=use_auth_token,
            revision=dataset_config.revision,
            ignore_patterns=["*"],
        )
        return True
    except (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        FileNotFoundError,
        ConnectionError,
        HFValidationError,
        ValueError,
    ):
        return False


def _get_remote_filesystem(
    path: str,
) -> tuple[
    S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem | None, dict
]:
    """Get the appropriate filesystem for a remote path."""
    if path.startswith("s3://"):
        try:
            import s3fs

            storage_options = {"anon": False}
            return s3fs.S3FileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("s3:// paths require s3fs to be installed") from exc

    elif path.startswith(("gs://", "gcs://")):
        try:
            import gcsfs

            storage_options = {"token": None}  # type: ignore
            return gcsfs.GCSFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError(
                "gs:// or gcs:// paths require gcsfs to be installed"
            ) from exc

    elif path.startswith(("adl://", "abfs://", "az://")):
        try:
            import adlfs

            storage_options = {"anon": False}
            return adlfs.AzureBlobFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError(
                "adl:// or abfs:// paths require adlfs to be installed"
            ) from exc

    elif path.startswith("oci://"):
        try:
            import ocifs

            storage_options = {}
            return ocifs.OCIFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("oci:// paths require ocifs to be installed") from exc

    return None, {}


def _load_from_local_path(
    dataset_config: DictDefault, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a local path."""
    local_path = Path(dataset_config.path)

    if local_path.is_dir():
        if dataset_config.data_files:
            dataset_type = get_dataset_type(dataset_config)
            return load_dataset(
                dataset_type,
                data_files=dataset_config.data_files,
                **load_dataset_kwargs,
            )
        try:
            return load_from_disk(dataset_config.path)
        except FileNotFoundError:
            return load_dataset(dataset_config.path, **load_dataset_kwargs)
    elif local_path.is_file():
        dataset_type = get_dataset_type(dataset_config)

        # For single file datasets, HF always creates only a "train" split
        if dataset_type in ("json", "csv", "text"):
            load_dataset_kwargs["split"] = "train"

        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            **load_dataset_kwargs,
        )
    else:
        raise ValueError(
            "Unhandled dataset load: local path exists, but is neither a directory or a file"
        )


def _load_from_hub(
    dataset_config: DictDefault, use_auth_token: bool, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from the HuggingFace Hub."""
    return load_dataset(
        dataset_config.path,
        data_files=dataset_config.data_files,
        token=use_auth_token,
        revision=dataset_config.revision,
        **load_dataset_kwargs,
    )


def _load_from_cloud(
    dataset_config: DictDefault,
    remote_fs: S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem,
    storage_options: dict,
    load_dataset_kwargs: dict,
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from cloud storage."""
    if remote_fs.isdir(dataset_config.path):
        return load_from_disk(
            dataset_config.path,
            storage_options=storage_options,
        )

    if remote_fs.isfile(dataset_config.path):
        dataset_type = get_dataset_type(dataset_config)
        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            storage_options=storage_options,
            **load_dataset_kwargs,
        )

    raise ValueError(
        f"Cloud path {dataset_config.path} is neither a directory nor a file"
    )


def _load_from_url(
    dataset_config: DictDefault, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a URL."""
    dataset_type = get_dataset_type(dataset_config)
    return load_dataset(
        dataset_type,
        data_files=dataset_config.path,
        **load_dataset_kwargs,
    )


def _load_from_data_files(
    dataset_config: DictDefault, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from data files."""
    file_path = None

    if isinstance(dataset_config.data_files, str):
        file_path = hf_hub_download(
            repo_id=dataset_config.path,
            repo_type="dataset",
            filename=dataset_config.data_files,
            revision=dataset_config.revision,
        )
    elif isinstance(dataset_config.data_files, list):
        file_path = [
            hf_hub_download(
                repo_id=dataset_config.path,
                repo_type="dataset",
                filename=file,
                revision=dataset_config.revision,
            )
            for file in dataset_config.data_files
        ]
    else:
        raise ValueError("data_files must be either a string or list of strings")

    return load_dataset("json", data_files=file_path, **load_dataset_kwargs)


def generate_split_fingerprints(
    dataset: Dataset, val_set_size: int | float, seed: int
) -> tuple[str, str]:
    """Generate consistent fingerprints for train/test splits."""
    fingerprint = dataset._fingerprint

    train_hash_input = f"{fingerprint}|{val_set_size}|train|{seed}"
    test_hash_input = f"{fingerprint}|{val_set_size}|test|{seed}"

    train_fingerprint = md5(train_hash_input)
    test_fingerprint = md5(test_hash_input)

    return train_fingerprint, test_fingerprint


def get_prepared_dataset_path(cfg: DictDefault, dataset_hash: str) -> Path:
    """Get standardized path for prepared datasets.

    Args:
        cfg: Configuration object.
        dataset_hash: Hash identifying the specific dataset configuration.

    Returns:
        Path where the prepared dataset should be stored.
    """
    base_path = cfg.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
    return Path(base_path) / dataset_hash


def create_train_validation_split(
    dataset: Dataset, cfg: DictDefault, val_set_size: int | float
) -> tuple[Dataset, Dataset]:
    """Create train/validation split with consistent fingerprinting.

    Args:
        dataset: Dataset to split.
        cfg: Configuration object containing seed and other settings.
        val_set_size: Size of validation set (absolute number or fraction).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    train_fingerprint, test_fingerprint = generate_split_fingerprints(
        dataset, val_set_size, cfg.seed
    )

    # Apply deduplication before splitting if configured
    if cfg.dataset_exact_deduplication:
        dataset, _ = deduplicate_and_log_datasets(dataset=dataset)

    split_dataset = dataset.train_test_split(
        test_size=val_set_size,
        shuffle=False,
        seed=cfg.seed,
        train_new_fingerprint=train_fingerprint,
        test_new_fingerprint=test_fingerprint,
    )

    return split_dataset["train"], split_dataset["test"]


def _generate_from_iterable_dataset(
    dataset: IterableDataset, worker_id: list[int], num_workers: list[int]
) -> Generator[Any, None, None]:
    """Generator function to correctly split the dataset for each worker"""
    for i, item in enumerate(dataset):
        if i % num_workers[0] == worker_id[0]:
            yield item


def save_preprocessed_dataset(
    cfg: DictDefault,
    dataset: Dataset,
    dataset_hash: str,
    split: str,
) -> None:
    """Save preprocessed dataset to disk and optionally push to the HF Hub."""
    prepared_ds_path = get_prepared_dataset_path(cfg, dataset_hash)
    num_workers = cfg.dataset_num_proc or get_default_process_count()
    if isinstance(dataset, IterableDataset):
        ds_from_iter = Dataset.from_generator(
            functools.partial(_generate_from_iterable_dataset, dataset),
            features=dataset.features,
            num_proc=num_workers,
            split=split,
            gen_kwargs={
                "worker_id": list(range(num_workers)),
                "num_workers": [num_workers] * num_workers,
            },
        )
        ds_from_iter.save_to_disk(
            str(prepared_ds_path),
            num_proc=num_workers,
            max_shard_size=None,
            num_shards=cfg.num_dataset_shards_to_save,
        )
    else:
        min_rows_per_proc = 256
        os.makedirs(prepared_ds_path, exist_ok=True)
        dataset.save_to_disk(
            str(prepared_ds_path),
            num_proc=min(max(1, len(dataset) // min_rows_per_proc), num_workers),
            max_shard_size=None,
            num_shards=cfg.num_dataset_shards_to_save,
        )
    if cfg.push_dataset_to_hub:
        LOG.info(
            "Pushing merged prepared dataset to Huggingface hub at "
            f"{cfg.push_dataset_to_hub} (version {dataset_hash})...",
            main_process_only=False,
        )
        dataset.push_to_hub(
            cfg.push_dataset_to_hub,
            dataset_hash,
            private=True,
        )


def load_preprocessed_dataset(cfg: DictDefault, dataset_hash: str) -> Dataset | None:
    """Load preprocessed dataset from disk if available.

    Args:
        cfg: Configuration object.
        dataset_hash: Hash identifying the dataset configuration.

    Returns:
        Loaded dataset if found and conditions are met, None otherwise.
    """
    prepared_ds_path = get_prepared_dataset_path(cfg, dataset_hash)

    if (
        cfg.dataset_prepared_path
        and any(prepared_ds_path.glob("*"))
        and not cfg.skip_prepare_dataset
        and not cfg.is_preprocess
    ):
        LOG.info(
            f"Loading prepared dataset from disk at {prepared_ds_path}...",
            main_process_only=True,
        )
        return load_from_disk(str(prepared_ds_path))

    LOG.info(
        f"Unable to find prepared dataset in {prepared_ds_path}",
        main_process_only=True,
    )
    return None


def try_load_from_hub(
    cfg: DictDefault, dataset_hash: str, split: str
) -> Dataset | None:
    """Try to load the prepared dataset from HuggingFace Hub."""
    try:
        LOG.info(
            "Attempting to load prepared dataset from HuggingFace Hub at "
            f"{cfg.push_dataset_to_hub} (version {dataset_hash})..."
        )
        dataset = load_dataset(
            cfg.push_dataset_to_hub,
            dataset_hash,
            token=cfg.hf_use_auth_token,
        )
        return dataset[split]
    except Exception:
        LOG.info("Unable to find prepared dataset in HuggingFace Hub")
        return None


def generate_dataset_hash_from_config(
    cfg: DictDefault, cfg_datasets: list, tokenizer_name: str
) -> str:
    """Generate a hash to uniquely identify a dataset configuration for SFT.

    Args:
        cfg: Main configuration object.
        cfg_datasets: List of dataset configurations.
        tokenizer_name: Name of the tokenizer being used.

    Returns:
        MD5 hash string representing the configuration.
    """
    config_str = (
        f"{cfg.sequence_len}@{cfg.sample_packing}@{cfg.eval_sample_packing}@"
        f"{cfg.group_by_length}@{cfg.kd_temperature or 1.0}|"
        f"{'|'.join(sorted([f'{d.path}:{d.type}:{d.shards}:{d.conversation}:{d.split}:{d.temperature or 1.0}' for d in cfg_datasets]))}"
        f"|{tokenizer_name}"
    )
    return str(md5(config_str))


def merge_datasets(datasets: list[Dataset], cfg: DictDefault) -> Dataset:
    """Merge multiple datasets into one with optional shuffling.

    Args:
        datasets: List of datasets to merge.
        cfg: Configuration object containing shuffle settings.

    Returns:
        Merged dataset.
    """
    if len(datasets) == 1:
        ds = datasets[0]

        # Do not shuffle if curriculum sampling is enabled or
        # shuffle_merged_datasets is disabled
        if cfg.curriculum_sampling or not cfg.shuffle_merged_datasets:
            return ds

        return ds.shuffle(seed=cfg.seed)

    # If enabled, shuffle each dataset independently before merging.
    # This allows curriculum learning strategies to be applied at the dataset level.
    if cfg.shuffle_before_merging_datasets:
        LOG.info("Shuffling each dataset individually before merging...")
        datasets = [ds.shuffle(seed=cfg.seed) for ds in datasets]

    LOG.info("Merging datasets...")
    merged_dataset = concatenate_datasets(datasets)

    if cfg.shuffle_merged_datasets:
        LOG.debug("Shuffling merged datasets...")
        if cfg.curriculum_sampling:
            LOG.warning(
                "Shuffling merged datasets with curriculum sampling is not recommended. "
                "This will randomize the order of samples."
            )
        merged_dataset = merged_dataset.shuffle(seed=cfg.seed)
    else:
        LOG.debug("Not shuffling merged datasets.")

    return merged_dataset

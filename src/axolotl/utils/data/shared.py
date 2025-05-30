"""Dataset loading shared utils."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.data.utils import deduplicate_and_log_datasets, md5
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
            load_dataset_kwargs["streaming"] = False
            return load_dataset(dataset_config.path, **load_dataset_kwargs)
    elif local_path.is_file():
        dataset_type = get_dataset_type(dataset_config)
        load_dataset_kwargs["streaming"] = False
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
    remote_fs: "S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem",
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
    fingerprint = dataset._fingerprint  # pylint: disable=protected-access

    train_hash_input = f"{fingerprint}|{val_set_size}|train|{seed}"
    test_hash_input = f"{fingerprint}|{val_set_size}|test|{seed}"

    train_fingerprint = md5(train_hash_input)
    test_fingerprint = md5(test_hash_input)

    return train_fingerprint, test_fingerprint


class DatasetPreparer:
    """Handles common dataset preparation tasks with distributed coordination.

    This class provides a standardized way to coordinate dataset preparation
    across multiple processes using file locks, ensuring only one process
    does the expensive preprocessing work while others wait and load the results.
    """

    def __init__(self, cfg: DictDefault):
        self.cfg = cfg
        self.dataset_prepared_path = (
            cfg.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
        )
        self.lock_file_path = Path(self.dataset_prepared_path) / "datasets_prep.lock"
        self.ready_flag_path = Path(self.dataset_prepared_path) / "datasets_ready.flag"

    def prepare_with_coordination(
        self, prepare_fn: Callable[[], Any], load_fn: Callable[[], Any] | None = None
    ) -> Any:
        """Execute dataset preparation with distributed coordination.

        Args:
            prepare_fn: Function to call for dataset preparation (first process only).
            load_fn: Optional function to call for loading (all processes).
                    If None, prepare_fn is used for both.

        Returns:
            Result from prepare_fn or load_fn.
        """
        if load_fn is None:
            load_fn = prepare_fn

        with FileLock(str(self.lock_file_path)):
            if not self.ready_flag_path.exists():
                # First process: do the preparation
                result = prepare_fn()
                self.ready_flag_path.touch()
                return result
            # Other processes: wait for completion then load
            while not self.ready_flag_path.exists():
                time.sleep(1)
            return load_fn()


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
    seed = cfg.seed if cfg.seed is not None else 42
    train_fingerprint, test_fingerprint = generate_split_fingerprints(
        dataset, val_set_size, seed
    )

    # Apply deduplication before splitting if configured
    if cfg.dataset_exact_deduplication:
        dataset, _ = deduplicate_and_log_datasets(dataset=dataset)

    split_dataset = dataset.train_test_split(
        test_size=val_set_size,
        shuffle=False,
        seed=seed,
        train_new_fingerprint=train_fingerprint,
        test_new_fingerprint=test_fingerprint,
    )

    return split_dataset["train"], split_dataset["test"]


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
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        return load_from_disk(str(prepared_ds_path))

    LOG.info(f"Unable to find prepared dataset in {prepared_ds_path}")
    return None


def save_preprocessed_dataset(
    cfg: DictDefault, dataset: Dataset, dataset_hash: str
) -> None:
    """Save preprocessed dataset to disk.

    Args:
        cfg: Configuration object.
        dataset: Dataset to save.
        dataset_hash: Hash identifying the dataset configuration.
    """
    if cfg.local_rank == 0 and not cfg.skip_prepare_dataset:
        prepared_ds_path = get_prepared_dataset_path(cfg, dataset_hash)
        LOG.info(f"Saving prepared dataset to disk... {prepared_ds_path}")
        os.makedirs(prepared_ds_path, exist_ok=True)
        dataset.save_to_disk(str(prepared_ds_path))


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


def generate_dataset_hash_from_yaml(config: Any) -> str:
    """Generate MD5 hash of configuration using YAML serialization.

    Args:
        config: Configuration object to hash.

    Returns:
        MD5 hash string.
    """
    config_str = yaml.dump(config, Dumper=yaml.Dumper)
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
        return datasets[0]

    LOG.info("Merging datasets")
    merged_dataset = concatenate_datasets(datasets)

    seed = cfg.seed if cfg.seed is not None else 42
    if cfg.shuffle_merged_datasets:
        LOG.debug("Shuffle merged datasets")
        merged_dataset = merged_dataset.shuffle(seed=seed)
    else:
        LOG.debug("NOT shuffling merged datasets")

    return merged_dataset

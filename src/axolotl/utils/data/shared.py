"""Dataset loading shared utils."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from axolotl.utils.data.utils import md5
from axolotl.utils.dict import DictDefault

if TYPE_CHECKING:
    from adlfs import AzureBlobFileSystem
    from gcsfs import GCSFileSystem
    from ocifs import OCIFileSystem
    from s3fs import S3FileSystem


def get_dataset_type(dataset_config: DictDefault) -> str:
    """Get the dataset type from the path if it's not specified."""
    dataset_type = "json"
    if dataset_config.ds_type:
        dataset_type = dataset_config.ds_type
    elif ".parquet" in dataset_config.path:
        dataset_type = "parquet"
    elif ".arrow" in dataset_config.path:
        dataset_type = "arrow"
    elif ".csv" in dataset_config.path:
        dataset_type = "csv"
    elif ".txt" in dataset_config.path:
        dataset_type = "text"

    return dataset_type


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
    if is_cloud_dataset and remote_fs:
        return _load_from_cloud(
            dataset_config, remote_fs, storage_options, load_dataset_kwargs
        )
    if dataset_config.path.startswith("https://"):
        return _load_from_url(dataset_config, storage_options, load_dataset_kwargs)
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
            print(dataset_type, dataset_config)
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
    dataset_config: DictDefault, storage_options: dict, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a URL."""
    dataset_type = get_dataset_type(dataset_config)
    return load_dataset(
        dataset_type,
        data_files=dataset_config.path,
        storage_options=storage_options,
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

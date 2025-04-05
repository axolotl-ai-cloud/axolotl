"""
dataset loading shared utils
"""

from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from axolotl.utils.dict import DictDefault


def get_ds_type(config_dataset: DictDefault):
    """
    Get the dataset type from the path if it's not specified
    """
    ds_type = "json"
    if config_dataset.ds_type:
        ds_type = config_dataset.ds_type
    elif ".parquet" in config_dataset.path:
        ds_type = "parquet"
    elif ".arrow" in config_dataset.path:
        ds_type = "arrow"
    elif ".csv" in config_dataset.path:
        ds_type = "csv"
    elif ".txt" in config_dataset.path:
        ds_type = "text"
    return ds_type


def datasets_w_name_generator(dataset_configs: list[DictDefault]):
    """
    Yields dataset configs handling multiple names or preprocess_shards

    Args:
        dataset_configs: list of dataset configs (equivalent to cfg.datasets)
    """
    for dataset in dataset_configs:
        if dataset.name and isinstance(dataset.name, list):
            # load_dataset doesn't properly handle multiple named configurations
            # at the same time for a given dataset
            for name in dataset.name:
                yield DictDefault({**dataset, "name": name})
        elif dataset.preprocess_shards and not dataset.shards:
            for shard in range(dataset.preprocess_shards):
                yield DictDefault(
                    {
                        **dataset,
                        "shards": dataset.preprocess_shards,
                        "shards_idx": shard,
                    }
                )
        else:
            yield dataset


def load_dataset_w_config(
    config_dataset: DictDefault, use_auth_token: bool, streaming=False
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from a config

    Args:
        config_dataset: single dataset config
        use_auth_token: whether to use HF auth token
        streaming: whether to stream the dataset
    """
    # pylint: disable=invalid-name
    ds: Optional[Union[Dataset, DatasetDict]] = None  # pylint: disable=invalid-name
    ds_from_hub = False
    try:
        # this is just a basic check to see if the path is a
        # valid HF dataset that's loadable
        snapshot_download(
            repo_id=config_dataset.path,
            repo_type="dataset",
            token=use_auth_token,
            revision=config_dataset.revision,
            ignore_patterns=["*"],
        )
        ds_from_hub = True
    except (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        FileNotFoundError,
        ConnectionError,
        HFValidationError,
        ValueError,
    ):
        pass

    ds_from_cloud = False
    storage_options: dict = {}
    remote_file_system = None
    if config_dataset.path.startswith("s3://"):
        try:
            import s3fs  # type: ignore
        except ImportError as exc:
            raise ImportError("s3:// paths require s3fs to be installed") from exc

        # Reads env, credentials from ~/.aws/credentials, or IAM metadata provider
        # https://s3fs.readthedocs.io/en/latest/index.html?highlight=storage_options#credentials
        storage_options = {"anon": False}
        remote_file_system = s3fs.S3FileSystem(**storage_options)
    elif config_dataset.path.startswith("gs://") or config_dataset.path.startswith(
        "gcs://"
    ):
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "gs:// or gcs:// paths require gcsfs to be installed"
            ) from exc

        # gcsfs will use default credentials from the environment else anon
        # https://gcsfs.readthedocs.io/en/latest/#credentials
        storage_options = {"token": None}
        remote_file_system = gcsfs.GCSFileSystem(**storage_options)
    elif (
        config_dataset.path.startswith("adl://")
        or config_dataset.path.startswith("abfs://")
        or config_dataset.path.startswith("az://")
    ):
        try:
            import adlfs
        except ImportError as exc:
            raise ImportError(
                "adl:// or abfs:// paths require adlfs to be installed"
            ) from exc

        # # Ensure you have the following environment variables set:
        # # Gen 1
        # storage_options = {
        #     "tenant_id": AZURE_STORAGE_TENANT_ID,
        #     "client_id": AZURE_STORAGE_CLIENT_ID,
        #     "client_secret": AZURE_STORAGE_CLIENT_SECRET,
        # }
        # # Gen 2
        # storage_options = {
        #     "account_name": AZURE_STORAGE_ACCOUNT_NAME,
        #     "account_key": AZURE_STORAGE_ACCOUNT_KEY,
        # }

        # Reads env
        # https://github.com/fsspec/adlfs?tab=readme-ov-file#setting-credentials
        storage_options = {"anon": False}
        remote_file_system = adlfs.AzureBlobFileSystem(**storage_options)
    elif config_dataset.path.startswith("oci://"):
        try:
            import ocifs
        except ImportError as exc:
            raise ImportError("oci:// paths require ocifs to be installed") from exc

        # https://ocifs.readthedocs.io/en/latest/getting-connected.html#Using-Environment-Variables
        remote_file_system = ocifs.OCIFileSystem(**storage_options)

    try:
        if remote_file_system and remote_file_system.exists(config_dataset.path):
            ds_from_cloud = True
    except (FileNotFoundError, ConnectionError):
        pass

    # gather extra args from the config
    load_ds_kwargs = {}
    if config_dataset.split:
        load_ds_kwargs["split"] = config_dataset.split
    else:
        load_ds_kwargs["split"] = None

    # prefer local dataset, even if hub exists
    local_path = Path(config_dataset.path)
    if local_path.exists():
        if local_path.is_dir():
            if config_dataset.data_files:
                ds_type = get_ds_type(config_dataset)
                ds = load_dataset(  # pylint: disable=invalid-name
                    ds_type,
                    name=config_dataset.name,
                    data_files=config_dataset.data_files,
                    streaming=streaming,
                    **load_ds_kwargs,
                )
            else:
                try:
                    ds = load_from_disk(
                        config_dataset.path
                    )  # pylint: disable=invalid-name
                except FileNotFoundError:
                    ds = load_dataset(
                        config_dataset.path,
                        name=config_dataset.name,
                        streaming=False,
                        **load_ds_kwargs,
                    )
        elif local_path.is_file():
            ds_type = get_ds_type(config_dataset)

            ds = load_dataset(  # pylint: disable=invalid-name
                ds_type,
                name=config_dataset.name,
                data_files=config_dataset.path,
                streaming=False,
                **load_ds_kwargs,
            )
        else:
            raise ValueError(
                "unhandled dataset load: local path exists, but is neither a directory or a file"
            )
    elif ds_from_hub:
        ds = load_dataset(
            config_dataset.path,
            name=config_dataset.name,
            streaming=streaming,
            data_files=config_dataset.data_files,
            token=use_auth_token,
            revision=config_dataset.revision,
            trust_remote_code=config_dataset.trust_remote_code,
            **load_ds_kwargs,
        )
    elif ds_from_cloud and remote_file_system:
        if remote_file_system.isdir(config_dataset.path):
            ds = load_from_disk(
                config_dataset.path,
                storage_options=storage_options,
            )
        elif remote_file_system.isfile(config_dataset.path):
            ds_type = get_ds_type(config_dataset)
            ds = load_dataset(
                ds_type,
                name=config_dataset.name,
                data_files=config_dataset.path,
                streaming=streaming,
                storage_options=storage_options,
                trust_remote_code=config_dataset.trust_remote_code,
                **load_ds_kwargs,
            )
    elif config_dataset.path.startswith("https://"):
        ds_type = get_ds_type(config_dataset)
        ds = load_dataset(
            ds_type,
            name=config_dataset.name,
            data_files=config_dataset.path,
            streaming=streaming,
            storage_options=storage_options,
            trust_remote_code=config_dataset.trust_remote_code,
            **load_ds_kwargs,
        )
    elif config_dataset.data_files:
        fp: str | list[str] | None = None
        if isinstance(config_dataset.data_files, str):
            fp = hf_hub_download(
                repo_id=config_dataset.path,
                repo_type="dataset",
                filename=config_dataset.data_files,
                revision=config_dataset.revision,
            )
        elif isinstance(config_dataset.data_files, list):
            fp = []
            for file in config_dataset.data_files:
                fp.append(
                    hf_hub_download(
                        repo_id=config_dataset.path,
                        repo_type="dataset",
                        filename=file,
                        revision=config_dataset.revision,
                    )
                )
        else:
            raise ValueError("data_files must be either a string or list of strings")
        ds = load_dataset(
            "json",
            name=config_dataset.name,
            data_files=fp,
            streaming=streaming,
            **load_ds_kwargs,
        )
    if not ds:
        raise ValueError("unhandled dataset load")

    return ds

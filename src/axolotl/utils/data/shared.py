"""
dataset loading shared utils
"""
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HFValidationError

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


def load_dataset_w_config(config_dataset, auth_token):
    # pylint: disable=invalid-name
    ds: Optional[Union[Dataset, DatasetDict]] = None  # pylint: disable=invalid-name
    ds_from_hub = False
    ds_trust_remote_code = config_dataset.trust_remote_code
    try:
        # this is just a basic check to see if the path is a
        # valid HF dataset that's loadable
        load_dataset(
            config_dataset.path,
            name=config_dataset.name,
            streaming=True,
            token=auth_token,
            revision=config_dataset.revision,
            trust_remote_code=ds_trust_remote_code,
        )
        ds_from_hub = True
    except (FileNotFoundError, ConnectionError, HFValidationError, ValueError):
        pass

    ds_from_cloud = False
    storage_options = {}
    remote_file_system = None
    if config_dataset.path.startswith("s3://"):
        try:
            import aiobotocore.session  # type: ignore
            import s3fs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "s3:// paths require aiobotocore and s3fs to be installed"
            ) from exc

        # Takes credentials from ~/.aws/credentials for default profile
        s3_session = aiobotocore.session.AioSession(profile="default")
        storage_options = {"session": s3_session}
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
    # TODO: Figure out how to get auth creds passed
    # elif config_dataset.path.startswith("adl://") or config_dataset.path.startswith("abfs://"):
    #     try:
    #         import adlfs
    #     except ImportError as exc:
    #        raise ImportError(
    #            "adl:// or abfs:// paths require adlfs to be installed"
    #        ) from exc

    #     # Gen 1
    #     storage_options = {
    #         "tenant_id": TENANT_ID,
    #         "client_id": CLIENT_ID,
    #         "client_secret": CLIENT_SECRET,
    #     }
    #     # Gen 2
    #     storage_options = {
    #         "account_name": ACCOUNT_NAME,
    #         "account_key": ACCOUNT_KEY,
    #     }

    #     remote_file_system = adlfs.AzureBlobFileSystem(**storage_options)
    try:
        if remote_file_system and remote_file_system.exists(config_dataset.path):
            ds_from_cloud = True
    except (FileNotFoundError, ConnectionError):
        pass

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
                    streaming=False,
                    split=None,
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
                        split=None,
                    )
        elif local_path.is_file():
            ds_type = get_ds_type(config_dataset)

            ds = load_dataset(  # pylint: disable=invalid-name
                ds_type,
                name=config_dataset.name,
                data_files=config_dataset.path,
                streaming=False,
                split=None,
            )
        else:
            raise ValueError(
                "unhandled dataset load: local path exists, but is neither a directory or a file"
            )
    elif ds_from_hub:
        load_ds_kwargs = {}
        if config_dataset.split:
            load_ds_kwargs["split"] = config_dataset.split
        ds = load_dataset(
            config_dataset.path,
            name=config_dataset.name,
            streaming=False,
            data_files=config_dataset.data_files,
            token=auth_token,
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
                streaming=False,
                split=None,
                storage_options=storage_options,
                trust_remote_code=config_dataset.trust_remote_code,
            )
    elif config_dataset.path.startswith("https://"):
        ds_type = get_ds_type(config_dataset)
        ds = load_dataset(
            ds_type,
            name=config_dataset.name,
            data_files=config_dataset.path,
            streaming=False,
            split=None,
            storage_options=storage_options,
            trust_remote_code=config_dataset.trust_remote_code,
        )
    else:
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
            streaming=False,
            split=None,
        )
    if not ds:
        raise ValueError("unhandled dataset load")

    return ds

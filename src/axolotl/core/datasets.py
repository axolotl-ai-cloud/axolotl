import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from datasets import Dataset as Dataset_ds
from datasets import DatasetDict, IterableDataset, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

logger = logging.getLogger("axolotl")


class DsType(Enum):
    JSON = "json"
    ARROW = "arrow"
    PARQUET = "parquet"


@dataclass
class DatasetConfiguration:
    path: str
    type: str
    name: Optional[str] = field(
        default=None,
        metadata={"help": "the name of the dataset configuration to load."},
    )
    ds_type: Optional[DsType] = None
    data_files: Optional[Union[str, List[str]]] = None
    shards: Optional[int] = None
    test_size: Optional[float] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Generator["DatasetConfiguration", None, None]:
        if "name" in d and isinstance(d["name"], list):
            name = d.pop("name")
            for n in name:
                yield DatasetConfiguration(
                    **d,
                    name=n,
                )


def load_dataset_from_local(config: DatasetConfiguration) -> Optional[Dataset_ds]:
    local_path = Path(config.path)
    if not local_path.exists():
        return None
    ds = None
    if local_path.is_dir():
        if config.ds_type:
            # TODO dirs with arrow or parquet files could be loaded with `load_from_disk`
            ds = load_from_disk(config.path)
        else:
            ds = load_dataset(
                config.path,
                name=config.name,
                data_files=config.data_files,
                streaming=False,
                split=None,
            )
    elif local_path.is_file():
        ds_type = "json"
        if config.ds_type:
            ds_type = config.ds_type.value
        elif "parquet" in config.path:
            ds_type = "parquet"
        elif "arrow" in config.path:
            ds_type = "arrow"
        ds = load_dataset(
            ds_type,
            name=config.name,
            data_files=config.path,
            streaming=False,
            split=None,  # is this correct?
        )
    if not ds:
        raise ValueError(
            "unhandled dataset load: local path exists, but is neither a directory or a file"
        )
    return ds


# TODO should this be a DatasetDict?
class Dataset(Dataset_ds):
    _config: DatasetConfiguration

    def __init__(self, *args, config: DatasetConfiguration = None, **kwargs):
        self._config = config
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(
        config: DatasetConfiguration,
        token: bool = False,
        default_test_size: float = 0.1,
    ):
        ds = load_dataset_from_local(config)
        if not ds:
            try:
                ds = load_dataset(
                    config.path,
                    name=config.name,
                    data_files=config.data_files,
                    token=token,
                )
            except FileNotFoundError:
                pass
        if not ds:
            fp = hf_hub_download(
                repo_id=config.path,
                repo_type="dataset",
                filename=config.data_files,
                token=token,
            )
            ds = load_dataset(
                "json", name=config.name, data_files=fp, streaming=False, split=None
            )
        if not ds:
            raise ValueError("unhandled dataset load")
        test_size = config.test_size if config.test_size else default_test_size
        # determine if the dataset is pre-tokenized
        check_ds = ds["train"] if isinstance(ds, DatasetDict) and "train" in ds else ds
        is_ds_tokenized = False
        if "input_ids" in check_ds.features:
            is_ds_tokenized = True
            if "attention_mask" not in check_ds.features:
                logger.warning("`attention_mask` missing from pre-tokenized dataset")
            if "labels" not in check_ds.features:
                logger.warning("`labels` missing from pre-tokenized dataset")
        if test_size and (not isinstance(ds, DatasetDict) or "test" not in ds):
            ds.train_test_split(test_size=test_size, shuffle=False)
            pass


class DatasetCollection:
    datasets: List[Dataset] = []

    def __init__(self, datasets: Union[Dataset, List[Dataset]]):
        self.datasets = datasets if isinstance(datasets, list) else [datasets]

    def __iter__(self):
        for ds in self.datasets:
            for d in ds:
                yield d

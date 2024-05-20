"""data handling specific to DPO"""
import inspect
import logging
from functools import partial
from pathlib import Path
from typing import Any, List

import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.orpo import load as load_orpo
from axolotl.utils.data.utils import md5
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process, zero_first
from axolotl.utils.models import load_tokenizer

LOG = logging.getLogger("axolotl")


def _get_path(ds_hash, cfg):
    prepared_ds_path = (
        Path(cfg.dataset_prepared_path) / ds_hash
        if cfg.dataset_prepared_path
        else Path(DEFAULT_DATASET_PREPARED_PATH) / ds_hash
    )

    return prepared_ds_path


def _load_preprocessed_ds(cfg, sub_cfg):
    ds_hash = md5(yaml.dump(sub_cfg, Dumper=yaml.Dumper))
    prepared_ds_path = _get_path(ds_hash, cfg)
    dataset = None

    # pylint: disable=duplicate-code
    if (
        cfg.dataset_prepared_path
        and any(prepared_ds_path.glob("*"))
        and not cfg.is_preprocess
    ):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))

    return dataset


def _save_preprocessed_ds(cfg, sub_cfg, dataset):
    ds_hash = md5(yaml.dump(sub_cfg, Dumper=yaml.Dumper))
    prepared_ds_path = _get_path(ds_hash, cfg)

    if cfg.is_preprocess and is_main_process():
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset.save_to_disk(str(prepared_ds_path))


def map_dataset(cfg, data_set, ds_transform_fn, tokenizer):
    sig = inspect.signature(ds_transform_fn)
    if "tokenizer" in sig.parameters:
        if not tokenizer:
            tokenizer = load_tokenizer(cfg)
        ds_transform_fn = partial(ds_transform_fn, tokenizer=tokenizer)

    data_set = data_set.map(
        ds_transform_fn,
        desc="Mapping RL Dataset",
    )
    if isinstance(data_set, DatasetDict):
        data_set = data_set["train"]
    return data_set


def load_prepare_dpo_datasets(cfg):
    def load_split(dataset_cfgs, _cfg):
        split_datasets: List[Any] = []
        for i, ds_cfg in enumerate(dataset_cfgs):
            if ds_cfg["ds_type"] == "json":
                for data_file in ds_cfg["data_files"]:
                    data_files = {ds_cfg["split"]: data_file}
                    ds = load_dataset(  # pylint: disable=invalid-name
                        "json",
                        data_files=data_files,
                        split=ds_cfg["split"],
                    )
                    split_datasets.insert(i, ds)
            else:
                ds = load_dataset(  # pylint: disable=invalid-name
                    ds_cfg["path"],
                    split=ds_cfg["split"],
                )
                split_datasets.insert(i, ds)

        tokenizer = None

        for i, data_set in enumerate(split_datasets):
            _type = dataset_cfgs[i]["type"]
            if _type:
                if isinstance(_type, DictDefault):
                    _type = "user_defined.default"
                if _cfg.rl == "orpo":
                    ds_transform_fn = load_orpo(_type, _cfg, dataset_idx=i)
                elif _cfg.rl == "kto":
                    ds_transform_fn = load_kto(_type, _cfg, dataset_idx=i)
                else:
                    ds_transform_fn = load_dpo(_type, _cfg, dataset_idx=i)

                split_datasets[i] = map_dataset(
                    cfg, data_set, ds_transform_fn, tokenizer
                )
            elif _cfg.rl == "kto":
                ds_transform_fn = load_kto(_type, _cfg, dataset_idx=i)
                split_datasets[i] = map_dataset(
                    cfg, data_set, ds_transform_fn, tokenizer
                )
            else:
                # If no `type` is provided, assume the dataset is already in the expected format with
                # "prompt", "chosen" and "rejected" already preprocessed
                split_datasets[i] = data_set

        return concatenate_datasets(split_datasets)

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
            eval_dataset = None

        if not train_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.datasets, train_dataset)
        if eval_dataset and not eval_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.test_datasets, eval_dataset)

    return train_dataset, eval_dataset

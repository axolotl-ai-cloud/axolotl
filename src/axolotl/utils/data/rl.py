"""data handling specific to DPO"""

import inspect
import logging
from functools import partial
from pathlib import Path
from typing import Any, List, Union

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.orpo import load as load_orpo
from axolotl.utils.data.shared import datasets_w_name_generator, load_dataset_w_config
from axolotl.utils.data.utils import deduplicate_and_log_datasets, md5
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


def map_dataset(cfg, data_set, ds_transform_fn, tokenizer, **map_kwargs):
    sig = inspect.signature(ds_transform_fn)
    if "tokenizer" in sig.parameters:
        if not tokenizer:
            tokenizer = load_tokenizer(cfg)
        ds_transform_fn = partial(ds_transform_fn, tokenizer=tokenizer)

    if isinstance(data_set, DatasetDict):
        data_set = data_set["train"]

    data_set = data_set.map(
        ds_transform_fn,
        desc="Mapping RL Dataset",
        **map_kwargs,
    )

    return data_set


def drop_long_rl_seq(
    sample, rl, tokenizer, sequence_len  # pylint: disable=invalid-name
):
    if rl in ("dpo", "ipo", "orpo", "simpo"):
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

        return (len_prompt + len_chosen) <= sequence_len and (
            len_prompt + len_rejected
        ) <= sequence_len

    if rl == "kto":
        if not (sample.get("prompt") and sample.get("completion")):
            raise ValueError("Prompt and completion keys are required for KTO datasets")

        prompt = sample["prompt"]
        completion = sample["completion"]

        len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        len_completion = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )

        return (len_prompt + len_completion) <= sequence_len

    if rl == "grpo":
        return True

    raise ValueError("Unknown RL type")


def load_prepare_preference_datasets(cfg):
    def load_split(dataset_cfgs, _cfg):
        split_datasets: List[Any] = []
        use_auth_token = _cfg.hf_use_auth_token
        for config_dataset in datasets_w_name_generator(dataset_cfgs):
            ds: Union[Dataset, DatasetDict] = load_dataset_w_config(
                config_dataset, use_auth_token, streaming=False
            )
            split_datasets.append(ds)

        tokenizer = load_tokenizer(cfg)

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

                map_kwargs = {}
                if isinstance(ds_transform_fn, tuple):
                    ds_transform_fn, map_kwargs = ds_transform_fn
                split_datasets[i] = map_dataset(
                    cfg, data_set, ds_transform_fn, tokenizer, **map_kwargs
                )
            elif _cfg.rl == "kto":
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
                drop_long = partial(
                    drop_long_rl_seq,
                    rl=_cfg.rl,
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
                    LOG.warning(
                        f"Dropped {dropped} long samples from dataset index {i}"
                    )

        combined_datasets = concatenate_datasets(split_datasets)
        combined_datasets = combined_datasets.shuffle(seed=cfg.seed)

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
            eval_dataset = None

        if not train_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.datasets, train_dataset)
        if eval_dataset and not eval_is_preprocessed:
            _save_preprocessed_ds(cfg, cfg.test_datasets, eval_dataset)

    if cfg.dataset_exact_deduplication:
        train_dataset, eval_dataset, _ = deduplicate_and_log_datasets(
            train_dataset=train_dataset, eval_dataset=eval_dataset
        )

    return train_dataset, eval_dataset

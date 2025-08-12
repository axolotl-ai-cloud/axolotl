"""data handling specific to DPO"""

import inspect
import logging
from functools import partial
from pathlib import Path
from typing import Any, List, Union

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.loaders import load_tokenizer
from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.prompt_strategies.kto import load as load_kto
from axolotl.prompt_strategies.orpo import load as load_orpo
from axolotl.utils.data.shared import datasets_w_name_generator, load_dataset_w_config
from axolotl.utils.data.utils import deduplicate_and_log_datasets, md5
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process, zero_first
from axolotl.utils.schemas.enums import RLType

LOG = logging.getLogger(__name__)


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
        num_proc=cfg.dataset_processes,
        **map_kwargs,
    )

    return data_set


def drop_long_rl_seq(
    sample,
    rl,
    tokenizer,
    sequence_len,
    handling="drop",  # Use the default handling mode
):
    result = None

    if rl in (RLType.DPO, RLType.IPO, RLType.ORPO, RLType.SIMPO):
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
        # GRPO doesn't involve sequence length checks in the same way?
        # The original code returned True for drop. What should it return for truncate?
        # Let's assume for now it always passes.
        result = sample if handling == "truncate" else True
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

        tokenizer = load_tokenizer(cfg)

        for i, data_set in enumerate(split_datasets):
            _type = dataset_cfgs[i]["type"]
            if _type:
                if isinstance(_type, DictDefault):
                    _type = "user_defined.default"
                if _cfg.rl is RLType.ORPO:
                    ds_transform_fn = load_orpo(_type, _cfg, dataset_idx=i)
                elif _cfg.rl is RLType.KTO:
                    ds_transform_fn = load_kto(_type, _cfg, dataset_idx=i)
                else:
                    ds_transform_fn = load_dpo(_type, _cfg, dataset_idx=i)

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
        )

    return train_dataset, eval_dataset

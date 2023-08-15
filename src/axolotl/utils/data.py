"""Module containing data utilities"""
import functools
import hashlib
import logging
from hashlib import md5
from pathlib import Path
from typing import Tuple, Union

import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

from axolotl.datasets import ConstantLengthDataset, TokenizedPromptDataset
from axolotl.prompt_strategies import load
from axolotl.prompt_tokenizers import (
    AlpacaMultipleChoicePromptTokenizingStrategy,
    AlpacaPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    CompletionPromptTokenizingStrategy,
    GPTeacherPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    ShareGPTPromptTokenizingStrategy,
    SummarizeTLDRPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    CompletionPrompter,
    GPTeacherPrompter,
    JeopardyPrompter,
    MultipleChoiceConcisePrompter,
    MultipleChoiceExplainPrompter,
    ReflectAlpacaPrompter,
    ShareGPTPrompter,
    SummarizeTLDRPrompter,
)
from axolotl.utils.distributed import is_main_process, zero_first
from axolotl.utils.trainer import (
    calculate_total_num_steps,
    process_datasets_for_packing,
)

LOG = logging.getLogger("axolotl")
DEFAULT_DATASET_PREPARED_PATH = "last_run_prepared"


def prepare_dataset(cfg, tokenizer):
    if not cfg.pretraining_dataset:
        train_dataset, eval_dataset = load_prepare_datasets(
            tokenizer, cfg, DEFAULT_DATASET_PREPARED_PATH
        )
    else:
        train_dataset = load_pretraining_dataset(
            cfg.pretraining_dataset,
            tokenizer,
            max_tokens=cfg.sequence_len,
            seed=cfg.seed or 42,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = None

    with zero_first(is_main_process()):
        train_dataset, eval_dataset = process_datasets_for_packing(
            cfg, train_dataset, eval_dataset
        )
    if cfg.max_steps:
        total_num_steps = min(
            calculate_total_num_steps(cfg, train_dataset, tokenizer), cfg.max_steps
        )
        LOG.info(f"Maximum number of steps set at {total_num_steps}")
    else:
        total_num_steps = calculate_total_num_steps(cfg, train_dataset, tokenizer)
    return train_dataset, eval_dataset, total_num_steps


def load_tokenized_prepared_datasets(
    tokenizer, cfg, default_dataset_prepared_path
) -> DatasetDict:
    tokenizer_name = tokenizer.__class__.__name__
    ds_hash = str(
        md5(  # nosec
            (
                str(cfg.sequence_len)
                + "@"
                + "|".join(
                    sorted([f"{d.path}:{d.type}:{d.shards}" for d in cfg.datasets])
                )
                + "|"
                + tokenizer_name
            ).encode("utf-8")
        ).hexdigest()
    )
    prepared_ds_path = (
        Path(cfg.dataset_prepared_path) / ds_hash
        if cfg.dataset_prepared_path
        else Path(default_dataset_prepared_path) / ds_hash
    )
    dataset = None
    use_auth_token = cfg.hf_use_auth_token
    try:
        if cfg.push_dataset_to_hub:
            dataset = load_dataset(
                f"{cfg.push_dataset_to_hub}/{ds_hash}",
                use_auth_token=use_auth_token,
            )
            dataset = dataset["train"]
    except Exception:  # pylint: disable=broad-except # nosec
        pass

    if dataset:
        ...
    elif any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        LOG.info(f"Unable to find prepared dataset in {prepared_ds_path}")
        LOG.info("Loading raw datasets...")

        if cfg.seed:
            seed = cfg.seed
        else:
            LOG.info("No seed provided, using default seed of 42")
            seed = 42

        datasets = []
        # pylint: disable=invalid-name
        for d in cfg.datasets:
            ds: Union[Dataset, DatasetDict] = None
            ds_from_hub = False
            try:
                load_dataset(
                    d.path,
                    name=d.name,
                    streaming=True,
                    use_auth_token=use_auth_token,
                )
                ds_from_hub = True
            except FileNotFoundError:
                pass

            # prefer local dataset, even if hub exists
            local_path = Path(d.path)
            if local_path.exists():
                if local_path.is_dir():
                    # TODO dirs with arrow or parquet files could be loaded with `load_from_disk`
                    ds = load_dataset(
                        d.path,
                        name=d.name,
                        data_files=d.data_files,
                        streaming=False,
                        split=None,
                    )
                elif local_path.is_file():
                    ds = load_dataset(
                        "json",
                        name=d.name,
                        data_files=d.path,
                        streaming=False,
                        split=None,
                    )
                else:
                    raise ValueError(
                        "unhandled dataset load: local path exists, but is neither a directory or a file"
                    )
            elif ds_from_hub:
                ds = load_dataset(
                    d.path,
                    name=d.name,
                    streaming=False,
                    data_files=d.data_files,
                    use_auth_token=use_auth_token,
                )
            else:
                fp = hf_hub_download(
                    repo_id=d.path,
                    repo_type="dataset",
                    filename=d.data_files,
                )
                ds = load_dataset(
                    "json", name=d.name, data_files=fp, streaming=False, split=None
                )
            if not ds:
                raise ValueError("unhandled dataset load")
            # support for using a subset of the data
            if d.shards:
                if "train" in ds:
                    ds = ds.shuffle(seed=seed)["train"].shard(
                        num_shards=d.shards, index=0
                    )
                else:
                    ds = ds.shuffle(seed=seed).shard(num_shards=d.shards, index=0)
            d_type = d.type
            d_type_split = d_type.split(":")
            d_base_type = d_type_split[0]
            d_prompt_style = d_type_split[1] if len(d_type_split) > 1 else None
            if "train" in ds:
                ds = ds["train"]
            if ds_strategy := load(d.type, tokenizer, cfg):
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "alpaca":
                ds_strategy = AlpacaPromptTokenizingStrategy(
                    AlpacaPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "explainchoice":
                ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
                    MultipleChoiceExplainPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "concisechoice":
                ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
                    MultipleChoiceConcisePrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "summarizetldr":
                ds_strategy = SummarizeTLDRPromptTokenizingStrategy(
                    SummarizeTLDRPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "jeopardy":
                ds_strategy = JeopardyPromptTokenizingStrategy(
                    JeopardyPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "oasst":
                ds_strategy = OpenAssistantPromptTokenizingStrategy(
                    AlpacaPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "gpteacher":
                ds_strategy = GPTeacherPromptTokenizingStrategy(
                    GPTeacherPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "reflection":
                ds_strategy = AlpacaReflectionPTStrategy(
                    ReflectAlpacaPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "sharegpt":
                ds_strategy = ShareGPTPromptTokenizingStrategy(
                    ShareGPTPrompter(d_prompt_style),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            elif d_base_type == "completion":
                ds_strategy = CompletionPromptTokenizingStrategy(
                    CompletionPrompter(),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
                datasets.append(ds_wrapper)
            else:
                suffix = ""
                if ":load_" in d.type:
                    suffix = f" Did you mean {d.type.replace(':load_', '.load_')}?"
                LOG.error(f"unhandled prompt tokenization strategy: {d.type}. {suffix}")
                raise ValueError(
                    f"unhandled prompt tokenization strategy: {d.type} {suffix}"
                )
        LOG.info("merging datasets")
        dataset = concatenate_datasets(datasets)

        if len(datasets) > 1:
            LOG.info("shuffle merged datasets")
            dataset = dataset.shuffle(seed=seed)
        if cfg.local_rank == 0:
            LOG.info(f"Saving merged prepared dataset to disk... {prepared_ds_path}")
            dataset.save_to_disk(prepared_ds_path)
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Saving merged prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset.push_to_hub(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                )

    return dataset


def load_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    default_dataset_prepared_path,
) -> Tuple[Dataset, Dataset]:
    max_packed_sequence_len = (
        cfg.max_packed_sequence_len if cfg.max_packed_sequence_len else cfg.sequence_len
    )
    max_packed_sequence_len = min(
        max_packed_sequence_len, cfg.sequence_len
    )  # make sure we don't accidentally set it larger than sequence_len

    tokenizer_name = tokenizer.__class__.__name__
    if cfg.max_packed_sequence_len is not None:
        # see if we can go ahead and load the stacked dataset
        seed = f"@{str(cfg.seed)}" if cfg.seed else ""
        ds_hash = str(
            md5(  # nosec
                (
                    str(cfg.sequence_len)
                    + "@"
                    + str(max_packed_sequence_len)
                    + seed
                    + "|".join(
                        sorted([f"{d.path}:{d.type}:{d.shards}" for d in cfg.datasets])
                    )
                    + "|"
                    + tokenizer_name
                ).encode("utf-8")
            ).hexdigest()
        )
        prepared_ds_path = (
            Path(cfg.dataset_prepared_path) / ds_hash
            if cfg.dataset_prepared_path
            else Path(default_dataset_prepared_path) / ds_hash
        )

        dataset = None
        use_auth_token = cfg.hf_use_auth_token
        try:
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Checking for packed prepared dataset from hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset = load_dataset(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}",
                    use_auth_token=use_auth_token,
                )
                dataset = dataset["train"]
        except Exception:  # pylint: disable=broad-except # nosec
            pass

        if dataset:
            ...
        elif any(prepared_ds_path.glob("*")):
            LOG.info(
                f"Loading prepared packed dataset from disk at {prepared_ds_path}..."
            )
            dataset = load_from_disk(str(prepared_ds_path))
            LOG.info("Prepared packed dataset loaded from disk...")
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Saving packed prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset.push_to_hub(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                )
        else:
            dataset = load_tokenized_prepared_datasets(
                tokenizer, cfg, default_dataset_prepared_path
            )

            if cfg.seed:
                dataset = dataset.shuffle(seed=cfg.seed)

            constant_len_dataset = ConstantLengthDataset(
                tokenizer,
                [dataset],
                seq_length=max_packed_sequence_len,
            )
            LOG.info(f"packing master dataset to len: {cfg.max_packed_sequence_len}")
            dataset = Dataset.from_list(list(constant_len_dataset))

            # filter out bad data
            # TODO convert to dataset.filter(...)
            dataset = Dataset.from_list(
                [
                    d
                    for d in dataset
                    if len(d["input_ids"]) <= cfg.sequence_len
                    and len(d["input_ids"]) > 0
                    and len(d["input_ids"]) == len(d["attention_mask"])
                    and len(d["input_ids"]) == len(d["labels"])
                ]
            )

            if cfg.local_rank == 0:
                LOG.info(
                    f"Saving packed prepared dataset to disk... {prepared_ds_path}"
                )
                dataset.save_to_disk(prepared_ds_path)
                if cfg.push_dataset_to_hub:
                    LOG.info(
                        f"Saving packed prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                    )
                    dataset.push_to_hub(
                        f"{cfg.push_dataset_to_hub}/{ds_hash}",
                        private=True,
                    )
    else:
        dataset = load_tokenized_prepared_datasets(
            tokenizer, cfg, default_dataset_prepared_path
        )

    if cfg.dataset_shard_num and cfg.dataset_shard_idx is not None:
        LOG.info(
            f"Using index #{cfg.dataset_shard_idx} of {cfg.dataset_shard_num} shards"
        )
        dataset = dataset.shard(
            num_shards=cfg.dataset_shard_num,
            index=cfg.dataset_shard_idx,
        )

    if cfg.val_set_size:
        # ensure we end up with the same fingerprint by doing rank0 first and being able to cache
        to_hash_train = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(cfg.val_set_size)
            + "|"
            + "train"
            + "|"
            + str(cfg.seed or 42)
        )
        to_hash_test = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(cfg.val_set_size)
            + "|"
            + "test"
            + "|"
            + str(cfg.seed or 42)
        )
        train_fingerprint = hashlib.md5(
            to_hash_train.encode(), usedforsecurity=False
        ).hexdigest()
        test_fingerprint = hashlib.md5(
            to_hash_test.encode(), usedforsecurity=False
        ).hexdigest()

        with zero_first(is_main_process()):
            dataset = dataset.train_test_split(
                test_size=cfg.val_set_size,
                shuffle=False,
                seed=cfg.seed or 42,
                train_new_fingerprint=train_fingerprint,
                test_new_fingerprint=test_fingerprint,
            )

        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    return train_dataset, eval_dataset


def encode_pretraining(tokenizer, max_tokens, examples):
    res = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    new_input_ids = []
    new_attention_mask = []
    # Append EOS and PAD tokens to input_ids, and correct attention_mask
    for i, _ in enumerate(input_ids):
        input_ids[i] = torch.cat(
            (
                input_ids[i],
                torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id]),
            ),
            dim=0,
        )
        attention_mask[i] = torch.cat((attention_mask[i], torch.tensor([1, 0])), dim=0)

    # Concatenate tokens so that their lengths are less than max_tokens
    buffer_input_ids = torch.tensor([], dtype=torch.long)
    buffer_attention_mask = torch.tensor([], dtype=torch.long)

    for ids, mask in zip(input_ids, attention_mask):
        if buffer_input_ids.numel() == max_tokens:
            new_input_ids.append(buffer_input_ids)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        elif buffer_input_ids.numel() + ids.numel() <= max_tokens:
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        else:
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            new_input_ids.append(buffer_input_ids)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)

            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)

    if buffer_input_ids.numel() > 0:  # for any leftover tokens
        while buffer_input_ids.numel() < max_tokens:  # make all sequences equal in size
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
        new_input_ids.append(buffer_input_ids)
        new_attention_mask.append(buffer_attention_mask)

    ret = {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_input_ids],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }

    LOG.debug(len(ret["input_ids"]))
    return ret


def load_pretraining_dataset(path, tokenizer, max_tokens=2048, seed=42):
    encode = functools.partial(encode_pretraining, tokenizer, max_tokens)
    dataset = load_dataset(path, streaming=True, split="train")
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    # TODO dynamically figure out which columns/features to remove
    dataset = dataset.map(encode, batched=True, remove_columns=["text", "meta"])
    return dataset

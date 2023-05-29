"""Module containing data utilities"""

import logging
from hashlib import md5
from pathlib import Path
from typing import Tuple, Union

from datasets import (
    load_from_disk,
    load_dataset,
    Dataset,
    DatasetDict,
)
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

from axolotl.datasets import TokenizedPromptDataset, ConstantLengthDataset
from axolotl.prompt_strategies import load
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    GPTeacherPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    ShareGPTPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    CompletionPromptTokenizingStrategy,
    AlpacaMultipleChoicePromptTokenizingStrategy,
    SummarizeTLDRPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    ReflectAlpacaPrompter,
    ShareGPTPrompter,
    JeopardyPrompter,
    CompletionPrompter,
    MultipleChoiceExplainPrompter,
    SummarizeTLDRPrompter,
    MultipleChoiceConcisePrompter,
)


def load_tokenized_prepared_datasets(
    tokenizer, cfg, default_dataset_prepared_path
) -> DatasetDict:
    tokenizer_name = tokenizer.__class__.__name__
    ds_hash = str(
        md5(
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
                f"{cfg.push_dataset_to_hub}/{ds_hash}", use_auth_token=use_auth_token
            )
            dataset = dataset["train"]
    except Exception:  # pylint: disable=broad-except
        pass

    if dataset:
        ...
    elif any(prepared_ds_path.glob("*")):
        logging.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        logging.info("Prepared dataset loaded from disk...")
    else:
        logging.info(f"Unable to find prepared dataset in {prepared_ds_path}")
        logging.info("Loading raw datasets...")
        datasets = []
        # pylint: disable=invalid-name
        for d in cfg.datasets:
            ds: Union[Dataset, DatasetDict] = None
            ds_from_hub = False
            try:
                load_dataset(d.path, streaming=True, use_auth_token=use_auth_token)
                ds_from_hub = True
            except FileNotFoundError:
                pass

            # prefer local dataset, even if hub exists
            if Path(d.path).exists():
                ds: Dataset = load_dataset(
                    "json", data_files=d.path, streaming=False, split=None
                )
            elif ds_from_hub:
                if d.data_files:
                    ds: Dataset = load_dataset(
                        d.path,
                        streaming=False,
                        data_files=d.data_files,
                        use_auth_token=use_auth_token,
                    )
                else:
                    ds: Dataset = load_dataset(d.path, streaming=False, use_auth_token=use_auth_token)
            else:
                fp = hf_hub_download(
                    repo_id=d.path, repo_type="dataset", filename=d.data_files
                )
                ds: Dataset = load_dataset(
                    "json", data_files=fp, streaming=False, split=None
                )
            if not ds:
                raise ValueError("unhandled dataset load")
            # support for using a subset of the data
            if d.shards:
                if "train" in ds:
                    ds: DatasetDict = ds.shuffle(seed=42)["train"].shard(
                        num_shards=d.shards, index=0
                    )
                else:
                    ds: Dataset = ds.shuffle(seed=42).shard(
                        num_shards=d.shards, index=0
                    )
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
                logging.error(f"unhandled prompt tokenization strategy: {d.type}")
        logging.info("tokenizing, merging, and shuffling master dataset")

        samples = []
        for d in datasets:
            samples = samples + list(d)
        dataset = Dataset.from_list(samples).shuffle(seed=42)
        if cfg.local_rank == 0:
            logging.info(
                f"Saving merged prepared dataset to disk... {prepared_ds_path}"
            )
            dataset.save_to_disk(prepared_ds_path)
            if cfg.push_dataset_to_hub:
                logging.info(
                    f"Saving merged prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset.push_to_hub(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                )

    return dataset


def load_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase, cfg, default_dataset_prepared_path
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
            md5(
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
                logging.info(
                    f"Checking for packed prepared dataset from hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset = load_dataset(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", use_auth_token=use_auth_token
                )
                dataset = dataset["train"]
        except Exception:  # pylint: disable=broad-except
            pass

        if dataset:
            ...
        elif any(prepared_ds_path.glob("*")):
            logging.info(
                f"Loading prepared packed dataset from disk at {prepared_ds_path}..."
            )
            dataset = load_from_disk(str(prepared_ds_path))
            logging.info("Prepared packed dataset loaded from disk...")
            if cfg.push_dataset_to_hub:
                logging.info(
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
            logging.info(
                f"packing master dataset to len: {cfg.max_packed_sequence_len}"
            )
            dataset = Dataset.from_list(list(constant_len_dataset))

            # filter out bad data
            dataset = Dataset.from_list(
                [
                    d
                    for d in dataset
                    if len(d["input_ids"]) < cfg.sequence_len
                    and len(d["input_ids"]) > 0
                    and len(d["input_ids"]) == len(d["attention_mask"])
                    and len(d["input_ids"]) == len(d["labels"])
                ]
            )

            if cfg.local_rank == 0:
                logging.info(
                    f"Saving packed prepared dataset to disk... {prepared_ds_path}"
                )
                dataset.save_to_disk(prepared_ds_path)
                if cfg.push_dataset_to_hub:
                    logging.info(
                        f"Saving packed prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                    )
                    dataset.push_to_hub(
                        f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                    )
    else:
        dataset = load_tokenized_prepared_datasets(
            tokenizer, cfg, default_dataset_prepared_path
        )

    if cfg.dataset_shard_num and cfg.dataset_shard_idx is not None:
        logging.info(
            f"Using index #{cfg.dataset_shard_idx} of {cfg.dataset_shard_num} shards"
        )
        dataset = dataset.shard(
            num_shards=cfg.dataset_shard_num, index=cfg.dataset_shard_idx
        )

    dataset = dataset.train_test_split(test_size=cfg.val_set_size, shuffle=False)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    return train_dataset, eval_dataset

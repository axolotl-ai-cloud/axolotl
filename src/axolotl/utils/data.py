import logging
from hashlib import md5
from pathlib import Path

from datasets import (
    load_from_disk,
    load_dataset,
    IterableDataset,
    Dataset,
    concatenate_datasets,
)
from huggingface_hub import hf_hub_download

from axolotl.datasets import TokenizedPromptDataset, ConstantLengthDataset
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    GPTeacherPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    ShareGPTPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    CompletionPromptTokenizingStrategy, AlpacaMultipleChoicePromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    ReflectAlpacaPrompter,
    ShareGPTPrompter,
    JeopardyPrompter,
    CompletionPrompter, MultipleChoiceExplainPrompter,
)


def load_tokenized_prepared_datasets(tokenizer, cfg, default_dataset_prepared_path):
    ds_hash = str(
        md5(
            (
                str(cfg.sequence_len)
                + "@"
                + "|".join(sorted([f"{d.path}:{d.type}" for d in cfg.datasets]))
            ).encode("utf-8")
        ).hexdigest()
    )
    prepared_ds_path = (
        Path(cfg.dataset_prepared_path) / ds_hash
        if cfg.dataset_prepared_path
        else Path(default_dataset_prepared_path) / ds_hash
    )

    if any(prepared_ds_path.glob("*")):
        logging.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        logging.info("Prepared dataset loaded from disk...")
    else:
        logging.info(f"Unable to find prepared dataset in {prepared_ds_path}")
        logging.info("Loading raw datasets...")
        datasets = []
        for d in cfg.datasets:
            ds = None
            ds_from_hub = False
            try:
                load_dataset(d.path, streaming=True)
                ds_from_hub = True
            except FileNotFoundError:
                pass

            # prefer local dataset, even if hub exists
            if Path(d.path).exists():
                ds: IterableDataset = load_dataset(
                    "json", data_files=d.path, streaming=True, split=None
                )
            elif ds_from_hub:
                if d.data_files:
                    ds = load_dataset(d.path, streaming=True, data_files=d.data_files)
                else:
                    ds = load_dataset(d.path, streaming=True)
            else:
                fp = hf_hub_download(
                    repo_id=d.path, repo_type="dataset", filename=d.data_files
                )
                ds = load_dataset("json", data_files=fp, streaming=True, split=None)
            if not ds:
                raise Exception("unhandled dataset load")

            if d.type == "alpaca":
                ds_strategy = AlpacaPromptTokenizingStrategy(
                    AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "explainchoice":
                ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
                    MultipleChoiceExplainPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "jeopardy":
                ds_strategy = JeopardyPromptTokenizingStrategy(
                    JeopardyPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "oasst":
                ds_strategy = OpenAssistantPromptTokenizingStrategy(
                    AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "gpteacher":
                ds_strategy = GPTeacherPromptTokenizingStrategy(
                    GPTeacherPrompter(),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "reflection":
                ds_strategy = AlpacaReflectionPTStrategy(
                    ReflectAlpacaPrompter(),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "sharegpt":
                ds_strategy = ShareGPTPromptTokenizingStrategy(
                    ShareGPTPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            elif d.type == "completion":
                ds_strategy = CompletionPromptTokenizingStrategy(
                    CompletionPrompter(),
                    tokenizer,
                    cfg.train_on_inputs,
                    cfg.sequence_len,
                )
                ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
                datasets.append(ds_wrapper)
            else:
                logging.error(f"unhandled prompt tokenization strategy: {d.type}")
        logging.info("tokenizing, merging, and shuffling master dataset")

        samples = []
        for d in datasets:
            samples = samples + [i for i in d]
        dataset = Dataset.from_list(samples).shuffle(seed=42)
        if cfg.local_rank == 0:
            logging.info(
                f"Saving merged prepared dataset to disk... {prepared_ds_path}"
            )
            dataset.save_to_disk(prepared_ds_path)

    return dataset


def load_prepare_datasets(tokenizer, cfg, default_dataset_prepared_path):
    max_packed_sequence_len = (
        cfg.max_packed_sequence_len if cfg.max_packed_sequence_len else cfg.sequence_len
    )
    max_packed_sequence_len = min(
        max_packed_sequence_len, cfg.sequence_len
    )  # make sure we don't accidentally set it larger than sequence_len

    if cfg.max_packed_sequence_len is not None:
        # see if we can go ahead and load the stacked dataset

        ds_hash = str(
            md5(
                (
                    str(cfg.sequence_len)
                    + "@"
                    + str(max_packed_sequence_len)
                    + "|".join(sorted([f"{d.path}:{d.type}" for d in cfg.datasets]))
                ).encode("utf-8")
            ).hexdigest()
        )
        prepared_ds_path = (
            Path(cfg.dataset_prepared_path) / ds_hash
            if cfg.dataset_prepared_path
            else Path(default_dataset_prepared_path) / ds_hash
        )

        if any(prepared_ds_path.glob("*")):
            logging.info(
                f"Loading prepared packed dataset from disk at {prepared_ds_path}..."
            )
            dataset = load_from_disk(str(prepared_ds_path))
            logging.info("Prepared packed dataset loaded from disk...")
        else:
            dataset = load_tokenized_prepared_datasets(
                tokenizer, cfg, default_dataset_prepared_path
            )

            constant_len_dataset = ConstantLengthDataset(
                tokenizer,
                [dataset],
                seq_length=max_packed_sequence_len,
            )
            logging.info(
                f"packing master dataset to len: {cfg.max_packed_sequence_len}"
            )
            dataset = Dataset.from_list([_ for _ in constant_len_dataset])

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

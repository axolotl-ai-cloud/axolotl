import logging
from hashlib import md5
from pathlib import Path

from datasets import load_from_disk, load_dataset, IterableDataset, Dataset

from axolotl.datasets import TokenizedPromptDataset, ConstantLengthDataset
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    GPTeacherPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    ShareGPTPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    ReflectAlpacaPrompter,
    ShareGPTPrompter,
)


def load_prepare_datasets(tokenizer, cfg, default_dataset_prepared_path):
    max_packed_sequence_len = (
        cfg.max_packed_sequence_len if cfg.max_packed_sequence_len else cfg.sequence_len
    )
    max_packed_sequence_len = min(
        max_packed_sequence_len, cfg.sequence_len
    )  # make sure we don't accidentally set it larger than sequence_len
    ds_hash = str(
        md5(
            (
                str(max_packed_sequence_len)
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
        logging.info("Loading prepared dataset from disk...")
        dataset = load_from_disk(str(prepared_ds_path))
        logging.info("Prepared dataset loaded from disk...")
    else:
        logging.info("Loading raw datasets...")
        datasets = []
        for d in cfg.datasets:
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
                ds = load_dataset(d.path, streaming=True)
            else:
                raise Exception("unhandled dataset load")

            if d.type == "alpaca":
                ds_strategy = AlpacaPromptTokenizingStrategy(
                    AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
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
            else:
                logging.error(f"unhandled prompt tokenization strategy: {d.type}")
        constant_len_dataset = ConstantLengthDataset(
            tokenizer,
            datasets,
            seq_length=max_packed_sequence_len,
        )
        logging.info("merging, packing, shuffling, and splitting master dataset")
        dataset = Dataset.from_list([_ for _ in constant_len_dataset]).train_test_split(
            test_size=cfg.val_set_size, shuffle=True, seed=42
        )

        if cfg.local_rank == 0:
            logging.info(f"Saving prepared dataset to disk... {prepared_ds_path}")
            dataset.save_to_disk(prepared_ds_path)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    return train_dataset, eval_dataset

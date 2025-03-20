"""data handling specific to SFT"""

import functools
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import PreTrainedTokenizerBase

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.datasets import TokenizedPromptDataset, wrap_dataset_for_tokenized_prompt
from axolotl.prompt_strategies import load
from axolotl.prompt_strategies.bradley_terry import load as bradley_terry_load
from axolotl.prompt_tokenizers import (
    AlpacaMultipleChoicePromptTokenizingStrategy,
    AlpacaPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    DatasetWrappingStrategy,
    GPTeacherPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    SummarizeTLDRPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    JeopardyPrompter,
    MultipleChoiceConcisePrompter,
    MultipleChoiceExplainPrompter,
    Prompter,
    ReflectAlpacaPrompter,
    SummarizeTLDRPrompter,
    UnsupportedPrompter,
)
from axolotl.utils.data.pretraining import wrap_pretraining_dataset
from axolotl.utils.data.shared import datasets_w_name_generator, load_dataset_w_config
from axolotl.utils.data.utils import (
    deduplicate_and_log_datasets,
    drop_long_seq_in_dataset,
    md5,
    retry_on_request_exceptions,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_local_main_process, zero_first
from axolotl.utils.trainer import (
    calculate_total_num_steps,
    process_datasets_for_packing,
)

LOG = logging.getLogger(__name__)


@retry_on_request_exceptions(max_retries=3, delay=5)
def prepare_dataset(cfg, tokenizer, processor=None, preprocess_iterable=None):
    prompters = []
    if not cfg.pretraining_dataset:
        with zero_first(is_local_main_process()):
            if cfg.test_datasets:
                train_dataset, _, prompters = load_prepare_datasets(
                    tokenizer,
                    cfg,
                    DEFAULT_DATASET_PREPARED_PATH,
                    split="train",
                    processor=processor,
                    preprocess_iterable=preprocess_iterable,
                )
                _, eval_dataset, _ = load_prepare_datasets(
                    tokenizer,
                    cfg,
                    DEFAULT_DATASET_PREPARED_PATH,
                    split="test",
                    processor=processor,
                    preprocess_iterable=preprocess_iterable,
                )
            else:
                train_dataset, eval_dataset, prompters = load_prepare_datasets(
                    tokenizer,
                    cfg,
                    DEFAULT_DATASET_PREPARED_PATH,
                    processor=processor,
                    preprocess_iterable=preprocess_iterable,
                )
    else:
        # Load streaming dataset if pretraining_dataset is given
        path = cfg.pretraining_dataset
        split = "train"
        name = None
        data_files = None
        skip = 0
        if isinstance(cfg.pretraining_dataset, list) and isinstance(
            cfg.pretraining_dataset[0], dict
        ):
            path = cfg.pretraining_dataset[0]["path"]
            name = cfg.pretraining_dataset[0]["name"]
            skip = cfg.pretraining_dataset[0]["skip"]
            if "split" in cfg.pretraining_dataset[0]:
                split = cfg.pretraining_dataset[0]["split"]

            data_files = cfg.pretraining_dataset[0].get("data_files")

        ds_wrapper_partial = functools.partial(
            get_dataset_wrapper,
            cfg.pretraining_dataset[0],
            tokenizer,
            cfg,
            cfg.pretraining_dataset[0]["type"] or "pretrain",
        )

        iter_ds = load_dataset(
            path, streaming=True, split=split, name=name, data_files=data_files
        )
        if skip:
            LOG.info(f"Skipping {skip} samples from the dataset")
            iter_ds = iter_ds.skip(skip)
        train_dataset = wrap_pretraining_dataset(
            iter_ds,
            tokenizer,
            cfg,
            ds_wrapper_partial,
            max_tokens=cfg.sequence_len,
            batch_size=cfg.micro_batch_size,
            seed=cfg.seed or 42,
            buffer_size=cfg.pretrain_multipack_buffer_size or 10_000,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")

        # Load eval dataset (non-streaming) if specified
        eval_dataset = None
        if cfg.test_datasets:
            _, eval_dataset, _ = load_prepare_datasets(
                tokenizer,
                cfg,
                DEFAULT_DATASET_PREPARED_PATH,
                split="test",
                processor=processor,
                preprocess_iterable=preprocess_iterable,
            )

        if cfg.dataset_exact_deduplication:
            LOG.info("Deduplication not available for pretrained datasets")

        return train_dataset, eval_dataset, cfg.max_steps, prompters

    if eval_dataset and cfg.sample_packing and cfg.eval_sample_packing is not False:
        total_eval_steps = calculate_total_num_steps(cfg, eval_dataset, update=False)
        if total_eval_steps == 0:
            raise ValueError(
                "eval dataset split is too small for sample_packing. You should set `eval_sample_packing: False`. "
            )

    if cfg.max_steps:
        total_num_steps = min(
            calculate_total_num_steps(cfg, train_dataset), cfg.max_steps
        )
        LOG.info(f"Maximum number of steps set at {total_num_steps}")
    else:
        total_num_steps = calculate_total_num_steps(cfg, train_dataset)

    return train_dataset, eval_dataset, total_num_steps, prompters


def load_tokenized_prepared_datasets(
    tokenizer,
    cfg,
    default_dataset_prepared_path,
    split="train",
    processor=None,
    preprocess_iterable: Optional[bool] = None,
) -> Tuple[DatasetDict, List[Prompter]]:
    cfg_datasets = cfg.test_datasets if split == "test" else cfg.datasets
    tokenizer_name = cfg.tokenizer_config

    ds_hash = str(
        md5(
            (
                str(cfg.sequence_len)
                + "@"
                + str(cfg.sample_packing)
                + "@"
                + str(cfg.eval_sample_packing)
                + "@"
                + str(cfg.group_by_length)
                + "@"
                + str(cfg.kd_temperature or 1.0)
                + "|".join(
                    sorted(
                        [
                            f"{d.path}:{d.type}:{d.shards}:{d.conversation}:{d.split}:{d.temperature or 1.0}"
                            for d in cfg_datasets
                        ]
                    )
                )
                + "|"
                + tokenizer_name
            )
        )
    )
    prepared_ds_path = (
        Path(cfg.dataset_prepared_path) / ds_hash
        if cfg.dataset_prepared_path
        else Path(default_dataset_prepared_path) / ds_hash
    )
    dataset = None
    prompters = []
    use_auth_token = cfg.hf_use_auth_token
    try:
        if cfg.push_dataset_to_hub:
            LOG.info(
                f"Attempting to load prepared dataset from Huggingface hub at {cfg.push_dataset_to_hub} (version {ds_hash})..."
            )
            dataset = load_dataset(
                cfg.push_dataset_to_hub,
                ds_hash,
                token=use_auth_token,
            )
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-except # nosec
        pass

    # pylint: disable=duplicate-code
    if dataset:
        # This is for the case where we already loaded a pretokenized dataset from the hub
        ...
    elif (
        cfg.dataset_prepared_path
        and any(prepared_ds_path.glob("*"))
        and not cfg.is_preprocess
        and not cfg.skip_prepare_dataset
    ):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        if cfg.push_dataset_to_hub:
            LOG.info("Unable to find prepared dataset in Huggingface hub")
        if cfg.is_preprocess:
            LOG.info(
                f"Skipping prepared dataset in {prepared_ds_path} for pre-processing..."
            )
        else:
            LOG.info(f"Unable to find prepared dataset in {prepared_ds_path}")
        LOG.info("Loading raw datasets...")
        if not cfg.is_preprocess:
            LOG.warning(
                "Processing datasets during training can lead to VRAM instability. Please pre-process your dataset."
            )

        if cfg.seed:
            seed = cfg.seed
        else:
            LOG.info("No seed provided, using default seed of 42")
            seed = 42

        datasets = []

        streaming_ds = False
        if preprocess_iterable:
            streaming_ds = True
        # pylint: disable=invalid-name
        for config_dataset in datasets_w_name_generator(cfg_datasets):
            ds: Union[Dataset, DatasetDict] = load_dataset_w_config(
                config_dataset, use_auth_token, streaming=streaming_ds
            )

            d_base_type = d_prompt_style = None
            d_type = config_dataset.type
            if isinstance(d_type, str):
                d_type_split = d_type.split(":")
                d_base_type = d_type_split[0]
                d_prompt_style = d_type_split[1] if len(d_type_split) > 1 else None

            if isinstance(ds, DatasetDict):
                if config_dataset.split and config_dataset.split in ds:
                    ds = ds[config_dataset.split]
                elif split in ds:
                    ds = ds[split]
                else:
                    raise ValueError(
                        f"no {split} split found for dataset {config_dataset.path}, you may specify a split with 'split: `"
                    )

            # support for using a subset of the data
            if config_dataset.shards:
                shards_idx = config_dataset.get("shards_idx", 0)
                ds = ds.shuffle(seed=seed).shard(
                    num_shards=config_dataset.shards, index=shards_idx
                )

            dataset_wrapper, dataset_prompter = get_dataset_wrapper(
                config_dataset=config_dataset,
                tokenizer=tokenizer,
                cfg=cfg,
                d_base_type=d_base_type,
                dataset=ds,
                d_prompt_style=d_prompt_style,
                processor=processor,
            )
            datasets.append(dataset_wrapper)
            prompters.append(dataset_prompter)

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            LOG.info("merging datasets")
            dataset = concatenate_datasets(datasets)

        if len(datasets) > 1:
            if cfg.shuffle_merged_datasets:
                LOG.debug("shuffle merged datasets")
                dataset = dataset.shuffle(seed=seed)
            else:
                LOG.debug("NOT shuffling merged datasets")

        if not cfg.skip_prepare_dataset:
            dataset = drop_long_seq_in_dataset(dataset, cfg)

            if cfg.sample_packing:
                dataset, _ = process_datasets_for_packing(cfg, dataset, None)

        if cfg.local_rank == 0 and not cfg.skip_prepare_dataset:
            LOG.info(f"Saving merged prepared dataset to disk... {prepared_ds_path}")
            if isinstance(dataset, IterableDataset):

                def gen_from_iter_ds(_ds, _=None):
                    yield from _ds

                ds_from_iter = Dataset.from_generator(
                    functools.partial(gen_from_iter_ds, dataset),
                    features=dataset.features,
                    num_proc=cfg.dataset_processes,
                    split=split,
                    gen_kwargs={"_": list(range(cfg.dataset_processes))},
                )
                ds_from_iter.save_to_disk(str(prepared_ds_path))
            else:
                os.makedirs(prepared_ds_path, exist_ok=True)
                dataset.save_to_disk(str(prepared_ds_path))
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Pushing merged prepared dataset to Huggingface hub at {cfg.push_dataset_to_hub} (version {ds_hash})..."
                )
                dataset.push_to_hub(
                    cfg.push_dataset_to_hub,
                    ds_hash,
                    private=True,
                )

    return dataset, prompters


def load_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    default_dataset_prepared_path,
    split="train",
    processor=None,
    preprocess_iterable: Optional[bool] = False,
) -> Tuple[Dataset, Dataset, List[Prompter]]:
    dataset, prompters = load_tokenized_prepared_datasets(
        tokenizer,
        cfg,
        default_dataset_prepared_path,
        split=split,
        processor=processor,
        preprocess_iterable=preprocess_iterable,
    )

    if cfg.dataset_shard_num and cfg.dataset_shard_idx is not None:
        LOG.info(
            f"Using index #{cfg.dataset_shard_idx} of {cfg.dataset_shard_num} shards"
        )
        dataset = dataset.shard(
            num_shards=cfg.dataset_shard_num,
            index=cfg.dataset_shard_idx,
        )

    val_set_size = (
        int(cfg.val_set_size) if cfg.val_set_size > 1 else float(cfg.val_set_size)
    )

    if split == "train" and val_set_size:
        # ensure we end up with the same fingerprint by doing rank0 first and being able to cache
        to_hash_train = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(val_set_size)
            + "|"
            + "train"
            + "|"
            + str(cfg.seed or 42)
        )
        to_hash_test = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(val_set_size)
            + "|"
            + "test"
            + "|"
            + str(cfg.seed or 42)
        )
        train_fingerprint = md5(to_hash_train)
        test_fingerprint = md5(to_hash_test)
        if cfg.dataset_exact_deduplication:
            _, _, dataset = deduplicate_and_log_datasets(dataset=dataset)
        dataset = dataset.train_test_split(
            test_size=val_set_size,
            shuffle=False,
            seed=cfg.seed or 42,
            train_new_fingerprint=train_fingerprint,
            test_new_fingerprint=test_fingerprint,
        )

        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    elif split == "test":
        if cfg.dataset_exact_deduplication:
            _, eval_dataset, _ = deduplicate_and_log_datasets(eval_dataset=dataset)
        else:
            eval_dataset = dataset
        train_dataset = None
    else:
        if cfg.dataset_exact_deduplication:
            train_dataset, _, _ = deduplicate_and_log_datasets(train_dataset=dataset)
        else:
            train_dataset = dataset
        eval_dataset = None
    return train_dataset, eval_dataset, prompters


def get_dataset_wrapper(
    config_dataset,
    tokenizer,
    cfg,
    d_base_type,
    dataset,
    d_prompt_style=None,
    processor=None,  # pylint: disable=unused-argument
):
    dataset_wrapper = None
    dataset_prompter = None

    ds_kwargs = {
        "process_count": cfg.dataset_processes,
        "keep_in_memory": cfg.dataset_keep_in_memory is True,
    }

    LOG.info(
        f"Loading dataset with base_type: {d_base_type} and prompt_style: {d_prompt_style}"
    )

    if (
        isinstance(dataset, Dataset)
        and "input_ids" in dataset.features
        and "attention_mask" in dataset.features
        and "labels" in dataset.features
    ):
        # dataset is already tokenized, just drop it straight in
        dataset_prompter = UnsupportedPrompter()
        dataset_wrapper = dataset
    elif isinstance(config_dataset.type, DictDefault):
        ds_strategy = load(
            "user_defined", tokenizer, cfg, config_dataset.type.to_dict()
        )
        dataset_prompter = UnsupportedPrompter()
        dataset_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
    elif cfg.skip_prepare_dataset:
        dataset_wrapper = dataset
    elif ds_strategy := config_dataset.type.startswith(
        "bradley_terry"
    ) and bradley_terry_load(
        config_dataset.type.split(".", 1)[1], tokenizer, cfg, config_dataset
    ):
        dataset_prompter = UnsupportedPrompter()
        dataset_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
    elif config_dataset.type.startswith("stepwise_supervised"):
        dataset_prompter = UnsupportedPrompter()
        ds_strategy = load(config_dataset.type, tokenizer, cfg, config_dataset)
        # we need to explicitly cast boolean labels to int
        # for compatibility with how trl's PRMTrainer works
        dataset = dataset.cast_column("labels", Sequence(Value("int64")))
        dataset_wrapper = TokenizedPromptDataset(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
    elif ds_strategy := load(
        config_dataset.type, tokenizer, cfg, config_dataset, processor=processor
    ):
        if isinstance(ds_strategy, DatasetWrappingStrategy):
            dataset_wrapper = ds_strategy.wrap_dataset(dataset, **ds_kwargs)
        else:
            dataset_prompter = UnsupportedPrompter()
            dataset_wrapper = wrap_dataset_for_tokenized_prompt(
                ds_strategy,
                dataset,
                **ds_kwargs,
            )
    elif d_base_type == "alpaca":
        dataset_prompter = AlpacaPrompter(d_prompt_style)
        ds_strategy = AlpacaPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "explainchoice":
        dataset_prompter = MultipleChoiceExplainPrompter(d_prompt_style)
        ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "concisechoice":
        dataset_prompter = MultipleChoiceConcisePrompter(d_prompt_style)
        ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "summarizetldr":
        dataset_prompter = SummarizeTLDRPrompter(d_prompt_style)
        ds_strategy = SummarizeTLDRPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "jeopardy":
        dataset_prompter = JeopardyPrompter(d_prompt_style)
        ds_strategy = JeopardyPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "oasst":
        dataset_prompter = AlpacaPrompter(d_prompt_style)
        ds_strategy = OpenAssistantPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "gpteacher":
        dataset_prompter = GPTeacherPrompter(d_prompt_style)
        ds_strategy = GPTeacherPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    elif d_base_type == "reflection":
        dataset_prompter = ReflectAlpacaPrompter(d_prompt_style)
        ds_strategy = AlpacaReflectionPTStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = wrap_dataset_for_tokenized_prompt(
            ds_strategy,
            dataset,
            **ds_kwargs,
        )
        dataset_wrapper = ds_wrapper
    else:
        suffix = ""
        if ":load_" in config_dataset.type:
            suffix = f" Did you mean {config_dataset.type.replace(':load_', '.load_')}?"
        LOG.error(
            f"unhandled prompt tokenization strategy: {config_dataset.type}. {suffix}"
        )
        raise ValueError(
            f"unhandled prompt tokenization strategy: {config_dataset.type} {suffix}"
        )

    return dataset_wrapper, dataset_prompter

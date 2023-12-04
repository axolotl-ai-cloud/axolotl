"""Module containing data utilities"""
import functools
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

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

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.datasets import ConstantLengthDataset, TokenizedPromptDataset
from axolotl.prompt_strategies import load
from axolotl.prompt_tokenizers import (
    AlpacaMultipleChoicePromptTokenizingStrategy,
    AlpacaPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
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
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process, zero_first
from axolotl.utils.trainer import (
    calculate_total_num_steps,
    process_datasets_for_packing,
)

LOG = logging.getLogger("axolotl")


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def prepare_dataset(cfg, tokenizer):
    prompters = []
    if not cfg.pretraining_dataset:
        with zero_first(is_main_process()):
            train_dataset, eval_dataset, prompters = load_prepare_datasets(
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
        return train_dataset, eval_dataset, cfg.max_steps, prompters

    with zero_first(is_main_process()):
        train_dataset, eval_dataset = process_datasets_for_packing(
            cfg, train_dataset, eval_dataset, tokenizer
        )

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
    tokenizer, cfg, default_dataset_prepared_path
) -> Tuple[DatasetDict, List[Prompter]]:
    tokenizer_name = tokenizer.__class__.__name__
    ds_hash = str(
        md5(
            (
                str(cfg.sequence_len)
                + "@"
                + "|".join(
                    sorted(
                        [
                            f"{d.path}:{d.type}:{d.shards}:{d.conversation}"
                            for d in cfg.datasets
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
            dataset = load_dataset(
                f"{cfg.push_dataset_to_hub}/{ds_hash}",
                token=use_auth_token,
            )
            dataset = dataset["train"]
    except Exception:  # pylint: disable=broad-except # nosec
        pass

    if dataset:
        ...
    elif cfg.dataset_prepared_path and any(prepared_ds_path.glob("*")):
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

        def for_d_in_datasets(dataset_configs):
            for dataset in dataset_configs:
                if dataset.name and isinstance(dataset.name, list):
                    for name in dataset.name:
                        yield DictDefault({**dataset, "name": name})
                else:
                    yield dataset

        # pylint: disable=invalid-name
        for config_dataset in for_d_in_datasets(cfg.datasets):
            ds: Union[Dataset, DatasetDict] = None
            ds_from_hub = False
            try:
                load_dataset(
                    config_dataset.path,
                    name=config_dataset.name,
                    streaming=True,
                    token=use_auth_token,
                )
                ds_from_hub = True
            except (FileNotFoundError, ConnectionError):
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
            elif config_dataset.path.startswith(
                "gs://"
            ) or config_dataset.path.startswith("gcs://"):
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
                if remote_file_system and remote_file_system.exists(
                    config_dataset.path
                ):
                    ds_from_cloud = True
            except (FileNotFoundError, ConnectionError):
                pass

            # prefer local dataset, even if hub exists
            local_path = Path(config_dataset.path)
            if local_path.exists():
                if local_path.is_dir():
                    # TODO dirs with arrow or parquet files could be loaded with `load_from_disk`
                    ds = load_dataset(
                        config_dataset.path,
                        name=config_dataset.name,
                        data_files=config_dataset.data_files,
                        streaming=False,
                        split=None,
                    )
                elif local_path.is_file():
                    ds_type = get_ds_type(config_dataset)

                    ds = load_dataset(
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
                ds = load_dataset(
                    config_dataset.path,
                    name=config_dataset.name,
                    streaming=False,
                    data_files=config_dataset.data_files,
                    token=use_auth_token,
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
                    )
            else:
                if isinstance(config_dataset.data_files, str):
                    fp = hf_hub_download(
                        repo_id=config_dataset.path,
                        repo_type="dataset",
                        filename=config_dataset.data_files,
                    )
                elif isinstance(config_dataset.data_files, list):
                    fp = []
                    for file in config_dataset.data_files:
                        fp.append(
                            hf_hub_download(
                                repo_id=config_dataset.path,
                                repo_type="dataset",
                                filename=file,
                            )
                        )
                else:
                    raise ValueError(
                        "data_files must be either a string or list of strings"
                    )
                ds = load_dataset(
                    "json",
                    name=config_dataset.name,
                    data_files=fp,
                    streaming=False,
                    split=None,
                )
            if not ds:
                raise ValueError("unhandled dataset load")
            # support for using a subset of the data
            if config_dataset.shards:
                if "train" in ds:
                    ds = ds.shuffle(seed=seed)["train"].shard(
                        num_shards=config_dataset.shards, index=0
                    )
                else:
                    ds = ds.shuffle(seed=seed).shard(
                        num_shards=config_dataset.shards, index=0
                    )

            d_base_type = d_prompt_style = None
            d_type = config_dataset.type
            if isinstance(d_type, str):
                d_type_split = d_type.split(":")
                d_base_type = d_type_split[0]
                d_prompt_style = d_type_split[1] if len(d_type_split) > 1 else None
            if "train" in ds:
                ds = ds["train"]
            elif (
                isinstance(ds, DatasetDict)
                and config_dataset.train_on_split
                and config_dataset.train_on_split in ds
            ):
                ds = ds[config_dataset.train_on_split]
            elif isinstance(ds, DatasetDict):
                raise ValueError(
                    f"no train split found for dataset {config_dataset.path}, you may specify a split with 'train_on_split: `"
                )

            dataset_wrapper, dataset_prompter = get_dataset_wrapper(
                config_dataset=config_dataset,
                dataset=ds,
                tokenizer=tokenizer,
                cfg=cfg,
                d_base_type=d_base_type,
                d_prompt_style=d_prompt_style,
            )
            datasets.append(dataset_wrapper)
            prompters.append(dataset_prompter)

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

    return dataset, prompters


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


def load_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    default_dataset_prepared_path,
) -> Tuple[Dataset, Dataset, List[Prompter]]:
    max_packed_sequence_len = (
        cfg.max_packed_sequence_len if cfg.max_packed_sequence_len else cfg.sequence_len
    )
    max_packed_sequence_len = min(
        max_packed_sequence_len, cfg.sequence_len
    )  # make sure we don't accidentally set it larger than sequence_len

    tokenizer_name = tokenizer.__class__.__name__
    prompters: List[Prompter] = []
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
                )
            )
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
                    token=use_auth_token,
                )
                dataset = dataset["train"]
        except Exception:  # pylint: disable=broad-except # nosec
            pass

        if dataset:
            ...
        elif cfg.dataset_prepared_path and any(prepared_ds_path.glob("*")):
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
            dataset, prompters = load_tokenized_prepared_datasets(
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
        dataset, prompters = load_tokenized_prepared_datasets(
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
        train_fingerprint = md5(to_hash_train)
        test_fingerprint = md5(to_hash_test)

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

    return train_dataset, eval_dataset, prompters


def get_dataset_wrapper(
    config_dataset, dataset, tokenizer, cfg, d_base_type, d_prompt_style
):
    dataset_wrapper = None
    dataset_prompter = None

    if (
        "input_ids" in dataset.features
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
        dataset_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
        )
    elif ds_strategy := load(config_dataset.type, tokenizer, cfg, config_dataset):
        dataset_prompter = UnsupportedPrompter()
        dataset_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
        )
    elif d_base_type == "alpaca":
        dataset_prompter = AlpacaPrompter(d_prompt_style)
        ds_strategy = AlpacaPromptTokenizingStrategy(
            dataset_prompter,
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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
        ds_wrapper = TokenizedPromptDataset(
            ds_strategy, dataset, process_count=cfg.dataset_processes
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


def encode_pretraining(
    tokenizer: PreTrainedTokenizerBase, max_tokens: int, examples: List[str]
) -> Dict[str, List]:
    res = tokenizer(
        examples,
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
    dataset = dataset.map(
        encode,
        batched=True,
        input_columns="text",
        # remove all the existing columns after mapping since they end up having
        # a different length than the encoded/tokenized column
        remove_columns=dataset.features.keys(),
    )
    return dataset

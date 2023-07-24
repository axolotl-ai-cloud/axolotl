"""Axolotl CLI option definitions"""

from os.path import exists
from typing import Any, Callable, List, Optional, Tuple

import click

from axolotl.utils.config import (
    SubParamEntry,
    is_integer,
    option_factory,
    parse_and_validate_sub_params,
)
from axolotl.utils.dict import DictDefault


def seed_option(**kwargs: Any) -> Callable:
    """
    Seed is used to control the determinism of operations in the axolotl.
    A value of -1 will randomly select a seed.
    """
    return option_factory(
        "--seed",
        envvar="AXOLOTL_SEED",
        type=click.types.INT,
        help=seed_option.__doc__,
        override_kwargs=kwargs,
    )


def dataset_option(**kwargs: Any) -> Callable:
    """
    Dataset to load, multiple datasets can be specified in a single command.

    Example:\n
        --dataset=data/GPTeacher/Instruct,gpteacher
        --dataset=data/GPTeacher/Roleplay,gpteacher
    """

    # List of valid sub parameters for --dataset and simple validation
    sub_param_spec = {
        "path": SubParamEntry(
            validation=exists,
            fail_msg="path validation failed for %VALUE%, the path must exist on the filesystem",
            required=False,
        ),
        "type": SubParamEntry(
            validation=bool,
            fail_msg="type validation failed for '%VALUE%'",
            required=True,
        ),
        "data_files": SubParamEntry(
            validation=exists,
            fail_msg="data_files validation failed for '%VALUE%'",
            required=False,
        ),
        "shards": SubParamEntry(
            validation=is_integer,
            fail_msg="shards validation failed for '%VALUE%', an integer is required",
            required=False,
        ),
        "name": SubParamEntry(
            validation=bool,
            fail_msg="name validation failed for '%VALUE%'",
            required=False,
        ),
    }

    # pylint: disable=unused-argument
    def parse_dataset_callback(
        ctx: Any, param: Any, dataset_def_value: Tuple[str]
    ) -> Optional[List[DictDefault]]:
        # Each value is a tuple of strings like ("path1,type1", "path2,type2", ...)
        # This function splits each string by comma and verifies that the path exists.
        datasets: List[DictDefault] = []
        for item in dataset_def_value:
            datasets.append(
                parse_and_validate_sub_params(
                    unparsed_input=item,
                    param_name="dataset",
                    sub_param_def=sub_param_spec,
                )
            )

        # Since non-None CLI get merged into our global options we need to return a None when no dataset
        # options are detected. This way the CLI logic will use defaults from the yaml configuration file
        return datasets if len(datasets) > 0 else None

    return option_factory(
        "--dataset",
        "datasets",
        envvar="AXOLOTL_DATASETS",  # Here, the correct envvar should be AXOLOTL_DATASETS
        type=click.types.STRING,
        callback=parse_dataset_callback,
        multiple=True,
        help=dataset_option.__doc__,  # Here, the correct help should be from dataset_option itself.
        override_kwargs=kwargs,
    )


def base_model_option(**kwargs: Any) -> Callable:
    """
    The huggingface model that contains *.pt, *.safetensors, or *.bin files or a path to the model on the disk
    """
    return option_factory(
        "--base_model",
        envvar="AXOLOTL_BASE_MODEL",
        type=click.types.STRING,
        help=base_model_option.__doc__,
        override_kwargs=kwargs,
    )


def base_model_config_option(**kwargs: Any) -> Callable:
    """
    Useful when the base_model repo on HuggingFace hub doesn't include configuration .json files. When
    empty, defaults to config in base_model
    """
    return option_factory(
        "--base_model_config",
        envvar="AXOLOTL_BASE_MODEL_CONFIG",
        type=click.types.STRING,
        help=base_model_config_option.__doc__,
        override_kwargs=kwargs,
    )


def model_type_option(**kwargs: Any) -> Callable:
    """Specify the model type to load, ex: AutoModelForCausalLM"""
    return option_factory(
        "--model_type",
        envvar="AXOLOTL_MODEL_TYPE",
        type=click.types.STRING,
        help=model_type_option.__doc__,
        override_kwargs=kwargs,
    )


def tokenizer_type_option(**kwargs: Any) -> Callable:
    """Specify the tokenizer type to load, ex: AutoTokenizer"""
    return option_factory(
        "--tokenizer_type",
        envvar="AXOLOTL_TOKENIZER_TYPE",
        type=click.types.STRING,
        help=tokenizer_type_option.__doc__,
        override_kwargs=kwargs,
    )


def train_on_inputs_option(**kwargs: Any) -> Callable:
    """Controls whether to mask out or include the human's prompt from the training labels"""
    return option_factory(
        "--train_on_inputs/--no-train_on_inputs",
        envvar="AXOLOTL_TRAIN_ON_INPUTS",
        type=click.types.BOOL,
        help=train_on_inputs_option.__doc__,
        override_kwargs=kwargs,
    )


def micro_batch_size_option(**kwargs: Any) -> Callable:
    """Configures the train, batch inference, and batch validation micro batch size"""
    return option_factory(
        "--micro_batch_size",
        envvar="AXOLOTL_MICRO_BATCH_SIZE",
        type=click.types.INT,
        help=micro_batch_size_option.__doc__,
        override_kwargs=kwargs,
    )


def pretraining_dataset_option(**kwargs: Any) -> Callable:
    """Path to pretraining dataset"""
    return option_factory(
        "--pretraining_dataset",
        envvar="AXOLOTL_PRETRAINING_DATASET",
        type=click.types.STRING,
        help=pretraining_dataset_option.__doc__,
        override_kwargs=kwargs,
    )


def dataset_prepared_path_option(**kwargs: Any) -> Callable:
    """Cached generate / inference dataset will be saved to this path"""
    return option_factory(
        "--dataset_prepared_path",
        envvar="AXOLOTL_DATASET_PREPARED_PATH",
        type=click.types.STRING,
        help=dataset_prepared_path_option.__doc__,
        override_kwargs=kwargs,
    )


def max_packed_sequence_len_option(**kwargs: Any) -> Callable:
    """Concatenate training samples together up to this value"""
    return option_factory(
        "--max_packed_sequence_len",
        envvar="AXOLOTL_MAX_PACKED_SEQUENCE_LEN",
        type=click.types.INT,
        help=max_packed_sequence_len_option.__doc__,
        override_kwargs=kwargs,
    )


def sequence_len_option(**kwargs: Any) -> Callable:
    """The maximum length of an input to train with"""
    return option_factory(
        "--sequence_len",
        envvar="AXOLOTL_SEQUENCE_LEN",
        type=click.types.INT,
        help=sequence_len_option.__doc__,
        override_kwargs=kwargs,
    )


def split_name_option(**kwargs: Any) -> Callable:
    """The default dataset split name to use when loading from a HF dataset."""
    return option_factory(
        "--split_name",
        envvar="AXOLOTL_SPLIT_NAME",
        type=click.types.STRING,
        help=split_name_option.__doc__,
        override_kwargs=kwargs,
    )

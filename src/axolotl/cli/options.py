"""Axolotl CLI option definitions"""

from typing import Any, Callable, List, Optional, Tuple

import click

from axolotl.utils.config import (
    SubParamEntry,
    multiple_list_callback,
    option_factory,
    parse_and_validate_sub_params,
    parse_float,
    parse_integer,
)
from axolotl.utils.dict import DictDefault


def seed_option(**kwargs: Any) -> Callable:
    """
    Seed is used to control the determinism of operations in the axolotl.
    A value of -1 will randomly select a seed.
    """

    # pylint: disable=unused-argument
    def parse_seed_callback(ctx: Any, param: Any, seed_value: int) -> Optional[int]:
        return seed_value if seed_value != -1 else None

    return option_factory(
        "--seed",
        envvar="AXOLOTL_SEED",
        type=click.types.INT,
        callback=parse_seed_callback,
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
            parser=str,
        ),
        "type": SubParamEntry(
            parser=str,
            required=True,
        ),
        "data_files": SubParamEntry(
            parser=str,
        ),
        "shards": SubParamEntry(
            parser=parse_integer,
        ),
        "name": SubParamEntry(
            parser=str,
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
            parse_result = parse_and_validate_sub_params(
                unparsed_input=item,
                param_name="dataset",
                sub_param_def=sub_param_spec,
            )
            if parse_result is not None:
                datasets.append(parse_result)

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
        "--train_on_inputs/--no_train_on_inputs",
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


def generation_config_option(**kwargs: Any) -> Callable:
    """
    Sets inferencing GenerationConfig options
    """

    sub_param_spec = {
        "use_cache": SubParamEntry(
            parser=bool,
        ),
        "num_return_sequences": SubParamEntry(
            parser=parse_integer,
        ),
        "do_sample": SubParamEntry(
            parser=bool,
        ),
        "num_beams": SubParamEntry(
            parser=parse_integer,
        ),
        "temperature": SubParamEntry(
            parser=parse_float,
        ),
        "top_p": SubParamEntry(
            parser=parse_float,
        ),
        "top_k": SubParamEntry(
            parser=parse_integer,
        ),
        "typical_p": SubParamEntry(parser=parse_float),
        "max_new_tokens": SubParamEntry(
            parser=parse_integer,
        ),
        "min_new_tokens": SubParamEntry(
            parser=parse_integer,
        ),
        "repetition_penalty": SubParamEntry(
            parser=parse_float,
        ),
        "prepend_bos": SubParamEntry(
            parser=bool,
        ),
        "renormalize_logits": SubParamEntry(
            parser=bool,
        ),
        "early_stopping": SubParamEntry(
            parser=bool,
        ),
        "max_time": SubParamEntry(
            parser=parse_float,
        ),
        "num_beam_groups": SubParamEntry(
            parser=parse_integer,
        ),
        "penalty_alpha": SubParamEntry(
            parser=parse_float,
        ),
        "epsilon_cutoff": SubParamEntry(
            parser=parse_float,
        ),
        "eta_cutoff": SubParamEntry(
            parser=parse_float,
        ),
        "diversity_penalty": SubParamEntry(
            parser=parse_float,
        ),
        "encoder_repetition_penalty": SubParamEntry(
            parser=parse_float,
        ),
        "length_penalty": SubParamEntry(
            parser=parse_float,
        ),
        "no_repeat_ngram_size": SubParamEntry(
            parser=parse_integer,
        ),
        "forced_bos_token_id": SubParamEntry(
            parser=parse_integer,
        ),
        "forced_eos_token_id": SubParamEntry(
            parser=parse_integer,
        ),
        "remove_invalid_values": SubParamEntry(
            parser=bool,
        ),
        "guidance_scale": SubParamEntry(
            parser=parse_float,
        ),
        "output_attentions": SubParamEntry(
            parser=bool,
        ),
        "output_hidden_states": SubParamEntry(
            parser=bool,
        ),
        "output_scores": SubParamEntry(
            parser=bool,
        ),
        "return_dict_in_generate": SubParamEntry(
            parser=bool,
        ),
        "pad_token_id": SubParamEntry(parser=parse_integer),
        "bos_token_id": SubParamEntry(
            parser=parse_integer,
        ),
        "eos_token_id": SubParamEntry(
            parser=parse_integer,
        ),
        "encoder_no_repeat_ngram_size": SubParamEntry(
            parser=parse_integer,
        ),
        "decoder_start_token_id": SubParamEntry(
            parser=parse_integer,
        ),
    }

    # pylint: disable=unused-argument
    def parse_callback(ctx: Any, param: Any, value: str) -> Optional[DictDefault]:
        return parse_and_validate_sub_params(
            unparsed_input=value,
            param_name="generation_config",
            sub_param_def=sub_param_spec,
        )

    return option_factory(
        "--generation_config",
        envvar="AXOLOTL_GENERATION_CONFIG",
        type=click.types.STRING,
        callback=parse_callback,
        help=generation_config_option.__doc__,
        override_kwargs=kwargs,
    )


def output_dir_option(**kwargs: Any) -> Callable:
    """Path that Axolotl will save output files to"""
    return option_factory(
        "--output_dir",
        envvar="AXOLOTL_OUTPUT_DIR",
        type=click.types.STRING,
        help=output_dir_option.__doc__,
        override_kwargs=kwargs,
    )


def adapter_option(**kwargs: Any) -> Callable:
    """Adapter type"""
    return option_factory(
        "--adapter",
        envvar="AXOLOTL_ADAPTER",
        type=click.types.Choice(["lora", "qlora"]),
        help=adapter_option.__doc__,
        override_kwargs=kwargs,
    )


def lora_model_dir_option(**kwargs: Any) -> Callable:
    """Directory to LoRA adapter"""
    return option_factory(
        "--lora_model_dir",
        envvar="AXOLOTL_LORA_MODEL_DIR",
        type=click.types.STRING,
        help=lora_model_dir_option.__doc__,
        override_kwargs=kwargs,
    )


def lora_target_modules_option(**kwargs: Any) -> Callable:
    """The names of the modules to apply Lora to"""

    return option_factory(
        "--lora_target_module",
        "lora_target_modules",
        envvar="AXOLOTL_LORA_TARGET_MODULES",
        type=click.types.UNPROCESSED,
        callback=multiple_list_callback,
        multiple=True,
        help=lora_target_modules_option.__doc__,
        override_kwargs=kwargs,
    )


def base_model_ignore_patterns_option(**kwargs: Any) -> Callable:
    """An ignore pattern if the model repo contains more than 1 model type (*.pt, etc)"""
    return option_factory(
        "--base_model_ignore_patterns",
        envvar="AXOLOTL_BASE_MODEL_IGNORE_PATTERNS",
        type=click.types.STRING,
        help=base_model_ignore_patterns_option.__doc__,
        override_kwargs=kwargs,
    )


def model_revision_option(**kwargs: Any) -> Callable:
    """A specific model revision from huggingface hub"""
    return option_factory(
        "--model_revision",
        envvar="AXOLOTL_MODEL_REVISION",
        type=click.types.STRING,
        help=model_revision_option.__doc__,
        override_kwargs=kwargs,
    )


def tokenizer_config_option(**kwargs: Any) -> Callable:
    """Overrides the model tokenizer configuration, must be a filesystem path"""
    return option_factory(
        "--tokenizer_config",
        envvar="AXOLOTL_TOKENIZER_CONFIG",
        type=click.types.STRING,
        help=tokenizer_config_option.__doc__,
        override_kwargs=kwargs,
    )


def trust_remote_code_option(**kwargs: Any) -> Callable:
    """Trust remote code for untrusted source"""
    return option_factory(
        "--trust_remote_code/--no_trust_remote_code",
        envvar="AXOLOTL_TRUST_REMOTE_CODE",
        type=click.types.BOOL,
        help=trust_remote_code_option.__doc__,
        override_kwargs=kwargs,
    )


def tokenizer_use_fast_option(**kwargs: Any) -> Callable:
    """Sets use_fast option for tokenizer loading from_pretrained"""
    return option_factory(
        "--tokenizer_use_fast/--no_tokenizer_use_fast",
        envvar="AXOLOTL_TOKENIZER_USE_FAST",
        type=click.types.BOOL,
        help=trust_remote_code_option.__doc__,
        override_kwargs=kwargs,
    )


def gptq_option(**kwargs: Any) -> Callable:
    """Enable 4-bit GPTQ training"""
    return option_factory(
        "--gptq/--no_gptq",
        envvar="AXOLOTL_GPTQ",
        type=click.types.BOOL,
        help=gptq_option.__doc__,
        override_kwargs=kwargs,
    )


def gptq_groupsize_option(**kwargs: Any) -> Callable:
    """GPTQ group size"""
    return option_factory(
        "--gptq_groupsize",
        envvar="AXOLOTL_GPTQ_GROUPSIZE",
        type=click.types.INT,
        help=gptq_groupsize_option.__doc__,
        override_kwargs=kwargs,
    )


def gptq_model_v1_option(**kwargs: Any) -> Callable:
    """Use GPTQ v1"""
    return option_factory(
        "--gptq_model_v1/--no_gptq_model_v1",
        envvar="AXOLOTL_GPTQ_MODEL_V1",
        type=click.types.BOOL,
        help=gptq_model_v1_option.__doc__,
        override_kwargs=kwargs,
    )


def load_in_8bit_option(**kwargs: Any) -> Callable:
    """Quantize the model down to 8 bits"""
    return option_factory(
        "--load_in_8bit/--no_load_in_8bit",
        envvar="AXOLOTL_LOAD_IN_8BIT",
        type=click.types.BOOL,
        help=load_in_8bit_option.__doc__,
        override_kwargs=kwargs,
    )

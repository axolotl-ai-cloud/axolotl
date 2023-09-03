"""Axolotl configuration utilities"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import click
import torch
import yaml

import axolotl
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def load_config(config: Path) -> DictDefault:
    """Loads configuration from a file or a directory.

    If the 'config' parameter is a directory, the function will use 'choose_config' function to
    determine which file to use. The configuration file is expected to be in YAML format.
    After loading the configuration, it validates it using 'validate_config' function.

    Parameters
    ----------
    config : Path
        A Path object that points to a configuration file or directory.

    Returns
    -------
    DictDefault
        A dictionary that contains the loaded configuration.

    Raises
    ------
    FileNotFoundError
        If 'config' doesn't point to a valid file or directory, or if no valid configuration file is found.
    """

    config_path = Path(config)

    with open(str(config_path), encoding="utf-8") as config_fp:
        loaded_config = DictDefault(yaml.safe_load(config_fp))

    validate_config(loaded_config)

    return loaded_config


def update_config(
    overrides: Union[Dict[str, Any], DictDefault],
    allow_none: bool = False,
    validate: bool = True,
) -> None:
    """Updates the global configuration and optionally performs validation. The intended use
    case is for merging CLI options into the global configuration.

    Raises an error when validation fails.

    Parameters
    ----------
    overrides : Union[Dict[str, Any], DictDefault]
        Dictionary of configuration overrides. Each key-value pair in the dictionary represents
        a configuration option and its new value. If 'allow_none' is False, key-value pairs where
        the value is None are ignored.

    allow_none : bool, optional
        Determines whether or not configuration options with a value of None should be updated.
        If False, any key-value pair in 'overrides' where the value is None is ignored.
        If True, these pairs are included in the update. Defaults to False.

    validate : bool, optional
        Determines whether or not to validate the configuration after performing the update.
        If True, the function 'validate_config' is called after the update. If this function
        raises an error, the error is propagated to the caller of 'update_config'.
        Defaults to True.
    """
    updates = {
        key: value
        for key, value in overrides.items()
        if allow_none or value is not None
    }
    axolotl.cfg.update(**updates)

    if validate:
        validate_config(cfg=axolotl.cfg)


def option_factory(
    *args,
    override_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Callable:
    """Factory function to generate a decorator for Click options with proper defaults.

    Parameters
    ----------
    args : variable length argument list of str
        Parameter declarations for the Click option (e.g., '--seed').

    override_kwargs : Dict[str, Any], optional
        A dictionary of override parameters for the Click option, by default None.

    **kwargs:
        Additional keyword arguments for the Click option.

    Returns
    -------
    Callable
        Decorator to be used with a function or command.
    """

    derived_kwargs = {
        # Apply Axolotl CLI option defaults...
        "required": False,
        "default": None,
        "show_envvar": True,
        "show_default": True,
        "allow_from_autoenv": True,
        # Then any overrides
        **kwargs,
        # Then any overrides, passed through by the command group using the option...
        **(override_kwargs if override_kwargs is not None else {}),
    }

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func = click.option(
            *args,
            **derived_kwargs,
        )(func)

        return func

    # Note that MyPy doesn't like dynamically adding parameters like this however it is a justified
    # tradeoff to simplify testing and dynamic introspection of Axolotl CLI options
    decorator.click_params = DictDefault(  # type: ignore[attr-defined]
        {
            "args": args,
            "kwargs": derived_kwargs,
        }
    )

    return decorator


def option_group_factory(options: List[Callable], **kwargs) -> Callable:
    """
    Factory function to generate a decorator that applies a group of Click options.

    Parameters
    ----------
    options : List[Callable]
        List of decorator functions that add Click options.

    Returns
    -------
    Callable
        Decorator to be used with a function or command.
    """

    def decorator(func: Callable) -> Callable:
        for option in reversed(options):
            func = option(**kwargs)(func)
        return func

    return decorator


def parse_integer(obj: Any) -> int:
    """Parses a string into a float with strong validation

    Parameters
    ----------
    obj : Any
        The object to be checked for being an integer.

    Returns
    -------
    int
        The parsed int value
    """

    try:
        int_value = int(obj)
        if int_value != float(obj):
            raise ValueError(f"{obj} is not an integer")

        return int_value
    except TypeError as ex:
        raise ValueError(f"Cannot parse {obj} to an integer") from ex


def parse_float(obj: Any) -> float:
    """Parses a string into a float with strong validation

    Parameters
    ----------
    obj : Any
        The object to be checked for being a float.

    Returns
    -------
    float
        The parsed float value
    """

    try:
        float_value = float(obj)
        str_value = str(obj)
        # check if the conversion to string and back to float remains the same
        return float_value == float(str_value)
    except (ValueError, TypeError):
        return False


def parse_bool(obj: Any) -> bool:
    return bool(obj)


@dataclass
class SubParamEntry:
    """A class representing a SubParamEntry, which is used to specify details about a CLI sub-parameter.

    Attributes
    ----------
    parser : Callable[[str], Union[int, float, bool, str]]
        A parser function for the sub-parameter. It takes any value and returns a boolean.
    fail_msg : Optional[str]
        The message to display when the validation / parsing for the sub-parameter fails.
    required : bool
        Indicates whether the sub-parameter is required. If True, an exception will be raised if it is not present.
    """

    parser: Callable[[str], Union[int, float, bool, str]] = str
    fail_msg: Optional[str] = None
    required: bool = False


def parse_and_validate_sub_params(
    unparsed_input: Optional[str],
    param_name: str,
    sub_param_def: Dict[str, SubParamEntry],
) -> Optional[DictDefault]:
    """Parses a string of sub-parameters from a Click CLI command and validates them.

    The intended use of this method is to parse a dictionary of sub options from the Axolotl
    yaml (such as generation_config).

    Parameters
    ----------
    unparsed_input : Optional[str]
        The input string containing the sub-parameters, formatted as "key=value" pairs separated by commas.
    param_name : str
        The name of the parameter that the sub-parameters belong to. This is used in error messages.
    sub_param_def : Dict[str, SubParamEntry]
        A dictionary that defines the expected sub-parameters. The keys are the names of the sub-parameters, and the
        values are SubParamEntry objects that specify the validation function, fail message, and whether the sub-parameter
        is required.

    Returns
    -------
    Optional[DictDefault]
        A dictionary with the parsed and validated sub-parameters. The keys are the names of the sub-parameters, and the
        values are the parsed values.

    Raises
    ------
    click.BadParameter
        Raised when a sub-parameter is not formatted correctly, is not recognized, fails validation, or is required but missing.
    """

    # Nothing to parse if option value was not provided
    if unparsed_input is None:
        return None

    try:
        # Split the input string by "," to separate the "key=value" pairs,
        # then split each pair by "=" to get the key and value separately.
        # Note: Leading and trailing commas are removed before splitting.
        parsed_dict = DictDefault(
            {
                x.split("=")[0]: x.split("=")[1]
                for x in unparsed_input.strip().strip(",").split(",")
            }
        )
    except Exception as ex:
        # If splitting fails, raise a BadParameter exception with an appropriate error message.
        raise click.BadParameter(
            f"Unable to parse {param_name} spec '{unparsed_input}'. Sub-param values must be in this format: {','.join([f'{x}=VALUE' for x in sub_param_def.keys()])}"
        ) from ex

    for parsed_key, unparsed_value in parsed_dict.items():
        # Check if each parsed key is defined in sub_param_def.
        # If not, raise a BadParameter exception with an appropriate error message.
        if parsed_key not in sub_param_def:
            raise click.BadParameter(
                f"Unknown {param_name} sub-parameter: '{parsed_key}' in {param_name} spec: '{unparsed_input}'"
            )

        # Check if each parsed value passes its validation function.
        # If not, raise a BadParameter exception with the corresponding fail message.
        try:
            parsed_dict[parsed_key] = sub_param_def[parsed_key].parser(unparsed_value)

        except Exception as ex:
            # Insight - many parse failures will have the same message. This allows for much cleaner
            # code in options.py
            default_fail_msg = sub_param_def[parsed_key].fail_msg
            if default_fail_msg is not None:
                fail_msg = default_fail_msg.replace("%VALUE%", unparsed_value, 1)
            else:
                fail_msg = f"{parsed_key} parsing failed for '{unparsed_value}'"

            raise click.BadParameter(fail_msg) from ex

    # Check if each required sub-parameter is present in parsed_dict.
    # If not, raise a BadParameter exception with an appropriate error message.
    for subparam_key in sub_param_def.keys():
        if sub_param_def[subparam_key].required and subparam_key not in parsed_dict:
            raise click.BadParameter(
                f"The '{subparam_key}' sub-parameter is missing from {param_name} spec: '{unparsed_input}'"
            )

    # Return the parsed and validated sub-parameters.
    return parsed_dict


# pylint: disable=unused-argument
def multiple_list_callback(
    ctx: Any, param: Any, value: Tuple[str]
) -> Optional[Tuple[str]]:
    """Ensures that None is returned for lists where multiple==True, without
    this callback the default yaml configuration value will always be overridden"""

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            # No option provided, fall back to defaults
            return None

        if all((x == "" for x in value)):
            # Option was provided but empty, interpret as "set to nothing" and overide default
            return cast(Tuple[str], tuple())

    # Remove blanks
    return cast(Optional[Tuple[str]], tuple(x for x in value if x != ""))


def choose_device(cfg):
    def get_device():
        try:
            if torch.cuda.is_available():
                return f"cuda:{cfg.local_rank}"

            if torch.backends.mps.is_available():
                return "mps"

            raise SystemError("No CUDA/mps device found")
        except Exception:  # pylint: disable=broad-exception-caught
            return "cpu"

    cfg.device = get_device()
    if cfg.device_map != "auto":
        if cfg.device.startswith("cuda"):
            cfg.device_map = {"": cfg.local_rank}
        else:
            cfg.device_map = {"": cfg.device}

    # in `accelerate launch`, we need to not pass through any device map and let
    # accelerate figure out which parts of the model to put on which gpu
    accelerate_vars = [var for var in os.environ if var.startswith("ACCELERATE_USE_")]
    if accelerate_vars:
        cfg.device_map = None


def normalize_config(cfg):
    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32 or False

    if cfg.bf16 or cfg.bfloat16:
        cfg.torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        cfg.torch_dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float32

    log_gpu_memory_usage(LOG, "baseline", cfg.device)


def validate_config(cfg):
    if cfg.max_packed_sequence_len and cfg.sample_packing:
        raise ValueError(
            "please set only one of max_packed_sequence_len (deprecated soon) or sample_packing"
        )
    if cfg.max_packed_sequence_len:
        LOG.warning(
            str(
                PendingDeprecationWarning(
                    "max_packed_sequence_len will be deprecated in favor of sample_packing"
                )
            )
        )

    if cfg.gradient_accumulation_steps and cfg.batch_size:
        raise ValueError(
            "please set only one of gradient_accumulation_steps or batch_size"
        )
    if cfg.batch_size:
        LOG.warning(
            "%s\n%s",
            "batch_size is not recommended. Please use gradient_accumulation_steps instead.",
            "To calculate the equivalent gradient_accumulation_steps, divide batch_size / micro_batch_size / number of gpus.",
        )
    if cfg.load_4bit:
        raise ValueError(
            "cfg.load_4bit parameter has been deprecated and replaced by cfg.gptq"
        )

    if cfg.adapter == "qlora":
        if cfg.merge_lora:
            # can't merge qlora if loaded in 8bit or 4bit
            if cfg.load_in_8bit:
                raise ValueError("Can't merge qlora if loaded in 8bit")

            if cfg.gptq:
                raise ValueError("Can't merge qlora if gptq")

            if cfg.load_in_4bit:
                raise ValueError("Can't merge qlora if loaded in 4bit")

        else:
            if cfg.load_in_8bit:
                raise ValueError("Can't load qlora in 8bit")

            if cfg.gptq:
                raise ValueError("Can't load qlora if gptq")

            if not cfg.load_in_4bit:
                raise ValueError("Require cfg.load_in_4bit to be True for qlora")

    if not cfg.load_in_8bit and cfg.adapter == "lora":
        LOG.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")

    if cfg.trust_remote_code:
        LOG.warning(
            "`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model."
        )

    if cfg.push_dataset_to_hub and cfg.hf_use_auth_token is not True:
        raise ValueError(
            "Require cfg.hf_use_auth_token to be True for push_dataset_to_hub"
        )

    if (cfg.base_model and "falcon" in cfg.base_model.lower()) and cfg.fsdp:
        raise ValueError("FSDP is not supported for falcon models")

    if (
        cfg.base_model and "mpt" in cfg.base_model.lower()
    ) and cfg.gradient_checkpointing:
        raise ValueError("gradient_checkpointing is not supported for MPT models")

    if cfg.flash_optimum is True:
        if cfg.adapter:
            LOG.warning("BetterTransformers probably doesn't work with PEFT adapters")
        if cfg.fp16 or cfg.bf16:
            raise ValueError("AMP is not supported with BetterTransformer")
        if cfg.float16 is not True and cfg.bloat16 is not True:
            LOG.warning(
                "You should probably set bfloat16 or float16 to true to "
                "load the model in float16 for BetterTransformers"
            )
        if int(torch.__version__.split(".", maxsplit=1)[0]) < 2:
            LOG.warning("torch>=2.0.0 required")
            raise ValueError(
                f"flash_optimum for BetterTransformers may not be used with {torch.__version__}"
            )

    if cfg.pretraining_dataset and cfg.group_by_length:
        LOG.warning(
            "You probably want to disable group_by_length as it will force a streamed dataset to download completely."
        )

    if any([cfg.adam_beta1, cfg.adam_beta2, cfg.adam_epsilon]) and (
        not cfg.optimizer or "adamw" not in cfg.optimizer
    ):
        LOG.warning("adamw hyperparameters found, but no adamw optimizer set")

    if cfg.push_to_hub_model_id:
        raise ValueError(
            "push_to_hub_model_id is deprecated. Please use hub_model_id instead."
        )

    if cfg.gptq and cfg.model_revision:
        raise ValueError(
            "model_revision is not supported for GPTQ models. "
            + "Please download the model from HuggingFace Hub manually for correct branch, "
            + "point to its path, and remove model_revision from the config."
        )

    if cfg.sample_packing and cfg.sdp_attention:
        # incompatible due to bug w/ accelerate causing 0.0 loss when using llama2
        raise ValueError(
            "sample_packing not compatible with sdp_attention. Use flash_attention"
        )

    if cfg.sample_packing and cfg.xformers_attention:
        raise ValueError(
            "sample_packing not compatible with xformers_attention. Use flash_attention"
        )

    # TODO
    # MPT 7b
    # https://github.com/facebookresearch/bitsandbytes/issues/25
    # no 8bit adaAmw w bf16

    # GPT-NeoX
    # evals broken when extending context len
    # File "/root/miniconda3/envs/py3.9/lib/python3.9/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py", line 162, in forward                        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    # File "/root/miniconda3/envs/py3.9/lib/python3.9/site-packages/optimum/bettertransformer/models/attention.py", line 74, in gpt2_wrapped_scaled_dot_product
    # attention_mask = causal_mask + attention_mask
    # RuntimeError: The size of tensor a (2048) must match the size of tensor b (8132) at non-singleton dimension 3

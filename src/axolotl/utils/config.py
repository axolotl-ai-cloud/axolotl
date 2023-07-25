"""Axolotl configuration utilities"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import click
import yaml

import axolotl
from axolotl.utils.data import load_prepare_datasets, load_pretraining_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.validation import validate_config

LOG = logging.getLogger(__name__)


def choose_config(path: Path) -> str:
    yaml_files = list(path.glob("*.ya?ml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

    print("Choose a YAML file:")
    for idx, file in enumerate(yaml_files):
        print(f"{idx + 1}. {file}")

    chosen_file = None
    while chosen_file is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(yaml_files):
                chosen_file = yaml_files[choice - 1]
            else:
                print("Invalid choice. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return Path(chosen_file)


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
    derived_config_file = (
        choose_config(config_path) if config_path.is_dir() else str(config_path)
    )

    with open(derived_config_file, encoding="utf-8") as config_fp:
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


def startup_load_dataset(cfg, tokenizer):
    if not cfg.pretraining_dataset:
        # Ideally the "last_run_prepared" default would be set in the configuration
        # file, keeping the logic from finetune.py to maintain compatibility
        derived_default_dataset_prepared_path = (
            cfg.dataset_prepared_path
            if cfg.dataset_prepared_path
            else "last_run_prepared"
        )
        LOG.info(
            "Loading dataset: %s",
            derived_default_dataset_prepared_path,
        )
        train_dataset, eval_dataset = load_prepare_datasets(
            tokenizer, cfg, derived_default_dataset_prepared_path
        )
    else:
        LOG.info("Loading pretraining dataset")
        train_dataset = load_pretraining_dataset(
            cfg.pretraining_dataset,
            tokenizer,
            max_tokens=cfg.sequence_len,
            seed=cfg.seed,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = None

    return train_dataset, eval_dataset


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
    fail_msg : str
        The message to display when the validation for the sub-parameter fails.
    required : bool
        Indicates whether the sub-parameter is required. If True, an exception will be raised if it is not present.
    """

    parser: Callable[[str], Union[int, float, bool, str]]
    fail_msg: str
    required: bool


def parse_and_validate_sub_params(
    unparsed_input: str, param_name: str, sub_param_def: Dict[str, SubParamEntry]
) -> DictDefault:
    """Parses a string of sub-parameters from a Click CLI command and validates them.

    Parameters
    ----------
    unparsed_input : str
        The input string containing the sub-parameters, formatted as "key=value" pairs separated by commas.
    param_name : str
        The name of the parameter that the sub-parameters belong to. This is used in error messages.
    sub_param_def : Dict[str, SubParamEntry]
        A dictionary that defines the expected sub-parameters. The keys are the names of the sub-parameters, and the
        values are SubParamEntry objects that specify the validation function, fail message, and whether the sub-parameter
        is required.

    Returns
    -------
    DictDefault
        A dictionary with the parsed and validated sub-parameters. The keys are the names of the sub-parameters, and the
        values are the parsed values.

    Raises
    ------
    click.BadParameter
        Raised when a sub-parameter is not formatted correctly, is not recognized, fails validation, or is required but missing.
    """
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
    except (ValueError, IndexError) as ex:
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
            raise click.BadParameter(
                sub_param_def[parsed_key].fail_msg.replace("%VALUE%", unparsed_value, 1)
            ) from ex

    # Check if each required sub-parameter is present in parsed_dict.
    # If not, raise a BadParameter exception with an appropriate error message.
    for subparam_key in sub_param_def.keys():
        if sub_param_def[subparam_key].required and subparam_key not in parsed_dict:
            raise click.BadParameter(
                f"The '{subparam_key}' sub-parameter is missing from {param_name} spec: '{unparsed_input}'"
            )

    # Return the parsed and validated sub-parameters.
    return parsed_dict

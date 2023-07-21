"""Axolotl configuration utilities"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import click

import yaml

import axolotl
from axolotl.utils.dict import DictDefault
from axolotl.utils.validation import validate_config


def choose_config(path: Path) -> str:
    yaml_files = list(path.glob("*.ya?ml"))

    if not yaml_files:
        raise ValueError("No YAML config files found in the specified directory. Are you using a .yml extension?")

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
    derived_config_file = choose_config(config_path) if config_path.is_dir() else config_path

    with open(derived_config_file, encoding="utf-8") as config_fp:
        loaded_config = DictDefault(yaml.safe_load(config_fp))

    validate_config(loaded_config)

    return loaded_config


def update_config(
    overrides: Union[Dict[str, Any], DictDefault], allow_none: bool = False, validate: bool = True
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
    updates = {key: value for key, value in overrides.items() if allow_none or value is not None}
    axolotl.cfg.update(**updates)

    if validate:
        validate_config(cfg=axolotl.cfg)


def option_factory(
    *args: str,
    override_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> Callable:
    """Factory function to generate a decorator for Click options with proper defaults.

    Parameters
    ----------
    args : variable length argument list of str
        Parameter declarations for the Click option (e.g., '--seed').

    override_kwargs : Dict[str, Any], optional
        A dictionary of override parameters for the Click option, by default None.

    **kwargs : Dict[str, Any]
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

    def decorator(func: Callable) -> Callable:
        func = click.option(
            *args,
            **derived_kwargs,
        )(func)

        return func

    # For testing
    decorator.click_params: DictDefault = DictDefault(
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

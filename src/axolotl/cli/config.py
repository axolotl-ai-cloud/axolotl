"""Configuration loading and processing."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
import torch
import yaml
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.integrations.base import PluginManager
from axolotl.utils.comet_ import setup_comet_env_vars
from axolotl.utils.config import (
    normalize_cfg_datasets,
    normalize_config,
    validate_config,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.mlflow_ import setup_mlflow_env_vars
from axolotl.utils.trainer import prepare_opinionated_env, prepare_optim_env
from axolotl.utils.wandb_ import setup_wandb_env_vars

LOG = logging.getLogger(__name__)


def check_remote_config(config: Union[str, Path]) -> Union[str, Path]:
    """
    First, determines if the passed config is a valid HTTPS URL. Then, attempts to query
    for it and parse its content, first as JSON, then as YAML (YAML is preferred).
    Finally, the parsed content is written to a local file and its path is returned.

    Args:
        config: HTTPS URL to a YAML or JSON file.

    Returns:
        Either the original `config` if it's not a valid HTTPS URL, or the path to the
        downloaded remote config.

    Raises:
        ValueError: If the remote configuration is neither valid JSON or YAML.
        RuntimeError: If some request-related exception occurs from the file download.
        Exception: Catch-all for any other exception.
    """
    # Check if the config is a valid HTTPS URL to a .yml or .yaml file
    if not (isinstance(config, str) and config.startswith("https://")):
        return config  # Return the original value if it's not a valid URL

    filename = os.path.basename(urlparse(config).path)
    temp_dir = tempfile.mkdtemp()

    try:
        response = requests.get(config, timeout=30)
        response.raise_for_status()  # Check for HTTP errors

        content = response.content
        try:
            # Try parsing as JSON first to catch cases where JSON content is mistakenly
            # considered YAML.
            json.loads(content)

            # Log a warning but do not raise an error; JSON is technically valid YAML.
            # This can happen when you forget to point to a raw GitHub link.
            LOG.warning(
                f"Warning: The content of the file at {config} is JSON, which is technically valid YAML but might not be intended."
            )
        except json.JSONDecodeError:
            # If it's not valid JSON, verify it's valid YAML
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as err:
                raise ValueError(
                    f"Failed to parse the content at {config} as YAML: {err}"
                ) from err

        # Write the content to a file if it's valid YAML (or JSON treated as YAML)
        output_path = Path(temp_dir) / filename
        with open(output_path, "wb") as file:
            file.write(content)
        LOG.info(
            f"Using the following config obtained from {config}: \n\n{content.decode('utf-8')}\n"
        )
        return output_path

    except requests.RequestException as err:
        # This catches all requests-related exceptions including HTTPError
        raise RuntimeError(f"Failed to download {config}: {err}") from err
    except Exception as err:
        # Catch-all for any other exceptions
        raise err


def choose_config(path: Path) -> str:
    """
    Helper method for choosing a `axolotl` config YAML file (considering only files
    ending with `.yml` or `.yaml`). If more than one config file exists in the passed
    `path`, the user is prompted to choose one.

    Args:
        path: Directory in which config file(s) are stored.

    Returns:
        Path to either (1) the sole YAML file, or (2) if more than one YAML files exist,
        the user-selected YAML file.

    Raises:
        ValueError: If no YAML files are found in the given `path`.
    """
    yaml_files = list(path.glob("*.yml")) + list(path.glob("*.yaml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

    if len(yaml_files) == 1:
        print(f"Using default YAML file '{yaml_files[0]}'")
        return str(yaml_files[0])

    print("Choose a YAML file:")
    for idx, file in enumerate(yaml_files):
        print(f"{idx + 1}. {file}")

    chosen_file = None
    while chosen_file is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(yaml_files):
                chosen_file = str(yaml_files[choice - 1])
            else:
                print("Invalid choice. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return chosen_file


def prepare_plugins(cfg: DictDefault):
    """
    Registers the plugins for the given configuration.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
    """
    if cfg.get("plugins"):
        plugin_manager = PluginManager.get_instance()
        for plugin_name in cfg["plugins"]:
            plugin_manager.register(plugin_name)


def load_cfg(config: Union[str, Path] = Path("examples/"), **kwargs) -> DictDefault:
    """
    Loads the `axolotl` configuration stored at `config`, validates it, and performs
    various setup.

    Args:
        config: Path (local or remote) to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.

    Returns:
        `DictDefault` mapping configuration keys to values.
    """
    config = check_remote_config(config)
    if Path(config).is_dir():
        config = choose_config(Path(config))

    # Load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))

    # If there are any options passed in the cli, if it is something that seems valid
    # from the yaml, then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    cfg.axolotl_config_path = config

    try:
        device_props = torch.cuda.get_device_properties("cuda")
        gpu_version = "sm_" + str(device_props.major) + str(device_props.minor)
    except:  # pylint: disable=bare-except # noqa: E722
        gpu_version = None

    prepare_plugins(cfg)

    cfg = validate_config(
        cfg,
        capabilities={
            "bf16": is_torch_bf16_gpu_available(),
            "n_gpu": int(os.environ.get("WORLD_SIZE", 1)),
            "compute_capability": gpu_version,
        },
        env_capabilities={
            "torch_version": str(torch.__version__).split("+", maxsplit=1)[0]
        },
    )

    prepare_optim_env(cfg)
    prepare_opinionated_env(cfg)
    normalize_config(cfg)
    normalize_cfg_datasets(cfg)
    setup_wandb_env_vars(cfg)
    setup_mlflow_env_vars(cfg)
    setup_comet_env_vars(cfg)

    return cfg

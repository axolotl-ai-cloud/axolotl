"""
Runpod launcher utils
"""

import os

import yaml


def get_output_dir(run_id):
    path = f"fine-tuning/{run_id}"
    return path


def make_valid_config(input_args):
    """
    Creates and saves updated config file, returns the path to the new config
    :param input_args: dict of input args
    :return: str, path to the updated config file
    """
    # Load default config
    with open("config/config.yaml", "r", encoding="utf-8") as fin:
        all_args = yaml.safe_load(fin)

    if not input_args:
        print("No args provided, using defaults")
    else:
        all_args.update(input_args)

    # Create updated config path
    updated_config_path = "config/updated_config.yaml"

    # Save updated config to new file
    with open(updated_config_path, "w", encoding="utf-8") as f:
        yaml.dump(all_args, f)

    return updated_config_path


def set_config_env_vars(args: dict):
    """
    Convert API arguments into environment variables.
    Handles nested dictionaries, lists, and special values.

    Args:
        args (dict): The arguments dictionary from the API request
    """

    def process_value(value):
        """Convert Python values to string format for environment variables"""
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, dict)):
            return str(value)
        return str(value)

    def set_env_vars(data, prefix=""):
        """Recursively set environment variables from nested dictionary"""
        for key, value in data.items():
            env_key = prefix + key.upper()

            # Handle special cases
            if isinstance(value, dict):
                # For nested dictionaries (like special_tokens)
                set_env_vars(value, f"{env_key}_")
            elif isinstance(value, list):
                # Handle list of dictionaries (like datasets)
                if value and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        set_env_vars(item, f"{env_key}_{i}_")
                else:
                    # For simple lists (like lora_target_modules)
                    os.environ[env_key] = process_value(value)
            else:
                # Handle all other cases
                os.environ[env_key] = process_value(value)

    # Clear any existing related environment variables
    # This prevents old values from persisting
    for key in list(os.environ.keys()):
        if key.startswith(
            ("BASE_MODEL", "MODEL_TYPE", "TOKENIZER_TYPE", "DATASET", "LORA_", "WANDB_")
        ):
            del os.environ[key]

    # Set new environment variables
    set_env_vars(args)

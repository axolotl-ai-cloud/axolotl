"""Utilities for handling sweeps over configs for axolotl train CLI command"""

import random
from copy import deepcopy
from itertools import product


def generate_sweep_configs(
    base_config: dict[str, list], sweeps_config: dict[str, list]
) -> list[dict[str, list]]:
    """
    Recursively generates all possible configurations by applying sweeps to the base config.

    Args:
        base_config (dict): The original configuration dictionary
        sweeps_config (dict): Dictionary where keys are parameters and values are either:
            - lists of values to sweep independently
            - or for paired values, a list of dicts under the '_' key

    Returns:
        list: List of all possible configuration dictionaries

    Example:
        sweeps_config = {
            'learning_rate': [0.1, 0.01],
            '_': [
                {'load_in_8bit': True, 'adapter': 'lora'},
                {'load_in_4bit': True, 'adapter': 'qlora'}
            ]
        }
    """
    # Separate paired values from regular sweeps
    paired_values = sweeps_config.get("_", [])
    regular_sweeps = {k: v for k, v in sweeps_config.items() if k != "_"}

    # Process regular sweeps
    param_names = list(regular_sweeps.keys())
    param_values = list(regular_sweeps.values())

    # Generate combinations for regular sweeps
    regular_combinations = list(product(*param_values)) if param_values else [()]

    # Combine regular sweeps with paired values
    all_combinations = []
    for reg_combo in regular_combinations:
        if paired_values:
            for paired_set in paired_values:
                new_config = {}
                # new_config = deepcopy(base_config)
                # Combine regular parameters with paired parameters
                full_combo = {**dict(zip(param_names, reg_combo)), **paired_set}
                for param_name, param_value in full_combo.items():
                    new_config[param_name] = param_value
                print(new_config)
                all_combinations.append(new_config)
        else:
            # If no paired values, just use regular combinations
            # new_config = deepcopy(base_config)
            new_config = {}
            for param_name, param_value in zip(param_names, reg_combo):
                new_config[param_name] = param_value
            print(new_config)
            all_combinations.append(new_config)

    # randomize the order of trials
    random.seed(42)
    random.shuffle(all_combinations)

    # Generate a new config for each combination
    result_configs = []
    for combination in all_combinations:
        new_config = deepcopy(base_config)
        for param_name, param_value in combination.items():
            new_config[param_name] = param_value
        result_configs.append(new_config)

    return result_configs

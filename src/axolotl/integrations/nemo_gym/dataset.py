# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Dataset loading for NeMo Gym JSONL files.

Converts NeMo Gym JSONL format into HuggingFace Datasets compatible
with TRL's GRPOTrainer.
"""

from __future__ import annotations

import json
import os
import random

from datasets import Dataset

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load_nemo_gym_datasets(
    gym_dir: str,
    dataset_configs: list[dict],
) -> Dataset:
    """Load and merge NeMo Gym JSONL datasets.

    Each dataset config should have:
        - path: JSONL file path (absolute, or relative to gym_dir)
        - server_name: Name of the NeMo Gym resource server for reward verification
        - max_samples (optional): Max number of samples to use from this dataset

    The output dataset has columns:
        - prompt: list[dict] chat format
        - resources_server_ref: dict with {"name": server_name}
        - verify_extra: dict with original JSONL data for verify requests

    Args:
        gym_dir: Path to the NeMo Gym directory.
        dataset_configs: List of dataset configuration dicts.

    Returns:
        A HuggingFace Dataset ready for GRPOTrainer.
    """
    all_examples = []

    for ds_cfg in dataset_configs:
        path = ds_cfg["path"]
        server_name = ds_cfg["server_name"]
        max_samples = ds_cfg.get("max_samples")

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(gym_dir, path)
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"NeMo Gym dataset not found at {path}. "
                "Ensure the dataset file exists or run the appropriate "
                "NeMo Gym dataset creation script."
            )

        LOG.info(f"Loading NeMo Gym dataset from {path} (server: {server_name})")

        with open(path) as f:
            lines = f.readlines()

        if max_samples and len(lines) > max_samples:
            lines = random.sample(lines, max_samples)

        for line in lines:
            data = json.loads(line)
            task_prompt = data["responses_create_params"]["input"][0]["content"]

            all_examples.append(
                {
                    "prompt": [{"role": "user", "content": task_prompt}],
                    "resources_server_ref": {"name": server_name},
                    "verify_extra": data,
                }
            )

    random.shuffle(all_examples)
    LOG.info(f"Loaded {len(all_examples)} NeMo Gym examples total")

    return Dataset.from_list(all_examples)

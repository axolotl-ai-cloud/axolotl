# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Dataset loading for NeMo Gym JSONL files.

Converts NeMo Gym JSONL format into HuggingFace Datasets compatible
with TRL's GRPOTrainer. Supports multi-environment routing via:
  1. Per-dataset server_name (all rows in a file go to one server)
  2. Per-row agent_ref.name (each row specifies its own server)
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
    """Load and merge NeMo Gym JSONL datasets with multi-environment support.

    Each dataset config should have:
        - path: JSONL file path (absolute, or relative to gym_dir)
        - server_name: Default NeMo Gym server for this dataset.
          Can be overridden per-row if the JSONL has an "agent_ref" field.
        - max_samples (optional): Max number of samples to use from this dataset

    Per-row routing: If a JSONL row has an "agent_ref": {"name": "..."} field,
    that takes precedence over the dataset-level server_name. This allows mixing
    environments within a single dataset file (matching TRL's pattern).

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
        default_server = ds_cfg.get("server_name", "")
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

        LOG.info(
            f"Loading NeMo Gym dataset from {path} (default server: {default_server})"
        )

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        if max_samples and len(lines) > max_samples:
            lines = random.sample(lines, max_samples)  # nosec B311

        for line in lines:
            data = json.loads(line)

            # Extract user prompt from the input messages
            inputs = data.get("responses_create_params", {}).get("input", [])
            task_prompt = ""
            for inp in inputs:
                if isinstance(inp, dict) and inp.get("role") in ("user",):
                    task_prompt = inp.get("content", "")
                    break
            if not task_prompt and inputs:
                # Fallback: use the last input's content
                task_prompt = (
                    inputs[-1].get("content", "")
                    if isinstance(inputs[-1], dict)
                    else ""
                )

            # Per-row agent routing: agent_ref.name can override dataset-level server_name.
            # NeMo Gym datasets may use agent names (e.g., "reasoning_gym_simple_agent")
            # which differ from resource server names (e.g., "reasoning_gym").
            # The dataset-level server_name is always the fallback.
            row_agent_ref = data.get("agent_ref", {})
            server_name = default_server
            if row_agent_ref and row_agent_ref.get("name"):
                # Use per-row name, but only if it looks like a resource server name.
                # Agent names typically have "_simple_agent" or "_agent" suffix.
                row_name = row_agent_ref["name"]
                if row_agent_ref.get("type") != "responses_api_agents":
                    # Not an agent — could be a direct resource server reference
                    server_name = row_name

            all_examples.append(
                {
                    "prompt": [{"role": "user", "content": task_prompt}],
                    "resources_server_ref": {"name": server_name},
                    "verify_extra": data,
                }
            )

    random.shuffle(all_examples)

    # Log environment distribution
    env_counts: dict[str, int] = {}
    for ex in all_examples:
        name = ex["resources_server_ref"]["name"]
        env_counts[name] = env_counts.get(name, 0) + 1
    LOG.info(f"Loaded {len(all_examples)} NeMo Gym examples: {env_counts}")

    return Dataset.from_list(all_examples)

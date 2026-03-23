# Copyright 2026 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Input arguments for the NeMo Gym integration plugin.
"""

from pydantic import BaseModel, Field, model_validator


class NemoGymArgs(BaseModel):
    """Configuration args for the NeMo Gym integration."""

    nemo_gym_enabled: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Enable NeMo Gym integration for environment-based RL rewards."
        },
    )
    nemo_gym_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Path to the NeMo Gym repository clone. "
                "If not set and nemo_gym_auto_clone is True, clones to ~/Gym."
            )
        },
    )
    nemo_gym_auto_clone: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Automatically clone the NeMo Gym repository if not present. "
                "Defaults to True when nemo_gym_enabled is set."
            )
        },
    )
    nemo_gym_config_paths: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "List of NeMo Gym resource server config YAML paths, relative to nemo_gym_dir. "
                "Example: ['resources_servers/reasoning_gym/configs/resources_only.yaml']"
            )
        },
    )
    nemo_gym_head_port: int | None = Field(
        default=11000,
        json_schema_extra={
            "description": "Port for the NeMo Gym head server. Defaults to 11000."
        },
    )
    nemo_gym_server_timeout: int | None = Field(
        default=360,
        json_schema_extra={
            "description": "Timeout in seconds waiting for NeMo Gym servers to start. Defaults to 360."
        },
    )
    nemo_gym_verify_timeout: int | None = Field(
        default=30,
        json_schema_extra={
            "description": "Timeout in seconds for individual /verify requests. Defaults to 30."
        },
    )
    nemo_gym_run_timeout: int | None = Field(
        default=300,
        json_schema_extra={
            "description": (
                "Timeout in seconds for each agent /run request (one multi-turn rollout). "
                "Prevents stuck generations (e.g. model looping on <think> tags) from "
                "blocking training indefinitely. Defaults to 300 (5 minutes)."
            )
        },
    )
    nemo_gym_datasets: list[dict] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "List of NeMo Gym dataset configs. Each entry has 'path' (JSONL file path "
                "relative to nemo_gym_dir) and optionally 'server_name' (default resource server). "
                "If the JSONL rows have agent_ref.name, that takes precedence per row, "
                "enabling multi-environment training from a single dataset file. "
                "Optional 'max_samples' to limit per dataset."
            )
        },
    )
    nemo_gym_auto_start: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Automatically start NeMo Gym resource servers. Defaults to True. "
                "Set to False if servers are already running externally."
            )
        },
    )
    nemo_gym_model_name: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Model name to report in verify requests. "
                "Defaults to the base_model from the main config."
            )
        },
    )
    nemo_gym_multi_turn: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Enable multi-turn rollouts via NeMo Gym. When True, uses TRL's "
                "rollout_func to run multi-step interactions with tool execution. "
                "Requires use_vllm=True in TRL config. The model generates responses, "
                "tool calls are executed against resource servers, and results are "
                "fed back for the next turn. Final reward comes from /verify."
            )
        },
    )
    nemo_gym_max_turns: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Maximum number of turns per multi-turn rollout. Defaults to 10. "
                "Each turn consists of a model generation + optional tool execution."
            )
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_nemo_gym_config(cls, data):
        if data.get("nemo_gym_enabled"):
            if not data.get("nemo_gym_config_paths") and data.get(
                "nemo_gym_auto_start", True
            ):
                raise ValueError(
                    "nemo_gym_config_paths is required when nemo_gym_enabled=True "
                    "and nemo_gym_auto_start is not False."
                )
            if not data.get("nemo_gym_datasets"):
                raise ValueError(
                    "nemo_gym_datasets is required when nemo_gym_enabled=True. "
                    "Provide at least one dataset with 'path' and 'server_name'."
                )
        return data

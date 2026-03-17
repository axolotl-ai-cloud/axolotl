# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
NeMo Gym Plugin for Axolotl.

Integrates NVIDIA NeMo Gym environments as reward sources for GRPO training.
Handles server lifecycle, dataset loading, and reward function wiring.

Supports two modes:
  - Single-turn (default): reward_fn calls /verify after each generation
  - Multi-turn (nemo_gym_multi_turn: true): rollout_func orchestrates
    multi-step interactions with tool execution via resource servers
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.common.datasets import TrainDatasetMeta

LOG = get_logger(__name__)


class NemoGymPlugin(BasePlugin):
    """Plugin for NVIDIA NeMo Gym integration with Axolotl.

    When enabled, this plugin:
    1. Clones and sets up the NeMo Gym repo (if needed)
    2. Starts NeMo Gym resource servers
    3. Loads datasets from NeMo Gym JSONL files
    4. For single-turn: creates a reward function calling /verify
    5. For multi-turn: creates a rollout_func with tool execution and env_mask
    """

    def __init__(self):
        super().__init__()
        self._gym_dir = None
        self._global_config = None
        self._verify_endpoints = None
        self._server_base_urls = None
        self._reward_fn = None
        self._dataset_lookup = None

    def get_input_args(self):
        return "axolotl.integrations.nemo_gym.NemoGymArgs"

    def pre_model_load(self, cfg):
        """Apply monkeypatches before trainer creation."""
        if not cfg.nemo_gym_enabled:
            return

        # When using LoRA sync server mode, skip NCCL communicator init
        # (the async trainer does this too, but we need it for standard GRPO)
        trl_cfg = getattr(cfg, "trl", None)
        if trl_cfg and getattr(trl_cfg, "vllm_lora_sync", False):
            self._patch_skip_nccl_init()

    def _patch_skip_nccl_init(self):
        """Monkeypatch VLLMClient.init_communicator to no-op for LoRA sync.

        LoRA sync uses HTTP (POST /set_lora_adapter/) instead of NCCL broadcast.
        The NCCL communicator is not needed and fails with vLLM V1 engine.
        """
        try:
            from trl.generation.vllm_client import VLLMClient

            VLLMClient._original_init_communicator = VLLMClient.init_communicator
            VLLMClient.init_communicator = lambda self, **kwargs: LOG.info(
                "Skipping NCCL init_communicator (LoRA sync mode)"
            )
            LOG.info("Patched VLLMClient.init_communicator to no-op for LoRA sync")
        except Exception as exc:
            LOG.warning(f"Failed to patch VLLMClient: {exc}")

    def register(self, cfg):
        if not cfg.get("nemo_gym_enabled"):
            return

        LOG.info("NeMo Gym integration enabled")
        gym_dir = cfg.get("nemo_gym_dir") or os.path.expanduser("~/Gym")
        auto_clone = cfg.get("nemo_gym_auto_clone", True)
        auto_start = cfg.get("nemo_gym_auto_start", True)
        head_port = cfg.get("nemo_gym_head_port", 11000)
        server_timeout = cfg.get("nemo_gym_server_timeout", 360)

        from .server import (
            ensure_gym_repo,
            ensure_gym_venv,
            get_server_base_url,
            get_server_configs,
            get_verify_endpoint,
            start_servers,
            wait_for_resource_servers,
        )

        self._gym_dir = ensure_gym_repo(gym_dir, auto_clone=auto_clone)

        if auto_start:
            config_paths = cfg.get("nemo_gym_config_paths", [])
            ensure_gym_venv(self._gym_dir)
            start_servers(
                self._gym_dir,
                config_paths,
                head_port=head_port,
                timeout=server_timeout,
            )

        self._global_config = get_server_configs(head_port=head_port)
        wait_for_resource_servers(self._global_config)

        # Build endpoint maps
        self._verify_endpoints = {}
        self._server_base_urls = {}
        for server_name in self._global_config:
            try:
                self._verify_endpoints[server_name] = get_verify_endpoint(
                    self._global_config, server_name
                )
                self._server_base_urls[server_name] = get_server_base_url(
                    self._global_config, server_name
                )
            except (ValueError, KeyError, TypeError):
                pass  # Skip non-server config entries

        # Pre-build dataset lookup for multi-turn (needs to happen at register time,
        # not load_datasets, because load_datasets may not be called if axolotl config
        # has its own datasets field)
        if cfg.get("nemo_gym_multi_turn") and cfg.get("nemo_gym_datasets"):
            from .dataset import load_nemo_gym_datasets

            gym_dir = cfg.get("nemo_gym_dir") or os.path.expanduser("~/Gym")
            dataset = load_nemo_gym_datasets(gym_dir, cfg["nemo_gym_datasets"])
            self._dataset_lookup = {}
            for i in range(len(dataset)):
                row = dataset[i]
                prompt_text = row["prompt"][0]["content"]
                self._dataset_lookup[prompt_text] = row
            LOG.info(f"Built dataset lookup with {len(self._dataset_lookup)} entries")

        multi_turn = cfg.get("nemo_gym_multi_turn", False)
        LOG.info(
            f"NeMo Gym ready with servers: {list(self._verify_endpoints.keys())} "
            f"(multi_turn={'enabled' if multi_turn else 'disabled'})"
        )

    def load_datasets(
        self, cfg, preprocess=False
    ) -> Union["TrainDatasetMeta", None]:
        if not cfg.nemo_gym_enabled:
            return None

        from axolotl.common.datasets import TrainDatasetMeta

        from .dataset import load_nemo_gym_datasets

        dataset_configs = cfg.nemo_gym_datasets
        dataset = load_nemo_gym_datasets(self._gym_dir, dataset_configs)

        # Build prompt → row lookup for multi-turn rollout_func
        # (rollout_func only receives prompt text, needs to look up row data)
        self._dataset_lookup = {}
        for i in range(len(dataset)):
            row = dataset[i]
            prompt_text = row["prompt"][0]["content"]
            self._dataset_lookup[prompt_text] = row

        return TrainDatasetMeta(
            train_dataset=dataset,
            eval_dataset=None,
            total_num_steps=0,  # computed later by the builder
        )

    def get_training_args(self, cfg):
        """Pass through vLLM colocate settings that axolotl doesn't map natively."""
        args = {}
        # Always pass vLLM settings from the vllm config block to TRL training args
        if cfg.vllm:
            vllm_cfg = cfg.vllm
            max_len = getattr(vllm_cfg, "max_model_len", None)
            gpu_util = getattr(vllm_cfg, "gpu_memory_utilization", None)
            tp_size = getattr(vllm_cfg, "tensor_parallel_size", None)
            if max_len:
                args["vllm_max_model_length"] = max_len
            if gpu_util:
                args["vllm_gpu_memory_utilization"] = gpu_util
            if tp_size:
                args["vllm_tensor_parallel_size"] = tp_size
        if args:
            LOG.info(f"NeMo Gym plugin injecting vLLM training args: {args}")
        return args if args else None

    def post_trainer_create(self, cfg, trainer):
        """Wire NeMo Gym into the trainer (reward_fn or rollout_func)."""
        if not cfg.nemo_gym_enabled:
            return

        model_name = cfg.nemo_gym_model_name or cfg.base_model or "axolotl-model"
        verify_timeout = cfg.nemo_gym_verify_timeout or 30
        multi_turn = cfg.nemo_gym_multi_turn or False

        # When using LoRA sync, make sync_weights a no-op
        # (weight sync is handled by _sync_lora_adapter via POST /set_lora_adapter/)
        # TRL's _generate_single_turn calls sync_weights when global_step != _last_loaded_step.
        # We override by making _last_loaded_step always match, and by replacing sync_weights.
        trl_cfg = getattr(cfg, "trl", None)
        if trl_cfg and getattr(trl_cfg, "vllm_lora_sync", False):
            if hasattr(trainer, "vllm_generation") and trainer.vllm_generation:
                # Replace sync_weights with a simple no-op on the instance
                # This must account for the trl_vllm monkeypatch wrapper
                trainer.vllm_generation.sync_weights = lambda: None
                # Also override the class-level method for any other call paths
                type(trainer.vllm_generation).sync_weights = lambda self: None
                LOG.info("Patched sync_weights to no-op for LoRA sync")

        if multi_turn:
            self._wire_multi_turn(cfg, trainer, model_name, verify_timeout)
        else:
            self._wire_single_turn(trainer, model_name, verify_timeout)

    def _wire_single_turn(self, trainer, model_name, verify_timeout):
        """Inject single-turn reward function into the trainer."""
        from .rewards import create_nemo_gym_reward_fn

        self._reward_fn = create_nemo_gym_reward_fn(
            global_config=self._global_config,
            verify_endpoints=self._verify_endpoints,
            model_name=model_name,
            verify_timeout=verify_timeout,
        )

        if hasattr(trainer, "reward_funcs"):
            trainer.reward_funcs.append(self._reward_fn)
            trainer.reward_func_names.append("nemo_gym")
            trainer.reward_processing_classes.append(None)
            LOG.info(
                f"Added NeMo Gym reward function (single-turn). "
                f"Total reward functions: {len(trainer.reward_funcs)}"
            )
        else:
            LOG.warning(
                "Trainer does not have reward_funcs attribute. "
                "NeMo Gym reward function not injected. "
                "Ensure you are using a GRPO trainer."
            )

    def _wire_multi_turn(self, cfg, trainer, model_name, verify_timeout):
        """Inject multi-turn rollout_func into the trainer."""
        from .multi_turn import create_multi_turn_rollout_func

        max_turns = cfg.nemo_gym_max_turns or 10
        tool_timeout = cfg.nemo_gym_verify_timeout or 30

        rollout_func = create_multi_turn_rollout_func(
            global_config=self._global_config,
            verify_endpoints=self._verify_endpoints,
            server_base_urls=self._server_base_urls,
            dataset_lookup=self._dataset_lookup or {},
            model_name=model_name,
            max_turns=max_turns,
            verify_timeout=verify_timeout,
            tool_timeout=tool_timeout,
        )

        # Wire the rollout_func into both the trainer and vllm_generation.
        # TRL checks trainer.rollout_func at init time (line 706-711) and wraps it
        # into a closure that gets passed to VLLMGeneration. Since we inject after
        # init, we need to set it on both objects.
        if hasattr(trainer, "vllm_generation") and trainer.vllm_generation:
            bound_rollout = lambda prompts: rollout_func(prompts, trainer)  # noqa: E731
            trainer.vllm_generation.rollout_func = bound_rollout
            # Also set on trainer itself (for any code that checks trainer.rollout_func)
            trainer.rollout_func = rollout_func
            LOG.info(
                f"NeMo Gym multi-turn rollout function injected "
                f"(max_turns={max_turns})"
            )
        else:
            LOG.warning(
                "Trainer does not support rollout_func. "
                "Multi-turn requires a GRPO trainer with vLLM."
            )

        # Also add a reward function that reads pre-computed rewards from rollout
        def nemo_gym_reward_from_rollout(completions, **kwargs):
            rewards = kwargs.get("nemo_gym_reward")
            if rewards is not None:
                return [float(r) for r in rewards]
            return [0.0 for _ in completions]

        if hasattr(trainer, "reward_funcs"):
            trainer.reward_funcs.append(nemo_gym_reward_from_rollout)
            trainer.reward_func_names.append("nemo_gym")
            trainer.reward_processing_classes.append(None)

    def post_train_unload(self, cfg):
        """Cleanup NeMo Gym servers if we started them."""
        if cfg.get("nemo_gym_enabled") and cfg.get("nemo_gym_auto_start", True):
            from .server import _cleanup_servers

            _cleanup_servers()

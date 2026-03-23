# Copyright 2026 Axolotl AI. All rights reserved.
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
        self._agent_servers = {}

    def get_input_args(self):
        return "axolotl.integrations.nemo_gym.NemoGymArgs"

    def pre_model_load(self, cfg):
        """Apply monkeypatches before trainer creation."""
        if not cfg.nemo_gym_enabled:
            return

        # Always skip NCCL communicator init in NeMo Gym mode.
        # NeMo Gym uses its own vLLM server (standard OpenAI API), not the TRL
        # colocate/NCCL path. The NCCL init fails with vLLM V1 and standard servers.
        trl_cfg = getattr(cfg, "trl", None)
        if trl_cfg and getattr(trl_cfg, "vllm_mode", "server") == "server":
            self._patch_skip_nccl_init()

    def _patch_skip_nccl_init(self):
        """Monkeypatch VLLMClient.init_communicator to no-op.

        NeMo Gym uses its own vLLM server (standard OpenAI API or custom LoRA
        serve script). The NCCL communicator is not needed and fails with both
        vLLM V1 engine and standard OpenAI server mode.
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
            get_agent_servers,
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
        wait_for_resource_servers(self._global_config, timeout=server_timeout)

        # Build endpoint maps for resource servers (/verify)
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
                pass

        # Discover agent servers (/run) for multi-turn
        self._agent_servers = get_agent_servers(self._global_config)

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
                # Use last message content as key (matches data_producer lookup)
                prompt_text = row["prompt"][-1]["content"]
                self._dataset_lookup[prompt_text] = row
            LOG.info(f"Built dataset lookup with {len(self._dataset_lookup)} entries")

        multi_turn = cfg.get("nemo_gym_multi_turn", False)
        LOG.info(
            f"NeMo Gym ready with servers: {list(self._verify_endpoints.keys())} "
            f"(multi_turn={'enabled' if multi_turn else 'disabled'})"
        )

    def load_datasets(self, cfg, preprocess=False) -> Union["TrainDatasetMeta", None]:
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
            # Use last message content as key (matches data_producer lookup)
            prompt_text = row["prompt"][-1]["content"]
            self._dataset_lookup[prompt_text] = row

        return TrainDatasetMeta(
            train_dataset=dataset,
            eval_dataset=None,
            total_num_steps=0,  # computed later by the builder
        )

    def get_training_args(self, cfg):
        """Pass through vLLM settings and force async trainer for multi-turn."""
        args = {}
        # Pass vLLM settings from vllm config block to TRL training args
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

        # Force async trainer for multi-turn: NemoGymDataProducer needs the
        # data producer protocol. Setting use_data_producer=True selects
        # AxolotlAsyncGRPOTrainer which supports _create_data_producer().
        # With async_prefetch=False this runs synchronously — no threading.
        if cfg.nemo_gym_multi_turn and self._agent_servers:
            args["use_data_producer"] = True
            LOG.info(
                "NeMo Gym multi-turn: forcing use_data_producer=True for data producer protocol"
            )

        # Dataloader workers fork subprocesses that can't handle the async
        # HTTP connections to NeMo Gym agents. Force num_workers=0.
        if getattr(cfg, "dataloader_num_workers", None) not in (None, 0):
            LOG.warning(
                f"NeMo Gym: overriding dataloader_num_workers={cfg.dataloader_num_workers} → 0 "
                "(forked workers can't use NeMo Gym agent connections)"
            )
        cfg.dataloader_num_workers = 0

        if args:
            LOG.info(f"NeMo Gym plugin injecting training args: {args}")
        return args if args else None

    def post_trainer_create(self, cfg, trainer):
        """Wire NeMo Gym into the trainer (reward_fn or rollout_func)."""
        if not cfg.nemo_gym_enabled:
            return

        model_name = cfg.nemo_gym_model_name or cfg.base_model or "axolotl-model"
        verify_timeout = cfg.nemo_gym_verify_timeout or 30
        multi_turn = cfg.nemo_gym_multi_turn or False

        # Handle weight sync. NeMo Gym skips NCCL init, so we need to either:
        # - Install LoRA sync (when vllm_lora_sync=True)
        # - Or no-op sync_weights (when using standard vLLM server)
        trl_cfg = getattr(cfg, "trl", None)
        if hasattr(trainer, "vllm_generation") and trainer.vllm_generation:
            vllm_gen = trainer.vllm_generation
            if trl_cfg and getattr(trl_cfg, "vllm_lora_sync", False):
                self._setup_lora_sync(trainer)
                # Verify the vLLM server supports runtime LoRA loading
                self._check_lora_endpoint(vllm_gen)
            else:
                # No NCCL, no LoRA sync — skip all weight sync paths
                vllm_gen.sync_weights = lambda: LOG.debug(
                    "Weight sync skipped (NeMo Gym mode)"
                )
                type(vllm_gen).sync_weights = lambda self: LOG.debug(
                    "Weight sync skipped (NeMo Gym mode)"
                )
                # Also patch the async trainer's internal sync method
                if hasattr(trainer, "_maybe_sync_vllm_weights"):
                    trainer._maybe_sync_vllm_weights = lambda: LOG.debug(
                        "Async weight sync skipped (NeMo Gym mode)"
                    )
                LOG.info("Disabled weight sync (NeMo Gym mode, no LoRA sync)")

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
        """Replace the data producer with NemoGymDataProducer.

        The plugin forces use_data_producer=True (in get_training_args) which
        selects AxolotlAsyncGRPOTrainer. Here we swap its data_producer with
        our NemoGymDataProducer that calls agent /run instead of vLLM generate.
        """
        if not self._agent_servers:
            LOG.warning(
                "No NeMo Gym agent servers discovered. Multi-turn requires agent servers "
                "started via ng_run with an agent config. Falling back to single-turn."
            )
            self._wire_single_turn(trainer, model_name, verify_timeout)
            return

        if not hasattr(trainer, "data_producer") or trainer.data_producer is None:
            LOG.warning(
                "Trainer has no data_producer. NeMo Gym multi-turn requires "
                "use_data_producer=true (should be auto-set by plugin)."
            )
            return

        from axolotl.core.trainers.grpo.async_trainer import AsyncDataProducer

        from .data_producer import NemoGymDataProducer

        # Get the current producer's config and params
        current = trainer.data_producer
        # Unwrap AsyncDataProducer to get the inner producer's config
        if isinstance(current, AsyncDataProducer):
            inner = current._inner
        else:
            inner = current

        nemo_producer = NemoGymDataProducer(
            config=inner.config,
            prompt_dataset=inner._dataset,
            num_generations=inner._num_generations,
            generation_batch_size=inner._generation_batch_size,
            train_batch_size=inner._train_batch_size,
            steps_per_generation=inner._steps_per_generation,
            shuffle_dataset=inner._shuffle_dataset,
            seed=inner._seed,
            agent_servers=self._agent_servers,
            dataset_lookup=self._dataset_lookup or {},
            request_timeout=float(cfg.nemo_gym_run_timeout or 300),
        )
        nemo_producer.set_trainer(trainer)

        # Re-wrap in AsyncDataProducer if async prefetch is enabled
        if getattr(trainer.args, "async_prefetch", False):
            nemo_producer = AsyncDataProducer(
                nemo_producer,
                background_produce_kwargs={"skip_policy_logps": True},
            )

        trainer.data_producer = nemo_producer
        LOG.info(
            f"NeMo Gym data producer installed "
            f"(agent servers: {list(self._agent_servers.keys())}, "
            f"async={'yes' if getattr(trainer.args, 'async_prefetch', False) else 'no'})"
        )

        # Passthrough reward function — agent /run already computed rewards
        from .rewards import reward_env

        if hasattr(trainer, "reward_funcs"):
            trainer.reward_funcs.append(reward_env)
            trainer.reward_func_names.append("nemo_gym")
            trainer.reward_processing_classes.append(None)

    @staticmethod
    def _check_lora_endpoint(vllm_gen):
        """Verify the vLLM server supports runtime LoRA loading."""
        import requests as http_requests

        if not hasattr(vllm_gen, "vllm_client") or vllm_gen.vllm_client is None:
            return  # Non-main rank in multi-GPU — client only exists on rank 0
        base_url = vllm_gen.vllm_client.base_url
        try:
            # Send a dummy load request — if the endpoint exists, we get a
            # proper error (400/404 about the adapter), not a route 404.
            resp = http_requests.post(
                f"{base_url}/v1/load_lora_adapter",
                json={"lora_name": "__probe__", "lora_path": "/nonexistent"},
                timeout=5,
            )
            if (
                resp.status_code == 404
                and "Not Found" in resp.text
                and "adapter" not in resp.text.lower()
            ):
                LOG.warning(
                    "vLLM server does not expose /v1/load_lora_adapter. "
                    "Set VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 when starting vLLM, e.g.:\n"
                    "  VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 python -m vllm.entrypoints.openai.api_server "
                    "--enable-lora --max-lora-rank 64 ..."
                )
        except Exception:
            pass  # Server might not be up yet, sync will warn later

    def _setup_lora_sync(self, trainer):
        """Replace sync_weights with LoRA adapter sync via filesystem + HTTP.

        If the async trainer is detected (has ``_sync_lora_adapter``), delegates
        to it — that method already handles multi-GPU (FSDP/DeepSpeed state_dict
        gather, broadcast sync dir, barrier).

        Otherwise installs a standalone closure for the non-async GRPO path that
        saves the adapter and POSTs to ``/v1/load_lora_adapter``.
        """
        vllm_gen = trainer.vllm_generation

        # Async trainer path: delegate to its _sync_lora_adapter (multi-GPU safe)
        if hasattr(trainer, "_sync_lora_adapter"):

            def lora_sync_weights():
                trainer._sync_lora_adapter()

            vllm_gen.sync_weights = lora_sync_weights
            type(vllm_gen).sync_weights = lambda self: lora_sync_weights()
            LOG.info(
                "Installed LoRA adapter sync "
                "(delegates to async trainer._sync_lora_adapter)"
            )
            return

        # Non-async standard GRPO path: standalone closure
        import os
        import shutil
        import tempfile

        import requests as http_requests

        base_model = getattr(trainer.args, "model_name_or_path", None) or "axolotl-lora"
        sync_state = {"version": 0, "sync_dir": tempfile.mkdtemp(prefix="lora_sync_")}

        def lora_sync_weights():
            """Save LoRA adapter and load it into vLLM."""
            accelerator = vllm_gen.accelerator
            model = vllm_gen.model

            if vllm_gen.mode != "server":
                return

            sync_state["version"] += 1
            version = sync_state["version"]
            adapter_path = os.path.join(sync_state["sync_dir"], f"v{version}")

            wrapped_model = getattr(trainer, "model_wrapped", model)
            state_dict = accelerator.get_state_dict(wrapped_model)

            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(adapter_path, state_dict=state_dict)

                base_url = vllm_gen.vllm_client.base_url
                resp = http_requests.post(
                    f"{base_url}/v1/load_lora_adapter",
                    json={
                        "lora_name": base_model,
                        "lora_path": adapter_path,
                        "load_inplace": True,
                    },
                    timeout=30,
                )
                if resp.status_code != 200:
                    resp = http_requests.post(
                        f"{base_url}/set_lora_adapter/",
                        json={
                            "lora_name": "active_lora",
                            "lora_int_id": version,
                            "lora_path": adapter_path,
                        },
                        timeout=30,
                    )
                    if resp.status_code != 200:
                        LOG.warning(
                            f"Failed to set LoRA adapter: "
                            f"{resp.status_code} {resp.text}"
                        )
                        return

                try:
                    vllm_gen.vllm_client.reset_prefix_cache()
                except Exception as exc:
                    LOG.warning("Failed to reset prefix cache: %s", exc)

                if version > 1:
                    old = os.path.join(sync_state["sync_dir"], f"v{version - 1}")
                    if os.path.exists(old):
                        shutil.rmtree(old, ignore_errors=True)

                LOG.info(f"Synced LoRA adapter v{version} to vLLM ({adapter_path})")

            if accelerator.num_processes > 1:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()

        vllm_gen.sync_weights = lora_sync_weights
        type(vllm_gen).sync_weights = lambda self: lora_sync_weights()
        LOG.info("Installed LoRA adapter sync (standalone fallback)")

    def post_train_unload(self, cfg):
        """Cleanup NeMo Gym servers if we started them."""
        if cfg.get("nemo_gym_enabled") and cfg.get("nemo_gym_auto_start", True):
            from .server import _cleanup_servers

            _cleanup_servers()

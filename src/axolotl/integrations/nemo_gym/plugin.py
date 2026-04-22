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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.common.datasets import TrainDatasetMeta

LOG = get_logger(__name__)


# ---- vLLM weight-sync transport probe ------------------------------------


@dataclass
class VLLMWeightSyncCapabilities:
    """What weight-sync routes a vLLM server actually exposes.

    Discovered once at ``pre_model_load`` time by fetching the server's
    ``/openapi.json``. Drives the transport-selection table below.
    """

    nccl: bool = False  # /init_communicator/ + /update_named_param/
    lora_filesystem: bool = False  # /v1/load_lora_adapter (vLLM native)
    lora_axolotl: bool = False  # /set_lora_adapter/ (axolotl serve_lora extension)
    http_full: bool = False  # /http_update_weights/ (axolotl serve_lora extension)
    probed: bool = False
    probe_error: str | None = None
    routes: list[str] = field(default_factory=list)

    @property
    def any_full_param_sync(self) -> bool:
        """True if at least one transport can push full-model weights."""
        return self.nccl or self.http_full

    @property
    def any_lora_sync(self) -> bool:
        """True if at least one transport can push LoRA adapters."""
        return self.lora_filesystem or self.lora_axolotl or self.nccl


def probe_vllm_weight_sync(
    base_url: str, timeout: float = 5.0
) -> VLLMWeightSyncCapabilities:
    """Detect which weight-sync routes the configured vLLM server exposes.

    Uses the server's FastAPI ``/openapi.json`` — every weight-sync transport
    we care about is mounted as a POST route there. Falls back to all-False
    on any error so the caller can still decide what to do (typically: raise
    a clear error rather than silently no-op).
    """
    import requests

    caps = VLLMWeightSyncCapabilities()
    try:
        r = requests.get(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout)
        r.raise_for_status()
        spec = r.json()
        routes = sorted((spec.get("paths") or {}).keys())
        caps.routes = routes
        caps.nccl = "/init_communicator/" in routes and "/update_named_param/" in routes
        caps.lora_filesystem = "/v1/load_lora_adapter" in routes
        caps.lora_axolotl = "/set_lora_adapter/" in routes
        caps.http_full = "/http_update_weights/" in routes
        caps.probed = True
    except Exception as exc:
        caps.probe_error = f"{type(exc).__name__}: {exc}"
        LOG.warning(
            "NeMo Gym: failed to probe vLLM /openapi.json at %s — %s. "
            "Will fall back to LoRA-only behavior.",
            base_url,
            caps.probe_error,
        )
    return caps


def select_weight_sync_transport(
    caps: VLLMWeightSyncCapabilities,
    *,
    has_lora: bool,
    vllm_lora_sync_pref: bool,
) -> str:
    """Pick the right transport for a (server caps, model type) combo.

    Returns one of: ``"lora_filesystem"``, ``"nccl"``, ``"http_full"``, or
    ``"none"``. The caller decides what to do with ``"none"`` (typically:
    raise an error explaining the misconfiguration).

    Selection table:
        LoRA model + lora endpoint + lora-sync pref    → lora_filesystem
        LoRA model + lora endpoint                     → lora_filesystem
        LoRA model + nccl endpoint                     → nccl (broadcast merged adapter)
        Full model + nccl endpoint                     → nccl
        Full model + http endpoint                     → http_full
        anything else                                  → none
    """
    if has_lora:
        if (caps.lora_filesystem or caps.lora_axolotl) and vllm_lora_sync_pref:
            return "lora_filesystem"
        if caps.lora_filesystem or caps.lora_axolotl:
            return "lora_filesystem"
        if caps.nccl:
            return "nccl"
        return "none"
    # Full-parameter model
    if caps.nccl:
        return "nccl"
    if caps.http_full:
        return "http_full"
    return "none"


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
        self._vllm_caps: VLLMWeightSyncCapabilities | None = None

    def get_input_args(self):
        return "axolotl.integrations.nemo_gym.NemoGymArgs"

    def pre_model_load(self, cfg):
        """Probe vLLM weight-sync routes and conditionally bypass NCCL init.

        Replaces the previous unconditional ``init_communicator`` monkey-patch
        with a probe of the configured vLLM server's ``/openapi.json``. We only
        bypass NCCL init when the server we're talking to actually lacks the
        ``/init_communicator/`` route (i.e. stock ``vllm serve``); against
        TRL/axolotl serve modules that DO expose NCCL routes, we leave the
        standard TRL flow alone so full-finetune training can sync weights.
        """
        if not cfg.nemo_gym_enabled:
            return

        trl_cfg = getattr(cfg, "trl", None)
        if not (trl_cfg and getattr(trl_cfg, "vllm_mode", "server") == "server"):
            return

        host = getattr(trl_cfg, "vllm_server_host", None) or "127.0.0.1"
        port = getattr(trl_cfg, "vllm_server_port", None) or 8000
        base_url = f"http://{host}:{port}"
        self._vllm_caps = probe_vllm_weight_sync(base_url)

        if self._vllm_caps.probed:
            LOG.info(
                "NeMo Gym: vLLM weight-sync probe @ %s — nccl=%s lora_native=%s "
                "lora_axolotl=%s http_full=%s",
                base_url,
                self._vllm_caps.nccl,
                self._vllm_caps.lora_filesystem,
                self._vllm_caps.lora_axolotl,
                self._vllm_caps.http_full,
            )

        # Only bypass NCCL init when the server doesn't speak it. If NCCL is
        # available we leave VLLMClient.init_communicator alone so the
        # standard TRL sync flow can run for full-parameter training.
        if not self._vllm_caps.nccl:
            self._patch_skip_nccl_init()

    def _patch_skip_nccl_init(self):
        """Monkeypatch VLLMClient.init_communicator to no-op.

        Only called when the configured vLLM server doesn't expose
        ``/init_communicator/`` (e.g. stock ``vllm serve``). In that case
        TRL's standard ``init_communicator`` would 404 inside trainer
        construction; we no-op it so the LoRA filesystem path can install
        its own sync in ``post_trainer_create``.
        """
        try:
            from trl.generation.vllm_client import VLLMClient

            VLLMClient._original_init_communicator = VLLMClient.init_communicator
            VLLMClient.init_communicator = lambda self, **kwargs: LOG.info(
                "Skipping NCCL init_communicator (server has no /init_communicator/)"
            )
            LOG.info(
                "Patched VLLMClient.init_communicator to no-op (server has no NCCL routes)"
            )
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

        # Pick a weight-sync transport based on what the configured vLLM
        # server actually exposes (see ``pre_model_load`` probe) and what
        # kind of model we're training. The selection table is documented
        # in ``select_weight_sync_transport``.
        trl_cfg = getattr(cfg, "trl", None)
        if hasattr(trainer, "vllm_generation") and trainer.vllm_generation:
            vllm_gen = trainer.vllm_generation
            adapter = getattr(cfg, "adapter", None)
            has_lora = adapter in ("lora", "qlora")
            vllm_lora_sync_pref = bool(
                trl_cfg and getattr(trl_cfg, "vllm_lora_sync", False)
            )
            caps = self._vllm_caps or VLLMWeightSyncCapabilities()
            transport = select_weight_sync_transport(
                caps,
                has_lora=has_lora,
                vllm_lora_sync_pref=vllm_lora_sync_pref,
            )

            if transport == "lora_filesystem":
                self._setup_lora_sync(trainer)
                self._check_lora_endpoint(vllm_gen)
                LOG.info("NeMo Gym weight sync: LoRA filesystem")
            elif transport == "nccl":
                # Standard TRL NCCL path. We leave ``VLLMClient.init_communicator``
                # alone (pre_model_load only patched it when the probe found no
                # NCCL route) so the trainer's normal weight-sync flow runs.
                LOG.info(
                    "NeMo Gym weight sync: NCCL (server exposes /init_communicator/)"
                )
            elif transport == "http_full":
                # Full-parameter HTTP sync — implementation lands in step 3.
                # For now, fail loudly so users know the path is detected but
                # not yet wired up, instead of silently no-oping like before.
                raise NotImplementedError(
                    "NeMo Gym + full fine-tune + HTTP weight sync is detected "
                    "but the client-side sync helper is not yet implemented "
                    "(planned). Use `adapter: lora|qlora` for now, or use a "
                    "vLLM serve module that exposes /init_communicator/ for "
                    "NCCL sync."
                )
            else:  # transport == "none"
                # No viable sync path. Build a precise error so the user knows
                # exactly what's missing and how to fix it.
                if not caps.probed:
                    msg = (
                        "could not probe the vLLM server's "
                        f"/openapi.json: {caps.probe_error}. "
                        "Verify that vLLM is reachable at "
                        f"{getattr(trl_cfg, 'vllm_server_host', '?')}:"
                        f"{getattr(trl_cfg, 'vllm_server_port', '?')}."
                    )
                elif has_lora:
                    msg = (
                        "the vLLM server has neither NCCL routes "
                        "(/init_communicator/) nor a LoRA-loading route "
                        "(/v1/load_lora_adapter or /set_lora_adapter/). "
                        "Restart vLLM with `--enable-lora --max-lora-rank N "
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` for the stock "
                        "server, or use `axolotl vllm-serve` for the "
                        "NCCL-capable serve module."
                    )
                else:
                    msg = (
                        "the vLLM server exposes no full-parameter sync route "
                        "(/init_communicator/ for NCCL or /http_update_weights/ "
                        "for HTTP). Use `axolotl vllm-serve` (which has both) "
                        "or set `adapter: lora|qlora`."
                    )
                raise ValueError(
                    f"NeMo Gym: no usable weight-sync transport — {msg} Without "
                    "weight sync the trainer's gradient updates never reach the "
                    "rollout policy (functionally a no-op trainer)."
                )

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

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
  - Single-turn (default): a reward function calls /verify after each generation
  - Multi-turn (nemo_gym_multi_turn: true, async GRPO): agent servers run whole
    rollouts, with tool execution, behind their /run endpoint
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

    nccl: bool = False  # /init_weight_transfer_engine + /update_weights (dev mode)
    lora_filesystem: bool = False  # /v1/load_lora_adapter (runtime LoRA updating)
    probed: bool = False
    probe_error: str | None = None
    routes: list[str] = field(default_factory=list)


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
        caps.nccl = (
            "/init_weight_transfer_engine" in routes and "/update_weights" in routes
        )
        caps.lora_filesystem = "/v1/load_lora_adapter" in routes
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

    Returns one of: ``"lora_filesystem"``, ``"nccl"``, or ``"none"``. The caller
    decides what to do with ``"none"`` (typically: raise an error explaining the
    misconfiguration).

    Selection table:
        LoRA model + lora endpoint + lora-sync pref    → lora_filesystem
        LoRA model + nccl endpoint, no lora-sync pref  → nccl (broadcast merged adapter)
        LoRA model + lora endpoint only                → lora_filesystem
        Full model + nccl endpoint                     → nccl
        anything else                                  → none
    """
    if has_lora and caps.lora_filesystem and (vllm_lora_sync_pref or not caps.nccl):
        return "lora_filesystem"
    return "nccl" if caps.nccl else "none"


def _on_async_grpo(cfg) -> bool:
    from axolotl.core.trainers.grpo import GRPOStrategy

    return GRPOStrategy.get_async_backend(cfg) == "trl"


class NemoGymPlugin(BasePlugin):
    """Plugin for NVIDIA NeMo Gym integration with Axolotl.

    When enabled, this plugin:
    1. Clones and sets up the NeMo Gym repo (if needed)
    2. Starts NeMo Gym resource servers
    3. Loads datasets from NeMo Gym JSONL files
    4. For single-turn: adds a reward function calling /verify
    5. For multi-turn: runs async GRPO's rollouts through agent /run endpoints
    """

    def __init__(self):
        super().__init__()
        self._gym_dir = None
        self._global_config = None
        self._verify_endpoints = None
        self._server_base_urls = None
        self._agent_servers = {}
        self._multi_turn = False
        self._vllm_caps: VLLMWeightSyncCapabilities | None = None

    def get_input_args(self):
        return "axolotl.integrations.nemo_gym.NemoGymArgs"

    def pre_model_load(self, cfg):
        """Probe vLLM weight-sync routes and conditionally bypass NCCL init.

        Replaces the previous unconditional ``init_communicator`` monkey-patch
        with a probe of the configured vLLM server's ``/openapi.json``. We only
        bypass NCCL init when the server we're talking to actually lacks the
        ``/init_weight_transfer_engine`` route (i.e. stock ``vllm serve`` without
        dev mode); against servers that DO expose it (``axolotl vllm-serve``),
        we leave the standard TRL flow alone so full-finetune training can
        sync weights.
        """
        if not cfg.nemo_gym_enabled:
            return
        # Async GRPO picks its weight sync from the server's `/server_info` itself.
        if _on_async_grpo(cfg):
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
                "NeMo Gym: vLLM weight-sync probe @ %s: nccl=%s lora_filesystem=%s",
                base_url,
                self._vllm_caps.nccl,
                self._vllm_caps.lora_filesystem,
            )

        # Only bypass NCCL init when the server doesn't speak it. If NCCL is
        # available we leave VLLMClient.init_communicator alone so the
        # standard TRL sync flow can run for full-parameter training.
        if not self._vllm_caps.nccl:
            self._patch_skip_nccl_init()

    def _patch_skip_nccl_init(self):
        """Monkeypatch VLLMClient.init_communicator to no-op.

        Only called when the configured vLLM server doesn't expose
        ``/init_weight_transfer_engine`` (e.g. stock ``vllm serve``). In that case
        TRL's standard ``init_communicator`` would 404 inside trainer
        construction; we no-op it so the LoRA filesystem path can install
        its own sync in ``post_trainer_create``.
        """
        try:
            from trl.generation.vllm_client import VLLMClient

            VLLMClient._original_init_communicator = VLLMClient.init_communicator
            VLLMClient.init_communicator = lambda self, *args, **kwargs: LOG.info(
                "Skipping NCCL init_communicator (server has no "
                "/init_weight_transfer_engine)"
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
        self._multi_turn = bool(cfg.get("nemo_gym_multi_turn"))
        if self._multi_turn and not self._agent_servers:
            LOG.warning(
                "No NeMo Gym agent servers discovered. Multi-turn requires agent servers "
                "started via ng_run with an agent config. Falling back to single-turn."
            )
            self._multi_turn = False

        LOG.info(
            f"NeMo Gym ready with servers: {list(self._verify_endpoints.keys())} "
            f"(multi_turn={'enabled' if self._multi_turn else 'disabled'})"
        )

    def load_datasets(self, cfg, preprocess=False) -> Union["TrainDatasetMeta", None]:
        if not cfg.nemo_gym_enabled:
            return None

        from axolotl.common.datasets import TrainDatasetMeta

        from .dataset import load_nemo_gym_datasets

        dataset_configs = cfg.nemo_gym_datasets
        dataset = load_nemo_gym_datasets(self._gym_dir, dataset_configs)

        return TrainDatasetMeta(
            train_dataset=dataset,
            eval_dataset=None,
            total_num_steps=0,  # computed later by the builder
        )

    @staticmethod
    def _model_name(cfg) -> str:
        return cfg.nemo_gym_model_name or cfg.base_model or "axolotl-model"

    def get_reward_funcs(self, cfg):
        """Single-turn rollouts are scored by the resource servers' /verify."""
        if not cfg.nemo_gym_enabled or self._multi_turn:
            return None
        from .rewards import NemoGymVerifyReward

        return [
            NemoGymVerifyReward(
                self._verify_endpoints or {},
                model_name=self._model_name(cfg),
                timeout=cfg.nemo_gym_verify_timeout or 30,
            )
        ]

    def get_training_args(self, cfg):
        """Run multi-turn rollouts through the agents, or pass through vLLM settings."""
        if not cfg.nemo_gym_enabled:
            return None
        args = {}
        if _on_async_grpo(cfg):
            if self._multi_turn:
                args["rollout_loop"] = (
                    "axolotl.integrations.nemo_gym.rollout.NemoGymRolloutLoop"
                )
                args["rollout_loop_kwargs"] = {
                    "agent_servers": dict(self._agent_servers),
                    "run_timeout": float(cfg.nemo_gym_run_timeout or 300),
                }
                if cfg.adapter in ("lora", "qlora"):
                    # The agents' model proxy names the policy by a fixed model name.
                    args["vllm_lora_alias"] = self._model_name(cfg)
        elif cfg.vllm:
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
        """Check and install weight sync for trainers that generate with vLLM."""
        if not cfg.nemo_gym_enabled:
            return

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
                    "NeMo Gym weight sync: NCCL (server exposes /init_weight_transfer_engine)"
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
                        "the vLLM server has neither native weight-transfer "
                        "routes (/init_weight_transfer_engine) nor "
                        "/v1/load_lora_adapter. Start it with `axolotl "
                        "vllm-serve`, which enables the native weight-transfer "
                        "routes and, with trl.vllm_lora_sync: true, runtime "
                        "LoRA loading; or run `vllm serve` with "
                        "VLLM_SERVER_DEV_MODE=1 --weight-transfer-config "
                        '\'{"backend":"nccl"}\' / --enable-lora + '
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True."
                    )
                else:
                    msg = (
                        "the vLLM server exposes no full-parameter sync route "
                        "(/init_weight_transfer_engine). Start it with `axolotl "
                        "vllm-serve`, or run `vllm serve` with "
                        "VLLM_SERVER_DEV_MODE=1 --weight-transfer-config "
                        '\'{"backend":"nccl"}\', or set `adapter: lora|qlora`.'
                    )
                raise ValueError(
                    f"NeMo Gym: no usable weight-sync transport — {msg} Without "
                    "weight sync the trainer's gradient updates never reach the "
                    "rollout policy (functionally a no-op trainer)."
                )

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

        If the data producer trainer is detected (has ``_sync_lora_adapter``),
        delegates to it — that method already handles multi-GPU (FSDP/DeepSpeed
        state_dict gather, broadcast sync dir, barrier).

        Otherwise installs a standalone closure for the non-async GRPO path that
        saves the adapter and POSTs to ``/v1/load_lora_adapter``.
        """
        vllm_gen = trainer.vllm_generation

        # Data producer trainer path: delegate to its _sync_lora_adapter (multi-GPU safe)
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
        import tempfile

        from axolotl.utils.vllm_lora_sync import publish_lora_adapter

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

                lora_name = publish_lora_adapter(
                    vllm_gen.vllm_client,
                    sync_state["sync_dir"],
                    version,
                    trainer.args.vllm_server_timeout,
                )
                LOG.info(
                    f"Synced LoRA adapter v{version} to vLLM as {lora_name} "
                    f"({adapter_path})"
                )

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

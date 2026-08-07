# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Axolotl plugin that routes SFT training to a remote Arctic Platform server."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoConfig, PreTrainedModel, Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ArcticSFTPlugin(BasePlugin):
    """Replaces local SFT forward/backward/step with ``ArcticSFTClient`` calls.

    Activated from the axolotl YAML::

        plugins:
          - axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin

        arctic_sft:
          host: localhost
          port: 8765
          training_gpus: 2
          launch_local_server: true
          server_cuda_visible_devices: "0,1"
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.arctic_platform.sft.args.ArcticSFTArgs"

    def register(self, cfg: dict):
        # The remote trainer consumes input_ids/attention_mask/labels directly,
        # so keep every tokenized column instead of letting HF drop "unused" ones.
        if cfg.get("remove_unused_columns") is None:
            cfg["remove_unused_columns"] = False

    def pre_model_load(self, cfg: DictDefault):
        """Skip local weight loading — the server owns the model weights."""
        LOG.info(
            "ArcticSFT plugin active: training dispatched to a remote Arctic "
            "Platform server. Skipping local model weight loading."
        )

        from axolotl.loaders import ModelLoader

        def _stub_build_model(loader_self) -> bool:
            base_model = loader_self.cfg.base_model
            LOG.info(f"Skipping local model weight loading for: {base_model}")

            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=loader_self.cfg.get("trust_remote_code", False),
            )

            class _Stub(PreTrainedModel):
                config_class = type(config)
                _no_split_modules: list[str] = []
                supports_gradient_checkpointing = False

                def __init__(self, cfg):
                    super().__init__(cfg)
                    vocab_size = getattr(cfg, "vocab_size", 32000)
                    self.embed_tokens = torch.nn.Embedding(vocab_size, 1)

                def get_input_embeddings(self):
                    return self.embed_tokens

                def set_input_embeddings(self, value):
                    pass

                def get_output_embeddings(self):
                    return None

            loader_self.model = _Stub(config)
            return True

        ModelLoader._build_model = _stub_build_model  # type: ignore[method-assign,assignment]

    def get_trainer_cls(self, cfg: DictDefault) -> type[Trainer] | None:
        from .trainer import ArcticSFTTrainer

        return ArcticSFTTrainer

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        model._arctic_sft_remote = True

    def post_train(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        LOG.info(
            "ArcticSFT: skipping local model save (weights live on the remote "
            "server). Checkpoints are written server-side under checkpoint_path."
        )

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        """Build the ArcticSFTClientConfig and attach it to the trainer."""
        from .args import ArcticSFTConfig
        from .trainer import ArcticSFTTrainer

        if not isinstance(trainer, ArcticSFTTrainer):
            return

        acfg = cfg.arctic_sft
        if isinstance(acfg, dict):
            acfg = ArcticSFTConfig(**acfg)
        elif acfg is None:
            raise ValueError(
                "ArcticSFTPlugin is enabled but no `arctic_sft:` config block was found."
            )

        client_config = self._build_client_config(cfg, acfg)

        if getattr(cfg, "generate_samples", False) and int(getattr(acfg, "sampling_gpus", 0) or 0) <= 0:
            raise ValueError(
                "arctic_sft: generate_samples=true requires arctic_sft.sampling_gpus > 0 "
                "(vLLM sampling job, same topology as RL)."
            )

        trainer._arctic_client_config = client_config
        trainer._arctic_loss_fn = acfg.loss_fn
        trainer._arctic_logits_optimization = acfg.logits_optimization
        trainer._arctic_logits_optimization_peak_mem_gib = acfg.logits_optimization_peak_mem_size_in_gib
        trainer._arctic_learning_rate = float(
            cfg.learning_rate if cfg.learning_rate is not None else 1e-5
        )
        trainer._arctic_export_hf = bool(acfg.export_hf)
        # Only a plugin-synthesized ds_config gets its LR-schedule horizon patched
        # with the runtime step count; never clobber a user-supplied ds_config.
        trainer._arctic_autoset_horizon = acfg.ds_config is None
        trainer.axolotl_cfg = cfg

        self._install_generation_callback(cfg, trainer)

    @staticmethod
    def _install_generation_callback(cfg: DictDefault, trainer: Trainer) -> None:
        """Replace stock local ``SFTGenerationCallback`` with remote Arctic one."""
        from axolotl.utils.callbacks.generation import SFTGenerationCallback

        from .callbacks import ArcticSFTGenerationCallback

        handler = getattr(trainer, "callback_handler", None)
        if handler is not None and getattr(handler, "callbacks", None) is not None:
            handler.callbacks = [
                cb for cb in handler.callbacks if not isinstance(cb, SFTGenerationCallback)
            ]
        if getattr(cfg, "generate_samples", False):
            trainer.add_callback(ArcticSFTGenerationCallback(trainer))
            LOG.info("Arctic SFT sample generation enabled (remote sampling job)")

    @classmethod
    def _build_client_config(cls, cfg: DictDefault, acfg):
        """Map the nested arctic_sft block + top-level axolotl knobs onto ArcticSFTClientConfig."""
        from .deps import require_arctic_sft_client

        _, ArcticSFTClientConfig = require_arctic_sft_client()

        model_name = acfg.model_name or cfg.base_model
        max_length = int(cfg.sequence_len or 2048)
        grad_accum = int(cfg.gradient_accumulation_steps or 1)
        micro_bs = int(cfg.micro_batch_size or 1)
        cls._validate_micro_batch(micro_bs, acfg.training_gpus)

        # Honor explicit overrides from the arctic_sft block, else synthesize
        # server configs from the top-level axolotl knobs.
        checkpoint_path = cls._resolve_checkpoint_path(cfg, acfg)
        # Optimizer + LR schedule are folded directly into ds_config (DeepSpeed
        # config-json), matching current arctic-platform which no longer accepts
        # a separate high-level training_config on ArcticSFTClientConfig.
        ds_config = acfg.ds_config or cls._synth_ds_config(cfg, acfg, micro_bs, grad_accum)
        ds_worker_config = acfg.ds_worker_config or cls._synth_ds_worker_config(cfg, acfg)

        return ArcticSFTClientConfig(
            backend=acfg.backend,
            comm_protocol=acfg.comm_protocol,
            model_name=model_name,
            # Server JobConfig.seed is a required int (defaults to 42); never pass None.
            seed=int(cfg.seed) if cfg.seed is not None else 42,
            max_seq_len=max_length,
            training_gpus=acfg.training_gpus,
            sampling_gpus=int(getattr(acfg, "sampling_gpus", 0) or 0),
            colocate=bool(getattr(acfg, "colocate", False)),
            vllm_config=getattr(acfg, "vllm_config", None),
            host=acfg.host,
            port=acfg.port,
            launch_local_server=acfg.launch_local_server,
            server_cuda_visible_devices=acfg.server_cuda_visible_devices,
            startup_timeout=acfg.startup_timeout,
            job_ready_timeout=acfg.job_ready_timeout,
            request_timeout=acfg.request_timeout,
            ds_config=ds_config,
            ds_worker_config=ds_worker_config,
            checkpoint_path=checkpoint_path,
            training_job_id=acfg.training_job_id,
            sampling_job_id=getattr(acfg, "sampling_job_id", None),
        )

    @staticmethod
    def _validate_micro_batch(micro_bs: int, training_gpus: int) -> None:
        # The server shards each batch across `training_gpus` DP ranks and then
        # splits each rank's shard into `grad_accum` microbatches, so the
        # micro-batch must divide evenly across the training GPUs.
        if micro_bs < training_gpus or micro_bs % training_gpus != 0:
            raise ValueError(
                f"arctic_sft: micro_batch_size ({micro_bs}) must be a positive multiple of "
                f"training_gpus ({training_gpus}) so the server can split it across DP ranks."
            )

    @staticmethod
    def _resolve_checkpoint_path(cfg: DictDefault, acfg):
        # The server requires a checkpoint_path for every new training job. When
        # the user didn't set one explicitly, derive it from axolotl's output_dir
        # so `arctic_sft: {training_gpus: N}` works without extra boilerplate.
        checkpoint_path = acfg.checkpoint_path
        if not checkpoint_path and acfg.training_job_id is None:
            output_dir = cfg.output_dir or "./outputs"
            checkpoint_path = f"{str(output_dir).rstrip('/')}/arctic_sft_ckpt"
        return checkpoint_path

    @staticmethod
    def _synth_ds_config(cfg: DictDefault, acfg, micro_bs: int, grad_accum: int) -> dict:
        # ZeRO-2 default sized from the axolotl batch knobs, with bf16 (or fp16)
        # mixed precision enabled to mirror axolotl's DeepSpeed training path
        # (BF16_Optimizer / fp32 master weights).
        if cfg.get("fp16") and not cfg.get("bf16"):
            mixed_precision = {"fp16": {"enabled": True}}
        else:
            mixed_precision = {"bf16": {"enabled": True}}
        # `micro_batch_size` is the axolotl per-step batch *before* the server
        # shards it across `training_gpus` DP ranks. DeepSpeed's
        # `train_micro_batch_size_per_gpu` is per-rank, so it must be the
        # post-shard size; `train_batch_size` is then the global effective
        # batch (per-gpu x gpus x grad_accum == micro_bs x grad_accum).
        micro_bs_per_gpu = micro_bs // acfg.training_gpus
        ds_config: dict = {
            "train_micro_batch_size_per_gpu": micro_bs_per_gpu,
            "train_batch_size": micro_bs_per_gpu * acfg.training_gpus * grad_accum,
            "gradient_accumulation_steps": grad_accum,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
            },
            **mixed_precision,
        }
        # Optimizer + gradient clipping live in ds_config (DeepSpeed config-json),
        # matching arctic-platform's SFT demos (the server no longer expands a
        # separate high-level training_config).
        ds_config["optimizer"] = ArcticSFTPlugin._synth_optimizer(cfg)
        ds_config["gradient_clipping"] = float(
            cfg.max_grad_norm if cfg.max_grad_norm is not None else 0.0
        )
        # LR schedule needs the total optimizer-step horizon. Prefer max_steps;
        # when only num_epochs is set the horizon is unknown here, so the trainer
        # re-applies the schedule with the resolved step count at train() start.
        horizon = int(cfg.max_steps) if (cfg.max_steps and int(cfg.max_steps) > 0) else 0
        ArcticSFTPlugin._apply_scheduler(ds_config, cfg, horizon)
        return ds_config

    @staticmethod
    def _synth_optimizer(cfg: DictDefault) -> dict:
        """DeepSpeed AdamW optimizer block from top-level axolotl knobs."""
        params: dict = {
            "lr": float(cfg.learning_rate if cfg.learning_rate is not None else 1e-5),
            "betas": [
                float(cfg.adam_beta1 if cfg.adam_beta1 is not None else 0.9),
                float(cfg.adam_beta2 if cfg.adam_beta2 is not None else 0.999),
            ],
            "eps": float(cfg.adam_epsilon if cfg.adam_epsilon is not None else 1e-8),
            "weight_decay": float(cfg.weight_decay if cfg.weight_decay is not None else 0.0),
        }
        # Map torch AdamW variants onto DeepSpeed's torch_adam flag (else FusedAdam).
        optimizer_name = str(cfg.optimizer or "").lower()
        if optimizer_name in ("adamw_torch", "adamw_torch_fused", "adamw_hf", "adamw_torch_8bit"):
            params["torch_adam"] = True
        return {"type": "AdamW", "params": params}

    @staticmethod
    def _apply_scheduler(ds_config: dict, cfg: DictDefault, horizon: int) -> None:
        """Set (or clear) ds_config["scheduler"] from axolotl warmup / LR-scheduler knobs.

        ``horizon`` is the total number of optimizer steps. Idempotent: safe to
        call again with a resolved ``horizon`` once the trainer knows the real
        step count (num_epochs-only configs)."""
        lr = float(cfg.learning_rate if cfg.learning_rate is not None else 1e-5)
        if cfg.warmup_steps is not None and int(cfg.warmup_steps) > 0:
            warmup_steps = float(int(cfg.warmup_steps))
        else:
            warmup_steps = float(cfg.warmup_ratio or 0.0) * horizon
        sched_name = str(cfg.lr_scheduler or "constant").lower()

        if sched_name.startswith("cosine") and horizon > 0:
            ds_config["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": horizon,
                    "warmup_num_steps": warmup_steps,
                    "warmup_min_ratio": 0.0,
                    "cos_min_ratio": float(cfg.cosine_min_lr_ratio or 0.0),
                    "warmup_type": "linear",
                },
            }
        elif warmup_steps > 0:
            ds_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": lr,
                    "warmup_num_steps": warmup_steps,
                    "warmup_type": "linear",
                },
            }
        else:
            # Constant LR (no warmup / no cosine): leave DeepSpeed on the raw LR.
            ds_config.pop("scheduler", None)

    @staticmethod
    def _synth_ds_worker_config(cfg: DictDefault, acfg) -> dict:
        # attn_implementation is top-level only (same as native axolotl). Default
        # FA2 matches DeepSpeedWorker when unset.
        attn = cfg.get("attn_implementation") or "flash_attention_2"
        # Top-level gradient_checkpointing wins; nested arctic_sft.* is fallback.
        gc = cfg.get("gradient_checkpointing")
        if gc is None:
            enable_gc = bool(acfg.gradient_checkpointing)
        else:
            # axolotl may use True / "offload" / "offload_disk".
            enable_gc = bool(gc) and str(gc).lower() not in ("false", "0", "none", "")
        return {
            "attn_implementation": attn,
            "enable_gradient_checkpointing": enable_gc,
            "zorro_train_enable": False,
        }

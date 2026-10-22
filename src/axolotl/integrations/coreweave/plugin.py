"""Publish saved model and LoRA checkpoints to CoreWeave Dedicated Inference."""

import json
from pathlib import Path

from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin
from axolotl.utils.schemas.checkpoint_publish import CheckpointPublishConfig

from .args import CoreWeaveConfig
from .checkpoints import CheckpointPublisher
from .client import CoreWeaveClient
from .destination import CoreWeaveDestination


def load_weights(directory):
    """Read a complete HF checkpoint, rejecting partial or adapter-only saves."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        if "delta_encoding" in manifest.get("metadata", {}):
            raise ValueError("Expected full model weights, received a delta checkpoint")
        files = set(manifest["weight_map"].values())
    else:
        manifest = None
        files = {"model.safetensors"}
    weights = {}
    for filename in sorted(files):
        if Path(filename).name != filename:
            raise ValueError("Checkpoint shards must be in the checkpoint directory")
        shard = load_file(directory / filename)
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names in checkpoint shards")
        if manifest is not None:
            expected = {
                name
                for name, file in manifest["weight_map"].items()
                if file == filename
            }
            if shard.keys() != expected:
                raise ValueError("Checkpoint index does not match shard tensors")
        weights.update(shard)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return weights, config


class CoreWeaveCheckpointCallback(TrainerCallback):
    """Publish only completed Hugging Face saves on the main process."""

    def __init__(
        self, config, *, base_model=None, revision=None, merged_format="preserve"
    ):
        self.config = config
        self.base_model, self.revision = base_model, revision
        self.merged_format = merged_format
        self.publisher = None
        self.worker = None

    def _get_publisher(self, output_dir):
        if self.publisher is None:
            publisher_type = CheckpointPublisher
            options = {}
            if self.base_model is not None:
                from .lora import LoRACheckpointPublisher

                publisher_type = LoRACheckpointPublisher
                options = {
                    "base_model": self.base_model,
                    "revision": self.revision,
                    "merged_format": self.merged_format,
                }
            self.publisher = publisher_type(
                self.config,
                CoreWeaveClient(self.config),
                Path(output_dir) / "coreweave-checkpoints",
                **options,
            )
        return self.publisher

    def on_save(self, args, state, control, **kwargs):
        from .rollouts import check_rank_errors

        error = None
        if state.is_world_process_zero:
            try:
                directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                loader = load_weights
                if self.base_model is not None:
                    from .lora import adapter_checkpoint

                    loader = adapter_checkpoint
                if self.config.async_publish:
                    from .async_publisher import CoreWeaveBackgroundPublisher

                    if self.worker is None:
                        self.worker = CoreWeaveBackgroundPublisher(
                            lambda: self._get_publisher(args.output_dir),
                            loader,
                            Path(args.output_dir) / "coreweave-checkpoints" / ".queue",
                        )
                    self.worker.submit(directory, state.global_step)
                else:
                    weights, model_config = loader(directory)
                    self._get_publisher(args.output_dir).publish(
                        weights, model_config, state.global_step
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = str(exc)
        check_rank_errors(error)
        return control

    def _check_worker(self, control, *, finish=False):
        from .rollouts import check_rank_errors

        error = None
        try:
            if self.worker is not None:
                if finish:
                    self.worker.close()
                else:
                    self.worker.check()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = str(exc)
        check_rank_errors(error)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._check_worker(control)

    def on_train_end(self, args, state, control, **kwargs):
        return self._check_worker(control, finish=True)

    def close(self):
        if self.worker is not None:
            self.worker.close(cancel_pending=True)


def publish_config(cfg, config):
    """Apply ``checkpoint_publish`` to the CoreWeave destination's chain settings."""
    settings = cfg.get("checkpoint_publish")
    if not settings:
        return config, None
    if isinstance(settings, BaseModel):
        settings = settings.model_dump()
    if hasattr(settings, "to_dict"):
        settings = settings.to_dict()
    settings = CheckpointPublishConfig.model_validate(dict(settings))
    if settings.destination is not None:
        raise ValueError(
            "Omit checkpoint_publish.destination when publishing to CoreWeave"
        )
    if cfg.get("adapter") == "lora":
        CoreWeaveDestination.check_export(settings.export)
    if settings.every:
        if cfg.get("adapter") != "lora" or settings.merged_format != "preserve":
            raise ValueError(
                "checkpoint_publish.every publishes merged LoRA in the base's own "
                "format; it needs adapter: lora and merged_format: preserve"
            )
        if cfg.get("nvfp4_merge_aware") and cfg.get("nvfp4_merge_aware_start_step"):
            raise ValueError(
                "checkpoint_publish.every needs merge-aware NVFP4 from the first step; "
                "unset nvfp4_merge_aware_start_step"
            )
    if settings.merge_compile:
        raise ValueError(
            "checkpoint_publish.merge_compile is not supported by this merge path"
        )
    if settings.recovery is not None and cfg.get("adapter") != "lora":
        raise ValueError("checkpoint_publish.recovery saves LoRA adapters only")
    max_deltas = min(config.max_deltas, settings.max_deltas)
    if not settings.delta:
        max_deltas = 0
    return config.model_copy(update={"max_deltas": max_deltas}), settings


def step_publisher_factory(cfg, config, settings, revision):
    """Build each rank's per-step publisher once training has its sharded LoRA."""

    def factory(*, model, comm, pieces):
        import torch

        from axolotl.core.publishing.callback import live_adapter_config, resolve_base
        from axolotl.core.publishing.pinned_base import pin_base
        from axolotl.core.publishing.step_publisher import StepPublisher

        output = Path(cfg.get("output_dir") or ".") / "coreweave-checkpoints"
        workdir = Path(settings.workdir) if settings.workdir else output / "step"
        base = pin_base(
            resolve_base(cfg["base_model"], revision),
            workdir / f"base-rank{comm.rank:05d}",
            revision=revision,
        )
        adapter_config = live_adapter_config(model)
        if cfg.get("use_sonicmoe") and cfg.get("nvfp4_merge_aware"):
            from axolotl.integrations.kernels.merge_aware_callback import (
                merge_aware_metadata,
            )

            adapter_config["nvfp4_merge_aware"] = merge_aware_metadata()
        return StepPublisher(
            base=base,
            adapter_config=adapter_config,
            pieces=pieces,
            destination=CoreWeaveDestination(config, CoreWeaveClient(config)),
            workdir=workdir,
            state_dir=output / "step-state",
            comm=comm,
            device=(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            ),
            max_deltas=config.max_deltas,
            delta=settings.delta,
            max_shard_size=config.max_shard_size,
        )

    return factory


class CoreWeavePlugin(BasePlugin):
    """Upload full/delta checkpoints from ordinary Axolotl checkpoint saves."""

    def get_input_args(self):
        return "axolotl.integrations.coreweave.args.CoreWeaveArgs"

    def post_train_unload(self, cfg):
        for name in ("_checkpoint_callback", "_recovery_callback"):
            if callback := getattr(self, name, None):
                callback.close()

    def register(self, cfg):
        if not cfg.get("coreweave"):
            raise ValueError("CoreWeavePlugin requires a coreweave configuration")
        if (
            cfg.get("adapter") not in (None, "lora")
            or cfg.get("load_in_4bit")
            or cfg.get("load_in_8bit")
        ):
            raise ValueError(
                "CoreWeave publishing requires full model weights or LoRA trained on "
                "the base checkpoint's own weights"
            )
        config = CoreWeaveConfig.model_validate(cfg["coreweave"])
        publish_config(cfg, config)
        if cfg.get("adapter") == "lora":
            if not cfg.get("base_model"):
                raise ValueError("CoreWeave LoRA publishing requires base_model")
            if config.rollouts:
                raise ValueError(
                    "CoreWeave LoRA publishing supports checkpoint saves only"
                )
            if cfg.get("save_safetensors") is False or cfg.get("trust_remote_code"):
                raise ValueError(
                    "CoreWeave LoRA publishing requires safetensors and a built-in model architecture"
                )
        if config.rollouts:
            trl = cfg.get("trl") or {}
            if cfg.get("rl") != "grpo":
                raise ValueError("CoreWeave rollouts currently require rl: grpo")
            if (
                (cfg.get("context_parallel_size") or 1) > 1
                or trl.get("async_prefetch")
                or trl.get("use_data_producer")
            ):
                raise ValueError(
                    "CoreWeave rollouts require synchronous GRPO without context parallelism"
                )
            if trl.get("use_vllm"):
                raise ValueError(
                    "Set trl.use_vllm: false when using CoreWeave rollouts"
                )
            if trl.get("temperature") is not None and trl["temperature"] <= 0:
                raise ValueError("CoreWeave rollouts require a positive temperature")
            if cfg.get("is_multimodal") or cfg.get("nemo_gym_enabled"):
                raise ValueError(
                    "CoreWeave rollouts require text-only GRPO without another rollout plugin"
                )
            if (
                trl.get("top_k") not in (None, 0, -1)
                or trl.get("min_p") not in (None, 0)
                or trl.get("repetition_penalty") not in (None, 1)
            ):
                raise ValueError(
                    "CoreWeave rollouts support temperature and top_p sampling"
                )

    def post_trainer_create(self, cfg, trainer):
        config = CoreWeaveConfig.model_validate(cfg.coreweave)
        if config.rollouts:
            from .correction import install_sampler_correction
            from .rollouts import CoreWeaveRollouts

            if getattr(trainer, "rollout_func", None) is not None or getattr(
                trainer, "tools", None
            ):
                raise ValueError(
                    "CoreWeave rollouts cannot be combined with another rollout function or tools"
                )
            install_sampler_correction(trainer)
            trainer.rollout_func = CoreWeaveRollouts(config)

    def add_callbacks_post_trainer(self, cfg, trainer):
        config = CoreWeaveConfig.model_validate(cfg.coreweave)
        if config.rollouts:
            return []
        config, settings = publish_config(cfg, config)
        revision = cfg.get("revision_of_model")
        commit = getattr(getattr(trainer.model, "config", None), "_commit_hash", None)
        if isinstance(commit, str) and commit:
            revision = commit
        recovery = None
        if settings is not None and settings.recovery is not None:
            from axolotl.core.publishing.recovery import RecoverySchedule

            recovery = RecoverySchedule.from_config(
                settings.recovery,
                output_dir=cfg.get("output_dir") or ".",
                workdir=settings.workdir,
            )
        if settings is not None and settings.every:
            from axolotl.core.publishing.callback import StepPublishCallback

            self._checkpoint_callback = StepPublishCallback(
                step_publisher_factory(cfg, config, settings, revision),
                settings.every,
                trainer=trainer,
                recovery=recovery,
            )
            return [self._checkpoint_callback]
        self._checkpoint_callback = CoreWeaveCheckpointCallback(
            config,
            base_model=cfg.get("base_model") if cfg.get("adapter") == "lora" else None,
            revision=revision,
            merged_format=settings.merged_format if settings else "preserve",
        )
        if recovery is None:
            return [self._checkpoint_callback]
        from axolotl.core.publishing.callback import StepPublishCallback

        self._recovery_callback = StepPublishCallback(
            None, 0, trainer=trainer, recovery=recovery
        )
        return [self._checkpoint_callback, self._recovery_callback]

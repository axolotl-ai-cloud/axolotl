"""Publish saved model and LoRA checkpoints to CoreWeave Dedicated Inference."""

import json
from pathlib import Path

from safetensors.torch import load_file
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin

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
    from axolotl.core.publishing.callback import publish_settings

    settings = publish_settings(
        cfg, weight_sync_steps(cfg) if async_rollouts(cfg) else None
    )
    if settings is None:
        return config, None
    if settings.destination is not None:
        raise ValueError(
            "Omit checkpoint_publish.destination when publishing to CoreWeave"
        )
    if cfg.get("adapter") == "lora":
        CoreWeaveDestination.check_export(settings.export)
    max_deltas = min(config.max_deltas, settings.max_deltas)
    if not settings.delta:
        max_deltas = 0
    return config.model_copy(update={"max_deltas": max_deltas}), settings


def step_publisher_factory(cfg, config, settings, revision):
    """Each rank's per-step publisher, hot-loading through CoreWeave."""
    from axolotl.core.publishing.callback import (
        step_publisher_factory as core_factory,
    )

    return core_factory(
        cfg,
        settings,
        lambda: CoreWeaveDestination(config, CoreWeaveClient(config)),
        revision=revision,
        output=Path(cfg.get("output_dir") or ".") / "coreweave-checkpoints",
        max_deltas=config.max_deltas,
        max_shard_size=config.max_shard_size,
    )


ASYNC_ROLLOUT_LOOP = (
    "axolotl.integrations.coreweave.async_rollouts.CoreWeaveRolloutLoop"
)


def _get(obj, key):
    if obj is None:
        return None
    return obj.get(key) if hasattr(obj, "get") else getattr(obj, key, None)


def async_rollouts(cfg) -> bool:
    """Whether rollouts come from CoreWeave in async GRPO (``trl.async_prefetch``)."""
    return bool(
        _get(_get(cfg, "coreweave"), "rollouts")
        and _get(_get(cfg, "trl"), "async_prefetch")
    )


def weight_sync_steps(cfg) -> int:
    trl = _get(cfg, "trl")
    return _get(trl, "weight_sync_steps") or _get(trl, "vllm_sync_interval") or 1


def check_async_rollouts(cfg):
    """Async GRPO samples from the deployment the trainer hot-loads every publish."""
    trl = cfg.get("trl") or {}
    if cfg.get("rl") != "grpo":
        raise ValueError("CoreWeave rollouts currently require rl: grpo")
    if cfg.get("adapter") != "lora":
        raise ValueError(
            "Async CoreWeave rollouts publish merged LoRA at every weight sync: set "
            "adapter: lora"
        )
    if not trl.get("token_budget"):
        raise ValueError("Async CoreWeave rollouts need trl.token_budget")
    if trl.get("rollout_loop") or trl.get("vllm_lora_sync"):
        raise ValueError(
            "Async CoreWeave rollouts use their own rollout loop and weight sync; "
            "unset trl.rollout_loop and trl.vllm_lora_sync"
        )
    if trl.get("temperature") is not None and trl["temperature"] <= 0:
        raise ValueError("CoreWeave rollouts require a positive temperature")
    if cfg.get("is_multimodal") or cfg.get("nemo_gym_enabled"):
        raise ValueError(
            "CoreWeave rollouts require text-only GRPO without another rollout plugin"
        )
    if (
        trl.get("top_p") not in (None, 1, 1.0)
        or trl.get("top_k") not in (None, 0, -1)
        or trl.get("min_p") not in (None, 0)
        or trl.get("repetition_penalty") not in (None, 1)
    ):
        # The gateway does not return the candidate set top_p/top_k sample from.
        raise ValueError("Async CoreWeave rollouts support temperature sampling only")


class CoreWeavePlugin(BasePlugin):
    """Upload full/delta checkpoints from ordinary Axolotl checkpoint saves."""

    def get_input_args(self):
        return "axolotl.integrations.coreweave.args.CoreWeaveArgs"

    def post_train_unload(self, cfg):
        if callback := getattr(self, "_checkpoint_callback", None):
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
            if config.rollouts and not async_rollouts(cfg):
                raise ValueError(
                    "CoreWeave LoRA rollouts need async GRPO (trl.async_prefetch)"
                )
            if cfg.get("save_safetensors") is False or cfg.get("trust_remote_code"):
                raise ValueError(
                    "CoreWeave LoRA publishing requires safetensors and a built-in model architecture"
                )
        if async_rollouts(cfg):
            check_async_rollouts(cfg)
        elif config.rollouts:
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

    def get_training_args(self, cfg):
        if not async_rollouts(cfg):
            return None
        config = CoreWeaveConfig.model_validate(cfg.coreweave)
        return {
            "rollout_loop": ASYNC_ROLLOUT_LOOP,
            "rollout_loop_kwargs": {
                "gateway_url": config.gateway_url,
                "model": config.model,
                "api_token_env": config.api_token_env,
            },
            "external_weight_sync": True,
        }

    def post_trainer_create(self, cfg, trainer):
        config = CoreWeaveConfig.model_validate(cfg.coreweave)
        if async_rollouts(cfg):
            if not getattr(trainer.args, "external_weight_sync", False):
                raise ValueError(
                    "Async CoreWeave rollouts need AxolotlAsyncGRPOTrainer with "
                    "external_weight_sync"
                )
        elif config.rollouts:
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
        rollouts = async_rollouts(cfg)
        if config.rollouts and not rollouts:
            return []
        config, settings = publish_config(cfg, config)
        revision = cfg.get("revision_of_model")
        commit = getattr(getattr(trainer.model, "config", None), "_commit_hash", None)
        if isinstance(commit, str) and commit:
            revision = commit
        if settings is not None and settings.every:
            from axolotl.core.publishing.callback import StepPublishCallback

            on_published = on_discarded = None
            if rollouts:
                from .async_rollouts import FIRST_PUBLISH_FAILED

                served = False

                def on_published(step):
                    nonlocal served
                    served = True
                    # Rank 0 runs the rollout worker; every replica serves `step`.
                    if worker := getattr(trainer, "rollout_worker", None):
                        worker.update_model_version(step // settings.every)

                def on_discarded(step):
                    # Nothing is served yet, so no rollout could ever start.
                    if not served and (
                        worker := getattr(trainer, "rollout_worker", None)
                    ):
                        worker.update_model_version(FIRST_PUBLISH_FAILED)

            self._checkpoint_callback = StepPublishCallback(
                step_publisher_factory(cfg, config, settings, revision),
                settings.every,
                trainer=trainer,
                publish_at_start=rollouts,
                on_published=on_published,
                on_discarded=on_discarded,
            )
            return [self._checkpoint_callback]
        self._checkpoint_callback = CoreWeaveCheckpointCallback(
            config,
            base_model=cfg.get("base_model") if cfg.get("adapter") == "lora" else None,
            revision=revision,
            merged_format=settings.merged_format if settings else "preserve",
        )
        return [self._checkpoint_callback]

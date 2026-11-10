"""Trainer callback for per-step publishing from a staged LoRA snapshot."""

import json
from collections.abc import Callable
from numbers import Number
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

from .coalesce import CoalescingScheduler
from .comm import GlooComm
from .lora_shards import Piece, collect_model_pieces
from .staging import LoRAStager

LOG = get_logger(__name__)
METRIC_PREFIX = "checkpoint_publish/"
# Also in the log line: metrics logged after training ends are not printed.
_SUMMARY = (
    "full",
    "artifact_bytes",
    "coalesced_steps",
    "seconds_build",
    "seconds_encode",
    "seconds_upload",
    "seconds_publish_index",
    "seconds_total",
)


def _gather(value) -> list:
    if dist.is_available() and dist.is_initialized():
        out: list = [None] * dist.get_world_size()
        dist.all_gather_object(out, value)
        return out
    return [value]


def check_rank_errors(error: str | None):
    """Raise on every rank when any rank failed (training thread, default group)."""
    failed = [message for message in _gather(error) if message is not None]
    if failed:
        raise RuntimeError(f"Checkpoint publishing failed: {'; '.join(failed)}")


def live_adapter_config(model: torch.nn.Module, adapter: str = "default") -> dict:
    """The adapter config as ``adapter_config.json`` would hold it."""
    for module in (model, *model.modules()):
        configs = getattr(module, "peft_config", None)
        if isinstance(configs, dict) and adapter in configs:
            return json.loads(
                json.dumps(configs[adapter].to_dict(), default=sorted, sort_keys=True)
            )
    raise ValueError("Per-step publishing needs a PEFT LoRA model")


def resolve_base(base_model: str, revision: str | None = None) -> Path:
    source = Path(base_model)
    if source.is_dir():
        return source
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            base_model,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )


PublisherFactory = Callable[..., Any]


def publish_settings(cfg):
    """``cfg.checkpoint_publish`` validated for per-step publishing, or ``None``."""
    from pydantic import BaseModel

    from axolotl.utils.schemas.checkpoint_publish import CheckpointPublishConfig

    settings = cfg.get("checkpoint_publish")
    if not settings:
        return None
    if isinstance(settings, BaseModel):
        settings = settings.model_dump()
    if hasattr(settings, "to_dict"):
        settings = settings.to_dict()
    settings = CheckpointPublishConfig.model_validate(dict(settings))
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
    return settings


def step_publisher_factory(
    cfg,
    settings,
    destination: Callable[[], Any],
    *,
    revision: str | None,
    output: Path,
    max_deltas: int,
    max_shard_size: int = 1 << 30,
) -> PublisherFactory:
    """Build each rank's per-step publisher once training has its sharded LoRA."""

    def factory(*, model, comm, pieces):
        from .pinned_base import pin_base
        from .step_publisher import StepPublisher

        workdir = Path(settings.workdir) if settings.workdir else output / "step"
        # Agreed before StepPublisher's first collective, which ranks that pinned
        # would otherwise wait in for the one that failed.
        base, error = None, None
        try:
            base = pin_base(
                resolve_base(cfg["base_model"], revision),
                workdir / f"base-rank{comm.rank:05d}",
                revision=revision,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {comm.rank}: {exc!r}"
        failed = [e for e in comm.all_gather_object(error) if e is not None]
        if failed:
            raise RuntimeError(f"Pinning the base failed: {'; '.join(failed)}")
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
            destination=destination(),
            workdir=workdir,
            state_dir=output / "step-state",
            comm=comm,
            device=(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            ),
            max_deltas=max_deltas,
            delta=settings.delta,
            max_shard_size=max_shard_size,
            num_threads=cfg.get("merge_num_threads") or 2,
        )

    return factory


def destination_callbacks(cfg, trainer, revision=None) -> list:
    """Per-step publishing to ``checkpoint_publish.destination`` (local dir or S3)."""
    settings = publish_settings(cfg)
    if settings is None or not settings.destination or not settings.every:
        return []
    from .destinations import build_destination

    output = Path(cfg.get("output_dir") or ".") / "published"
    factory = step_publisher_factory(
        cfg,
        settings,
        lambda: build_destination(settings.destination),
        revision=revision,
        output=output,
        max_deltas=settings.max_deltas if settings.delta else 0,
    )
    return [StepPublishCallback(factory, settings.every, trainer=trainer)]


class StepPublishCallback(TrainerCallback):
    """Snapshot the LoRA every ``every`` optimizer steps and publish it asynchronously.

    ``factory(model=, comm=, pieces=)`` builds the rank's ``StepPublisher``. The
    only training-thread cost is the host copy of this rank's LoRA slices.

    With ``publish_at_start`` the weights training starts from (step 0, or the
    resumed step) are published at once. ``on_published(step)`` runs on the
    publisher thread after each step every destination serves.

    Publish metrics join the trainer's next training log row instead of a
    ``trainer.log`` of their own: on rank 0 alone, that call resets
    ``control.should_log`` there, so rank 0 skips the step's log and its
    all-gathers while the other ranks wait in them.
    """

    def __init__(
        self,
        factory: PublisherFactory,
        every: int,
        *,
        trainer=None,
        adapter: str = "default",
        arrival_timeout: float = 600.0,
        publish_at_start: bool = False,
        on_published: Callable[[int], None] | None = None,
    ):
        if every <= 0:
            raise ValueError("every must be positive")
        self.factory, self.every, self.trainer = factory, every, trainer
        self.adapter, self.arrival_timeout = adapter, arrival_timeout
        self.publish_at_start = publish_at_start
        self.on_published = on_published
        self.publisher = None
        self.scheduler: CoalescingScheduler | None = None
        self._pending: dict[str, float] = {}
        trainer_log = getattr(trainer, "log", None)
        if trainer_log is not None:

            def log(logs, *args, **kwargs):
                if "loss" in logs and self._pending:
                    logs = {**logs, **self._pending}
                    self._pending = {}
                return trainer_log(logs, *args, **kwargs)

            trainer.log = log

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        error = None
        pieces: list[tuple[Piece, torch.Tensor]] = []
        try:
            if model is None:
                raise ValueError("The trainer did not pass the model to callbacks")
            pieces = collect_model_pieces(model, self.adapter)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {_rank()}: {exc!r}"
        check_rank_errors(error)
        comm = GlooComm.create()
        self.publisher = self.factory(
            model=model, comm=comm, pieces=[piece for piece, _ in pieces]
        )
        stager = LoRAStager([tensor for _, tensor in pieces], slots=2)
        LOG.info(
            "Per-step publishing every %d steps; staging %.2f GiB of LoRA per rank",
            self.every,
            stager.nbytes / 2**30,
        )
        self.scheduler = CoalescingScheduler(
            stager,
            self.publisher.publish,
            comm,
            restore=self.publisher.restore,
            arrival_timeout=self.arrival_timeout,
            on_published=self.on_published,
        )
        self.scheduler.start(restore_step=state.global_step or None)
        if self.publish_at_start:
            error = None
            try:
                self.scheduler.offer(state.global_step)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = f"rank {_rank()}: {exc!r}"
            check_rank_errors(error)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.scheduler is not None:
            self.scheduler.wait_training()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.scheduler is None or state.global_step % self.every:
            return control
        error = None
        try:
            self.scheduler.offer(state.global_step)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {_rank()}: {exc!r}"
        check_rank_errors(error)
        self._log(state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.scheduler is None:
            return control
        error = None
        try:
            self.scheduler.close(flush=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {_rank()}: {exc!r}"
        finally:
            if self.publisher is not None:
                self.publisher.close()
            self._log(state)
            self.scheduler = None
            if self._pending and hasattr(self.trainer, "log"):
                # no step log follows, and no rank waits in one
                pending, self._pending = self._pending, {}
                self.trainer.log(pending)
        check_rank_errors(error)
        return control

    def close(self):
        if self.scheduler is not None:
            self.scheduler.close(flush=False)
            self.scheduler = None

    def _log(self, state):
        assert self.scheduler is not None
        for metrics in self.scheduler.drain_metrics():
            if not state.is_world_process_zero:
                continue
            if metrics.get("action") == "discard":
                LOG.warning(
                    "Publish of step %s was discarded: %s",
                    metrics.get("step"),
                    "; ".join(metrics.get("errors", [])),
                )
            values = {
                METRIC_PREFIX + key: float(value)
                for key, value in metrics.items()
                if isinstance(value, Number)
            }
            LOG.info(
                "Published step %s as %s (%s): %s",
                metrics.get("step"),
                metrics.get("identity"),
                metrics.get("action"),
                ", ".join(
                    f"{key}={value:.3g}"
                    for key, value in metrics.items()
                    if key in _SUMMARY and isinstance(value, Number)
                ),
            )
            self._pending.update(values)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

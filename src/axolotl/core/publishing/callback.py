"""Trainer callback for per-step publishing and recovery saves from one LoRA snapshot."""

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
from .comm import GlooComm, LocalComm
from .lora_shards import Piece, collect_model_pieces
from .recovery import OptimizerStager, RecoveryJob, RecoverySchedule, RecoveryWriter
from .staging import LoRAStager

LOG = get_logger(__name__)
METRIC_PREFIX = "checkpoint_publish/"


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


class StepPublishCallback(TrainerCallback):
    """Snapshot the LoRA every ``every`` optimizer steps and publish it asynchronously.

    ``factory(model=, comm=, pieces=)`` builds the rank's ``StepPublisher``. The
    only training-thread cost is the host copy of this rank's LoRA slices.

    With ``recovery`` the same snapshot also feeds asynchronous recovery saves on
    their own cadence (``every`` may then be 0 to save without publishing). A
    step that is both reuses one staged copy: the recovery writer holds the slot
    until its shard is on disk. A recovery step is skipped on every rank when
    any rank is still writing the previous one.
    """

    def __init__(
        self,
        factory: PublisherFactory | None,
        every: int,
        *,
        trainer=None,
        adapter: str = "default",
        arrival_timeout: float = 600.0,
        recovery: RecoverySchedule | None = None,
    ):
        if every < 0 or (every == 0 and recovery is None):
            raise ValueError("every must be positive unless recovery saves are set")
        if every and factory is None:
            raise ValueError("Per-step publishing needs a publisher factory")
        self.factory, self.every, self.trainer = factory, every, trainer
        self.adapter, self.arrival_timeout = adapter, arrival_timeout
        self.recovery = recovery
        self.publisher = None
        self.scheduler: CoalescingScheduler | None = None
        self.writer: RecoveryWriter | None = None
        self.optimizer_stager: OptimizerStager | None = None
        self.recovery_skipped = 0

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
        piece_list = [piece for piece, _ in pieces]
        comm = GlooComm.create() if self.every else LocalComm()
        recovery_comm = GlooComm.create() if self.recovery is not None else None
        if self.every:
            assert self.factory is not None
            self.publisher = self.factory(model=model, comm=comm, pieces=piece_list)
        slots = (2 if self.every else 0) + (1 if self.recovery is not None else 0)
        stager = LoRAStager([tensor for _, tensor in pieces], slots=slots)
        LOG.info(
            "Per-step publishing every %d steps, recovery saves every %s; staging "
            "%.2f GiB of LoRA per rank in %d slots",
            self.every,
            self.recovery.every if self.recovery is not None else "never",
            stager.nbytes / 2**30,
            slots,
        )
        self.scheduler = CoalescingScheduler(
            stager,
            self.publisher.publish if self.publisher is not None else None,
            comm,
            restore=self.publisher.restore if self.publisher is not None else None,
            arrival_timeout=self.arrival_timeout,
        )
        if self.recovery is not None:
            assert recovery_comm is not None and model is not None
            self.writer = RecoveryWriter(
                self.recovery.directory,
                piece_list,
                recovery_comm,
                keep=self.recovery.keep,
                consolidate_dir=self.recovery.consolidate_dir,
                adapter_config=live_adapter_config(model, self.adapter),
            )
            if self.recovery.optimizer_every:
                self.optimizer_stager = OptimizerStager()
        if self.every:
            self.scheduler.start(restore_step=state.global_step or None)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.scheduler is not None:
            self.scheduler.wait_training()
        if self.optimizer_stager is not None:
            self.optimizer_stager.wait_training()
        return control

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if self.scheduler is None:
            return control
        step = state.global_step
        publish = bool(self.every) and step % self.every == 0
        save = self.recovery is not None and step % self.recovery.every == 0
        if not (publish or save):
            return control
        if save:
            save = self._agree_to_save(step)
        error = None
        try:
            if save:
                self._save(step, optimizer, publish=publish)
            elif publish:
                self.scheduler.offer(step)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {_rank()}: {exc!r}"
        check_rank_errors(error)
        self._log(state)
        return control

    def on_train_end(self, args, state, control, optimizer=None, **kwargs):
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
        recovery_error = None
        try:
            check_rank_errors(error)
            if self.writer is not None:
                try:
                    self._final_save(state.global_step, optimizer)
                    self.writer.close()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    recovery_error = f"rank {_rank()}: {exc!r}"
        finally:
            self._log(state)
            if self.writer is not None and self.writer.thread.is_alive():
                try:
                    self.writer.close(timeout=60)
                except Exception:  # pylint: disable=broad-exception-caught
                    LOG.exception("Recovery checkpoint writer failed")
            self.scheduler = None
            self.writer = None
        check_rank_errors(recovery_error)
        return control

    def close(self):
        if self.scheduler is not None:
            self.scheduler.close(flush=False)
            self.scheduler = None
        if self.writer is not None:
            try:
                self.writer.close(timeout=60)
            except Exception:  # pylint: disable=broad-exception-caught
                LOG.exception("Recovery checkpoint writer failed")
            self.writer = None

    def _agree_to_save(self, step: int) -> bool:
        """Whether every rank's writer is idle; raises everywhere if any writer failed."""
        assert self.writer is not None
        error, busy = None, False
        try:
            busy = self.writer.busy()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"rank {_rank()}: {exc!r}"
        replies = _gather((error, busy))
        failed = [message for message, _ in replies if message is not None]
        if failed:
            raise RuntimeError(f"Recovery checkpointing failed: {'; '.join(failed)}")
        if any(flag for _, flag in replies):
            self.recovery_skipped += 1
            if _rank() == 0:
                LOG.warning(
                    "Skipping the recovery save of step %d: the previous one is "
                    "still being written",
                    step,
                )
            return False
        return True

    def _save(self, step: int, optimizer, *, publish: bool, consolidate=None):
        """Stage ``step`` for the writer; always submits, so ranks stay in lockstep.

        A staging failure is carried into the job and fails it on every rank.
        """
        assert self.recovery is not None and self.writer is not None
        assert self.scheduler is not None
        scheduler = self.scheduler
        if consolidate is None:
            every = self.recovery.consolidate_every
            consolidate = bool(every) and step % every == 0
        slot = None
        failure: BaseException | None = None
        optimizer_state = None
        try:
            slot = scheduler.snapshot(step, publish=publish, hold=True)
            if slot is None:
                raise RuntimeError(f"No free staging slot for step {step}")
            every = self.recovery.optimizer_every
            if every and step % every == 0:
                if optimizer is None:
                    raise ValueError("The trainer did not pass the optimizer")
                assert self.optimizer_stager is not None
                self.optimizer_stager.stage(optimizer)
                optimizer_state = self.optimizer_stager
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failure = exc

        def lora():
            if failure is not None:
                raise failure
            assert slot is not None
            return scheduler.buffers(slot)

        def release():
            if slot is not None:
                scheduler.unhold(slot)

        self.writer.submit(
            RecoveryJob(step, lora, release, optimizer_state, consolidate)
        )

    def _final_save(self, step: int, optimizer):
        assert self.recovery is not None and self.writer is not None
        self.writer.wait()
        if not self.recovery.final:
            return
        if self.writer.last_written != step:
            self._save(step, optimizer, publish=False, consolidate=True)
        elif self.writer.last_consolidated != step:
            self.writer.submit(RecoveryJob(step, None, lambda: None, consolidate=True))
        self.writer.wait()

    def _log(self, state):
        assert self.scheduler is not None
        if self.writer is not None:
            for metrics in self.writer.drain_metrics():
                if not state.is_world_process_zero:
                    continue
                metrics["recovery_skipped"] = self.recovery_skipped
                LOG.info(
                    "Recovery checkpoint of step %s: %s",
                    metrics.get("recovery_step"),
                    {k: v for k, v in metrics.items() if k != "recovery_step"},
                )
                if self.trainer is not None:
                    self.trainer.log(
                        {
                            METRIC_PREFIX + key: float(value)
                            for key, value in metrics.items()
                            if isinstance(value, Number)
                        }
                    )
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
                "Published step %s as %s (%s)",
                metrics.get("step"),
                metrics.get("identity"),
                metrics.get("action"),
            )
            if self.trainer is not None:
                self.trainer.log(values)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

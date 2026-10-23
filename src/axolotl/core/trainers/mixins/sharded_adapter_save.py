"""Sharded FSDP2 adapter checkpoints (``adapter_checkpoint_format: sharded``).

Each rank writes its adapter slices; rank 0 consolidates ``adapter_model.safetensors`` in the
background. Until that finishes the checkpoint is incomplete, so this mixin holds back what
treats it as complete: the ``on_save`` callbacks (checkpoint mirror, publishers) and
``save_total_limit`` rotation run once rank 0 reports the consolidation done, polled at step
ends, and training end waits for every outstanding consolidation.

The end-of-training export (:meth:`ShardedAdapterSaveMixin.save_final_adapter`) goes
through the same writer into ``output_dir`` but consolidates synchronously: it returns
only once ``adapter_model.safetensors`` is in place.
"""

import contextlib
import copy
import os
import shutil
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _distributed():
    return dist.is_available() and dist.is_initialized()


def rotate_checkpoints_excluding(
    excluded,
    output_dir,
    save_total_limit=None,
    best_model_checkpoint=None,
    use_mtime=False,
    checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
):
    """``transformers``' rotation over the checkpoints not in ``excluded``, which are neither
    counted nor deleted."""
    from transformers.trainer_utils import sort_checkpoints

    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints = [
        path
        for path in sort_checkpoints(output_dir, checkpoint_prefix, use_mtime)
        if os.path.abspath(path) not in excluded
    ]
    if len(checkpoints) <= save_total_limit:
        return
    protected = {checkpoints[-1]}
    if best_model_checkpoint is not None:
        protected.add(str(Path(best_model_checkpoint)))
    keep = max(save_total_limit, len(protected))
    remaining = len(checkpoints)
    for checkpoint in checkpoints:
        if remaining <= keep:
            break
        if checkpoint not in protected:
            shutil.rmtree(checkpoint, ignore_errors=True)
            remaining -= 1


class ShardedAdapterSaveMixin(Trainer):
    """Write PEFT adapter checkpoints as per-rank shards under FSDP2."""

    def _sharded_adapter_checkpoints(self) -> bool:
        cfg = getattr(self, "axolotl_cfg", None)
        return getattr(cfg, "adapter_checkpoint_format", None) == "sharded"

    def _sharded_final_save(self) -> bool:
        cfg = getattr(self, "axolotl_cfg", None)
        final = getattr(cfg, "adapter_final_save_format", None)
        return (final or getattr(cfg, "adapter_checkpoint_format", None)) == "sharded"

    def _final_adapter_save(self) -> bool:
        return getattr(self, "_axolotl_final_adapter_save", False)

    def _hub_save(self) -> bool:
        return getattr(self, "_axolotl_hub_save", False)

    def _sharded_adapter_save_applies(self) -> bool:
        if self._final_adapter_save() or self._hub_save():
            enabled = self._sharded_final_save()
        else:
            enabled = self._sharded_adapter_checkpoints() and getattr(
                self, "_axolotl_saving_checkpoint", False
            )
        if not (enabled and _distributed()):
            return False
        from accelerate.utils.modeling import is_peft_model

        plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        return (
            getattr(plugin, "fsdp_version", None) == 2
            and "FULL_STATE_DICT" in str(getattr(plugin, "state_dict_type", ""))
            and is_peft_model(self.model)
            and len(self.model.peft_config) == 1
        )

    def _consolidation_state(self):
        if not hasattr(self, "_axolotl_unconsolidated"):
            self._axolotl_unconsolidated: set[str] = set()
            self._axolotl_failed_consolidations: set[str] = set()
            self._axolotl_deferred_saves: deque = deque()
            self._axolotl_deferred_rotation = None
            self._axolotl_last_checkpoint = None

    def save_final_adapter(self, output_dir: str | None = None) -> bool:
        """Collective: the end-of-training adapter export through the sharded writer.

        Returns once ``adapter_model.safetensors`` is consolidated in ``output_dir``, or
        ``False`` without saving anything when the sharded path does not apply, so the
        caller can save through the gather.
        """
        self._axolotl_final_adapter_save = True
        self._axolotl_final_adapter_saved = False
        try:
            if not self._sharded_adapter_save_applies():
                return False
            self.save_model(output_dir or self.args.output_dir)
        finally:
            self._axolotl_final_adapter_save = False
        return self._axolotl_final_adapter_saved

    def push_to_hub(self, *args, **kwargs):
        # its save_model passes _internal_call like a checkpoint, then uploads at once
        self._axolotl_hub_save = True
        try:
            return super().push_to_hub(*args, **kwargs)
        finally:
            self._axolotl_hub_save = False

    def _save_model_native(
        self, output_dir: str | None = None, _internal_call: bool = False
    ):
        if self._sharded_adapter_save_applies():
            final = self._final_adapter_save()
            if self._save_sharded_adapter(
                output_dir or self.args.output_dir, final or self._hub_save()
            ):
                if final:
                    self._axolotl_final_adapter_saved = True
                    if getattr(self.args, "push_to_hub", False) and not _internal_call:
                        self.push_to_hub(
                            commit_message="Model save",
                            revision=self.args.hub_revision,
                        )
                return None
            if final:
                # nothing saved: save_final_adapter returns False so the gather runs instead
                return None
        return super()._save_model_native(output_dir, _internal_call)

    def _save_sharded_adapter(self, output_dir, final=False) -> bool:
        from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import _adapter_values
        from axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter import (
            CONSOLIDATIONS,
            EXPORT_DIRNAME,
            mark_sharded_export,
            move_export,
            write_adapter_shards,
        )
        from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
            persist_native_metadata_after_save,
        )

        self._consolidation_state()
        rank0 = dist.get_rank() == 0
        if rank0:
            if final:
                CONSOLIDATIONS.wait()
            else:
                CONSOLIDATIONS.wait(output_dir)
                CONSOLIDATIONS.wait_for_capacity()
        shards = write_adapter_shards(self.model, output_dir)
        if shards is None:
            LOG.warning_once(
                "adapter_checkpoint_format: sharded does not cover this adapter; "
                "saving it through the rank 0 gather"
            )
            return False
        error = None
        if rank0:
            try:
                export = shards.directory / EXPORT_DIRNAME
                # PEFT looks modules_to_save copies up by name: every name needs an entry
                placeholder = {
                    name: torch.empty(0) for name in _adapter_values(self.model)
                }
                self._save(str(export), state_dict=placeholder)
                persist_native_metadata_after_save(self.model, str(export))
                move_export(export, output_dir)
                future = CONSOLIDATIONS.submit(shards.directory)
                if final:
                    future.result()
            except Exception as exc:  # pylint: disable=broad-except
                error = f"{type(exc).__name__}: {exc}"
                if final:
                    error += (
                        f"; the adapter shards stay in {shards.directory}, "
                        "complete_interrupted_consolidation() in "
                        "axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter finishes them"
                    )
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise RuntimeError(f"Distributed adapter export failed: {error}")
        if final:
            return True
        mark_sharded_export(output_dir)
        self._axolotl_unconsolidated.add(os.path.abspath(output_dir))
        return True

    @contextlib.contextmanager
    def _gated_rotation(self):
        import transformers.trainer as hf_trainer

        original = getattr(hf_trainer, "rotate_checkpoints", None)
        if original is None:
            raise RuntimeError(
                "adapter_checkpoint_format: sharded holds back save_total_limit rotation "
                "through transformers.trainer.rotate_checkpoints, which this transformers "
                "version does not have"
            )

        def rotate(output_dir, *args, **kwargs):
            if self._axolotl_unconsolidated:
                self._axolotl_deferred_rotation = (output_dir, args, kwargs)
                return None
            if self._axolotl_failed_consolidations:
                return rotate_checkpoints_excluding(
                    self._axolotl_failed_consolidations, output_dir, *args, **kwargs
                )
            return original(output_dir, *args, **kwargs)

        hf_trainer.rotate_checkpoints = rotate
        try:
            yield
        finally:
            hf_trainer.rotate_checkpoints = original

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        if not self._sharded_adapter_checkpoints():
            return super()._save_checkpoint(model, trial, *args, **kwargs)
        self._consolidation_state()
        self._install_consolidation_gate()
        self._axolotl_last_checkpoint = os.path.abspath(
            os.path.join(
                self._get_output_dir(trial=trial),
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
        )
        with self._gated_rotation():
            return super()._save_checkpoint(model, trial, *args, **kwargs)

    def _install_consolidation_gate(self):
        handler = self.callback_handler
        if getattr(handler, "_axolotl_consolidation_gate", False):
            return
        on_save, on_step_end = handler.on_save, handler.on_step_end
        on_train_end = handler.on_train_end

        def gated_on_save(args, state, control, **kwargs):
            if not (self._axolotl_unconsolidated or self._axolotl_deferred_saves):
                return on_save(args, state, control, **kwargs)
            control.should_save = False
            self._axolotl_deferred_saves.append(
                (self._axolotl_last_checkpoint, copy.copy(state), kwargs)
            )
            return control

        def gated_on_step_end(args, state, control, **kwargs):
            control = self._release_consolidated(on_save, args, control)
            return on_step_end(args, state, control, **kwargs)

        def gated_on_train_end(args, state, control, **kwargs):
            control = self._release_consolidated(on_save, args, control, wait=True)
            if self._axolotl_failed_consolidations:
                LOG.error(
                    "These checkpoints have no consolidated adapter: "
                    f"{', '.join(sorted(self._axolotl_failed_consolidations))}. Their "
                    "shards are kept; resuming from one consolidates it first."
                )
            return on_train_end(args, state, control, **kwargs)

        handler.on_save = gated_on_save
        handler.on_step_end = gated_on_step_end
        handler.on_train_end = gated_on_train_end
        handler._axolotl_consolidation_gate = True

    def _release_consolidated(self, on_save, args, control, wait=False):
        """Run the rotations and ``on_save`` events held back for finished consolidations,
        in save order. Collective while anything is held back."""
        from axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter import (
            CONSOLIDATIONS,
        )

        if not (self._axolotl_deferred_saves or self._axolotl_unconsolidated):
            return control
        rank0 = not _distributed() or dist.get_rank() == 0
        if wait and rank0:
            CONSOLIDATIONS.wait()
        status = [
            {path: CONSOLIDATIONS.status(path) for path in self._axolotl_unconsolidated}
            if rank0
            else None
        ]
        if _distributed() and dist.get_world_size() > 1:
            dist.broadcast_object_list(status, src=0)
        finished = False
        for path, done in status[0].items():
            if done is None:
                continue
            finished = True
            self._axolotl_unconsolidated.discard(path)
            if not done:
                self._axolotl_failed_consolidations.add(path)
        # on_save first: rotation may delete a checkpoint whose on_save is still held back
        while self._axolotl_deferred_saves:
            path, state, kwargs = self._axolotl_deferred_saves[0]
            if path in self._axolotl_unconsolidated:
                break
            self._axolotl_deferred_saves.popleft()
            if path in self._axolotl_failed_consolidations:
                LOG.error(
                    f"{path} has no consolidated adapter; skipping its on_save callbacks "
                    "(checkpoint mirror, publishers)"
                )
                continue
            control = on_save(args, state, control, **kwargs)
        if self._axolotl_deferred_rotation and finished:
            output_dir, rotate_args, rotate_kwargs = self._axolotl_deferred_rotation
            if not self._axolotl_unconsolidated:
                self._axolotl_deferred_rotation = None
            # still-consolidating checkpoints neither count toward the limit nor get deleted
            rotate_checkpoints_excluding(
                self._axolotl_unconsolidated | self._axolotl_failed_consolidations,
                output_dir,
                *rotate_args,
                **rotate_kwargs,
            )
        return control

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self._sharded_adapter_checkpoints():
            from axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter import (
                complete_interrupted_consolidation,
            )

            error = None
            if not _distributed() or dist.get_rank() == 0:
                try:
                    complete_interrupted_consolidation(resume_from_checkpoint)
                except Exception as exc:  # pylint: disable=broad-except
                    error = f"{type(exc).__name__}: {exc}"
            if _distributed():
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, error)
                error = next((item for item in errors if item is not None), None)
            if error is not None:
                raise RuntimeError(
                    f"Cannot consolidate the adapter shards in {resume_from_checkpoint}: {error}"
                )
        return super()._load_from_checkpoint(resume_from_checkpoint, model)

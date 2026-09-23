# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Remote SFT trainer: axolotl data pipeline + ArcticSFTClient for fwd_bwd/step.

The local model is a stub (``ArcticSFTPlugin.pre_model_load``); compute is remote.
GAS microbatches are sent as a list on the wire (no client concat / server re-split).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import torch
from transformers.trainer_utils import TrainOutput

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _metric_scalar(x: Any) -> float:
    """Scalar or 1-element list/tuple → float."""
    if isinstance(x, (list, tuple)):
        return float(x[0]) if x else 0.0
    return float(x)


def _loggable_metrics(metrics: dict) -> dict[str, float]:
    """Numeric backend metrics that HF Trainer / wandb accept as scalars."""
    logs: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("_") or value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            logs[key] = float(value)
        elif isinstance(value, (list, tuple)):
            logs[key] = _metric_scalar(value)
    return logs


def _count_valid_targets(batch: dict[str, torch.Tensor]) -> int:
    """HF-shifted valid targets (``labels[:, 1:] != -100``) for token-weighted eval."""
    labels = batch.get("labels")
    if labels is None:
        return 0
    return (labels[:, 1:] != -100).sum().item()


class ArcticSFTTrainer(AxolotlTrainer):
    """Trainer that sends preprocessed SFT batches to a remote Arctic server."""

    _arctic_client_config: Any
    _arctic_loss_fn: str
    _arctic_learning_rate: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arctic_client_config = None
        self._client = None
        self._pad_token_id: Optional[int] = None
        self._arctic_autoset_horizon = False
        self._arctic_final_saved = False

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._arctic_client_config is None:
            raise RuntimeError(
                "ArcticSFTTrainer._arctic_client_config not set. "
                "Ensure ArcticSFTPlugin is registered (plugins: [...ArcticSFTPlugin])."
            )

        from .deps import require_arctic_sft_client

        ArcticSFTClient, _ = require_arctic_sft_client()
        self._client = ArcticSFTClient(self._arctic_client_config)
        LOG.info(
            f"Arctic SFT training session created: "
            f"model={self._arctic_client_config.model_name}, "
            f"training_gpus={self._arctic_client_config.training_gpus}, "
            f"transport={self._arctic_client_config.backend.protocol}, "
            f"job={self._client.jobs.training}"
        )
        return self._client

    def _pad_id(self) -> int:
        if self._pad_token_id is not None:
            return self._pad_token_id
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        pad = getattr(tok, "pad_token_id", None) if tok is not None else None
        if pad is None:
            pad = getattr(tok, "eos_token_id", 0) if tok is not None else 0
        self._pad_token_id = pad
        return self._pad_token_id

    # ------------------------------------------------------------------- batch
    def _normalize_microbatch(
        self,
        batch: dict[str, torch.Tensor],
        pad_id: int,
        *,
        sample_packing: bool,
    ) -> dict[str, torch.Tensor]:
        """CPU-normalize one dataloader batch into a wire microbatch (no cross-GAS pad)."""
        ids = batch["input_ids"].detach().to("cpu", torch.long)
        labels = batch["labels"].detach().to("cpu", torch.long)
        out: dict[str, torch.Tensor] = {
            "input_ids": ids.contiguous(),
            "labels": labels.contiguous(),
        }

        pos = batch.get("position_ids")
        if pos is not None:
            out["position_ids"] = pos.detach().to("cpu", torch.long).contiguous()

        attn = batch.get("attention_mask")
        if attn is not None:
            out["attention_mask"] = attn.detach().to("cpu", torch.long).contiguous()
        elif not sample_packing:
            # Dense batches: synthesize a 0/1 mask. Never for packed batches.
            out["attention_mask"] = torch.ones_like(ids)

        return out

    def _build_wire_batch(self, batches: list[dict[str, torch.Tensor]]) -> dict:
        """Wire envelope: CPU tensors as a GAS microbatch list (H3), not concatenated.

        With sample packing, forward ``position_ids`` and never synthesize an all-ones
        ``attention_mask`` (that would force dense cross-segment attention).
        """
        pad_id = self._pad_id()
        sample_packing = bool(batches) and all("position_ids" in b for b in batches)
        if any("position_ids" in b for b in batches) and not sample_packing:
            raise ValueError(
                "arctic_sft: position_ids present on only some GAS microbatches; "
                "enable sample_packing consistently or drop position_ids"
            )

        microbatches = [
            self._normalize_microbatch(b, pad_id, sample_packing=sample_packing)
            for b in batches
        ]

        meta: dict = {"pad_token_id": pad_id, "gas_microbatches": True}
        if sample_packing:
            meta["sample_packing"] = True

        processing: dict = {"loss_fn": self._arctic_loss_fn}
        # Only "compute"/"memory" change server behavior; default "sft" ignores this.
        if self._arctic_logits_optimization != "none":
            processing["config"] = {
                "logits_optimization": self._arctic_logits_optimization,
                "logits_optimization_peak_mem_size_in_gib": self._arctic_logits_optimization_peak_mem_gib,
            }

        return {
            "batch": microbatches,
            "meta": meta,
            "processing": processing,
        }

    def _resolve_schedule(self, num_batches: int) -> tuple[int, int, int]:
        """``(grad_accum, num_train_epochs, max_steps)``.

        Trailing partial GAS groups are dropped (DeepSpeed rejects short shards).
        Re-applies the ds_config LR schedule with the resolved optimizer-step
        horizon before the client is created (DeepSpeed bakes it at engine init).
        """
        grad_accum = self.args.gradient_accumulation_steps
        steps_per_epoch = max(num_batches // grad_accum, 1)
        # Same as HF Trainer: max_steps wins; otherwise ceil(num_epochs * steps/epoch)
        # so a fractional last epoch is a partial pass, not a truncated whole epoch.
        if self.args.max_steps and self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = math.ceil(max_steps / steps_per_epoch)
        else:
            max_steps = math.ceil(self.args.num_train_epochs * steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)

        remainder = num_batches % grad_accum
        if remainder:
            LOG.warning(
                f"Arctic SFT: dropping the trailing {remainder} batch(es) this "
                f"epoch (num_batches={num_batches} not a multiple of "
                f"gradient_accumulation_steps={grad_accum}). The server's fixed "
                f"gradient_accumulation_steps requires full accumulation groups; "
                f"size the dataset/batching to avoid a remainder if you need "
                f"every sample."
            )

        if self._arctic_autoset_horizon:
            from .plugin import ArcticSFTPlugin

            ArcticSFTPlugin._apply_scheduler(
                self._arctic_client_config.training.ds_config,
                self.axolotl_cfg,
                max_steps,
            )

        return grad_accum, num_train_epochs, max_steps

    # -------------------------------------------------------------------- loop
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        train_dataloader = self.get_train_dataloader()
        num_batches = len(train_dataloader)
        grad_accum, num_train_epochs, max_steps = self._resolve_schedule(num_batches)

        client = self._get_client()

        LOG.info(
            f"Arctic SFT remote training: {num_batches} batches/epoch, "
            f"grad_accum={grad_accum}, max_steps={max_steps}, "
            f"epochs={num_train_epochs}, loss_fn={self._arctic_loss_fn}"
        )

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True

        # Server restores weights/optimizer/LR; skip already-consumed accum groups locally.
        resume_step = self._resume_from_checkpoint(resume_from_checkpoint, client)
        self.state.global_step = resume_step

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control  # type: ignore[has-type]
        )

        lr = self._arctic_learning_rate
        global_step = resume_step
        accum_groups_to_skip = resume_step
        accum_groups_seen = 0
        total_loss = 0.0
        start_time = time.time()

        def _run_step(pending: list[dict[str, torch.Tensor]], epoch: int, batch_idx: int) -> bool:
            """One optimizer step over ``pending``. Returns True if training should stop."""
            nonlocal global_step, total_loss

            self.control = self.callback_handler.on_step_begin(
                self.args, self.state, self.control
            )

            from arctic_platform.sft import merge_sft_step_metrics

            wire = self._build_wire_batch(pending)
            out = client.fwd_bwd(wire)
            # LR is server-side; ``lr`` is logging-only.
            step_out = client.step()
            # Same merge as RL ``update_actor``: loss-fn + optimizer metrics.
            metrics = merge_sft_step_metrics(out, step_out)
            logs = _loggable_metrics(metrics)
            if "loss" not in logs:
                logs["loss"] = _metric_scalar(out.get("avg_loss", 0.0))
            step_loss = logs["loss"]

            global_step += 1
            total_loss += step_loss
            self.state.global_step = global_step
            self.state.epoch = epoch + (batch_idx + 1) / num_batches

            log_interval = self.args.logging_steps
            if log_interval and global_step % log_interval == 0:
                logs["learning_rate"] = lr
                logs["epoch"] = self.state.epoch
                self.log(logs)

            self.control = self.callback_handler.on_step_end(
                self.args, self.state, self.control
            )
            # DefaultFlowCallback sets should_save / should_evaluate from save_steps / eval_steps.
            self._maybe_save_and_evaluate()
            return self.control.should_training_stop

        for epoch in range(num_train_epochs):
            if global_step >= max_steps:
                break

            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )

            pending: list[dict[str, torch.Tensor]] = []
            stop = False
            for batch_idx, batch in enumerate(train_dataloader):
                if global_step >= max_steps:
                    stop = True
                    break

                pending.append(batch)
                if len(pending) < grad_accum:
                    continue

                if accum_groups_seen < accum_groups_to_skip:
                    accum_groups_seen += 1
                    pending = []
                    continue

                stop = _run_step(pending, epoch, batch_idx)
                pending = []
                if stop:
                    break

            # Trailing partial group is dropped (fixed-gas server contract).

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_save_and_evaluate()
            if stop or self.control.should_training_stop:
                break

        if global_step > 0:
            self._save_remote_checkpoint()
            self._arctic_final_saved = True
            if self.args.load_best_model_at_end and self.state.best_global_step:
                self._load_best_checkpoint()

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)
        LOG.info(
            f"Arctic SFT training complete: {global_step} steps, {elapsed:.1f}s, "
            f"avg_loss={avg_loss:.4f}"
        )

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

        return TrainOutput(
            global_step=global_step,
            training_loss=avg_loss,
            metrics={"train_loss": avg_loss, "train_runtime": elapsed},
        )

    # -------------------------------------------------- checkpoint / eval / resume
    def _maybe_save_and_evaluate(self) -> None:
        """Act on ``control.should_save`` / ``should_evaluate``; clear flags after."""
        if self.control.should_evaluate:
            if self._has_eval_dataset():
                self.evaluate()
            self.control.should_evaluate = False
        if self.control.should_save:
            self._save_step_checkpoint()
            self.control.should_save = False

    def _has_eval_dataset(self) -> bool:
        return self.eval_dataset is not None

    def _save_step_checkpoint(self) -> None:
        """Remote engine save + local ``trainer_state.json`` + ``on_save`` + rotation."""
        self._save_remote_checkpoint()
        self._write_local_trainer_state()
        self._prune_local_checkpoints()
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _write_local_trainer_state(self) -> None:
        """HF-layout ``checkpoint-{step}/trainer_state.json`` for axolotl auto-resume discovery.

        Weights/optimizer live on the server (``load_checkpoint``); only the step is local.
        """
        try:
            ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
        except Exception:  # noqa: BLE001 — best-effort; don't crash the run on save
            LOG.exception("Arctic SFT: writing local trainer_state.json failed.")

    def _prune_local_checkpoints(self) -> None:
        """Honor ``save_total_limit`` on local ``checkpoint-*`` dirs under ``output_dir``."""
        limit = self.args.save_total_limit
        if not limit or limit <= 0 or not self.args.output_dir:
            return
        try:
            from arctic_platform.common.utils.checkpoint import prune_checkpoint_dirs

            prune_checkpoint_dirs(self.args.output_dir, limit)
        except Exception:  # noqa: BLE001 — best-effort rotation
            LOG.exception("Arctic SFT: pruning local checkpoints failed.")

    def _save_remote_checkpoint(self, *, export_hf: bool | None = None) -> None:
        try:
            do_export = self._arctic_export_hf if export_hf is None else export_hf
            limit = self.args.save_total_limit
            self._get_client().save_checkpoint(
                step=self.state.global_step,
                export_hf=do_export,
                save_total_limit=limit,
            )
            LOG.info(
                f"Arctic SFT: remote checkpoint saved "
                f"(step={self.state.global_step})."
            )
        except Exception:  # noqa: BLE001 — best-effort; don't crash the run on save
            LOG.exception("Arctic SFT: remote checkpoint save failed.")

    def _load_best_checkpoint(self) -> None:
        best = self.state.best_global_step or 0
        if best <= 0:
            return
        try:
            self._get_client().load_checkpoint(step=best)
            LOG.info(f"Arctic SFT: loaded best checkpoint (step={best}).")
        except Exception:  # noqa: BLE001
            LOG.exception("Arctic SFT: loading best checkpoint failed.")

    def _resume_from_checkpoint(self, resume_from_checkpoint: Any, client) -> int:
        """``client.load_checkpoint``; return restored ``global_step`` (0 = fresh).

        Trust only the server's returned step — a stale local ``trainer_state.json``
        must not skip groups against a fresh engine.
        """
        if not resume_from_checkpoint:
            return 0
        path = resume_from_checkpoint if isinstance(resume_from_checkpoint, str) else None
        step = None
        if isinstance(path, str):
            base = os.path.basename(path.rstrip("/"))
            if base.startswith("checkpoint-") and base.split("-", 1)[-1].isdigit():
                step = int(base.split("-", 1)[-1])
                path = None  # use job checkpoint_path + step
        try:
            resp = client.load_checkpoint(path, step=step) or {}
            restored = int(resp.get("global_step") or 0)
        except Exception:  # noqa: BLE001 — best-effort; log and start fresh
            LOG.exception("Arctic SFT: remote checkpoint load failed; starting fresh.")
            return 0
        LOG.info(f"Arctic SFT: resuming at global_step={restored}.")
        return restored

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict:
        """Remote ``fwd_no_grad``; token-weighted mean of per-batch losses → ``{prefix}_loss``."""
        eval_dl = self.get_eval_dataloader(eval_dataset)
        client = self._get_client()

        loss_sum, n_tokens = 0.0, 0
        for batch in eval_dl:
            wire = self._build_wire_batch([batch])
            out = client.fwd_no_grad(wire)
            metrics = out.get("metrics") or {}
            loss = _metric_scalar(metrics.get("loss", out.get("avg_loss", 0.0)))
            tokens = _count_valid_targets(batch)
            loss_sum += loss * tokens
            n_tokens += tokens

        eval_loss = loss_sum / n_tokens if n_tokens else 0.0
        out_metrics = {f"{metric_key_prefix}_loss": eval_loss}
        self.log(out_metrics)
        self._maybe_update_best_metric(out_metrics, metric_key_prefix)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, out_metrics
        )
        return out_metrics

    def _maybe_update_best_metric(self, metrics: dict, metric_key_prefix: str) -> None:
        """Track ``TrainerState.best_*`` for ``load_best_model_at_end``."""
        if not self.args.load_best_model_at_end:
            return
        key = self.args.metric_for_best_model or f"{metric_key_prefix}_loss"
        if key not in metrics:
            alt = f"{metric_key_prefix}_{key}" if not key.startswith(metric_key_prefix) else key
            if alt not in metrics:
                return
            key = alt
        value = metrics[key]
        greater = self.args.greater_is_better
        if greater is None:
            greater = "loss" not in key
        best = self.state.best_metric
        improved = best is None or (value > best if greater else value < best)
        if improved:
            self.state.best_metric = value
            self.state.best_global_step = self.state.global_step
            self.state.best_model_checkpoint = f"checkpoint-{self.state.global_step}"

    def save_model(self, output_dir=None, _internal_call=False):
        # Weights are remote; request an HF export when configured / on explicit save.
        self._save_remote_checkpoint(export_hf=True)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raise NotImplementedError(
            "ArcticSFTTrainer dispatches to a remote server; compute_loss is unused."
        )

    def __del__(self):
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001
                pass

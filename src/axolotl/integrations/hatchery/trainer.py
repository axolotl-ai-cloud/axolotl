# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Remote trainer that dispatches to Tinker or Hatchery API."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
from transformers.trainer_utils import TrainOutput

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .args import HatcheryConfig
from .data import batch_to_datums_sft, datums_to_tinker

LOG = get_logger(__name__)


def _extract_loss(result) -> float:
    """Extract loss:sum from a forward_backward result.

    Tinker's cross_entropy (and other losses) return the SUM of per-token
    losses, not the mean. This is by design — it lets users control
    normalization via the weights tensor. The trainer logs this raw sum;
    users who want per-token loss should divide by number of active tokens.
    """
    if hasattr(result, "metrics"):
        metrics = result.metrics or {}
        return float(metrics.get("loss:sum", metrics.get("loss", 0.0)))
    if isinstance(result, dict):
        metrics = result.get("metrics", {})
        return float(metrics.get("loss:sum", metrics.get("loss", 0.0)))
    return 0.0


def _create_training_client(args: HatcheryConfig, base_model: str):
    """Create a training client for either Tinker or Hatchery backend."""
    if args.backend == "tinker":
        import tinker

        api_key = args.api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError(
                "Tinker API key required. Set `hatchery.api_key` in config "
                "or TINKER_API_KEY env var."
            )
        os.environ["TINKER_API_KEY"] = api_key

        service = tinker.ServiceClient(project_id=args.project_id)
        return service.create_lora_training_client(
            base_model=base_model,
            rank=args.lora_rank,
            train_mlp=args.train_mlp,
            train_attn=args.train_attn,
            train_unembed=args.train_unembed,
        )

    from hatchery.core.client import TinkerClient

    base_url = args.base_url or os.environ.get("HATCHERY_URL", "http://127.0.0.1:8420")
    token = args.api_key or os.environ.get("HATCHERY_API_KEY", "dev")

    client = TinkerClient(base_url=base_url, token=token, timeout=args.future_timeout)
    return client.create_lora_training_client(
        base_model=base_model,
        rank=args.lora_rank,
        train_attn=args.train_attn,
        train_mlp=args.train_mlp,
        train_unembed=args.train_unembed,
    )


class HatcheryTrainer(AxolotlTrainer):
    """Trainer that sends preprocessed batches to a remote training API.

    Replaces local forward/backward with remote API calls to Tinker or
    Hatchery. Uses axolotl's full data preprocessing pipeline (tokenization,
    chat templates, packing, etc.) but offloads compute to remote GPUs.
    """

    hatchery_args: Optional[HatcheryConfig] = None
    _base_model_name: Optional[str] = None
    _training_client: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_training_client(self):
        """Lazily create the remote training session."""
        if self._training_client is not None:
            return self._training_client

        args = self.hatchery_args
        if args is None:
            raise RuntimeError(
                "HatcheryTrainer.hatchery_args not set. "
                "Ensure the HatcheryPlugin is registered."
            )

        base_model = self._base_model_name
        if not base_model:
            raise RuntimeError("HatcheryTrainer._base_model_name not set.")

        self._training_client = _create_training_client(args, base_model)

        LOG.info(
            f"Remote training session created: backend={args.backend}, "
            f"model={base_model}, rank={args.lora_rank}"
        )
        return self._training_client

    def _send_batch(self, batch: dict[str, torch.Tensor]):
        """Convert batch to datums and send forward_backward to remote.

        Returns (future, n_active_tokens) where n_active_tokens counts
        the completion tokens in this batch (for loss normalization).
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")

        n_active = int((labels[:, 1:] != -100).sum().item())
        datums = batch_to_datums_sft(input_ids, labels, attention_mask)

        tc = self._get_training_client()
        args = self.hatchery_args
        assert args is not None  # validated by _get_training_client
        send_datums = datums_to_tinker(datums)

        future = tc.forward_backward(
            send_datums,
            loss_fn=args.loss_fn,
            loss_fn_config=args.loss_fn_config,
        )
        return future, n_active

    def _do_optim_step(self):
        """Send optimizer step to remote using axolotl's training params."""
        import tinker.types as tt

        tc = self._get_training_client()
        return tc.optim_step(tt.AdamParams(**self._optim_params))

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Any = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """Main training loop — sends batches to remote API."""
        args = self.hatchery_args
        if args is None:
            raise RuntimeError("hatchery_args not configured")

        train_dataloader = self.get_train_dataloader()
        num_batches = len(train_dataloader)

        grad_accum = self.args.gradient_accumulation_steps
        num_train_epochs = int(self.args.num_train_epochs)
        steps_per_epoch = max(num_batches // grad_accum, 1)
        max_steps = (
            self.args.max_steps
            if self.args.max_steps > 0
            else steps_per_epoch * num_train_epochs
        )

        LOG.info(
            f"Remote training: {num_batches} batches/epoch, "
            f"{grad_accum} grad_accum, {max_steps} max steps, "
            f"{num_train_epochs} epochs"
        )

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True

        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
        )

        global_step = 0
        total_loss = 0.0
        start_time = time.time()

        for _epoch in range(num_train_epochs):
            if global_step >= max_steps:
                break

            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )

            pending_fb_futures = []
            accum_count = 0

            for batch_idx, batch in enumerate(train_dataloader):
                if global_step >= max_steps:
                    break

                self.control = self.callback_handler.on_step_begin(
                    self.args, self.state, self.control
                )

                fb_future, n_active = self._send_batch(batch)
                pending_fb_futures.append((fb_future, n_active))
                accum_count += 1

                if accum_count >= grad_accum:
                    step_loss_sum = 0.0
                    step_active = 0
                    for fut, n_act in pending_fb_futures:
                        result = fut.result(timeout=args.future_timeout)
                        step_loss_sum += _extract_loss(result)
                        step_active += n_act

                    optim_future = self._do_optim_step()
                    if not args.pipeline:
                        optim_future.result(timeout=args.future_timeout)

                    step_loss = (
                        step_loss_sum / step_active
                        if step_active > 0
                        else step_loss_sum
                    )

                    global_step += 1
                    total_loss += step_loss
                    self.state.global_step = global_step
                    self.state.epoch = _epoch + (batch_idx + 1) / num_batches

                    log_interval = self.args.logging_steps or 1
                    if global_step % log_interval == 0:
                        elapsed = time.time() - start_time
                        avg_loss = total_loss / global_step
                        LOG.info(
                            f"[step {global_step}/{max_steps}] "
                            f"loss/tok={step_loss:.4f} avg={avg_loss:.4f} "
                            f"active={step_active} "
                            f"{elapsed / global_step:.2f}s/step"
                        )
                        self.log(
                            {
                                "loss": step_loss,
                                "learning_rate": self._optim_params["learning_rate"],
                                "epoch": self.state.epoch,
                            }
                        )

                    if args.save_steps and global_step % args.save_steps == 0:
                        self._save_remote_checkpoint(global_step)

                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    pending_fb_futures = []
                    accum_count = 0

                    if self.control.should_training_stop:
                        break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            if self.control.should_training_stop:
                break

        if global_step > 0:
            self._save_remote_checkpoint(global_step, name="final")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)

        LOG.info(
            f"Training complete: {global_step} steps, {elapsed:.1f}s total, "
            f"{elapsed / max(global_step, 1):.2f}s/step, avg_loss={avg_loss:.4f}"
        )

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

        return TrainOutput(
            global_step=global_step,
            training_loss=avg_loss,
            metrics={"train_loss": avg_loss, "train_runtime": elapsed},
        )

    def _save_remote_checkpoint(self, step: int, name: Optional[str] = None):
        """Save a checkpoint on the remote service."""
        tc = self._get_training_client()
        args = self.hatchery_args
        assert args is not None  # validated by _get_training_client
        ckpt_name = name or f"{args.save_name_prefix}-{step:06d}"
        try:
            future = tc.save_state(ckpt_name)
            future.result(timeout=args.future_timeout)
            LOG.info(f"Remote checkpoint saved: {ckpt_name}")
        except Exception as e:
            LOG.warning(f"Failed to save checkpoint {ckpt_name}: {e}")

    def save_model(self, output_dir=None, _internal_call=False):
        """No-op — model weights live on the remote API."""
        LOG.info("Hatchery: save_model skipped (weights are remote)")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raise NotImplementedError(
            "HatcheryTrainer uses remote API; compute_loss should not be called."
        )

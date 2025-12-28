"""Monkey patch for trainer.compute_loss to apply DFT loss in SFT training."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from axolotl.utils.distributed import is_distributed

from .dft_utils import compute_dft_loss


def patch_compute_loss_for_dft(trainer, cfg) -> None:
    """Patch a trainer instance to apply DFT loss when enabled."""
    original_compute_loss = trainer.compute_loss

    def compute_loss_with_dft(
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        if not getattr(trainer.args, "enable_dft_loss", False):
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if getattr(trainer.args, "orpo_alpha", None):
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if getattr(trainer.args, "label_smoothing_factor", 0.0) not in (0, 0.0, None):
            msg = (
                "DFT loss is currently incompatible with label smoothing "
                "(label_smoothing_factor > 0)."
            )
            raise ValueError(msg)

        labels = inputs.get("labels")
        if labels is None:
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if getattr(trainer.args, "include_tkps", False):
            inputs_key = "labels" if "labels" in inputs else "input_ids"
            num_tokens = (inputs[inputs_key] != -100).sum()
            if is_distributed():
                torch.distributed.all_reduce(
                    num_tokens, op=torch.distributed.ReduceOp.SUM
                )

            local_tokens = (inputs[inputs_key] != -100).sum().cpu()
            if hasattr(trainer.state, "num_tokens"):
                trainer.state.num_tokens = trainer.state.num_tokens + local_tokens
            else:
                trainer.state.num_tokens = local_tokens

            if hasattr(trainer.state, "total_tokens"):
                trainer.state.total_tokens += num_tokens
            else:
                trainer.state.total_tokens = num_tokens

        forward_inputs = dict(inputs)
        labels = forward_inputs.pop("labels")

        outputs = model(**forward_inputs)
        logits = _extract_logits(outputs)
        if logits is None:
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        loss = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=num_items_in_batch,
        )

        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss_with_dft


def _extract_logits(outputs: Any) -> torch.Tensor | None:
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        return outputs.get("logits")
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    if isinstance(outputs, SimpleNamespace) and hasattr(outputs, "logits"):
        return outputs.logits
    return None

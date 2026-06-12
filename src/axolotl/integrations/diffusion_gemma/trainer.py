"""Block-diffusion trainer for DiffusionGemma."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .diffusion import (
    DiffusionObjectiveConfig,
    corrupt_canvas,
    diffusion_loss,
    sample_timesteps,
)

LOG = get_logger(__name__)


def block_diffusion_step(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    cfg: DiffusionObjectiveConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float], object]:
    """Run one block-diffusion training step: corrupt → (self-condition) → forward → loss.

    Returns ``(loss, metrics, model_outputs)``. Kept model-agnostic and free of
    Trainer state so it can be unit-tested against a tiny model.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    canvas_labels = inputs["canvas_labels"]
    eligible = inputs["canvas_loss_mask"]
    device = input_ids.device
    batch, canvas_len = canvas_labels.shape

    # Multimodal inputs (image tokens live in the prompt prefix -> encoder).
    encoder_kwargs = {
        k: inputs[k]
        for k in ("pixel_values", "image_position_ids", "mm_token_type_ids")
        if k in inputs and inputs[k] is not None
    }

    timesteps = sample_timesteps(batch, device, cfg.timestep_eps, generator=generator)
    noised_canvas, corrupted_mask = corrupt_canvas(
        canvas_labels, timesteps, cfg, eligible_mask=eligible, generator=generator
    )

    decoder_attention_mask = torch.cat(
        [attention_mask, attention_mask.new_ones((batch, canvas_len))], dim=1
    )

    self_conditioning_logits = None
    draw = torch.rand((), generator=generator, device="cpu").item()
    if cfg.self_conditioning_prob > 0 and draw < cfg.self_conditioning_prob:
        with torch.no_grad():
            sc_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=noised_canvas,
                decoder_attention_mask=decoder_attention_mask,
                **encoder_kwargs,
            )
        self_conditioning_logits = sc_out.logits.detach()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=noised_canvas,
        decoder_attention_mask=decoder_attention_mask,
        self_conditioning_logits=self_conditioning_logits,
        **encoder_kwargs,
    )

    loss, metrics = diffusion_loss(
        outputs.logits, canvas_labels, corrupted_mask, timesteps, cfg
    )

    with torch.no_grad():
        if corrupted_mask.sum() > 0:
            preds = outputs.logits[corrupted_mask].argmax(dim=-1)
            metrics["diffusion/accuracy"] = (
                (preds == canvas_labels[corrupted_mask]).float().mean().item()
            )
        metrics["diffusion/self_conditioned"] = float(self_conditioning_logits is not None)
    return loss, metrics, outputs


class BlockDiffusionTrainer(AxolotlTrainer):
    """Trainer that denoises a target canvas conditioned on an encoder prefix.

    Expects batches from :class:`CanvasCollator`:
    ``input_ids`` / ``attention_mask`` (encoder prefix) and ``canvas_labels`` /
    ``canvas_loss_mask`` (the clean target block).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj_cfg: DiffusionObjectiveConfig | None = None

    def post_set_axolotl_cfg(self):
        bd = self.axolotl_cfg.block_diffusion
        vocab_size = self.model.config.get_text_config().vocab_size
        self._obj_cfg = DiffusionObjectiveConfig(
            vocab_size=vocab_size,
            corruption=bd.corruption,
            mask_token_id=bd.mask_token_id,
            timestep_eps=bd.timestep_eps,
            loss_weighting=bd.loss_weighting,
            self_conditioning_prob=bd.self_conditioning_prob,
        )
        LOG.info(
            f"DiffusionGemma block-diffusion: corruption={bd.corruption} "
            f"weighting={bd.loss_weighting} self_cond_p={bd.self_conditioning_prob}"
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        loss, metrics, outputs = block_diffusion_step(model, inputs, self._obj_cfg)
        train_eval: Literal["train", "eval"] = "train" if model.training else "eval"
        self.store_metrics(metrics, train_eval=train_eval)

        if return_outputs:
            return loss, outputs
        return loss

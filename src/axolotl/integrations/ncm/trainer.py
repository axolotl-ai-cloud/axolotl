"""Trainer and callback for NCM: extra metrics and periodic samples."""

import torch
from transformers import TrainerCallback

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .modeling import NCMGeneratorModel, NCMTokenizerModel

LOG = get_logger(__name__)


class NCMTrainer(AxolotlTrainer):
    """Logs reconstruction/codebook metrics (stage 1) or per-level losses (stage 2)."""

    # exp(loss) is not a token perplexity for either stage.
    reports_token_perplexity = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        split = "train" if model.training else "eval"
        prefix = "" if model.training else "eval_"
        unwrapped = self.accelerator.unwrap_model(model)
        if isinstance(unwrapped, NCMTokenizerModel):
            metrics = {
                "recon_loss": outputs.recon_loss.item(),
                "vq_loss": outputs.vq_loss.item(),
            }
            if outputs.logits is not None:  # None under fused cross entropy
                # The model crops or pads the batch to its block length.
                n = min(outputs.logits.shape[1], inputs["labels"].shape[1])
                labels = inputs["labels"][:, :n]
                valid = labels != -100
                hits = (outputs.logits[:, :n].argmax(-1) == labels) & valid
                metrics["recon_acc"] = (hits.sum() / valid.sum().clamp(min=1)).item()
            if split == "train":
                util = unwrapped.vq.codebook_utilization()
                metrics["codebook_util"] = (sum(util) / len(util), "max")
            self.store_metrics(
                {prefix + k: v for k, v in metrics.items()}, train_eval=split
            )
        elif isinstance(unwrapped, NCMGeneratorModel):
            first = unwrapped.first
            self.store_metrics(
                {
                    f"{prefix}level_loss/{first + i}": v.item()
                    for i, v in enumerate(outputs.level_losses)
                },
                train_eval=split,
            )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrapped, NCMTokenizerModel) and "loss" in logs:
            # Utilization is reported per logging window, not since step 0.
            unwrapped.vq.reset_utilization()


def truncate_at_eos(ids: list[int], eos_token_id: int | None) -> list[int]:
    """Blocks are fixed-length; positions after the first EOS are untrained filler."""
    return ids[: ids.index(eos_token_id)] if eos_token_id in ids else ids


class NCMSampleCallback(TrainerCallback):
    """Logs unconditional generator samples, decoded coarse to fine."""

    def __init__(self, tokenizer, every: int, num_samples: int, temperature: float):
        self.tokenizer = tokenizer
        self.every = every
        self.num_samples = num_samples
        self.temperature = temperature

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or state.global_step % self.every:
            return
        model = getattr(model, "module", model)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                ids = model.generate(
                    batch_size=self.num_samples, temperature=self.temperature
                )
            for i, row in enumerate(ids.tolist()):
                text = self.tokenizer.decode(
                    truncate_at_eos(row, self.tokenizer.eos_token_id),
                    skip_special_tokens=True,
                )
                LOG.info("NCM sample %d @ step %d: %r", i, state.global_step, text)
        finally:
            model.train(was_training)

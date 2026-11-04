"""Trainer that optimizes a letter-logit decision head on a causal LM."""

from __future__ import annotations

import os

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from .data import letter_token_ids
from .head import HEAD_FILENAME, DecisionHead, decision_logits, soft_cross_entropy


class CausalDecisionTrainer(AxolotlTrainer):
    """Soft-target CE over the answer-letter logits of a decision head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The loss is a per-batch mean, so Trainer must scale it by accumulation steps.
        self.model_accepts_loss_kwargs = False

    def post_set_axolotl_cfg(self):
        super().post_set_axolotl_cfg()
        decision = self.axolotl_cfg.causal_decision
        letter_ids = letter_token_ids(self.processing_class, decision.num_options)
        self._lm_head = self.model.get_output_embeddings()
        # Registered on the model so the optimizer and DDP see its parameters.
        self.model.decision_head = DecisionHead.from_lm_head(
            self.model, letter_ids, decision.rank
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        targets = inputs.pop("decision_targets")
        mask = inputs.pop("decision_mask")
        self._maybe_inject_mm_token_type_ids(inputs)
        logits, outputs = decision_logits(
            model, self.model.decision_head, self._lm_head, **inputs
        )
        loss = soft_cross_entropy(logits, targets, mask)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        torch.save(
            self.model.decision_head.state_dict(),
            os.path.join(output_dir or self.args.output_dir, HEAD_FILENAME),
        )

"""Trainer that optimizes a letter-logit decision head on a causal LM."""

from __future__ import annotations

import os

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from .data import letter_token_ids
from .head import HEAD_FILENAME, DecisionHead, decision_logits, soft_cross_entropy


class CausalDecisionTrainer(AxolotlTrainer):
    """Soft-target CE over the answer-letter logits of a decision head."""

    reports_token_perplexity = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The loss is a per-batch mean, so Trainer must scale it by accumulation steps.
        self.model_accepts_loss_kwargs = False

    def post_set_axolotl_cfg(self):
        super().post_set_axolotl_cfg()
        decision = self.axolotl_cfg.causal_decision
        letter_ids = letter_token_ids(self.processing_class, decision.num_options)
        self._lm_head = self.model.get_output_embeddings()
        head = DecisionHead.from_lm_head(self.model, letter_ids, decision.rank)
        # Registered on the model so the optimizer and DDP see its parameters.
        self.model.decision_head = head
        self.model._keys_to_ignore_on_save.update(
            f"decision_head.{name}" for name in head.state_dict()
        )
        # Vocab logits never reach the loss, and DDP rejects trainable params without grads.
        input_embeddings = self.model.get_input_embeddings().weight
        for param in self._lm_head.parameters():
            if param is not input_embeddings:
                param.requires_grad_(False)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += ["decision_targets", "num_options"]

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

    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        if self.args.should_save:
            self._save_head(self.args.output_dir)
        return output

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self._save_head(output_dir or self.args.output_dir)

    def _save_head(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.decision_head.state_dict(),
            os.path.join(output_dir, HEAD_FILENAME),
        )

    def _load_head(self, checkpoint_dir):
        state = torch.load(
            os.path.join(checkpoint_dir, HEAD_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
        self.model.decision_head.load_state_dict(state)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self._load_head(resume_from_checkpoint)

    def _load_best_model(self):
        super()._load_best_model()
        self._load_head(self.state.best_model_checkpoint)

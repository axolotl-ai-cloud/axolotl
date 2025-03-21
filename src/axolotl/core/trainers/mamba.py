"""Module for mamba trainer"""

import torch

from axolotl.core.trainers.base import AxolotlTrainer


class AxolotlMambaTrainer(AxolotlTrainer):
    """Mamba specific trainer to handle loss calculation"""

    tag_names = ["axolotl", "mamba"]

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,  # pylint: disable=unused-argument
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        return lm_loss

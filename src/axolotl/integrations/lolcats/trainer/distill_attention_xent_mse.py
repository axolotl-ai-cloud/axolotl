"""
Custom trainer class for distilling attentions ("attention transfer"). Can substitute for Hugging Face trainer.

In this implementation we support using either just the softmax attention outputs, or the softmax attention weights.
"""

from typing import Any

from torch import Tensor, nn, tensor

from axolotl.core.trainers.base import AxolotlTrainer


class DistillAttentionXentMSETrainer(AxolotlTrainer):
    """
    Custom trainer class for distilling attentions.
    - We compute and store the attention outputs and/or weights for each head and layer,
      for both the "teacher" softmax attentions and "student" learnable subquadratic attentions
    - We then train the student layers to minimize either MSE(outputs) or CrossEntropy(weights)
    """

    def __init__(
        self,
        model: nn.Module,
        mse_factor: float = 1e3,
        xent_factor: float = 0,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.criterion_xent = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_mse = nn.MSELoss(reduction="mean")
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor
        # self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss # NOTE: this config seems unnecessary

        self.model_accepts_loss_kwargs = False  # added to combat explosive loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> tuple[Tensor, dict]:
        """
        Attention distillation ("attention transfer")
        - For each layer and head, get attentions and train to
          minimize some combo of MSE and cross-entropy loss
        """
        # alias inputs to data
        data = inputs

        device = model.device

        # Filter out labels
        inputs = {k: v.to(device) for k, v in data.items() if k != "labels"}

        # Forward pass
        outputs = model(**inputs, output_attentions=True, use_cache=False)
        outputs = outputs.get("attentions")

        # Attentions are tuple[tuple[torch.Tensor, torch.Tensor]]
        # n_layers x (predicted_attns, true_attns)
        # predicted_attns and true_attns are shape (batch, n_heads, q_len, k_len)
        loss_mse = tensor(0.0, device=device)
        loss_xent = tensor(0.0, device=device)
        n_layers = 0  # Number of layers to distill
        softmax_layers = []
        for layer_idx, attns in enumerate(outputs):
            if attns is not None:
                if len(attns) != 2:
                    attns = attns.cpu()
                else:
                    if self.xent_factor > 0:
                        # Cross-entropy loss
                        a_pred, a_true = attns[0]
                        a_pred = a_pred.clamp(
                            min=1e-12
                        ).log()  # nn.CrossEntropy assumes unnormalized logits
                        k_len = a_true.shape[-1]  # batch, n_heads, q_len, k_len
                        # Compute mean cross-entropy over all queries
                        a_pred = a_pred.contiguous().view(-1, k_len)
                        a_true = a_true.contiguous().view(-1, k_len)
                        loss_xent += self.criterion_xent(a_pred, a_true)
                    if self.mse_factor > 0:
                        loss_mse += self.criterion_mse(*attns[1])
                    n_layers += 1
            else:
                softmax_layers.append(layer_idx)
        if n_layers > 0:
            loss_xent = loss_xent / n_layers * self.xent_factor
            loss_mse = loss_mse / n_layers * self.mse_factor
        loss = loss_xent + loss_mse

        if "position_ids" in data:
            outputs = {
                "loss_xent": loss_xent.item() if self.xent_factor > 0 else 0,
                "loss_mse": loss_mse if self.mse_factor > 0 else 0,
                "input_len": data["position_ids"].shape[1],
                "position_ids": data["position_ids"][0].detach().cpu().numpy(),
                "mse_factor": self.mse_factor,
                "xent_factor": self.xent_factor,
            }
        else:
            outputs = {
                "loss_xent": loss_xent.item() if self.xent_factor > 0 else 0,
                "loss_mse": loss_mse if self.mse_factor > 0 else 0,
                "mse_factor": self.mse_factor,
                "xent_factor": self.xent_factor,
            }
        return (loss, outputs) if return_outputs else loss

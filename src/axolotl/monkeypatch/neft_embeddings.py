"""
patches implemented through the trainer hooks to enable NEFT/noisy embeddings per https://arxiv.org/abs/2310.05914
"""
import torch
from peft import PeftModel
from transformers import PreTrainedModel


def patch_neft(alpha, model):
    embeddings = None
    if isinstance(model, PreTrainedModel):
        embeddings = model.get_input_embeddings()
    if isinstance(model, PeftModel):
        embeddings = model.base_model.get_input_embeddings()
    if not embeddings:
        raise ValueError(f"unhandled model class for neft: {model.__class__.__name__}")
    embeddings.noisy_embedding_alpha = alpha
    old_forward = embeddings.forward

    # This hack seems to be needed to properly use a custom forward pass
    # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
    bound_method = neft_forward.__get__(  # pylint: disable=no-value-for-parameter
        embeddings, embeddings.__class__
    )
    setattr(embeddings, "forward", bound_method)

    embeddings._old_forward = old_forward  # pylint: disable=protected-access
    return model


def unpatch_neft(model):
    embeddings = None
    if isinstance(model, PreTrainedModel):
        embeddings = model.get_input_embeddings()
    if isinstance(model, PeftModel):
        embeddings = model.base_model.get_input_embeddings()
    if not embeddings:
        raise ValueError(f"unhandled model class for neft: {model.__class__.__name__}")
    if hasattr(embeddings, "_old_forward"):
        embeddings.forward = embeddings._old_forward  # pylint: disable=protected-access
        del embeddings._old_forward  # pylint: disable=protected-access
        del embeddings.noisy_embedding_alpha


def neft_forward(self, inputs: torch.Tensor):
    embeddings = self._old_forward(inputs)  # pylint: disable=protected-access

    if self.training:
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = self.noisy_embedding_alpha / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(
            -mag_norm, mag_norm
        )

    return embeddings


def pretrain_hook(cfg, trainer):
    if cfg.noisy_embedding_alpha:
        trainer.model = patch_neft(cfg.noisy_embedding_alpha, trainer.model)


def post_train_hook(cfg, trainer):
    if cfg.noisy_embedding_alpha:
        unpatch_neft(trainer.model)

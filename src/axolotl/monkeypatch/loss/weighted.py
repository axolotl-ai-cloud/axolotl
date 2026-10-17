"""Per-token loss weights for causal LM SFT.

Each trained token's cross entropy is multiplied by its weight and the sum is
divided by the trained-token count (``num_items_in_batch`` across gradient
accumulation, else the micro-batch count), the same denominator as the
unweighted loss. Unit weights reproduce the unweighted loss, and a weight scales
a token's gradient by exactly that factor regardless of what it is batched with.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

_STATE_ATTR = "_axolotl_loss_weights_state"
_INSTALLED_ATTR = "_axolotl_weighted_loss_installed"


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if shift_labels is not None:
        raise ValueError(
            "Per-token loss weights do not support pre-shifted `shift_labels` "
            "(context/sequence parallelism)."
        )
    logits = logits.float()
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    shift_weights = F.pad(loss_weights, (0, 1), value=0.0)[..., 1:]

    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labels.reshape(-1).to(logits.device),
        ignore_index=ignore_index,
        reduction="none",
    )
    weighted_sum = (
        per_token * shift_weights.reshape(-1).to(per_token.device, per_token.dtype)
    ).sum()

    if num_items_in_batch is None:
        num_items_in_batch = (shift_labels != ignore_index).sum().clamp(min=1)
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(weighted_sum.device)
    return weighted_sum / num_items_in_batch


def _loss_owner(model):
    while True:
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def _install(owner) -> None:
    if owner.__dict__.get(_INSTALLED_ATTR):
        return
    original = owner.loss_function

    def loss_function(*args, **kwargs):
        state = owner.__dict__.get(_STATE_ATTR)
        if state is None:
            return original(*args, **kwargs)
        bound = dict(zip(("logits", "labels", "vocab_size"), args, strict=False))
        bound.update(kwargs)
        state["consumed"] = True
        return weighted_causal_lm_loss(
            bound["logits"],
            bound["labels"],
            state["loss_weights"],
            num_items_in_batch=bound.get("num_items_in_batch"),
            ignore_index=bound.get("ignore_index", -100),
            shift_labels=bound.get("shift_labels"),
        )

    owner.loss_function = loss_function
    owner.__dict__[_INSTALLED_ATTR] = True


@contextmanager
def loss_weights_applied(model, loss_weights: torch.Tensor):
    """Route ``model``'s own loss through ``weighted_causal_lm_loss`` for one forward.

    Going through the model's ``loss_function`` keeps model-specific terms such as
    MoE router auxiliary losses and logit soft-capping that a trainer-side loss
    computed from returned logits would drop.
    """
    owner = _loss_owner(model)
    _install(owner)
    state = {"loss_weights": loss_weights, "consumed": False}
    owner.__dict__[_STATE_ATTR] = state
    try:
        yield
    finally:
        owner.__dict__.pop(_STATE_ATTR, None)
    if not state["consumed"]:
        raise RuntimeError(
            "`loss_weights` were provided but the model's forward never called its "
            "`loss_function` (a fused or custom loss replaced it), so they would be "
            "silently ignored. Disable the fused loss or drop `message_field_weight` / "
            "`field_sample_weight`."
        )

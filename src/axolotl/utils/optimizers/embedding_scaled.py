"""
Scales the learning rate of the embedding layer by a factor of `embedding_lr_scale` or to `embedding_lr` if set.

Applies weight decay to parameters in `decay_parameters` and no weight decay to the rest.
"""


def create_embedding_scaled_optimizer(
    opt_model,
    embedding_lr_scale,
    embedding_lr,
    weight_decay,
    decay_parameters,
    optimizer_cls,
    optimizer_kwargs,
):
    params = {
        "embeddings": {},  # lm_head, embed_tokens,
        "to_weight_decay": {},  # LayerNorm and bias
        "no_weight_decay": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight") or any(
            embed_name in name for embed_name in ["embed_tokens", "lm_head"]
        ):
            params["embeddings"][name] = param
        elif name in decay_parameters:
            params["to_weight_decay"][name] = param
        else:
            params["no_weight_decay"][name] = param

    optimizer_grouped_parameters = []
    if params["to_weight_decay"]:
        optimizer_grouped_parameters.append(
            {
                "params": list(params["to_weight_decay"].values()),
                "weight_decay": weight_decay,
                "lr": optimizer_kwargs["lr"],
            }
        )

    if params["embeddings"]:
        lr = optimizer_kwargs["lr"]  # pylint: disable=invalid-name
        if embedding_lr_scale:
            lr *= embedding_lr_scale  # pylint: disable=invalid-name
        elif embedding_lr:
            lr = embedding_lr  # pylint: disable=invalid-name
        optimizer_grouped_parameters.append(
            {
                "params": list(params["embeddings"].values()),
                "weight_decay": 0.0,
                "lr": lr,
            }
        )

    if params["no_weight_decay"]:
        optimizer_grouped_parameters.append(
            {
                "params": list(params["no_weight_decay"].values()),
                "weight_decay": 0.0,
                "lr": optimizer_kwargs["lr"],
            }
        )

    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

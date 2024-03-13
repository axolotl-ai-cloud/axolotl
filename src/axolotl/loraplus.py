"""Module for LoRA+"""

# MIT License
#
# Copyright (c) 2024 nikhil-ghosh-berkeley
# https://github.com/nikhil-ghosh-berkeley/loraplus

import logging
from functools import reduce

from peft.tuners import lora
from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

LOG = logging.getLogger("axolotl.loraplus")


def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.
    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module


def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.

    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_cls (class): The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        loraplus_lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        loraplus_lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.

    Returns:
        An instance of the specified optimizer class configured with the model's parameters organized into groups with custom learning rates.
    """

    assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided."

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group, group_params in param_groups.items():
        assigned_param_groups += f"{group}\n {list(group_params.keys())}\n\n"
    LOG.info(assigned_param_groups)

    lr = optimizer_kwargs["lr"]  # pylint: disable=invalid-name
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                LOG.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                LOG.debug(f"bitsandbytes: will optimize {module} in fp32")
        LOG.info(f"skipped: {skipped/2**20}M params")

    return optimizer

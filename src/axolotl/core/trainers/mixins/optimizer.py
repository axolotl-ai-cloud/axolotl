"""Module for Axolotl trainer optimizer mixin"""

import logging

from peft.optimizers import create_loraplus_optimizer
from torch import nn
from transformers.trainer import Trainer
from transformers.utils import is_sagemaker_mp_enabled

from axolotl.integrations.base import BaseOptimizerFactory

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

LOG = logging.getLogger(__name__)


class OptimizerMixin(Trainer):
    """Mixin class for shared handling of building custom optimizers"""

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def create_optimizer_grouped_parameters(
        self, opt_model, optimizer_kwargs
    ) -> list[dict]:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        params: dict = {
            "to_weight_decay": {},  # LayerNorm and bias
            "embeddings": {},  # lm_head, embed_tokens,
            "no_weight_decay": {},
        }
        lr_groups_lookup = {}
        lr_groups_learning_rates = {}
        if self.args.lr_groups:
            for lr_group in self.args.lr_groups:
                group_name = lr_group["name"]
                group_modules = lr_group["modules"]
                for module in group_modules:
                    lr_groups_lookup[module] = group_name
                lr_groups_learning_rates[group_name] = lr_group["lr"]
                params[f"to_weight_decay_{group_name}"] = {}

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("modules_to_save.default.weight") or any(
                embed_name in name for embed_name in ["embed_tokens", "lm_head"]
            ):
                params["embeddings"][name] = param
            elif name in decay_parameters:
                lr_group_modules = [
                    group_modules
                    for group_modules in lr_groups_lookup
                    if group_modules in name
                ]
                if lr_groups_lookup and any(lr_group_modules):
                    lr_group_module = lr_group_modules[0]
                    group_name = lr_groups_lookup[lr_group_module]
                    params[f"to_weight_decay_{group_name}"][name] = param
                else:
                    params["to_weight_decay"][name] = param
            else:
                params["no_weight_decay"][name] = param
        optimizer_grouped_parameters = []
        if params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(params["to_weight_decay"].values()),
                    "weight_decay": self.args.weight_decay,
                    "lr": optimizer_kwargs["lr"],
                }
            )
        if params["embeddings"]:
            lr = optimizer_kwargs["lr"]  # pylint: disable=invalid-name
            if self.args.embedding_lr_scale:
                lr *= self.args.embedding_lr_scale  # pylint: disable=invalid-name
            elif self.args.embedding_lr:
                lr = self.args.embedding_lr  # pylint: disable=invalid-name
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
        for group_name, group_lr in lr_groups_learning_rates.items():
            if params[f"to_weight_decay_{group_name}"]:
                optimizer_grouped_parameters.append(
                    {
                        "params": list(
                            params[f"to_weight_decay_{group_name}"].values()
                        ),
                        "weight_decay": self.args.weight_decay,
                        "lr": group_lr,
                    }
                )

        return optimizer_grouped_parameters

    def create_optimizer(self):
        if (
            self.args.loraplus_lr_ratio is None
            and self.args.embedding_lr_scale is None
            and self.args.embedding_lr is None
            and self.args.lr_groups is None
            and self.optimizer_cls_and_kwargs is None
        ):
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if (
            not self.optimizer
            and self.optimizer_cls_and_kwargs is not None
            and issubclass(self.optimizer_cls_and_kwargs[0], BaseOptimizerFactory)
        ):
            optimizer_factory_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            self.optimizer = optimizer_factory_cls()(
                opt_model, self.args, **optimizer_kwargs
            )

        if not self.optimizer:
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            optimizer_grouped_parameters = self.create_optimizer_grouped_parameters(
                opt_model, optimizer_kwargs
            )

            if self.args.loraplus_lr_ratio is not None:
                loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
                loraplus_lr_embedding = getattr(
                    self.args, "loraplus_lr_embedding", 1e-6
                )
                self.optimizer = create_loraplus_optimizer(  # pylint: disable=attribute-defined-outside-init
                    opt_model,
                    optimizer_cls,
                    loraplus_lr_ratio=loraplus_lr_ratio,
                    loraplus_lr_embedding=loraplus_lr_embedding,
                    **optimizer_kwargs,
                )
            else:
                # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for GaLore optimizer.
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("params")

                # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for LOMO optimizer.
                if "model" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("model")

                # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
                # to avoid arguments conflicts.
                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop(
                        "optimizer_dict"
                    )

                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        LOG.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        LOG.debug(f"bitsandbytes: will optimize {module} in fp32")
                LOG.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(  # pylint: disable=attribute-defined-outside-init
                self.optimizer
            )

        return self.optimizer

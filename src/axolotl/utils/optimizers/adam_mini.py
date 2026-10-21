"""Factory for the Adam-mini optimizer (https://arxiv.org/abs/2406.16793)."""

import torch.distributed as dist

from axolotl.integrations.base import BaseOptimizerFactory


class AdamMiniOptimizerFactory(BaseOptimizerFactory):
    """Builds Adam-mini with attention head layout taken from the model config."""

    def __call__(self, opt_model, training_args=None, **optimizer_kwargs):
        from adam_mini import Adam_mini

        not_2d = [
            name
            for name, param in opt_model.named_parameters()
            if param.requires_grad and param.dim() > 2
        ]
        if not_2d:
            raise ValueError(
                "adam_mini only supports 1D/2D parameters, but "
                f"{not_2d[:5]} are not (e.g. fused MoE experts or conv layers). "
                "Pick a different optimizer."
            )

        config = opt_model.config.get_text_config()
        for key, attr in (
            ("dim", "hidden_size"),
            ("n_heads", "num_attention_heads"),
            ("n_kv_heads", "num_key_value_heads"),
        ):
            value = optimizer_kwargs.get(key)
            if value is None:
                value = getattr(config, attr, None)
            if value is not None:
                optimizer_kwargs[key] = int(value)
        # upstream defaults to printing every parameter block it partitions
        optimizer_kwargs.setdefault("verbose", False)
        optimizer = Adam_mini(opt_model.named_parameters(), **optimizer_kwargs)

        # upstream assumes head_dim == dim // n_heads and world_size == cuda.device_count()
        head_dim = getattr(config, "head_dim", None) or (
            optimizer.dim // optimizer.n_heads
        )
        optimizer.head_numel = optimizer.dim * head_dim
        optimizer.world_size = dist.get_world_size() if dist.is_initialized() else 1

        no_decay = {
            id(module.weight)
            for module in (
                opt_model.get_input_embeddings(),
                opt_model.get_output_embeddings(),
            )
            if module is not None
        }
        for group in optimizer.param_groups:
            if id(group["params"][0]) in no_decay:
                group["weight_decay"] = 0.0
        return optimizer

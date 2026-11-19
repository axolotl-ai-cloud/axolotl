"""Factory for the Adam-mini optimizer (https://arxiv.org/abs/2406.16793)."""

from axolotl.integrations.base import BaseOptimizerFactory


class AdamMiniOptimizerFactory(BaseOptimizerFactory):
    """Builds Adam-mini with attention head layout taken from the model config."""

    def __call__(self, opt_model, training_args=None, **optimizer_kwargs):
        from adam_mini import Adam_mini

        config = opt_model.config.get_text_config()
        optimizer_kwargs.setdefault("dim", config.hidden_size)
        optimizer_kwargs.setdefault("n_heads", config.num_attention_heads)
        optimizer_kwargs.setdefault(
            "n_kv_heads", getattr(config, "num_key_value_heads", None)
        )
        # upstream defaults to printing every parameter block it partitions
        optimizer_kwargs.setdefault("verbose", False)
        return Adam_mini(opt_model.named_parameters(), **optimizer_kwargs)

"""
DPO Specific Strategy for training
"""

from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer


class DPOStrategy:
    """
    Strategy for DPO training
    """

    @classmethod
    def get_trainer_class(cls):
        return AxolotlDPOTrainer

    @classmethod
    def get_training_args_class(cls):
        from axolotl.core.trainers.dpo.args import AxolotlDPOConfig

        return AxolotlDPOConfig

    @classmethod
    def set_training_args_kwargs(cls, cfg):
        training_args_kwargs = {}
        if cfg.rl == "ipo":
            training_args_kwargs["loss_type"] = "ipo"
        training_args_kwargs["max_length"] = cfg.sequence_len
        training_args_kwargs["max_completion_length"] = None
        training_args_kwargs["max_prompt_length"] = cfg.sequence_len
        training_args_kwargs["generate_during_eval"] = cfg.use_wandb
        if cfg.dpo_use_weighting is not None:
            training_args_kwargs["use_weighting"] = cfg.dpo_use_weighting
        return training_args_kwargs

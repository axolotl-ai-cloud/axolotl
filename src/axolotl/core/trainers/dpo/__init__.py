"""DPO Specific Strategy for training"""

from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer
from axolotl.utils.schemas.enums import RLType


class DPOStrategy:
    """Strategy for DPO training"""

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
        if cfg.rl is RLType.IPO:
            training_args_kwargs["loss_type"] = "ipo"
        # Label smoothing is not compatible with IPO
        if cfg.rl is RLType.DPO and cfg.dpo_label_smoothing:
            training_args_kwargs["label_smoothing"] = cfg.dpo_label_smoothing
        training_args_kwargs["max_completion_length"] = None
        training_args_kwargs["max_length"] = cfg.sequence_len
        training_args_kwargs["generate_during_eval"] = cfg.dpo_generate_during_eval
        if cfg.dpo_use_weighting is not None:
            training_args_kwargs["use_weighting"] = cfg.dpo_use_weighting
        if cfg.dpo_padding_free is not None:
            training_args_kwargs["padding_free"] = cfg.dpo_padding_free
        if cfg.dpo_norm_loss is not None:
            training_args_kwargs["dpo_norm_loss"] = cfg.dpo_norm_loss
        if cfg.dpo_use_logits_to_keep is not None:
            training_args_kwargs["use_logits_to_keep"] = cfg.dpo_use_logits_to_keep
        if cfg.dpo_use_liger_kernel is not None:
            training_args_kwargs["use_liger_kernel"] = cfg.dpo_use_liger_kernel
        return training_args_kwargs

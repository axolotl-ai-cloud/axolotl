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
        if cfg.rl is RLType.DPO:
            if cfg.dpo_loss_type is not None:
                training_args_kwargs["loss_type"] = cfg.dpo_loss_type

            if cfg.dpo_loss_weights is not None:
                training_args_kwargs["loss_weights"] = cfg.dpo_loss_weights

        if cfg.rl is RLType.IPO:
            training_args_kwargs["loss_type"] = ["ipo"]

        # Label smoothing is not compatible with IPO
        if cfg.rl is RLType.DPO and cfg.dpo_label_smoothing:
            training_args_kwargs["label_smoothing"] = cfg.dpo_label_smoothing
        training_args_kwargs["max_length"] = cfg.sequence_len
        if cfg.dpo_use_weighting is not None:
            training_args_kwargs["use_weighting"] = cfg.dpo_use_weighting
        if cfg.dpo_padding_free is not None:
            training_args_kwargs["padding_free"] = cfg.dpo_padding_free
        if cfg.dpo_use_liger_kernel is not None:
            training_args_kwargs["use_liger_kernel"] = cfg.dpo_use_liger_kernel
        if cfg.precompute_ref_log_probs is not None:
            training_args_kwargs["precompute_ref_log_probs"] = (
                cfg.precompute_ref_log_probs
            )
        return training_args_kwargs

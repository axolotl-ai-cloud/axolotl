"""Custom handling to not fail training if fsdp optimizer is not savable"""

import os

from transformers import Trainer

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CheckpointSaveMixin(Trainer):
    """Mixin to handle saving the optimizer and scheduler if they are not savable."""

    def _save_optimizer_and_scheduler(self, output_dir):
        try:
            super()._save_optimizer_and_scheduler(output_dir)
        except (NotImplementedError, KeyError) as exc:
            LOG.warning_once(
                f"Trainer does not support saving optimizer and scheduler:  {exc}\n"
                "Optimizer and scheduler states were not saved - resuming from checkpoints "
                "for this training run will not be possible.",
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if (
            resume_from_checkpoint is not None
            and self.is_fsdp_enabled
            and getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        ):
            self._align_fsdp2_state_dict_type(resume_from_checkpoint)
        super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _align_fsdp2_state_dict_type(self, checkpoint):
        """Set fsdp_plugin.state_dict_type to match the format actually saved in checkpoint.

        FSDP2 defaults to SHARDED_STATE_DICT, but users can change the setting between
        runs, causing load_fsdp_model/load_fsdp_optimizer to look for the wrong files.
        Auto-detecting from the checkpoint dir fixes the mismatch. The state_dict_config
        objects must also be replaced to stay consistent with the new type.
        """
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
        )
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        fsdp_plugin = self.accelerator.state.fsdp_plugin
        sharded_exists = os.path.isdir(os.path.join(checkpoint, "pytorch_model_fsdp_0"))
        full_exists = os.path.isfile(os.path.join(checkpoint, "pytorch_model_fsdp.bin"))

        if (
            sharded_exists
            and fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT
        ):
            LOG.warning(
                f"Checkpoint at {checkpoint} was saved with SHARDED_STATE_DICT but current "
                f"state_dict_type is {fsdp_plugin.state_dict_type}. Overriding to SHARDED_STATE_DICT."
            )
            fsdp_plugin.state_dict_type = StateDictType.SHARDED_STATE_DICT
            fsdp_plugin.state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
            fsdp_plugin.optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=True
            )
        elif (
            full_exists and fsdp_plugin.state_dict_type != StateDictType.FULL_STATE_DICT
        ):
            LOG.warning(
                f"Checkpoint at {checkpoint} was saved with FULL_STATE_DICT but current "
                f"state_dict_type is {fsdp_plugin.state_dict_type}. Overriding to FULL_STATE_DICT."
            )
            fsdp_plugin.state_dict_type = StateDictType.FULL_STATE_DICT
            fsdp_plugin.state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            fsdp_plugin.optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )

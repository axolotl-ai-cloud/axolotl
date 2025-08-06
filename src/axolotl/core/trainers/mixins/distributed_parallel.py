"""
Mixin for correctly saving fsdp
"""

from transformers import Trainer

from axolotl.utils.distributed import barrier


class DistributedParallelMixin(Trainer):
    """
    Mixin for correctly saving fsdp
    """

    def _save(self, output_dir: str | None = None, state_dict=None):
        if (
            state_dict is None
            and self.accelerator.parallelism_config
            and self.accelerator.parallelism_config.dp_shard_enabled
        ):
            state_dict = self.accelerator.get_state_dict(self.model)
        super()._save(output_dir, state_dict=state_dict)
        barrier()  # sometimes non-rank0 processes will continue training while rank0 is saving optimizer

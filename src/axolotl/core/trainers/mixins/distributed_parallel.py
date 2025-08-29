"""
Mixin for correctly saving fsdp
"""

from accelerate import PartialState
from transformers import Trainer


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

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if (
            self.accelerator.distributed_type == "FSDP"
            and self.accelerator.state.fsdp_plugin is None
        ):
            # handle Context Parallelism without FSDP
            self.accelerator.state.distributed_type = "MULTI_GPU"
            self.accelerator.state._shared_state["distributed_type"] = "MULTI_GPU"
            PartialState().distributed_type = "MULTI_GPU"

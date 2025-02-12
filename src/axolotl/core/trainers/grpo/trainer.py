"""
Axolotl GRPO trainer
"""
from accelerate.utils import is_peft_model
from accelerate.utils.other import is_compiled_module
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from axolotl.core.trainers.base import SchedulerMixin


# mypy: ignore-errors
class AxolotlGRPOTrainer(SchedulerMixin, GRPOTrainer):
    """
    Extend the base GRPOTrainer for axolotl helpers
    """

    _tag_names = ["trl", "grpo", "axolotl"]

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = (
                    unwrapped_model._orig_mod  # pylint: disable=protected-access
                )
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                state_dict = {
                    k.removeprefix("base_model.model.")
                    .removeprefix("base_model.model.")
                    .replace(".default", "")
                    .replace(".base_layer", "")
                    .replace(".modules_to_save", ""): v
                    for k, v in state_dict.items()
                    if unwrapped_model.prefix not in k and "original_module" not in k
                }
                unwrapped_model.unmerge_adapter()
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = (
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            llm_model.load_weights(state_dict.items())

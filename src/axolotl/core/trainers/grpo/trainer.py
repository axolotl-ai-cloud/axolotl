"""
Axolotl GRPO trainer
"""
from accelerate.utils import is_peft_model
from accelerate.utils.other import is_compiled_module
from transformers import PreTrainedModel
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation

from axolotl.core.trainers.base import SchedulerMixin


# mypy: ignore-errors
class AxolotlGRPOTrainer(SchedulerMixin, GRPOTrainer):
    """
    Extend the base GRPOTrainer for axolotl helpers
    """

    _tag_names = ["trl", "grpo", "axolotl"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # pylint: disable=access-member-before-definition
        # Enable gradient checkpointing if requested
        if kwargs["args"].gradient_checkpointing:
            # Ensure use_cache is disabled
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # Enable gradient checkpointing on the base model for PEFT
            if is_peft_model(self.model) and hasattr(
                self.model.base_model, "gradient_checkpointing_enable"
            ):
                self.model.base_model.gradient_checkpointing_enable()
            # Enable gradient checkpointing for non-PEFT models
            elif hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            self.model = self._enable_gradient_checkpointing(self.model, kwargs["args"])
        # pylint: enable=access-member-before-definition

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # pylint: disable=unused-argument,redefined-builtin
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        return model
        # pylint: enable=unused-argument,redefined-builtin

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
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.")
                    .removeprefix("base_model.model.")
                    .replace(".base_layer", ""): v
                    for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if unwrapped_model.prefix not in k
                }
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights(state_dict.items())
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

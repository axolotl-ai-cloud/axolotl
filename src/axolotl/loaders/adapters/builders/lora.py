from transformers import PreTrainedModel
from peft import LoraConfig, PeftModel

from axolotl.utils.logging import get_logger
from .base import BaseAdapterBuilder

LOG = get_logger(__name__)


class LoraAdapterBuilder(BaseAdapterBuilder):
    """Builder for LoRA adapters."""

    def build_config(self, model: PreTrainedModel, **kwargs) -> LoraConfig:
        """
        Build LoRA configuration.

        Args:
            model: The base model
            **kwargs: Additional configuration options

        Returns:
            LoraConfig: Configured LoRA adapter
        """
        target_modules = self.prepare_target_modules(model)
        target_parameters = self.prepare_target_parameters()

        config_kwargs = self.build_common_config_kwargs()

        config_kwargs.update(kwargs)

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=target_modules,
            target_parameters=target_parameters,
            layers_to_transform=self.cfg.peft_layers_to_transform,
            layers_pattern=self.cfg.peft_layers_pattern,
            lora_dropout=self.cfg.lora_dropout,
            fan_in_fan_out=self.cfg.lora_fan_in_fan_out,
            modules_to_save=self.cfg.lora_modules_to_save
            if self.cfg.lora_modules_to_save
            else None,
            bias="none",
            task_type="CAUSAL_LM",
            **config_kwargs,
        )
        return lora_config

    def build_model(
        self, model: PreTrainedModel, config: LoraConfig, *, inference: bool = False
    ) -> PeftModel:
        """
        Build LoRA model.

        Args:
            model: Base model
            config: LoRA configuration

        Returns:
            PeftModel: Model with LoRA adapter applied
        """
        self.setup_quantization_for_training(model)

        if self.cfg.lora_model_dir:
            LOG.debug("Loading pretrained PEFT - LoRA")
            model = self.load_pretrained_adapter(model, inference)
        else:
            model = self.create_peft_model(model, config)

        self.print_trainable_parameters(model)
        self.setup_quantization_for_training_post_build(model)

        return model

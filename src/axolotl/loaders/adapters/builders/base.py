import os
from abc import abstractmethod, ABC
from typing import Optional, Union, List, Any, Dict
from transformers import PreTrainedModel
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class BaseAdapterBuilder(ABC):
    """Base class for adapter builders"""

    def __init__(self, cfg: DictDefault):
        self.cfg = cfg
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    @abstractmethod
    def build_config(self, model: PreTrainedModel, **kwargs) -> PeftConfig:
        """Build the PEFT configuration"""
        target_modules = self.prepare_target_modules(model)
        target_parameters = self.prepare_target_parameters()

        config_kwargs = self.build_common_config_kwargs()
        config_kwargs.update(kwargs)

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=target_modules,
            target_parameters=target_parameters,
            **config_kwargs,
        )
        return lora_config

    @abstractmethod
    def build_model(self, model: PreTrainedModel, config: PeftConfig) -> PeftModel:
        """Build the PEFT model"""
        self.setup_quantization_for_training(model)

        if self.cfg.lora_model_dir:
            model = self.load_pretrained_adapter(model)
        else:
            model = self.create_peft_model(model, config)

        self.print_trainable_parameters(model)
        self.setup_quantization_for_training_post_build(model)

        return model

    def prepare_target_modules(
        self,
        model: PreTrainedModel,
        target_modules: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Prepare and validate target modules for the adapter.

        Args:
            model: The base model
            target_modules: User-specified target modules

        Returns:
            List[str]: Processed list of target modules
        """

        lora_target_modules: Union[str, List[str]] = (
            target_modules or self.cfg.lora_target_modules or []
        )

        if self.cfg.lora_target_linear:
            from axolotl.loaders.adapter import find_all_linear_names

            linear_names = find_all_linear_names(model)
            LOG.info(f"found linear modules: {repr(sorted(linear_names))}")
            lora_target_modules_as_list = (
                lora_target_modules
                if isinstance(lora_target_modules, list)
                else [lora_target_modules]
                if lora_target_modules
                else []
            )
            lora_target_modules = list(set(lora_target_modules_as_list + linear_names))
        elif isinstance(lora_target_modules, str):
            lora_target_modules = [lora_target_modules]
        elif lora_target_modules is None:
            lora_target_modules = []

        return lora_target_modules

    def prepare_target_parameters(
        self, target_parameters: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Prepare target parameters for the adapter.

        Args:
            target_parameters: User-specified target parameters

        Returns:
            List[str]: Processed list of target parameters
        """
        result = target_parameters or self.cfg.lora_target_parameters or []
        if isinstance(result, str):
            return [result]
        elif isinstance(result, list):
            return result
        else:
            return []

    def build_common_config_kwargs(self) -> Dict[str, Any]:
        """
        Build common configuration kwargs shared across adapter types.

        Returns:
            Dict[str, Any]: Common configuration parameters
        """
        config_kwargs = {}

        # LoftQ configuration
        loftq_bits = (
            self.cfg.peft
            and self.cfg.peft.loftq_config
            and self.cfg.peft.loftq_config.loftq_bits
        )
        if loftq_bits:
            from peft import LoftQConfig

            config_kwargs["loftq_config"] = LoftQConfig(loftq_bits=loftq_bits)
            config_kwargs["init_lora_weights"] = "loftq"

        # LoRA weight initialization
        if self.cfg.peft_init_lora_weights:
            config_kwargs["init_lora_weights"] = self.cfg.peft_init_lora_weights

        # DoRA configuration
        if self.cfg.peft_use_dora:
            config_kwargs["use_dora"] = self.cfg.peft_use_dora
            LOG.info("Initializing LoRA weights using DoRA. This might take longer.")

        # RSLoRA configuration
        if self.cfg.peft_use_rslora:
            config_kwargs["use_rslora"] = self.cfg.peft_use_rslora

        # Layer replication
        if self.cfg.peft_layer_replication:
            config_kwargs["layer_replication"] = self.cfg.peft_layer_replication

        return config_kwargs

    def setup_quantization_for_training(self, model: Union[PreTrainedModel, PeftModel]):
        """
        Setup quantization metadata for training.

        Args:
            model: The model to setup quantization for
        """
        from axolotl.loaders.adapter import setup_quantized_meta_for_peft

        if (
            self.cfg.fsdp_config
            and self.cfg.adapter
            and self.cfg.fsdp_config.cpu_ram_efficient_loading
            and self.rank != 0
        ):
            setup_quantized_meta_for_peft(model)

    def setup_quantization_for_training_post_build(
        self, model: Union[PreTrainedModel, PeftModel]
    ):
        """
        Setup quantization metadata after model building for training.

        Args:
            model: The model to setup quantization for
        """
        from axolotl.loaders.adapter import setup_quantized_peft_meta_for_training

        if (
            self.cfg.fsdp_config
            and self.cfg.adapter
            and self.cfg.fsdp_config.cpu_ram_efficient_loading
            and self.rank != 0
        ):
            setup_quantized_peft_meta_for_training(model)

    def load_pretrained_adapter(
        self, model: PreTrainedModel, inference: bool = False
    ) -> Union[PreTrainedModel, PeftModel]:
        """
        Load a pretrained adapter from a directory.

        Args:
            model: Base model to load adapter onto
            inference: Whether this is for inference mode

        Returns:
            PeftModel: Model with loaded adapter
        """

        if not self.cfg.lora_model_dir:
            return model

        LOG.debug(f"Loading pretrained PEFT - {self.__class__.__name__}")
        model_kwargs: Dict[str, Any] = {}

        if self.cfg.lora_on_cpu:
            model_kwargs["max_memory"] = {"cpu": "256GiB"}
            model_kwargs["device_map"] = {"": "cpu"}

        return PeftModel.from_pretrained(
            model,
            self.cfg.lora_model_dir,
            is_trainable=(not inference),
            **model_kwargs,
        )

    def create_peft_model(
        self, model: PreTrainedModel, config: PeftConfig
    ) -> PeftModel:
        """
        Create a PEFT model from base model and config.

        Args:
            model: Base model
            config: PEFT configuration

        Returns:
            PeftModel: Created PEFT model
        """
        return get_peft_model(model, config)

    def print_trainable_parameters(self, model: Union[PreTrainedModel, PeftModel]):
        """
        Print the number of trainable parameters in the model.

        Args:
            model: The model to analyze
        """
        if self.rank == 0:
            try:
                model.print_trainable_parameters()
            except AttributeError as exc:
                LOG.warning(
                    "Exception caught during model.print_trainable_parameters(): %s",
                    exc,
                )

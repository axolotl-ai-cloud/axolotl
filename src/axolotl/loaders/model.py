"""Model loader class implementation for loading, configuring, and patching various
models.
"""

import gc
import math
import os
from functools import cached_property
from importlib.util import find_spec
from typing import Any

import peft
import torch
import transformers
import transformers.modeling_utils
from accelerate import init_empty_weights
from peft import (
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AwqConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.integrations.deepspeed import (
    HfTrainerDeepSpeedConfig,
    is_deepspeed_zero3_enabled,
)

from axolotl.common.architectures import MOE_ARCH_BLOCK
from axolotl.integrations.base import PluginManager
from axolotl.loaders.adapter import load_adapter, load_lora
from axolotl.loaders.constants import MULTIMODAL_AUTO_MODEL_MAPPING
from axolotl.loaders.patch_manager import PatchManager
from axolotl.loaders.utils import (
    get_linear_embedding_layers,
    get_module_class_from_name,
    load_model_config,
)
from axolotl.models.mamba import fix_mamba_attn_for_loss
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import (
    get_device_count,
    get_device_type,
)
from axolotl.utils.logging import get_logger
from axolotl.utils.model_shard_quant import load_sharded_model_quant
from axolotl.utils.schemas.enums import RLType

LOG = get_logger(__name__)
PLUGIN_MANAGER = PluginManager.get_instance()


class ModelLoader:
    """Manages model configuration, initialization and application of patches during
    model loading.

    This class orchestrates the entire process of loading a model from configuration to
    final preparation. It handles device mapping, quantization, attention mechanisms,
    adapter integration, and various optimizations.

    The loading process includes:
        - Loading and validating model configuration
        - Applying monkey patches for optimizations / fixes
        - Setting up device mapping (including multi-GPU configurations)
        - Configuring quantization
        - Setting attention mechanisms (Flash Attention, SDPA, etc.)
        - Loading and initializing the model
        - Applying adapters (LoRA, QLoRA, etc.)

    Attributes:
        model: The loaded model instance (available after load() is called).
        model_kwargs: Dictionary of keyword arguments passed to model initialization.
        base_model: Name or path of the base model to load.
        model_type: Type of model to load (e.g., `AutoModelForCausalLM`).
        model_config: Configuration object for the model.
        auto_model_loader: class used for loading the model (default:
            `AutoModelForCausalLM`).
    """

    def __init__(
        self,
        cfg: DictDefault,
        tokenizer: PreTrainedTokenizerBase,
        *,
        inference: bool = False,
        reference_model: bool = False,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Initializes the ModelLoader.

        Args:
            cfg: Configuration dictionary with model and training settings.
            tokenizer: Tokenizer instance associated with the model.
            processor: Optional processor for multimodal models. Defaults to None.
            inference: Whether the model is being loaded for inference mode. Defaults
                to False.
            reference_model: Whether this is a reference model (used in setups like DPO
                training). Defaults to False.
            **kwargs: Additional keyword arguments (ignored).
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inference: bool = inference
        self.reference_model: bool = reference_model

        # Init model kwargs
        self.model_kwargs: dict[str, Any] = {}
        if cfg.overrides_of_model_kwargs:
            for key, val in cfg.overrides_of_model_kwargs.items():
                self.model_kwargs[key] = val

        # Init model
        self.model: PreTrainedModel | PeftModel | PeftMixedModel
        self.base_model = cfg.base_model
        self.model_type = cfg.type_of_model

        # Init model config
        self.model_config = load_model_config(cfg)
        self.auto_model_loader = AutoModelForCausalLM  # pylint: disable=invalid-name

        # Initialize the patch manager
        self.patch_manager = PatchManager(
            cfg=cfg,
            model_config=self.model_config,
            inference=inference,
        )

    @cached_property
    def has_flash_attn(self) -> bool:
        """Check if flash attention is installed."""
        return find_spec("flash_attn") is not None

    @cached_property
    def qlora_fsdp(self):
        """Property that determines if FSDP with QLoRA is enabled."""
        return self.cfg.fsdp and self.cfg.adapter == "qlora"

    def load(self) -> tuple[PreTrainedModel | PeftModelForCausalLM, PeftConfig | None]:
        """Load and prepare the model with all configurations and patches.

        Returns:
            A tuple with the loaded model and its LoRA configuration (if applicable).
        """
        # Initial setup and patches
        self.patch_manager.apply_pre_model_load_patches()
        self._apply_pre_model_load_setup()

        # Build the model
        PLUGIN_MANAGER.pre_model_load(self.cfg)
        skip_move_to_device = self._build_model()
        PLUGIN_MANAGER.post_model_build(self.cfg, self.model)

        # Post-build model configuration
        self._apply_post_model_load_setup()

        # Load adapters (LoRA, etc.)
        PLUGIN_MANAGER.pre_lora_load(self.cfg, self.model)
        lora_config = self._load_adapters()
        PLUGIN_MANAGER.post_lora_load(self.cfg, self.model)

        # Apply remaining patches and finalize
        self._apply_post_lora_load_setup(skip_move_to_device)
        self.patch_manager.apply_post_model_load_patches(self.model)
        PLUGIN_MANAGER.post_model_load(self.cfg, self.model)

        return self.model, lora_config

    def _apply_pre_model_load_setup(self):
        """Apply patches and setup configurations before model loading."""
        self._set_auto_model_loader()
        self._set_device_map_config()
        if self.cfg.revision_of_model:
            self.model_kwargs["revision"] = self.cfg.revision_of_model
        self._set_quantization_config()
        self._set_attention_config()

    def _apply_post_model_load_setup(self):
        """Configure the model after it has been loaded."""
        # Handle PeftModel if needed
        if (
            isinstance(self.model, (peft.PeftModel, peft.PeftModelForCausalLM))
            and not self.qlora_fsdp
        ):
            self.model = self.model.merge_and_unload()

        self._resize_token_embeddings()
        self._adjust_model_config()
        self._log_memory_usage()
        self._configure_embedding_dtypes()
        self._configure_qat()

    def _resize_token_embeddings(self):
        """Resize token embeddings if needed."""
        embeddings_len = (
            math.ceil(len(self.tokenizer) / 32) * 32
            if self.cfg.resize_token_embeddings_to_32x
            else len(self.tokenizer)
        )
        if hasattr(self.model, "get_input_embeddings") and (
            self.model.get_input_embeddings().num_embeddings < embeddings_len
            or (
                self.model.get_input_embeddings().num_embeddings > embeddings_len
                and self.cfg.shrink_embeddings
            )
        ):
            resize_kwargs = {}
            if self.cfg.mean_resizing_embeddings is not None and (
                self.model_config.model_type != "llava"
            ):
                resize_kwargs["mean_resizing"] = self.cfg.mean_resizing_embeddings
            self.model.resize_token_embeddings(embeddings_len, **resize_kwargs)
        else:
            self.model.tie_weights()

    def _adjust_model_config(self):
        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "max_position_embeddings")
            and self.model.config.max_position_embeddings
            and self.cfg.sequence_len > self.model.config.max_position_embeddings
        ):
            LOG.warning(
                "increasing model.config.max_position_embeddings from "
                f"{self.model.config.max_position_embeddings} to {self.cfg.sequence_len}"
            )
            self.model.config.max_position_embeddings = self.cfg.sequence_len

        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "bos_token_id")
            and self.model.config.bos_token_id
            and self.model.config.bos_token_id != self.tokenizer.bos_token_id
        ):
            self.model.config.bos_token_id = self.tokenizer.bos_token_id

        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "eos_token_id")
            and self.model.config.eos_token_id
            and self.model.config.eos_token_id != self.tokenizer.eos_token_id
        ):
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def _log_memory_usage(self):
        """Log device memory usage after model load."""
        if hasattr(self.model, "device") and self.model.device.type in (
            "cuda",
            "mps",
            "npu",
        ):
            log_gpu_memory_usage(LOG, "after model load", self.model.device)

    def _configure_embedding_dtypes(self):
        """Configure embedding module dtypes."""
        # Get embedding modules
        embedding_modules = get_linear_embedding_layers(self.cfg.model_config_type)

        # Initial dtype conversion
        if not self.cfg.fsdp:
            # We don't run this during FSDP because this will leave mixed and bfloat16
            # dtypes in the model which FSDP doesn't like
            if self.cfg.load_in_4bit and self.cfg.embeddings_skip_upcast:
                embedding_modules = []
            self._convert_embedding_modules_dtype(
                embedding_modules,
                dist_dtype=torch.float32,
                before_kbit_train_or_finetune=True,
            )

        # Handle DeepSpeed Zero3
        if is_deepspeed_zero3_enabled():
            self._set_z3_leaf_modules()

        # Apply gradient checkpointing if needed
        needs_fa2_dtype = self.cfg.adapter or self.cfg.fsdp
        if self.cfg.adapter in ["lora", "qlora"]:
            needs_fa2_dtype = True
            if self.cfg.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self.cfg.gradient_checkpointing_kwargs
                )

        self._prepare_model_for_quantization()

        # Convert dtypes if needed
        should_convert = (
            # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so
            # we need to convert them back to fp16/bf16 for flash-attn compatibility.
            (
                (needs_fa2_dtype or self.cfg.flash_attention or self.cfg.flex_attention)
                and not self.qlora_fsdp
            )
            # CCE requires embedding layers to be in fp16/bf16 for backward pass
            or self.cfg.cut_cross_entropy
        )

        if should_convert:
            LOG.info("Converting modules to %s", self.cfg.torch_dtype)
            self._convert_embedding_modules_dtype(
                embedding_modules=embedding_modules,
                dist_dtype=self.cfg.torch_dtype,
                before_kbit_train_or_finetune=False,
            )

    def _configure_qat(self):
        """Configure QAT."""
        if self.cfg.qat:
            from axolotl.utils.quantization import prepare_model_for_qat

            prepare_model_for_qat(
                self.model,
                self.cfg.qat.weight_dtype,
                self.cfg.qat.group_size,
                self.cfg.qat.activation_dtype,
                self.cfg.qat.quantize_embedding,
            )

    def _load_adapters(self) -> PeftConfig | None:
        """Load LoRA or other adapters."""
        # Load LoRA or adapter
        lora_config = None
        if not self.reference_model or self.cfg.lora_model_dir:
            # If we're not loading the reference model, then we're loading the model
            # for training. Then, the DPO trainer doesn't want the PEFT model loaded
            # over it, it just wants the LoRA / PEFT config.
            if (
                self.cfg.adapter
                and self.cfg.rl in [RLType.DPO, RLType.IPO, RLType.KTO]
                and not self.cfg.merge_lora
            ):
                _, lora_config = load_lora(
                    self.model, self.cfg, inference=False, config_only=True
                )
            else:
                self.model, lora_config = load_adapter(
                    self.model, self.cfg, self.cfg.adapter
                )

        return lora_config

    def _apply_post_lora_load_setup(self, skip_move_to_device: bool):
        """Apply final optimizations and patches."""
        # Place model on accelerator
        if (
            self.cfg.ddp
            and not self.cfg.load_in_8bit
            and not (self.cfg.rl and self.cfg.load_in_4bit)
            and not skip_move_to_device
        ):
            # TODO: validate this conditional
            self.model.to(f"{str(get_device_type())}:{self.cfg.local_rank}")

        if get_device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        if not any(
            param.requires_grad
            for _, param in self.model.named_parameters(recurse=True)
        ):
            LOG.warning("There are no parameters that require gradient updates")

        if self.cfg.flash_optimum:
            from optimum.bettertransformer import BetterTransformer

            self.model = BetterTransformer.transform(self.model)

        if self.cfg.adapter is not None:
            log_gpu_memory_usage(LOG, "after adapters", self.model.device)

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    def _set_auto_model_loader(self):
        """Set `self.auto_model_loader`. Defaults to `transformers.AutoModelForCausalLM`
        (set at `__init__`). When using a multimodal model, `self.auto_model_loader`
        should be set according to the type of the model.
        """
        if self.cfg.is_multimodal:
            self.auto_model_loader = MULTIMODAL_AUTO_MODEL_MAPPING.get(
                self.model_config.model_type, AutoModelForVision2Seq
            )

    def _set_device_map_config(self):
        """Setup `device_map` according to config"""
        device_map = self.cfg.device_map
        max_memory = self.cfg.max_memory

        if self.cfg.gpu_memory_limit:
            gpu_memory_limit = (
                str(self.cfg.gpu_memory_limit) + "GiB"
                if isinstance(self.cfg.gpu_memory_limit, int)
                else self.cfg.gpu_memory_limit
            )

            max_memory = {}
            num_device = get_device_count()
            for i in range(num_device):
                max_memory[i] = gpu_memory_limit
            max_memory["cpu"] = "256GiB"  # something sufficiently large to fit anything

        if max_memory is not None:
            # Based on https://github.com/togethercomputer/OpenChatKit/blob/main/inference/bot.py
            from accelerate import infer_auto_device_map

            with init_empty_weights():
                model_canvas = self.auto_model_loader.from_config(
                    self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                )
            model_canvas.tie_weights()
            device_map = infer_auto_device_map(
                model_canvas,
                max_memory=max_memory,
                dtype=self.cfg.torch_dtype,
            )
            # We can discard max_memory now as we have a device map set up
            max_memory = None

        self.model_kwargs["torch_dtype"] = self.cfg.torch_dtype

        if not is_deepspeed_zero3_enabled():
            self.model_kwargs["device_map"] = device_map

            cur_device = get_device_type()
            if "mps" in str(cur_device):
                self.model_kwargs["device_map"] = "mps:0"
            elif "npu" in str(cur_device):
                self.model_kwargs["device_map"] = "npu:0"

        # TODO: can we put the reference model on it's own gpu? I think we have to move
        # logits around to calculate loss
        # if cfg.rl:
        #     if torch.cuda.device_count() > 1:
        #         if reference_model:
        #             model_kwargs["device_map"] = "cuda:" + str(
        #                 torch.cuda.current_device() + 1
        #             )
        #         else:
        #             model_kwargs["device_map"] = "cuda:" + str(torch.cuda.current_device())

    def _set_quantization_config(self):
        """Set up quantization config (bitsandbytes, awq, gptq, etc.)"""
        self.model_kwargs["load_in_8bit"] = self.cfg.load_in_8bit
        self.model_kwargs["load_in_4bit"] = self.cfg.load_in_4bit

        if self.cfg.gptq:
            if not hasattr(self.model_config, "quantization_config"):
                LOG.warning(
                    "model config does not contain quantization_config information"
                )
            else:
                if self.cfg.gptq_disable_exllama is not None:
                    self.model_config.quantization_config["disable_exllama"] = (
                        self.cfg.gptq_disable_exllama
                    )
                self.model_kwargs["quantization_config"] = GPTQConfig(
                    **self.model_config.quantization_config
                )
        if (
            self.cfg.adapter in ["qlora", "lora"]
            and hasattr(self.model_config, "quantization_config")
            and self.model_config.quantization_config["quant_method"]
            in ["gptq", "awq", "bitsandbytes"]
        ):
            if self.model_config.quantization_config["quant_method"] == "gptq":
                self.model_kwargs["quantization_config"] = GPTQConfig(
                    **self.model_config.quantization_config
                )
            elif self.model_config.quantization_config["quant_method"] == "awq":
                self.model_kwargs["quantization_config"] = AwqConfig(
                    **self.model_config.quantization_config
                )
            elif (
                self.model_config.quantization_config["quant_method"] == "bitsandbytes"
            ):
                self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    **self.model_config.quantization_config
                )
        elif self.cfg.adapter == "qlora" and self.model_kwargs["load_in_4bit"]:
            bnb_config = {
                "load_in_4bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False,
                "bnb_4bit_compute_dtype": self.cfg.torch_dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_quant_storage": torch.bfloat16,
            }
            if self.cfg.model_config_type in ["jamba", "qwen2_moe"] and not (
                self.cfg.deepspeed or self.cfg.fsdp
            ):
                # for some reason, this causes the loss to be off by an order of magnitude
                # but deepspeed needs this still in bfloat16
                bnb_config["bnb_4bit_quant_storage"] = torch.float32

            if self.cfg.bnb_config_kwargs:
                bnb_config.update(self.cfg.bnb_config_kwargs)

            self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )
        elif self.cfg.adapter == "lora" and self.model_kwargs["load_in_8bit"]:
            bnb_config = {
                "load_in_8bit": True,
            }
            # Exclude mamba blocks from int8 quantization for jamba
            if self.cfg.model_config_type == "jamba":
                bnb_config["llm_int8_skip_modules"] = ["mamba"]
            self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )

        # no longer needed per https://github.com/huggingface/transformers/pull/26610
        if "quantization_config" in self.model_kwargs or self.cfg.gptq:
            self.model_kwargs.pop("load_in_8bit", None)
            self.model_kwargs.pop("load_in_4bit", None)

    def _set_attention_config(self):
        """Sample packing uses custom FA2 patch"""
        if self.cfg.flex_attention:
            self.model_kwargs["attn_implementation"] = "flex_attention"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "flex_attention"
            )

        elif self.cfg.flash_attention:
            if not self.cfg.sample_packing and self.cfg.s2_attention:
                pass
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "flash_attention_2"
            )
        elif self.cfg.sdp_attention:
            self.model_kwargs["attn_implementation"] = "sdpa"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "sdpa"
            )
        elif self.cfg.eager_attention:
            self.model_kwargs["attn_implementation"] = "eager"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "eager"
            )

        if self.cfg.low_cpu_mem_usage:
            self.model_kwargs["low_cpu_mem_usage"] = True

    def _configure_zero3_memory_efficient_loading(
        self,
    ) -> HfTrainerDeepSpeedConfig | None:
        """
        Set the deepspeed config to load the model into RAM first before moving to VRAM.

        IMPORTANT
        ==========

        We need to return `hf_ds_cfg` as it needs to exist before model loading for zero3.
        HfTrainerDeepSpeedConfig is a class that is used to configure the DeepSpeed training.
        It is not passed anywhere in the model loading function, just need to exist.
        """
        hf_ds_cfg = None

        if os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3":
            hf_ds_cfg = HfTrainerDeepSpeedConfig(self.cfg.deepspeed)
            hf_ds_cfg.fill_match(
                "train_micro_batch_size_per_gpu", self.cfg.micro_batch_size
            )
            hf_ds_cfg.fill_match(
                "gradient_accumulation_steps", self.cfg.gradient_accumulation_steps
            )
            hf_ds_cfg.fill_match(
                "train_batch_size",
                int(os.getenv("WORLD_SIZE", "1"))
                * self.cfg.micro_batch_size
                * self.cfg.gradient_accumulation_steps,
            )
            if "device_map" in self.model_kwargs:
                del self.model_kwargs["device_map"]

            transformers.modeling_utils.is_deepspeed_zero3_enabled = lambda: True
            transformers.integrations.deepspeed.is_deepspeed_zero3_enabled = (
                lambda: True
            )

        return hf_ds_cfg

    def _build_model(self) -> bool:
        """Load model, with load strategy depending on config."""
        skip_move_to_device = False
        if (
            self.qlora_fsdp
            and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
            and (
                self.cfg.model_config_type == "dbrx"
                or self.cfg.qlora_sharded_model_loading
            )
        ):
            quant_storage = self.cfg.torch_dtype
            quantization_config = getattr(
                self.model_config, "quantization_config", None
            )
            quantization_config = (
                quantization_config or self.model_kwargs["quantization_config"]
            )
            self.model = load_sharded_model_quant(
                self.base_model,
                self.model_config,
                self.cfg,
                quant_storage=quant_storage,
                quantization_config=quantization_config,
            )
            skip_move_to_device = True
        elif (
            self.model_config.model_type in ["llama", "llama4"]
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
        ):
            # TODO: Do we need to open this up for all models?
            if self.cfg.fsdp and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
                skip_move_to_device = True
                if "device_map" in self.model_kwargs:
                    del self.model_kwargs["device_map"]

            # Please don't remove underscore binding without reading the fn docstring.
            _ = self._configure_zero3_memory_efficient_loading()

            # Load model with random initialization if specified
            if self.cfg.random_init_weights:
                # AutoModel classes support the from_config method
                if self.auto_model_loader in [
                    AutoModelForCausalLM,
                    AutoModelForVision2Seq,
                ]:
                    self.model = self.auto_model_loader.from_config(
                        config=self.model_config,
                    )
                else:
                    self.model = self.auto_model_loader(config=self.model_config)
            else:
                self.model = self.auto_model_loader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    **self.model_kwargs,
                )
        elif self.model_type == "MambaLMHeadModel":
            # FIXME this is janky at best and hacked together to make it work
            MambaLMHeadModel = fix_mamba_attn_for_loss()  # pylint: disable=invalid-name

            self.model_kwargs["dtype"] = self.model_kwargs["torch_dtype"]
            self.model_kwargs["device"] = torch.cuda.current_device()
            self.model_kwargs.pop("torch_dtype", None)
            self.model_kwargs.pop("device_map", None)

            self.model = MambaLMHeadModel.from_pretrained(
                self.base_model,
                **self.model_kwargs,
            )
        elif (
            self.model_type
            and self.model_type != "AutoModelForCausalLM"
            and not self.cfg.trust_remote_code
        ):
            if self.cfg.gptq:
                self.model = self.auto_model_loader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
            else:
                self.model = getattr(transformers, self.model_type).from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
        else:
            if self.cfg.gptq:
                self.model = self.auto_model_loader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
            else:
                if (
                    self.cfg.fsdp
                    and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
                ):
                    # disabling either of these two still leads to VRAM spike before setting back down
                    skip_move_to_device = True
                    if "device_map" in self.model_kwargs:
                        del self.model_kwargs["device_map"]

                # Please don't remove underscore binding without reading the fn docstring.
                _ = self._configure_zero3_memory_efficient_loading()

                self.model = self.auto_model_loader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
        if is_deepspeed_zero3_enabled():
            skip_move_to_device = True

        return skip_move_to_device

    def _set_z3_leaf_modules(self):
        from deepspeed.utils import set_z3_leaf_modules

        if self.cfg.model_config_type in MOE_ARCH_BLOCK:
            moe_blocks = MOE_ARCH_BLOCK[self.cfg.model_config_type]
            moe_blocks = [moe_blocks] if isinstance(moe_blocks, str) else moe_blocks
            set_z3_leaf_modules(
                self.model,
                [
                    get_module_class_from_name(self.model, module_name)
                    for module_name in moe_blocks
                ],
            )

    def _prepare_model_for_quantization(self):
        """Prepare loaded model for quantization."""
        skip_prepare_model_for_kbit_training = False
        if self.cfg.model_config_type == "qwen" and self.cfg.adapter == "lora":
            # Qwen doesn't play nicely with LoRA if this is enabled
            skip_prepare_model_for_kbit_training = True

        loftq_bits = (
            self.cfg.peft
            and self.cfg.peft.loftq_config
            and self.cfg.peft.loftq_config.loftq_bits
        )
        if self.cfg.adapter == "lora" and loftq_bits:
            skip_prepare_model_for_kbit_training = True

        if (
            self.qlora_fsdp
            or (self.cfg.fsdp and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading)
            or is_deepspeed_zero3_enabled()
        ):
            # Make sure everything is in the same dtype
            skip_prepare_model_for_kbit_training = True

        if (
            not skip_prepare_model_for_kbit_training
            and self.cfg.adapter in ["lora", "qlora"]
            and (self.cfg.load_in_8bit or self.cfg.load_in_4bit)
        ):
            LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.cfg.gradient_checkpointing
            )

    def _convert_embedding_modules_dtype(
        self,
        embedding_modules: list[str],
        dist_dtype: torch.dtype,
        before_kbit_train_or_finetune: bool,
    ):
        for name, module in self.model.named_modules():
            if "norm" in name:
                module.to(dist_dtype)
            if before_kbit_train_or_finetune:
                if name.endswith(".gate"):
                    module.to(dist_dtype)
                if self.model_config.model_type == "btlm":
                    # don't upcast lm_head for btlm
                    continue
            if any(m in name for m in embedding_modules) and hasattr(module, "weight"):
                module.to(dist_dtype)

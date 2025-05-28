"""Adapter loading functionality, including LoRA / QLoRA and associated utils"""

import os
import types
from typing import Any

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Params4bit
from peft import (
    AdaptionPromptConfig,
    LoftQConfig,
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    get_peft_model,
)
from transformers import PreTrainedModel

from axolotl.loaders.utils import get_linear_embedding_layers
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def setup_quantized_meta_for_peft(model: torch.nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = (  # pylint: disable=protected-access
                param.quant_state.to
            )
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: torch.nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = (
                param.quant_state._orig_to  # pylint: disable=protected-access
            )
            param.quant_state._orig_to = None  # pylint: disable=protected-access


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    embedding_modules = get_linear_embedding_layers(model.config.model_type)
    output_embedding = embedding_modules[1]
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)


def load_lora(
    model: PreTrainedModel,
    cfg: DictDefault,
    inference: bool = False,
    config_only: bool = False,
) -> tuple[PreTrainedModel | PeftModel | PeftMixedModel | None, PeftConfig | None]:
    lora_target_modules = cfg.lora_target_modules or []

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(sorted(linear_names))}")
        lora_target_modules_as_list = (
            lora_target_modules
            if isinstance(lora_target_modules, list)
            else [lora_target_modules]
        )
        lora_target_modules = list(set(lora_target_modules_as_list + linear_names))

    lora_config_kwargs = {}
    loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if loftq_bits:
        lora_config_kwargs["loftq_config"] = LoftQConfig(loftq_bits=loftq_bits)
        lora_config_kwargs["init_lora_weights"] = "loftq"
    if cfg.peft_init_lora_weights:
        lora_config_kwargs["init_lora_weights"] = cfg.peft_init_lora_weights
    if cfg.peft_use_dora:
        lora_config_kwargs["use_dora"] = cfg.peft_use_dora
        LOG.info("Initializing LoRA weights using dora. This might take longer.")
    if cfg.peft_use_rslora:
        lora_config_kwargs["use_rslora"] = cfg.peft_use_rslora
    if cfg.peft_layer_replication:
        lora_config_kwargs["layer_replication"] = cfg.peft_layer_replication

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        layers_to_transform=cfg.peft_layers_to_transform,
        layers_pattern=cfg.peft_layers_pattern,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_config_kwargs,
    )

    if config_only:
        return None, lora_config

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_meta_for_peft(model)

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - LoRA")
        model_kwargs: Any = {}
        if cfg.lora_on_cpu:
            model_kwargs["max_memory"] = {"cpu": "256GiB"}
            model_kwargs["device_map"] = {"": "cpu"}
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
            **model_kwargs,
        )
    else:
        model = get_peft_model(model, lora_config)

    if rank == 0:
        try:
            model.print_trainable_parameters()
        except AttributeError as exc:
            LOG.warning(
                "Exception caught during model.print_trainable_parameters(): %s", exc
            )
    elif (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_peft_meta_for_training(model)

    return model, lora_config


def load_adapter(
    model: PreTrainedModel,
    cfg: DictDefault,
    adapter: str | None,
    inference: bool = False,
) -> tuple[PreTrainedModel | PeftModel | PeftMixedModel, PeftConfig | None]:
    if adapter is None:
        return model, None
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if adapter in ["lora", "qlora"]:
        peft_model, lora_config = load_lora(model, cfg, inference=inference)
        return peft_model, lora_config
    if adapter == "llama-adapter":
        peft_model, lora_config = load_llama_adapter(model, cfg)
        return peft_model, lora_config

    raise NotImplementedError(f"{adapter} PEFT adapter not available")


def load_llama_adapter(
    model: PreTrainedModel, cfg: DictDefault
) -> tuple[PeftModel | PeftMixedModel, PeftConfig]:
    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - llama_adapter")
        peft_model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        peft_model = get_peft_model(model, peft_config)

    peft_model.print_trainable_parameters()

    return peft_model, peft_config

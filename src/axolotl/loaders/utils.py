"""Utilities for axolotl.loaders module"""

import contextlib
from typing import Type

import addict
import torch
import transformers
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def get_module_class_from_name(
    module: torch.nn.Module, name: str
) -> Type[torch.nn.Module] | None:
    """Gets a class from a module by its name. Copied from `accelerate.utils.dataclasses`
    (https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L2805).

    Args:
        module: The module to get the class from.
        name: The name of the class.

    Returns:
        The class type of the matching module, or `None` if no match is found.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__

    if len(modules_children) == 0:
        return None

    for child_module in modules_children:
        module_class = get_module_class_from_name(child_module, name)
        if module_class is not None:
            return module_class

    return None


def check_model_config(cfg: DictDefault, model_config: PretrainedConfig):
    """Validates and adjusts model config based on `axolotl` config.

    This function performs several important checks and adjustments:
        - Disables model caching for better memory efficiency
        - Handles multimodal model-specific configurations
        - Validates quantization settings
        - Ensures proper LoRA configuration when using adapters with new tokens

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        model_config: The model's configuration object from `transformers`.

    Raises:
        ValueError: If a multimodal model lacks text configuration, if GPTQ settings
            are inconsistent, or if LoRA `modules_to_save` is improperly configured
            with new tokens.
    """
    if hasattr(model_config, "use_cache"):
        model_config.use_cache = False

    if cfg.is_multimodal:
        # For multimodal configs, use_cache is set in the text_config
        if hasattr(model_config, "get_text_config"):
            text_config = model_config.get_text_config()
            if hasattr(text_config, "use_cache"):
                text_config.use_cache = False
        else:
            raise ValueError(
                "No text config found for multimodal model. Please raise an Issue with model details."
            )

        # Check if image_size is not set and load image size from model config if available
        if (
            cfg.image_size is None
            and hasattr(model_config, "vision_config")
            and hasattr(model_config.vision_config, "image_size")
        ):
            image_size = model_config.vision_config.image_size
            if isinstance(image_size, list):
                cfg.image_size = tuple(image_size)
            else:
                cfg.image_size = image_size
            LOG.debug(f"Loaded image size: {cfg.image_size} from model config")

    quant_config_exists = (
        hasattr(model_config, "quantization_config")
        and model_config.quantization_config
    )

    # Detect compressed-tensors config
    is_compressed_tensors_config = (
        quant_config_exists
        and model_config.quantization_config.get("quant_method") == "compressed-tensors"
    )

    if is_compressed_tensors_config:
        if model_config.quantization_config.get("config_groups"):
            LOG.warning(
                "Found `config_groups` in a compressed-tensors config. "
                "QAT integration with llmcompressor is not tested."
            )
        # Skip further quant checks for compressed-tensors
        return

    quant_config_method_is_gptq = (
        quant_config_exists
        and "quant_method" in model_config.quantization_config
        and model_config.quantization_config["quant_method"] == "gptq"
    )

    if cfg.gptq and not quant_config_method_is_gptq:
        raise ValueError(
            "model_config.quantization_config is not set or quant_method is not set to gptq. "
            "Please make sure to point to a GPTQ model."
        )

    lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
    if (
        cfg.adapter
        and cfg.tokens
        and (
            not cfg.lora_modules_to_save
            or not all(x in cfg.lora_modules_to_save for x in lora_modules_to_save)
        )
    ):
        lora_modules_to_save_joined = ", ".join(
            map(lambda x: f"`{x}`", lora_modules_to_save)
        )
        raise ValueError(
            "`lora_modules_to_save` not properly set when adding new tokens. "
            f"Please include [{lora_modules_to_save_joined}] in `lora_modules_to_save`."
        )

    if (
        cfg.tensor_parallel_size
        and cfg.tensor_parallel_size > 1
        and hasattr(model_config, "tie_word_embeddings")
        and model_config.tie_word_embeddings
    ):
        raise ValueError(
            "Tensor parallelism is incompatible with models configured with `tie_word_embeddings` enabled. "
            "Please use a model without `tie_word_embeddings`, or disable tensor parallelism."
        )


def load_model_config(cfg: DictDefault) -> PretrainedConfig | addict.Dict:
    """Loads and configures a model configuration from HuggingFace or local sources.

    This function determines the appropriate model config source, loads it, applies any
    necessary overrides, and validates it for compatibility with the `axolotl` config.

    If `cfg.cls_model_config` is set, a custom config class from transformers will be
    used instead of `AutoConfig` (e.g., 'LlamaConfig', 'MistralConfig').

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        A configured model configuration object (`AutoConfig` instance), or a simple
            dictionary configuration for special cases like Mamba models.

    Raises:
        ValueError: If configuration loading fails for reasons other than special cases
            that are handled (e.g., Mamba models).
    """
    model_config_name = cfg.base_model_config or cfg.base_model
    if not model_config_name and cfg.tokenizer_config:
        model_config_name = cfg.tokenizer_config
    trust_remote_code = cfg.trust_remote_code is True
    config_kwargs = {}
    if cfg.revision_of_model:
        config_kwargs["revision"] = cfg.revision_of_model
    if cfg.num_labels:
        # num_labels is used to initialize classifier models
        config_kwargs["num_labels"] = cfg.num_labels

    config_cls = AutoConfig
    if cfg.cls_model_config:
        config_cls = getattr(transformers, cfg.cls_model_config)

    try:
        model_config = config_cls.from_pretrained(
            model_config_name,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )
    except ValueError as error:
        if "mamba" in model_config_name:
            return addict.Dict(
                {
                    "model_type": "mamba",
                }
            )
        raise error

    if cfg.overrides_of_model_config:
        for key, val in cfg.overrides_of_model_config.items():
            setattr(model_config, key, val)

    check_model_config(cfg, model_config)

    return model_config


def ensure_dtype(model: PreTrainedModel, dtype: torch.dtype = torch.bfloat16):
    """Ensures all modules in the model are converted to the specified data type."""
    for name, module in model.named_modules():
        weight_mismatch = False
        with contextlib.suppress(AttributeError):
            weight_mismatch = module.weight.dtype != dtype

        bias_mismatch = False
        with contextlib.suppress(AttributeError):
            bias_mismatch = module.bias.dtype != dtype

        if weight_mismatch:
            LOG.debug(
                f"Converting module {name}.weight: {module.weight.dtype} -> {dtype}"
            )
        if bias_mismatch:
            LOG.debug(f"Converting module {name}.bias: {module.bias.dtype} -> {dtype}")
        if weight_mismatch or bias_mismatch:
            module.to(dtype)


def _cpu_storage_like(tensor: torch.Tensor) -> torch.Tensor | None:
    """Real CPU storage matching a plain tensor's shape/dtype, or ``None`` for an opaque subclass."""
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        return None
    try:
        return torch.zeros_like(tensor, device="cpu")
    except (NotImplementedError, RuntimeError):
        # some dtypes (e.g. float4_e2m1fn_x2) have no CPU fill kernel
        return torch.empty_like(tensor, device="cpu")


def materialize_trainable_meta_params(model: torch.nn.Module) -> list[str]:
    """Gives meta-device trainable params real CPU storage, because accelerate's FSDP2 prepare keys
    its optimizer parameter remap on `data_ptr()` and every meta tensor reports 0."""
    from torch.distributed.tensor import DTensor

    replacements: dict[int, torch.nn.Parameter] = {}
    materialized: list[str] = []
    skipped: list[str] = []

    for name, param in model.named_parameters():
        if not (param.requires_grad and param.is_meta):
            continue

        if isinstance(param, DTensor):
            cpu_local = _cpu_storage_like(param._local_tensor)
            storage = (
                None
                if cpu_local is None
                else DTensor.from_local(
                    cpu_local,
                    param.device_mesh,
                    param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
            )
            kind = f"DTensor over {type(param._local_tensor).__name__}"
        else:
            storage = _cpu_storage_like(param)
            kind = type(param).__name__

        if storage is None:
            skipped.append(f"{name} ({kind})")
            continue
        replacements[id(param)] = torch.nn.Parameter(storage)
        materialized.append(name)

    # only non-rank-0 has anything to do here, where LOG's main-process default drops every line
    if skipped:
        LOG.warning(
            f"{len(skipped)} trainable params are on meta and cannot be materialized on "
            f"CPU; accelerate's FSDP2 optimizer remap may not reach them: {skipped[:5]}",
            main_process_only=False,
        )

    if not materialized:
        return materialized

    # meta -> cpu is not a compatible shallow copy, so `.data =` raises and the object is swapped
    for module in model.modules():
        for attr, param in module._parameters.items():
            if param is not None and id(param) in replacements:
                module._parameters[attr] = replacements[id(param)]

    LOG.info(
        f"materialized {len(materialized)} trainable meta params on CPU "
        "for FSDP2 optimizer mapping",
        main_process_only=False,
    )

    return materialized


def get_linear_embedding_layers(model_type: str) -> list[str]:
    """Returns layer names of linear embeddings needed for LoRA based on model type."""
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    if model_type == "nemotron_h":
        return ["embeddings", "lm_head"]
    if model_type == "bailing_hybrid":
        return ["word_embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]

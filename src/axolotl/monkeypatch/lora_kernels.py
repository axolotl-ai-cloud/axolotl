"""Module for patching custom LoRA Triton kernels and `torch.autograd` functions."""

import importlib
import inspect
import logging
import types
from typing import Generator, Tuple, Type

import torch
from accelerate.logging import get_logger
from peft import PeftModelForCausalLM
from torch import nn
from transformers import AutoConfig

from axolotl.kernels.lora import (
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)
from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.dict import DictDefault

LOG = get_logger(__name__)

ORIGINAL_QKV_CODE = """
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip(
    "\n"
)

PATCHED_QKV_CODE = """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip(
    "\n"
)

ORIGINAL_O_CODE = """
    attn_output = self.o_proj(attn_output)
""".lstrip(
    "\n"
)

PATCHED_O_CODE = """
    attn_output = self.apply_o(attn_output)
""".lstrip(
    "\n"
)

SUPPORTED_ACTIVATIONS = ["silu", "gelu"]
APPLY_FN_MAPPING = {
    "silu": apply_lora_mlp_swiglu,
    "gelu": apply_lora_mlp_geglu,
}


def original_apply_qkv(
    self: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Original implementation of QKV projection without optimizations.

    Args:
        self: The attention module instance.
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim].

    Returns:
        A tuple `(query_states, key_states, value_states)` containing the projected
            states for query, key, and value.
    """
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    return query_states, key_states, value_states


def original_apply_o(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Original implementation of output projection without optimizations.

    Args:
        self: The attention module instance.
        hidden_states: Input tensor of shape `[`batch_size, seq_len, hidden_dim]`.

    Returns:
        The output projection result.
    """
    attn_output = self.o_proj(hidden_states)

    return attn_output


def get_attention_cls_from_config(cfg: DictDefault) -> Type[nn.Module]:
    """
    Get the appropriate attention class by inspecting the model config.
    Uses dynamic import to support any model architecture that follows
    the standard transformers naming convention.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        The appropriate attention class for the model.

    Raises:
        ValueError: If `base_model` not specified or attention class cannot be imported
        ImportError: If the model module or attention class doesn't exist
    """
    if "base_model" not in cfg:
        raise ValueError("base_model must be specified in config")

    # Get model config without loading the model
    model_config = AutoConfig.from_pretrained(cfg["base_model"])
    model_type = model_config.model_type

    # Special case for model_type = "qwen2"
    if model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        return Qwen2Attention

    try:
        # Dynamically import the module and attention class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        module = __import__(
            module_path, fromlist=[f"{model_type.capitalize()}Attention"]
        )
        attention_cls = getattr(module, f"{model_type.capitalize()}Attention")

        return attention_cls
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Could not import attention class for model_type: {model_type}. "
            f"Error: {str(e)}"
        ) from e


# pylint: disable=protected-access
def patch_self_attn_lora(cfg: DictDefault):
    """
    Given an `axolotl` config, this method patches the inferred attention class forward
    pass with optimized LoRA implementations.

    It modifies the attention class to use optimized QKV and output projections. The
    original implementation is preserved and can be restored if needed.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.

    Raises:
        AssertionError: If the required code blocks are not found in the attention
            implementation.
    """
    attention_cls = get_attention_cls_from_config(cfg)

    # Check if already patched
    if hasattr(attention_cls, "_original_forward"):
        LOG.info(f"{attention_cls.__name__} already patched")
        return

    self_attn_forward = inspect.getsource(attention_cls.forward)
    attention_cls._original_forward = self_attn_forward
    self_attn_forward, _ = detab_code(self_attn_forward)

    assert ORIGINAL_QKV_CODE in self_attn_forward, "Original QKV code not found"
    assert ORIGINAL_O_CODE in self_attn_forward, "Original O code not found"

    self_attn_forward = self_attn_forward.replace(ORIGINAL_QKV_CODE, PATCHED_QKV_CODE)
    self_attn_forward = self_attn_forward.replace(ORIGINAL_O_CODE, PATCHED_O_CODE)
    self_attn_forward = self_attn_forward.replace(
        "def forward(",
        "def axolotl_attn_forward(",
        1,
    )

    # Load necessary imports
    module_name = attention_cls.__module__
    module = importlib.import_module(module_name)

    items_to_import = []
    for item in dir(module):
        if item in self_attn_forward:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        f"from {module_name} import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(self_attn_forward, globals())  # pylint: disable=exec-used  # nosec B102

    LOG.info(f"Patched attention class with LoRA optims: {attention_cls.__name__}")
    attention_cls.forward = (
        axolotl_attn_forward  # pylint: disable=undefined-variable  # noqa: F821
    )


def find_self_attn_in_layer(
    layer: nn.Module,
) -> Generator[Tuple[nn.Module], None, None]:
    # general case of most models
    if hasattr(layer, "self_attn"):
        if all(
            hasattr(layer.self_attn, proj)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            yield layer.self_attn


def find_mlp_in_layer(
    layer: nn.Module,
) -> Generator[Tuple[nn.Module, nn.Module, nn.Module, nn.Module], None, None]:
    # general case of most models
    if hasattr(layer, "mlp"):
        if all(
            hasattr(layer.mlp, proj) for proj in ["gate_proj", "up_proj", "down_proj"]
        ):
            yield layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj, layer.mlp
    # llama4 linearized experts
    if hasattr(layer, "feedforward") and hasattr(layer.feedforward, "shared_expert"):
        mlp = layer.feedforward.shared_expert
        yield mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp
    if hasattr(layer, "feedforward") and hasattr(layer.feedforward, "experts"):
        if all(
            hasattr(layer.feedforward.experts, proj)
            for proj in ["gate_projs", "up_projs", "down_projs"]
        ):
            for gate_proj, up_proj, down_proj in zip(
                layer.feedforward.experts.gate_projs,
                layer.feedforward.experts.up_projs,
                layer.feedforward.experts.down_projs,
            ):
                yield gate_proj, up_proj, down_proj, FakeMLP(
                    gate_proj, up_proj, down_proj
                )


def apply_lora_kernel_patches(
    model: PeftModelForCausalLM, cfg: DictDefault
) -> PeftModelForCausalLM:
    """
    Applies optimized Triton kernel patches to a PEFT model.

    Patches a PEFT model with optimized implementations for MLP and attention
    computations. The optimizations include custom Triton kernels for activation
    functions and specialized autograd functions for LoRA computations.

    Args:
        model: A PEFT model to be patched with optimized kernels.
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        PeftModelForCausalLM: The patched model with optimized kernels.

    Raises:
        TypeError: If the provided model is not a `PeftModelForCausalLM`.
        NotImplementedError: If the model type is not supported.
        AssertionError: If multiple adapters are active (currently unsupported).

    Note:
        The optimizations require LoRA adapters with no dropout and no bias terms. The
            function will skip patching if these conditions aren't met.
    """
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError("Model must be a PeftModelForCausalLM")

    # Get active LoRA adapter config
    if hasattr(model, "active_adapters"):
        assert (
            len(model.active_adapters) == 1
        ), "Axolotl currently does not support LoRA Triton kernels for multiple adapters"
        active_adapter = model.active_adapters[0]
    else:
        active_adapter = model.active_adapter
    lora_config = model.model.peft_config[active_adapter]

    # Only patch if conditions are met
    can_patch = lora_config.lora_dropout == 0 and lora_config.bias == "none"

    if not can_patch:
        LOG.warning("Cannot patch layers - requires no dropout and no bias")
        LOG.warning("Please specify `lora_dropout: 0` in your axolotl config file")
        return model

    # This needs to be reset after patching
    original_level = LOG.getEffectiveLevel()
    LOG.setLevel(logging.INFO)

    # Choose activation based on model type
    activation = None
    text_config = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    if hasattr(text_config, "hidden_act"):
        activation = text_config.hidden_act
    elif hasattr(text_config, "hidden_activation"):
        activation = text_config.hidden_activation

    # map activation to supported activation
    if "gelu" in activation:
        # gemma3 uses gelu_pytorch_tanh
        activation = "gelu"

    if activation not in SUPPORTED_ACTIVATIONS:
        raise NotImplementedError(f"Activation {activation} is not supported")

    layers = []
    # check for multimodal models first
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.model.layers
    else:
        raise NotImplementedError(
            f"Model type {model.config.model_type} is not supported yet. Please create an Issue."
        )

    # Patch each layer
    for layer in layers:
        # Add QKV, O fallback implementations to start
        # These will be overwritten later (if some conditions apply)
        for self_attn in find_self_attn_in_layer(layer):
            self_attn.apply_qkv = types.MethodType(original_apply_qkv, self_attn)
            self_attn.apply_o = types.MethodType(original_apply_o, self_attn)

            if cfg.lora_qkv_kernel:
                # Query, key, value patching
                layer_modules = [
                    getattr(self_attn, linear_proj)
                    for linear_proj in ["q_proj", "k_proj", "v_proj"]
                ]
                can_patch_qkv = all(
                    hasattr(module, "lora_A")
                    and getattr(module, "base_layer", module).bias is None
                    and len(getattr(module, "lora_magnitude_vector", []) or []) == 0
                    for module in layer_modules
                )

                if can_patch_qkv:
                    # Add optimized implementation
                    self_attn.apply_qkv = types.MethodType(apply_lora_qkv, self_attn)
                else:
                    LOG.warning_once(
                        "Cannot patch some attention QKV projections - requires LoRA adapters with no bias"
                    )
            if cfg.lora_o_kernel:
                # Output patching
                layer_modules = [
                    getattr(self_attn, linear_proj) for linear_proj in ["o_proj"]
                ]
                can_patch_o = all(
                    hasattr(module, "lora_A")
                    and getattr(module, "base_layer", module).bias is None
                    and len(getattr(module, "lora_magnitude_vector", []) or []) == 0
                    for module in layer_modules
                )

                if can_patch_o:
                    self_attn.apply_o = types.MethodType(apply_lora_o, self_attn)
                else:
                    LOG.warning_once(
                        "Cannot patch some attention output projection - requires LoRA adapters with no bias"
                    )
        for gate_proj, up_proj, down_proj, mlp in find_mlp_in_layer(layer):
            if cfg.lora_mlp_kernel:
                # MLP patching
                can_patch_mlp = all(
                    hasattr(proj, "lora_A")
                    and getattr(proj, "base_layer", proj).bias is None
                    and len(getattr(proj, "lora_magnitude_vector", []) or []) == 0
                    for proj in (gate_proj, up_proj, down_proj)
                )

                if can_patch_mlp:
                    apply_fn = APPLY_FN_MAPPING[activation]
                    layer.mlp.forward = types.MethodType(apply_fn, mlp)
                else:
                    LOG.warning_once(
                        "Cannot patch some MLP layers - requires LoRA adapters with no bias"
                    )

    LOG.setLevel(original_level)

    return model


class FakeMLP(nn.Module):
    """
    placeholder MLP for triton patching
    """

    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

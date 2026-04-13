"""Module for patching custom LoRA Triton kernels and `torch.autograd` functions."""

import importlib
import inspect
import logging
import types
from typing import Generator, Tuple, Type

import torch
from peft import PeftModelForCausalLM
from torch import nn
from transformers import AutoConfig

from axolotl.kernels.lora import (
    apply_lora_embedding,
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qk,
    apply_lora_qkv,
)
from axolotl.monkeypatch.utils import detab_code
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

QKV_PATCHES = [
    (
        """
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
    ),
    (
        """
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
    ),
    (
        """
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states, gate = torch.chunk(
        query_states.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
    ),
    # Gemma4: norm between proj and transpose, RoPE between norm and transpose,
    # conditional KV sharing (is_kv_shared_layer), v_proj may be None (attention_k_eq_v).
    # We only fuse the projection calls; norms, RoPE, and KV sharing stay as-is.
    (
        """
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
    if self.is_kv_shared_layer and past_key_values is not None:
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = query_states.view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
    if self.is_kv_shared_layer and past_key_values is not None:
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape) if self.v_proj is not None else key_states
""".lstrip("\n"),
    ),
    # Gemma4 (transformers >= 5.6): shared_kv_states parameter replaces
    # past_key_values.shared_layers, and v_norm added after k_norm.
    (
        """
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
    # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
    # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
    if self.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = query_states.view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
    # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
    # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
    if self.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape) if self.v_proj is not None else key_states
""".lstrip("\n"),
    ),
]

ORIGINAL_O_CODE = """
    attn_output = self.o_proj(attn_output)
""".lstrip("\n")

PATCHED_O_CODE = """
    attn_output = self.apply_o(attn_output)
""".lstrip("\n")

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


def original_apply_qkv_optional_v(
    self: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV projection for models where v_proj may be None (e.g. Gemma4 attention_k_eq_v).

    When v_proj is None, key_states are reused as value_states.
    """
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    if self.v_proj is not None:
        value_states = self.v_proj(hidden_states)
    else:
        value_states = key_states

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

    if model_type == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        return Qwen3VLTextAttention

    if model_type == "mllama":
        from transformers.models.mllama.modeling_mllama import MllamaTextSelfAttention

        return MllamaTextSelfAttention

    if model_type == "llama4":
        from transformers.models.llama4.modeling_llama4 import Llama4TextAttention

        return Llama4TextAttention

    if model_type == "mistral3":
        from transformers.models.mistral.modeling_mistral import MistralAttention

        return MistralAttention

    if model_type == "gemma3_text":
        from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

        return Gemma3Attention

    if model_type in ("gemma4", "gemma4_text"):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        return Gemma4TextAttention

    try:
        # Dynamically import the module and attention class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}Attention"])
        attention_cls = getattr(module, f"{model_cls_prefix}Attention")

        return attention_cls
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Axolotl could not import attention class for model_type: {model_type}. "
            "Please raise an Issue and turn off lora kernels to continue training. "
            f"Error: {str(e)}"
        ) from e


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

    assert any(qkv_options[0] in self_attn_forward for qkv_options in QKV_PATCHES), (
        "Original QKV code not found"
    )
    assert ORIGINAL_O_CODE in self_attn_forward, "Original O code not found"

    for qkv_orig, qkv_patched in QKV_PATCHES:
        if qkv_orig in self_attn_forward:
            self_attn_forward = self_attn_forward.replace(
                qkv_orig,
                qkv_patched,
            )
            break
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

    exec(
        f"from {module_name} import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(self_attn_forward, globals())

    LOG.info(f"Patched attention class with LoRA optims: {attention_cls.__name__}")
    attention_cls.forward = axolotl_attn_forward


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
                strict=False,
            ):
                yield (
                    gate_proj,
                    up_proj,
                    down_proj,
                    FakeMLP(gate_proj, up_proj, down_proj),
                )


def get_layers(model: PeftModelForCausalLM) -> list[nn.Module]:
    """
    Get the layers of the model. Handles text-only and multimodal models.

    Args:
        model: A PEFT model.

    Returns:
        A list of layers.
    """
    pretrained_model = model.model

    # check for multimodal models first
    if hasattr(pretrained_model, "language_model"):
        return pretrained_model.language_model.layers
    if hasattr(pretrained_model, "model"):
        if hasattr(pretrained_model.model, "language_model"):
            return pretrained_model.model.language_model.layers
        return pretrained_model.model.layers

    raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported yet. Please create an Issue."
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
        assert len(model.active_adapters) == 1, (
            "Axolotl currently does not support LoRA Triton kernels for multiple adapters"
        )
        active_adapter = model.active_adapters[0]
    else:
        active_adapter = model.active_adapter
    lora_config = model.model.peft_config[active_adapter]

    # Log what features are active
    if lora_config.lora_dropout > 0:
        LOG.info(f"LoRA kernels: dropout={lora_config.lora_dropout} enabled")
    if lora_config.bias != "none":
        LOG.info(f"LoRA kernels: bias={lora_config.bias} enabled")
    if lora_config.use_dora:
        LOG.info("LoRA kernels: DoRA enabled")

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
    elif hasattr(text_config, "mlp_hidden_act"):
        # Hybrid models (e.g. nemotron_h) use mlp_hidden_act instead of hidden_act
        activation = text_config.mlp_hidden_act

    # map activation to supported activation
    if activation and "gelu" in activation:
        # gemma3 uses gelu_pytorch_tanh
        activation = "gelu"

    layers = get_layers(model)

    # Patch each layer
    for layer in layers:
        # Add QKV, O fallback implementations to start
        # These will be overwritten later (if some conditions apply)
        for self_attn in find_self_attn_in_layer(layer):
            # Use v_proj-optional fallback for models where v_proj can be None
            # (e.g. Gemma4 with attention_k_eq_v=True)
            if getattr(self_attn, "v_proj", None) is None:
                self_attn.apply_qkv = types.MethodType(
                    original_apply_qkv_optional_v, self_attn
                )
            else:
                self_attn.apply_qkv = types.MethodType(original_apply_qkv, self_attn)
            self_attn.apply_o = types.MethodType(original_apply_o, self_attn)

            if cfg.lora_qkv_kernel:
                # Query, key, value patching
                # Filter out None projections (e.g. Gemma4 v_proj when attention_k_eq_v=True)
                has_v_proj = getattr(self_attn, "v_proj", None) is not None
                proj_names = (
                    ["q_proj", "k_proj", "v_proj"]
                    if has_v_proj
                    else ["q_proj", "k_proj"]
                )
                layer_modules = [getattr(self_attn, name) for name in proj_names]
                can_patch_qkv = all(
                    hasattr(module, "lora_A") for module in layer_modules
                )

                if can_patch_qkv:
                    if has_v_proj:
                        self_attn.apply_qkv = types.MethodType(
                            apply_lora_qkv, self_attn
                        )
                    else:
                        self_attn.apply_qkv = types.MethodType(apply_lora_qk, self_attn)
                else:
                    LOG.warning_once(
                        "Cannot patch some attention QKV projections - requires LoRA adapters"
                    )
            if cfg.lora_o_kernel:
                # Output patching
                layer_modules = [
                    getattr(self_attn, linear_proj) for linear_proj in ["o_proj"]
                ]
                can_patch_o = all(hasattr(module, "lora_A") for module in layer_modules)

                if can_patch_o:
                    self_attn.apply_o = types.MethodType(apply_lora_o, self_attn)
                else:
                    LOG.warning_once(
                        "Cannot patch some attention output projection - requires LoRA adapters"
                    )
        for gate_proj, up_proj, down_proj, mlp in find_mlp_in_layer(layer):
            if cfg.lora_mlp_kernel:
                # Check is inside lora_mlp_kernel guard so models with an
                # unsupported activation (e.g. nemotron_h uses relu2) can set
                # lora_mlp_kernel: false without hitting an error here.
                if activation not in SUPPORTED_ACTIVATIONS:
                    raise NotImplementedError(
                        f"Activation {activation!r} is not supported by lora_mlp_kernel. "
                        f"Set `lora_mlp_kernel: false` in your config or use a model with "
                        f"a supported activation ({SUPPORTED_ACTIVATIONS})."
                    )
                # MLP patching
                can_patch_mlp = all(
                    hasattr(proj, "lora_A") for proj in (gate_proj, up_proj, down_proj)
                )

                if can_patch_mlp:
                    apply_fn = APPLY_FN_MAPPING[activation]
                    layer.mlp.forward = types.MethodType(apply_fn, mlp)
                else:
                    LOG.warning_once(
                        "Cannot patch some MLP layers - requires LoRA adapters"
                    )

    # Patch embedding layers (model-level, not per-layer)
    if cfg.lora_embedding_kernel:
        _patch_embedding_layers(model, cfg)

    LOG.setLevel(original_level)

    return model


def _patch_embedding_layers(model: PeftModelForCausalLM, cfg: DictDefault):
    """Patch embedding layers with fused LoRA kernel.

    Handles both embed_tokens (nn.Embedding with lora_embedding_A/B) and
    lm_head (nn.Linear with lora_A/B, used when tied embeddings are untied by PEFT).
    """
    pretrained_model = model.model
    patched = 0

    # Find embedding modules - check common locations
    for attr_path in [
        ("model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
    ]:
        parent = pretrained_model
        for attr in attr_path:
            parent = getattr(parent, attr, None)
            if parent is None:
                break
        if parent is not None and hasattr(parent, "lora_embedding_A"):
            LOG.info(f"Patching embedding layer: {'.'.join(attr_path)}")
            parent.forward = types.MethodType(apply_lora_embedding, parent)
            patched += 1

    # lm_head with LoRA is a Linear layer - already handled by LoRA_O/LoRA_W kernels
    # when included in target_modules. No special embedding handling needed since
    # PEFT wraps it as a Linear (not Embedding) even for tied models.

    if not patched:
        LOG.debug("No embedding layers with LoRA found to patch")


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

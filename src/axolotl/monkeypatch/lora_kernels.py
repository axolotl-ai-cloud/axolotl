"""Module for patching custom LoRA Triton kernels and torch.autograd functions."""

import inspect
import logging
import types

from accelerate.logging import get_logger
from peft import PeftModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention

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


def original_apply_qkv(self, hidden_states):
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    return query_states, key_states, value_states


def original_apply_o(self, hidden_states):
    attn_output = self.o_proj(hidden_states)

    return attn_output


def get_self_attn_code() -> str:
    forward = inspect.getsource(LlamaAttention.forward)

    return forward


def check_self_attn_is_patchable() -> bool:
    qkv = get_self_attn_code()
    qkv, _ = detab_code(qkv)

    return ORIGINAL_QKV_CODE in qkv and ORIGINAL_O_CODE in qkv


self_attn_lora_patched = False  # pylint: disable=invalid-name


def patch_self_attn_lora():
    global self_attn_lora_patched  # pylint: disable=global-statement
    if self_attn_lora_patched:
        # prevent patching multiple times
        return
    self_attn_forward = get_self_attn_code()
    LlamaAttention._original_forward = (  # pylint: disable=protected-access
        self_attn_forward
    )
    self_attn_forward, _ = detab_code(self_attn_forward)
    assert ORIGINAL_QKV_CODE in self_attn_forward, "Original qkv code not found"
    assert ORIGINAL_O_CODE in self_attn_forward, "Original o code not found"

    self_attn_forward = self_attn_forward.replace(ORIGINAL_QKV_CODE, PATCHED_QKV_CODE)
    self_attn_forward = self_attn_forward.replace(ORIGINAL_O_CODE, PATCHED_O_CODE)
    self_attn_forward = self_attn_forward.replace(
        "def forward(",
        "def axolotl_attn_forward(",
        1,
    )

    # load imports necessary
    import transformers.models.llama.modeling_llama

    items_to_import = []
    for item in dir(transformers.models.llama.modeling_llama):
        if item in self_attn_forward:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.models.llama.modeling_llama import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(self_attn_forward, globals())  # pylint: disable=exec-used  # nosec B102
    self_attn_lora_patched = True
    LOG.info("patching attention for axolotl LoRA kernels", main_process_only=True)
    LlamaAttention.forward = (
        axolotl_attn_forward  # pylint: disable=undefined-variable  # noqa: F821
    )


def apply_lora_kernel_patches(model: PeftModelForCausalLM, cfg: DictDefault):
    """Patches a PEFT model with optimized MLP and Attention kernels"""
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError("Model must be a PeftModelForCausalLM")

    # Get active adapter config
    active_adapter = (
        model.active_adapters[0]
        if hasattr(model, "active_adapters")
        else model.active_adapter
    )
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
    model_type = model.config.model_type
    if model_type in ["llama", "mistral", "qwen2"]:
        activation = "swiglu"
    elif model_type in ["gemma", "gemma2"]:
        activation = "geglu"
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")

    # Patch each layer
    for layer in model.model.model.layers:
        # Add QKV, O fallback implementations to start
        # These will be overwritten later (if some conditions apply)
        layer.self_attn.apply_qkv = types.MethodType(
            original_apply_qkv, layer.self_attn
        )
        layer.self_attn.apply_o = types.MethodType(original_apply_o, layer.self_attn)

        if cfg.lora_mlp_kernel:
            # MLP patching
            gate_proj = layer.mlp.gate_proj
            up_proj = layer.mlp.up_proj
            down_proj = layer.mlp.down_proj

            can_patch_mlp = all(
                hasattr(proj, "lora_A")
                and getattr(proj, "base_layer", proj).bias is None
                and len(getattr(proj, "lora_magnitude_vector", []) or []) == 0
                for proj in (gate_proj, up_proj, down_proj)
            )

            if can_patch_mlp:
                if activation == "swiglu":
                    layer.mlp.forward = types.MethodType(
                        apply_lora_mlp_swiglu, layer.mlp
                    )
                else:
                    layer.mlp.forward = types.MethodType(
                        apply_lora_mlp_geglu, layer.mlp
                    )
            else:
                LOG.warning_once(
                    "Cannot patch some MLP layers - requires LoRA adapters with no bias"
                )
        if cfg.lora_qkv_kernel:
            # Query, key, value patching
            layer_modules = [
                getattr(layer.self_attn, linear_proj)
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
                layer.self_attn.apply_qkv = types.MethodType(
                    apply_lora_qkv, layer.self_attn
                )
            else:
                LOG.warning_once(
                    "Cannot patch some attention QKV projections - requires LoRA adapters with no bias"
                )
        if cfg.lora_o_kernel:
            # Output patching
            layer_modules = [
                getattr(layer.self_attn, linear_proj) for linear_proj in ["o_proj"]
            ]
            can_patch_o = all(
                hasattr(module, "lora_A")
                and getattr(module, "base_layer", module).bias is None
                and len(getattr(module, "lora_magnitude_vector", []) or []) == 0
                for module in layer_modules
            )

            if can_patch_o:
                layer.self_attn.apply_o = types.MethodType(
                    apply_lora_o, layer.self_attn
                )
            else:
                LOG.warning_once(
                    "Cannot patch some attention output projection - requires LoRA adapters with no bias"
                )

    LOG.setLevel(original_level)

    return model

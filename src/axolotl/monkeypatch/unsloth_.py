"""module for patching with unsloth optimizations"""

import inspect
import re
import types
from typing import Tuple

import torch
from accelerate.logging import get_logger
from peft import PeftModelForCausalLM
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    LlamaForCausalLM,
)

LOG = get_logger("axolotl.monkeypatch.unsloth")

ORIGINAL_CEL_CODE = """# Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
"""

PATCHED_CEL_CODE = """shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )
"""

ORIGINAL_QKV_CODE = """
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
""".lstrip(
    "\n"
)

PATCHED_QKV_CODE = """
    query_states, key_states, value_states = self.apply_qkv(self, hidden_states)
""".lstrip(
    "\n"
)

ORIGINAL_O_CODE = """
    attn_output = self.o_proj(attn_output)
""".lstrip(
    "\n"
)

PATCHED_O_CODE = """
    attn_output = self.apply_o(self, attn_output)
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


def get_forward_code() -> str:
    forward = inspect.getsource(LlamaForCausalLM.forward)
    return forward


def check_cel_is_patchable() -> bool:
    forward = get_forward_code()
    forward, _ = detab_code(forward)
    return ORIGINAL_CEL_CODE in forward


def get_self_attn_code() -> str:
    forward = inspect.getsource(LlamaFlashAttention2.forward)
    return forward


def check_self_attn_is_patchable() -> bool:
    qkv = get_self_attn_code()
    qkv, _ = detab_code(qkv)
    return ORIGINAL_QKV_CODE in qkv and ORIGINAL_O_CODE in qkv


def integrate_cross_entropy_loss_patch(model_type: str = "llama") -> None:
    if model_type == "llama":
        forward = get_forward_code()
        LlamaForCausalLM._original_forward = forward  # pylint: disable=protected-access
        forward, _ = detab_code(forward)
        assert ORIGINAL_CEL_CODE in forward, "Original forward code not found"

        forward = forward.replace(
            "@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)", ""
        )
        forward = forward.replace(
            "@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)",
            "",
        )
        forward = forward.replace(ORIGINAL_CEL_CODE, PATCHED_CEL_CODE)
        forward = forward.replace(
            "def forward(",
            "def fast_cross_entropy_loss_forward(",
            1,
        )

        # load imports necessary
        import transformers.models.llama.modeling_llama

        items_to_import = []
        for item in dir(transformers.models.llama.modeling_llama):
            if item in forward:
                items_to_import.append(item)

        exec(  # pylint: disable=exec-used  # nosec B102
            "from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss",
            globals(),
        )

        exec(  # pylint: disable=exec-used  # nosec B102
            "from transformers.models.llama.modeling_llama import ("
            + ", ".join(x for x in items_to_import)
            + ")",
            globals(),
        )
        exec(forward, globals())  # pylint: disable=exec-used  # nosec B102
        LOG.info("patching unsloth fast_cross_entropy_loss", main_process_only=True)
        LlamaForCausalLM.forward = fast_cross_entropy_loss_forward  # pylint: disable=undefined-variable  # noqa: F821
    else:
        raise ValueError("Unsupported model type")


def detab_code(code: str) -> Tuple[str, str]:
    spaces = re.match(r"([\s\t]{1,})", code).group(0)
    code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    return code, spaces


def patch_self_attn_lora():
    self_attn_forward = get_self_attn_code()
    LlamaFlashAttention2._original_forward = (  # pylint: disable=protected-access
        self_attn_forward
    )
    self_attn_forward, _ = detab_code(self_attn_forward)
    assert ORIGINAL_QKV_CODE in self_attn_forward, "Original qkv code not found"
    assert ORIGINAL_O_CODE in self_attn_forward, "Original o code not found"

    self_attn_forward = self_attn_forward.replace(ORIGINAL_QKV_CODE, PATCHED_QKV_CODE)
    self_attn_forward = self_attn_forward.replace(ORIGINAL_O_CODE, PATCHED_O_CODE)
    self_attn_forward = self_attn_forward.replace(
        "def forward(",
        "def unsloth_attn_forward(",
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
    LOG.info("patching unsloth attn lora", main_process_only=True)
    LlamaFlashAttention2.forward = (
        unsloth_attn_forward  # pylint: disable=undefined-variable  # noqa: F821
    )


def integrate_rope_embeddings():
    import transformers.models.llama.modeling_llama
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    def apply_rotary_pos_emb(  # pylint: disable=unused-argument
        q,  # pylint: disable=invalid-name
        k,  # pylint: disable=invalid-name
        cos,
        sin,
        position_ids=None,
        unsqueeze_dim=1,
    ):
        return fast_rope_embedding(q, k, cos, sin)

    LOG.info("patching unsloth RoPE embeddings", main_process_only=True)
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb


def integrate_lora_mlp_patch(peft_model: PeftModelForCausalLM):
    if peft_model.base_model.config.model_type in ["llama", "mistral"]:
        from unsloth.kernels import apply_lora_mlp_swiglu

        apply_lora_mlp = apply_lora_mlp_swiglu
    elif peft_model.base_model.config.model_type == "gemma":
        from unsloth.kernels import apply_lora_mlp_geglu_approx

        apply_lora_mlp = apply_lora_mlp_geglu_approx
    else:
        raise NotImplementedError(
            f"Model type {peft_model.base_model.config.model_type} not supported"
        )

    for idx, layer in enumerate(peft_model.model.model.layers):
        layer_modules = [
            getattr(layer.mlp, linear_proj)
            for linear_proj in ["gate_proj", "up_proj", "down_proj"]
        ]
        is_mlp_lora = all(hasattr(module, "lora_A") for module in layer_modules)
        mlp_no_bias = all(
            getattr(module, "base_layer", module).bias is None
            for module in layer_modules
        )
        mlp_not_dora = all(
            getattr(module, "lora_magnitude_vector", None) is None
            for module in layer_modules
        )

        if is_mlp_lora and mlp_no_bias and mlp_not_dora:
            layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
        else:
            LOG.warning("unable to apply unsloth lora mlp patch to layer %d", idx)


def integrate_lora_patch(peft_model: PeftModelForCausalLM, cfg):
    from unsloth.kernels import apply_lora_o, apply_lora_qkv

    for idx, layer in enumerate(peft_model.model.model.layers):
        if cfg.unsloth_lora_qkv:
            layer_modules = [
                getattr(layer.self_attn, linear_proj)
                for linear_proj in ["q_proj", "k_proj", "v_proj"]
            ]
            is_qkv_lora = all(hasattr(module, "lora_A") for module in layer_modules)
            qkv_no_bias = all(
                getattr(module, "base_layer", module).bias is None
                for module in layer_modules
            )
            qkv_not_dora = all(
                getattr(module, "lora_magnitude_vector", None) is None
                for module in layer_modules
            )

            if is_qkv_lora and qkv_no_bias and qkv_not_dora:
                layer.self_attn.apply_qkv = apply_lora_qkv
            else:
                layer.self_attn.apply_qkv = original_apply_qkv
                LOG.warning("unable to apply unsloth lora qkv patch to layer %d", idx)
        if cfg.unsloth_lora_o:
            layer_modules = [
                getattr(layer.self_attn, linear_proj) for linear_proj in ["o_proj"]
            ]
            is_o_lora = all(hasattr(module, "lora_A") for module in layer_modules)
            o_no_bias = all(
                getattr(module, "base_layer", module).bias is None
                for module in layer_modules
            )
            o_not_dora = all(
                getattr(module, "lora_magnitude_vector", None) is None
                for module in layer_modules
            )

            if is_o_lora and o_no_bias and o_not_dora:
                layer.self_attn.apply_o = apply_lora_o
            else:
                layer.self_attn.apply_o = original_apply_o
                LOG.warning(
                    "unable to apply unsloth lora o_proj patch to layer %d", idx
                )


def patch_unsloth_layernorm():
    try:
        import transformers.models.llama.modeling_llama
        from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

        class LlamaRMSNorm(nn.Module):
            """LlamaRMSNorm"""

            def __init__(self, hidden_size, eps=1e-6):
                """
                LlamaRMSNorm is equivalent to T5LayerNorm
                """
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                return Fast_RMS_Layernorm.apply(
                    hidden_states, self.weight, self.variance_epsilon, False
                )

        LOG.info("patching with unsloth.kernels.rms_layernorm")
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    except ImportError:
        LOG.warning("missing unsloth library")

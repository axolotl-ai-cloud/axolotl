import inspect
import re
from typing import Tuple

from transformers.models.llama.modeling_llama import LlamaForCausalLM

ORIGINAL_CEL_CODE = """    if labels is not None:
        # Shift so that tokens < n predict n
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

PATCHED_CEL_CODE = """    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )
"""


def get_forward_code() -> str:
    forward = inspect.getsource(LlamaForCausalLM.forward)
    return forward


def test_cel_is_patchable() -> bool:
    forward = get_forward_code()
    return ORIGINAL_CEL_CODE in forward


def integrate_cross_entropy_loss_patch():
    forward = get_forward_code()
    LlamaForCausalLM._original_forward = forward
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
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    exec(
        "from transformers.models.llama.modeling_llama import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(forward, globals())
    print("patching unsloth fast_cross_entropy_loss")
    LlamaForCausalLM.forward = fast_cross_entropy_loss_forward


def detab_code(code: str) -> Tuple[str, str]:
    spaces = re.match(r"([\s\t]{1,})", code).group(0)
    code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    return code, spaces


def integrate_lora_mlp_patch(peft_model, cfg):
    # TODO
    pass


def integrate_lora_qkv_patch(peft_model, cfg):
    # TODO
    pass


def integrate_lora_o_patch(peft_model, cfg):
    # TODO
    pass

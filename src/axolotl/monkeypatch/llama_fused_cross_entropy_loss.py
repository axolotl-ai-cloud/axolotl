"""patch for fused lm_head + cross entropy loss"""
import inspect
import re
from typing import Tuple

from transformers.models.llama.modeling_llama import LlamaForCausalLM

ORIGINAL_CEL_CODE = """
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
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
""".lstrip(
    "\n"
)

PATCHED_CEL_CODE = """
    if self.training:
        loss = FusedCrossEntropyLossFunction.apply(
                rearrange(hidden_states, 'b s d -> (b s) d'),
                self.lm_head.weight,
                rearrange(labels, 'b s -> (b s)'),
                8,
                -100,
                "mean",
            )
        logits = None
    else:
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
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
""".lstrip(
    "\n"
)


def get_forward_code() -> str:
    forward = inspect.getsource(LlamaForCausalLM.forward)
    return forward


def test_cel_is_patchable() -> bool:
    forward = get_forward_code()
    return ORIGINAL_CEL_CODE in forward


def integrate_cross_entropy_loss_patch():
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
        "def fused_cross_entropy_loss_forward(",
        1,
    )

    # load imports necessary
    import transformers.models.llama.modeling_llama

    items_to_import = []
    for item in dir(transformers.models.llama.modeling_llama):
        if item in forward:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from axolotl.kernels.efficient_cross_entropy_loss import FusedCrossEntropyLossFunction\n"
        + "from einops import rearrange",
        globals(),
    )

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.models.llama.modeling_llama import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(forward, globals())  # pylint: disable=exec-used  # nosec B102
    print("patching fused cross_entropy_loss")
    LlamaForCausalLM.forward = fused_cross_entropy_loss_forward  # pylint: disable=undefined-variable # noqa: F821


def detab_code(code: str) -> Tuple[str, str]:
    spaces = re.match(r"([\s\t]{1,})", code).group(0)
    code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    return code, spaces

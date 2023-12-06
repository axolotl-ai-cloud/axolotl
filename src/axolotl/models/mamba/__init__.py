# pylint: skip-file

from collections import namedtuple

from torch.nn import CrossEntropyLoss


def fix_mamba_attn_for_loss():
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    MambaLMHeadModel.forward = mamba_forward
    return MambaLMHeadModel  # pylint: disable=invalid-name


def mamba_forward(
    self,
    input_ids,
    position_ids=None,
    inference_params=None,
    num_last_tokens=0,
    labels=None,
):
    """
    "position_ids" is just to be compatible with Transformer generation. We don't use it.
    num_last_tokens: if > 0, only return the logits for the last n tokens
    """
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = lm_logits
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
        print(loss, shift_logits, shift_logits.dtype, shift_labels, shift_labels.dtype)
        return (loss,)

    else:
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

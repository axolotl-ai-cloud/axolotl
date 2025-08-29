"""
Generic FLCE patch for untested models similar to Llama
"""

from typing import Optional, Tuple, Union

import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.trainer.orpo_trainer import _FSDPForwardRedirection
from liger_kernel.utils import PEFT_AVAILABLE
from peft.utils import ModulesToSaveWrapper
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix


def lce_forward(
    self,
    *args,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    labels: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).
    """

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        *args,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None

    # if in training mode, don't materialize logits
    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        # By default, if in training mode, don't materialize logits
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        loss = lce_maybe_trainable_lm_head(
            self,
            hidden_states=kept_hidden_states,
            hidden_size=self.config.hidden_size,
            labels=labels,
            shift_labels=shift_labels,
            **kwargs,
        )

    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def lce_maybe_trainable_lm_head(
    self, hidden_states, hidden_size, labels, shift_labels, **loss_kwargs
):
    lm_head = self.lm_head

    # Unwrap the module if lm_head has been added as trainable module in PEFT LoRA configuration,
    # i.e. listed in the modules_to_save field of LoraConfig, so the lm_head weights are read
    # from the unwrapped module.
    # See https://huggingface.co/docs/peft/package_reference/lora for reference.
    if PEFT_AVAILABLE and isinstance(lm_head, ModulesToSaveWrapper):
        lm_head = lm_head.modules_to_save.default

    # If FSDP is used and lm_head is trainable, e.g., during full fine-tuning or with LoRA,
    # reading the lm_head module weights and calling the kernel must be done within FSDP forward pass
    # so the module entire parameters are summoned and kept in memory during the kernel execution.
    if isinstance(lm_head, FullyShardedDataParallel):
        return _FSDPForwardRedirection()(
            lm_head,
            _liger_for_causal_lm_loss,
            lm_head.module,
            hidden_states,
            hidden_size,
            labels,
            shift_labels,
            **loss_kwargs,
        )

    # FSDP is not used so we can read the lm_head weights and call the kernel directly
    return _liger_for_causal_lm_loss(
        lm_head=self.lm_head,
        hidden_states=hidden_states,
        hidden_size=hidden_size,
        labels=labels,
        shift_labels=shift_labels,
        **loss_kwargs,
    )


def _liger_for_causal_lm_loss(
    lm_head, hidden_states, hidden_size, labels, shift_labels, **loss_kwargs
):
    return LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=lm_head.weight,
        labels=labels,
        hidden_size=hidden_size,
        shift_labels=shift_labels,
        **loss_kwargs,
    )


def patch_lce_forward(
    model_type,
):
    try:
        # Dynamically import the module and MLP class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"])
        model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")

        model_cls.forward = lce_forward

    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import ForCausalLM class for model_type: {model_type}. "
            f"Error: {str(e)}"
        ) from e

"""
expands the binary attention mask per 3.2.2 of https://arxiv.org/pdf/2107.02027.pdf
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids, mask_2d_to_4d


def llama_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
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
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    masked_zero_one_mask = mask_2d_to_4d(mask, dtype, tgt_len)
    inverted_mask = 1.0 - masked_zero_one_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def hijack_expand_mask():
    import transformers

    transformers.models.llama.modeling_llama._expand_mask = (  # pylint: disable=protected-access
        _expand_mask
    )
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
        llama_for_causal_lm_forward
    )

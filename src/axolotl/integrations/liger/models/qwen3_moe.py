"""
Liger FLCE for Qwen3 MoE. Based on transformers v4.51.3.
"""

import sys
from copy import deepcopy
from typing import List, Optional, Union

import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func


def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> MoeCausalLMOutputWithPast:
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

    Returns:
    """

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
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
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    logits = None
    loss = None
    # if in training mode, don't materialize logits
    if self.training and (labels is not None):
        loss = LigerForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )

    else:  # if in inference mode materialize logits
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(
                loss.device
            )  # make sure to reside in the same device

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def apply_liger_kernel_to_qwen3_moe(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = False,
    glu_activation: bool = False,
    layer_norm: bool = False,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is False.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be False.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is False.
        glu_activation (bool): Whether to apply Liger's SwiGLU MLP. Default is False.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is False.
    """

    import transformers.models.qwen3_moe.modeling_qwen3_moe  # noqa: F401
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    modeling_qwen3_moe = sys.modules["transformers.models.qwen3_moe.modeling_qwen3_moe"]

    if rms_norm:
        modeling_qwen3_moe.Qwen3MoeRMSNorm = LigerRMSNorm

    if glu_activation:

        def _liger_swiglu_mlp_wrapper(config, intermediate_size=None, **kwargs):
            "Accepts intermediate_size to pass to LigerSwiGLUMLP"
            # clone config to avoid modifying the original
            config = deepcopy(config)
            if intermediate_size:
                config.intermediate_size = intermediate_size
            return LigerSwiGLUMLP(config, **kwargs)

        modeling_qwen3_moe.Qwen3MoeMLP = _liger_swiglu_mlp_wrapper

    if layer_norm:
        modeling_qwen3_moe.nn.LayerNorm = LigerLayerNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = lce_forward

"""
DeepseekV3 model with LigerFusedLinearCrossEntropyLoss
"""

# pylint: disable=duplicate-code

import sys
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    KwargsForCausalLM,
    logger,
)
from transformers.processing_utils import Unpack


def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> CausalLMOutputWithPast:
    r"""
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

    Example:

    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

    >>> model = DeepseekV3ForCausalLM.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
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

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    logits = None
    loss = None

    if self.training and (labels is not None):
        loss = LigerForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
    else:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# adapted from https://github.com/ScienceOne-AI/DeepSeek-671B-SFT-Guide/blob/ccf17c581b9c42eca007aae793e164b66a0fbaab/model/DeepSeek-V3-BF16/modeling_deepseek.py#L424
def moe_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    # compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), self.weight.type(torch.float32), None
    )
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    elif self.scoring_func == "softmax":
        scores = logits.softmax(dim=-1, dtype=torch.float32)
    else:
        raise NotImplementedError(
            f"insupportable scoring function for MoE gating: {self.scoring_func}"
        )

    # select top-k experts
    if self.topk_method == "noaux_tc":
        # assert not self.training
        scores_for_choice = scores.view(
            bsz * seq_len, -1
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
    elif self.topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    elif self.topk_method == "group_limited_greedy":
        group_scores = (
            scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weight, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1, sorted=False
        )

    # norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    else:
        topk_weight = topk_weight * self.routed_scaling_factor
    # expert-level computation auxiliary loss
    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (pi * fi).sum() * self.alpha
    else:
        aux_loss = None
    return topk_idx, topk_weight, aux_loss


def model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
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
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError(
            "The `past_key_values` should be either a `Cache` object or `None`."
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(  # pylint: disable=protected-access
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = (
                self._gradient_checkpointing_func(  # pylint: disable=protected-access
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_liger_kernel_to_deepseekv3(
    base_model: str,
    trust_remote_code: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = False,
    glu_activation: bool = False,
    layer_norm: bool = False,
    **kwargs,  # pylint: disable=unused-argument
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace DeepseekV3 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    # from transformers.models.deepseek_v3 import modeling_deepseek_v3
    from accelerate import init_empty_weights
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=trust_remote_code or False
        )
        modeling_mod = sys.modules[model.__class__.__module__]

    # patch moe
    modeling_mod.MoEGate.forward = moe_forward

    # patch model forward
    modeling_mod.DeepseekV3Model.forward = model_forward

    if rms_norm:
        modeling_mod.DeepseekV3RMSNorm = LigerRMSNorm
    if glu_activation:

        def _liger_swiglu_mlp_wrapper(config, intermediate_size=None, **kwargs):
            "Accepts intermediate_size to pass to LigerSwiGLUMLP"
            # clone config to avoid modifying the original
            config = deepcopy(config)
            if intermediate_size:
                setattr(config, "intermediate_size", intermediate_size)
            return LigerSwiGLUMLP(config, **kwargs)

        modeling_mod.DeepseekV3MLP = _liger_swiglu_mlp_wrapper
    if layer_norm:
        modeling_mod.nn.LayerNorm = LigerLayerNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        modeling_mod.DeepseekV3ForCausalLM.forward = lce_forward

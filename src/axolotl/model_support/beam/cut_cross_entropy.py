"""Beam (reflection) CCE patch. Adapted from cut_cross_entropy's gemma3 patch."""

# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import Union

import torch
import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
    patch_remote_model_class,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

_PATCH_OPTS: PatchOptions | None = None


def cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **loss_kwargs,
) -> MoeCausalLMOutputWithPast:
    # Strip PEFT-injected return_dict so it can't leak into self.model and force a tuple return.
    loss_kwargs.pop("return_dict", None)

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **loss_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    loss = None
    logits = None

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            softcap=self.final_logit_softcapping,
            **loss_kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = (logits.float() / cap).tanh() * cap

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def _patch(module_name: str, class_name: str):
    def patch_fn(
        maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
        patch_options: PatchOptions,
        remote_model_id: str | None = None,
    ) -> TransformersModelT | None:
        global _PATCH_OPTS  # pylint: disable=global-statement
        _PATCH_OPTS = patch_options

        if remote_model_id is not None:
            patch_remote_model_class(
                remote_model_id=remote_model_id,
                class_name=class_name,
                patch_fn=cce_forward,
            )
            return None

        import importlib

        modeling = importlib.import_module(module_name)
        model_cls = getattr(modeling, class_name)

        if isinstance(maybe_model, transformers.PreTrainedModel):
            assert isinstance(maybe_model, model_cls), (
                f"Expected a {class_name} model. Got {type(maybe_model)}."
            )
            maybe_model.forward = MethodType(cce_forward, maybe_model)
            return maybe_model

        model_cls.forward = cce_forward
        return None

    return patch_fn


patch_beam = _patch("transformers.models.beam.modeling_beam", "BeamForCausalLM")
patch_reflection = _patch(
    "transformers.models.reflection.modeling_reflection", "ReflectionForCausalLM"
)

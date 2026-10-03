"""Opt-in Cut Cross Entropy bridge for native Nemotron diffusion models."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

from torch import nn

_PENDING_OPTIONS: Any = None


def _unwrap(model):
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        candidate = getattr(model, "module", None)
        if candidate is None and hasattr(model, "get_base_model"):
            candidate = model.get_base_model()
        if candidate is None or candidate is model:
            break
        model = candidate
    return model


def patch_nemotron(maybe_model, patch_options, remote_model_id=None):
    """CCE PATCH_FNS entrypoint; bind options to the next native model class."""
    del remote_model_id
    global _PENDING_OPTIONS
    if isinstance(maybe_model, str) or not isinstance(maybe_model, nn.Module):
        _PENDING_OPTIONS = patch_options
    else:
        apply_nemotron_cce_patch(type(maybe_model), patch_options)
    return maybe_model


def reset_pending_nemotron_cce_options() -> None:
    global _PENDING_OPTIONS
    _PENDING_OPTIONS = None


def apply_pending_nemotron_cce_patch(model_class: type) -> None:
    global _PENDING_OPTIONS
    options, _PENDING_OPTIONS = _PENDING_OPTIONS, None
    if options is not None:
        apply_nemotron_cce_patch(model_class, options)


def get_cce_options(model) -> Any | None:
    if model is None:
        raise ValueError("Nemotron CCE options require a model instance")
    options = getattr(type(_unwrap(model)), "_axolotl_cce_options", None)
    if options is None:
        raise ValueError("Nemotron CCE is not enabled for this model")
    return options


def get_cce_head(model) -> nn.Linear:
    model = _unwrap(model)
    head = model.get_output_embeddings()
    if type(head) is not nn.Linear:
        raise ValueError("Nemotron CCE requires an unwrapped nn.Linear diffusion head")
    if hasattr(head, "lora_A") or hasattr(head, "lora_B"):
        raise ValueError("Nemotron CCE does not support a LoRA diffusion head")
    return head


def linear_token_loss(hidden, head: nn.Linear, targets, options):
    if not isinstance(head, nn.Linear):
        raise ValueError("Nemotron CCE requires an nn.Linear diffusion head")
    if options is None:
        raise ValueError("Nemotron CCE is not enabled for this model")
    from cut_cross_entropy import linear_cross_entropy

    kwargs = (
        dict(options.to_kwargs()) if hasattr(options, "to_kwargs") else dict(options)
    )
    kwargs.pop("reduction", None)
    c_grad_chunk_size = kwargs.pop(
        "c_grad_chunk_size", getattr(options, "c_grad_chunk_size", 0)
    )
    if c_grad_chunk_size:
        if (
            "c_grad_chunk_size"
            not in inspect.signature(linear_cross_entropy).parameters
        ):
            from axolotl.integrations.cut_cross_entropy import _CCE_INSTALL_MESSAGE

            raise ImportError(
                "The installed cut_cross_entropy does not support "
                "`cut_cross_entropy_c_grad_chunk_size`. " + _CCE_INSTALL_MESSAGE
            )
        kwargs["c_grad_chunk_size"] = c_grad_chunk_size
    return linear_cross_entropy(
        hidden,
        head.weight,
        targets.to(hidden.device),
        bias=head.bias,
        reduction="none",
        shift=0,
        **kwargs,
    )


def apply_nemotron_cce_patch(model_class: Any, options) -> None:
    if getattr(model_class, "_axolotl_cce_patched", False):
        model_class._axolotl_cce_options = options
        return
    original_forward = model_class.forward

    @wraps(original_forward)
    def forward(
        self, *args, cce_targets=None, cce_return_hidden_states=False, **kwargs
    ):
        if cce_targets is None and not cce_return_hidden_states:
            return original_forward(self, *args, **kwargs)
        if kwargs.get("labels") is not None:
            raise ValueError("Nemotron CCE explicit path does not accept labels")
        hidden_output = original_forward(
            self, *args, output_last_hidden_states_only=True, **kwargs
        )
        if cce_return_hidden_states and cce_targets is None:
            return hidden_output
        head = get_cce_head(self)
        loss = linear_token_loss(
            hidden_output.last_hidden_state, head, cce_targets, get_cce_options(self)
        )
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(loss=loss, logits=None)

    model_class.forward = forward
    model_class._axolotl_cce_options = options
    model_class._axolotl_cce_patched = True

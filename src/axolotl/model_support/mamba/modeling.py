"""Training-compatible FLA Mamba models with document-batched packing."""

import importlib
import inspect
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType, MethodType

import torch
from fla.models.mamba import MambaConfig, MambaForCausalLM
from fla.models.mamba2 import Mamba2Config, Mamba2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.models.mamba.modeling import (
    PackedSegments,
    _from_docs,
    _to_docs,
)
from axolotl.monkeypatch.models.mamba_utils import get_seq_idx

_KERNEL_LOCK = threading.RLock()


def _config(config, family):
    values = config.to_dict()
    for source, target in {
        "layer_norm_epsilon": "norm_eps",
        "time_step_rank": "dt_rank",
        "time_step_scale": "dt_scale",
        "time_step_min": "dt_min",
        "time_step_max": "dt_max",
        "time_step_init_scheme": "dt_init_scheme",
        "time_step_floor": "dt_init_floor",
        "time_step_limit": "dt_limit",
    }.items():
        if source in values:
            values[target] = values[source]
    values.update(
        mamba_backend="fla",
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,
    )
    return (MambaConfig if family == "mamba" else Mamba2Config)(**values)


@lru_cache(maxsize=2)
def _kernel_bindings(family):
    from kernels import get_kernel

    ssm = get_kernel("kernels-community/mamba-ssm", version=2)
    conv = get_kernel("kernels-community/causal-conv1d", version=1)
    names = {
        "selective_state_update": "ops.triton.selective_state_update",
    }
    if family == "mamba":
        names.update(
            mamba_inner_fn="ops.selective_scan_interface",
            selective_scan_fn="ops.selective_scan_interface",
        )
    else:
        scan_module = importlib.import_module(
            f"{ssm.__name__}.ops.triton.ssd_chunk_scan"
        )
        scan_kernel = scan_module._chunk_scan_fwd_kernel
        # Reusing a tuning result across layouts can exceed the GPU's shared memory.
        scan_kernel.keys = [
            name
            for name in scan_kernel.arg_names
            if not name.endswith("_ptr") and not name.startswith("BLOCK_SIZE_")
        ]
        names.update(
            mamba_chunk_scan_combined="ops.triton.ssd_combined",
            mamba_split_conv1d_scan_combined="ops.triton.ssd_combined",
        )
    return {
        **{
            name: getattr(importlib.import_module(f"{ssm.__name__}.{path}"), name)
            for name, path in names.items()
        },
        "causal_conv1d_fn": conv.causal_conv1d_fn,
        "causal_conv1d_update": conv.causal_conv1d_update,
        "is_fast_path_available": True,
    }


@contextmanager
def _construction_kernels(family, bindings):
    module = importlib.import_module(f"fla.layers.{family}")
    with _KERNEL_LOCK:
        originals = {key: getattr(module, key) for key in bindings}
        try:
            for key, value in bindings.items():
                setattr(module, key, value)
            yield
        finally:
            for key, value in originals.items():
                setattr(module, key, value)


def _bind(function, instance, bindings):
    function = inspect.unwrap(function.__func__)
    cloned = FunctionType(
        function.__code__,
        {**function.__globals__, **bindings},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return MethodType(cloned, instance)


def _block_forward(
    self,
    hidden_states,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
    output_attentions=False,
    **kwargs,
):
    residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
    hidden_states = self.norm(hidden_states.to(self.norm.weight.dtype))
    hidden_states, attentions, cache = self.mixer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )
    return residual + hidden_states, attentions, cache


def _adapter_cuda_forward(
    self, hidden_states, last_state=None, use_cache=False, attention_mask=None, **kwargs
):
    original = self._axolotl_cuda_forward
    if (
        self.training
        and not use_cache
        and not isinstance(self.out_proj, torch.nn.Linear)
    ):
        # FLA's CUDA prefill path invokes out_proj instead of reading its raw weight.
        output, _, _ = original(
            hidden_states, last_state, True, attention_mask, **kwargs
        )
        return output, None, None
    return original(hidden_states, last_state, use_cache, attention_mask, **kwargs)


def _packed_forward(self, hidden_states, *args, **kwargs):
    original = self._axolotl_fla_forward
    segments = kwargs.pop("_axolotl_segments", None)
    if segments is None:
        return original(hidden_states, *args, **kwargs)
    if args or kwargs.get("past_key_values") is not None:
        raise ValueError("Packed FLA Mamba requires uncached inputs")
    groups = []
    batch, length, _ = hidden_states.shape
    mask = kwargs.pop("attention_mask", None)
    kwargs["use_cache"] = False
    for index, valid in segments.plan:
        inputs = _to_docs(hidden_states.transpose(1, 2), index, valid).transpose(1, 2)
        attention_mask = valid.to(hidden_states.device)
        if mask is not None:
            attention_mask = (
                attention_mask
                & _to_docs(mask[:, None, :], index, valid).squeeze(1).bool()
            )
        output = original(inputs, attention_mask=attention_mask, **kwargs)[0]
        groups.append((index, valid, output.transpose(1, 2)))
    return _from_docs(groups, batch, length).transpose(1, 2), None, None


def _norm_forward(self, hidden_states, *args, **kwargs):
    original = self._axolotl_norm_forward
    if hidden_states.device.type != "cpu":
        return original(hidden_states, *args, **kwargs)
    dtype = hidden_states.dtype
    values = hidden_states.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
    return (values * self.weight.float()).to(dtype)


class _FlaCompatibility:
    _tied_weights_keys = {"lm_head.weight": "backbone.embeddings.weight"}
    _supports_loss_kwargs = True

    def __init__(self, config):
        if os.environ.get("FLA_CONV_BACKEND", "cuda") != "cuda":
            raise ValueError("FLA Mamba integration requires FLA_CONV_BACKEND=cuda")
        family = config.model_type
        bindings = _kernel_bindings(family) if torch.cuda.is_available() else {}
        with _construction_kernels(family, bindings):
            super().__init__(_config(config, family))
        from .adapters import enable_lora_projections

        enable_lora_projections()
        for block in self.backbone.layers:
            block.forward = MethodType(_block_forward, block)
            mixer = block.mixer
            mixer._axolotl_fla_forward = _bind(mixer.forward, mixer, bindings)
            mixer._axolotl_cuda_forward = _bind(
                mixer.cuda_kernels_forward, mixer, bindings
            )
            mixer.cuda_kernels_forward = _bind(
                MethodType(_adapter_cuda_forward, mixer), mixer, bindings
            )
            mixer.forward = MethodType(_packed_forward, mixer)
        self.lm_head.register_forward_pre_hook(
            lambda module, args: (args[0].to(module.weight.dtype),)
        )
        for norm in [self.backbone.norm_f, *(b.norm for b in self.backbone.layers)]:
            norm._axolotl_norm_forward = norm.forward
            norm.forward = MethodType(_norm_forward, norm)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        position_ids=None,
        shift_labels=None,
        num_items_in_batch=None,
        use_cache=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        packed = position_ids is not None and (position_ids[:, 1:] == 0).any()
        if packed:
            if past_key_values is not None:
                raise ValueError("Packed FLA Mamba requires uncached inputs")
            use_cache = False
            kwargs["_axolotl_segments"] = PackedSegments(get_seq_idx(position_ids))
        native_logits_to_keep = (
            logits_to_keep if labels is None and isinstance(logits_to_keep, int) else 0
        )
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=(attention_mask != 0).to(attention_mask.dtype)
            if attention_mask is not None
            else None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=None,
            use_cache=use_cache,
            return_dict=True,
            logits_to_keep=native_logits_to_keep,
            **kwargs,
        )
        if labels is not None:
            if shift_labels is None:
                shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[
                    ..., 1:
                ].clone()
                if packed:
                    shift_labels[:, :-1].masked_fill_(position_ids[:, 1:] == 0, -100)
            outputs["loss"] = self.loss_function(
                logits=outputs.logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
        if isinstance(logits_to_keep, torch.Tensor):
            outputs.logits = outputs.logits[:, logits_to_keep]
        elif logits_to_keep and not native_logits_to_keep:
            outputs.logits = outputs.logits[:, -logits_to_keep:]
        outputs = CausalLMOutputWithPast(**outputs)
        if return_dict is False or (
            return_dict is None and not self.config.return_dict
        ):
            return outputs.to_tuple()
        return outputs


class FlaMambaForCausalLM(_FlaCompatibility, MambaForCausalLM):
    """Mamba with FLA kernels and Axolotl loss/packing contracts."""


class FlaMamba2ForCausalLM(_FlaCompatibility, Mamba2ForCausalLM):
    """Mamba2 with FLA kernels and Axolotl loss/packing contracts."""

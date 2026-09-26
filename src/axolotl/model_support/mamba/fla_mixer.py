"""FLA computation over parameters owned by a Transformers mixer."""

import importlib
import inspect
from functools import lru_cache
from types import FunctionType, MethodType

import torch
from fla.layers.mamba import Mamba
from fla.layers.mamba2 import Mamba2

from axolotl.monkeypatch.models.mamba.modeling import _from_docs, _to_docs


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


def _adapter_cuda_forward(
    self, hidden_states, last_state=None, use_cache=False, attention_mask=None, **kwargs
):
    original = self._axolotl_cuda_forward
    if (
        self.training
        and not use_cache
        and (
            not isinstance(self.out_proj, torch.nn.Linear)
            or (getattr(self, "n_groups", 1) > 1 and not self.legacy_norm)
        )
    ):
        # The fused path bypasses adapted out_proj and groups RMSNorm by state group.
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
    if args or kwargs.get("cache_params") is not None:
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


def _legacy_norm_forward(self, hidden_states, gate=None):
    values = hidden_states.float()
    if gate is not None and not self._norm_before_gate:
        values = values * torch.nn.functional.silu(gate.float())
    grouped = values.reshape(*values.shape[:-1], -1, self._group_size)
    values = (
        grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
    ).reshape_as(values)
    values = values * self.weight.float()
    if gate is not None and self._norm_before_gate:
        values = values * torch.nn.functional.silu(gate.float())
    return values.to(hidden_states.dtype)


class _FlaMixerView:
    def __init__(self, owner, config):
        torch.nn.Module.__init__(self)
        # Avoid registering the parent twice; deepcopy preserves this ownership cycle.
        object.__setattr__(self, "_owner", owner)
        self.backend = "cuda"
        self.dt_rank = owner.time_step_rank
        family = config.model_type
        self.legacy_norm = getattr(
            config, "fla_mamba_legacy_norm", False
        ) or "FlaMamba2ForCausalLM" in (config.architectures or ())
        if family == "mamba2":
            self.dt_limit = tuple(owner.time_step_limit)
            self.dt_min = owner.time_step_min
            self.dt_max = owner.time_step_max
            self.D_has_hdim = False
            self.rmsnorm = True
            self.norm_before_gate = False
            owner.norm.eps = owner.norm.variance_epsilon
            if self.legacy_norm:
                # Persist the old normalization after save_pretrained renames the architecture.
                config.fla_mamba_legacy_norm = True
                self.norm_before_gate = getattr(config, "norm_before_gate", False)
                owner.norm.eps = getattr(config, "norm_eps", owner.norm.eps)
                owner.norm._group_size = owner.intermediate_size // owner.n_groups
                owner.norm._norm_before_gate = self.norm_before_gate
                owner.norm.forward = MethodType(_legacy_norm_forward, owner.norm)
        self._family = family
        self._axolotl_fla_forward = self._forward
        self.forward = MethodType(_packed_forward, self)
        if torch.cuda.is_available():
            self._bind_kernels()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_"):
                raise
            return getattr(object.__getattribute__(self, "_owner"), name)

    def _bind_kernels(self):
        bindings = _kernel_bindings(self._family)
        self.causal_conv1d_fn = bindings["causal_conv1d_fn"]
        self.causal_conv1d_update = bindings["causal_conv1d_update"]
        self._axolotl_cuda_forward = _bind(self.cuda_kernels_forward, self, bindings)
        self.cuda_kernels_forward = _bind(
            MethodType(_adapter_cuda_forward, self), self, bindings
        )

    def _forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        if cache_params is not None and cache_params.layers[self.layer_idx].record_past:
            raise ValueError("FLA Mamba cached decoding does not support record_past")
        previous = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx
        )
        last_state = None
        if previous:
            layer = cache_params.layers[self.layer_idx]
            if hidden_states.shape[1] != 1:
                raise ValueError("FLA Mamba supports single-token cached decoding")
            last_state = {
                "conv_state": layer.conv_states[0],
                "recurrent_state": layer.recurrent_states[0],
            }
        if hidden_states.is_cuda:
            if not hasattr(self, "_axolotl_cuda_forward"):
                self._bind_kernels()
            forward = self.cuda_kernels_forward
        else:
            forward = (
                self.slow_forward if self._family == "mamba" else self.torch_forward
            )
        output, conv_state, recurrent_state = forward(
            hidden_states, last_state, cache_params is not None, attention_mask
        )
        if cache_params is not None:
            if not previous:
                cache_params.update_conv_state(conv_state, self.layer_idx)
            else:
                cache_params.layers[self.layer_idx].conv_states[0].copy_(conv_state)
            cache_params.update_recurrent_state(recurrent_state, self.layer_idx)
        return output, None, None


class FlaMambaMixerView(_FlaMixerView, Mamba):
    pass


class FlaMamba2MixerView(_FlaMixerView, Mamba2):
    pass

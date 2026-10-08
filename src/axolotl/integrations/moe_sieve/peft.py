"""PEFT compatibility boundary for compact packed-expert adapters."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import ParamWrapper as PeftParamWrapper, _LoraParameterProxy
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .selection import validate_selection


@dataclass
class MoeSieveLoraConfig(LoraConfig):
    """Store selection inside every PEFT adapter checkpoint."""

    moe_sieve_selection: dict = field(default_factory=dict)
    moe_sieve_version: int = 1

    @classmethod
    def from_peft_type(cls, **kwargs):
        return cls(**kwargs)


class _SelectedFactorsProxy(_LoraParameterProxy):
    def __init__(self, indices, lhs, rhs, scaling):
        nn.Module.__init__(self)
        self.indices = indices
        self.lhs = lhs
        self.rhs = rhs
        self.scaling = scaling

    def forward(self, weight):
        with torch.autocast(device_type=weight.device.type, enabled=False):
            delta = (self.lhs @ self.rhs) * self.scaling
            return weight.index_add(0, self.indices, delta.to(weight.dtype))


# PEFT checks this exact class name when nesting wrappers for multiple parameters.
class ParamWrapper(PeftParamWrapper):
    """LoRA factors for K selected experts while retaining all E base experts."""

    def __init__(self, *args, selected_experts, **kwargs):
        self.selected_experts = tuple(selected_experts)
        self.global_selected_experts = self.selected_experts
        super().__init__(*args, **kwargs)

    def update_layer(self, adapter_name, r, lora_alpha, config, **kwargs):
        if self.lora_A:
            raise ValueError("MoE-Sieve currently supports one adapter per model")
        if config.init_lora_weights not in (True, False, "gaussian"):
            raise ValueError(
                "MoE-Sieve requires default or gaussian LoRA initialization"
            )
        if self._param_ndim != 3:
            raise ValueError("MoE-Sieve requires 3D packed expert weights")
        if self.get_param().dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                "MoE-Sieve requires unquantized floating-point expert weights"
            )
        num_experts = self.num_experts
        try:
            self.num_experts = len(self.selected_experts)
            super().update_layer(adapter_name, r, lora_alpha, config, **kwargs)
        finally:
            self.num_experts = num_experts

    def get_delta_factors(self, adapter_name):
        count = len(self.selected_experts)
        weight_a = self.lora_A[adapter_name].weight
        weight_b = self.lora_B[adapter_name].weight
        if not count:
            weight_a = (
                weight_a.to_local() if hasattr(weight_a, "to_local") else weight_a
            )
            weight_b = (
                weight_b.to_local() if hasattr(weight_b, "to_local") else weight_b
            )
        rank = self.r[adapter_name]
        weight_a = weight_a.reshape(count, rank, self.in_features)
        weight_b = weight_b.reshape(self.out_features, rank, count).permute(2, 0, 1)
        if self._did_swap_in_out_features:
            lhs, rhs = weight_b, weight_a
        else:
            lhs, rhs = weight_a.transpose(-2, -1), weight_b.transpose(-2, -1)
        dtype = self.get_param().dtype
        return lhs.to(dtype), rhs.to(dtype), self.scaling[adapter_name]

    def get_delta_weight(self, adapter_name, *args, **kwargs):
        lhs, rhs, scaling = self.get_delta_factors(adapter_name)
        param = self.get_param()
        with torch.autocast(device_type=param.device.type, enabled=False):
            delta = (lhs @ rhs) * scaling
        indices = torch.tensor(
            self.selected_experts, device=param.device, dtype=torch.long
        )
        return torch.zeros_like(param).index_add(0, indices, delta)

    def kernel_lora_factors(self, weight_a, weight_b):
        """Supply existing kernels with zero factors for frozen experts."""
        rank = self.r[self.active_adapters[0]]
        count = len(self.selected_experts)
        if not count:
            weight_a = (
                weight_a.to_local() if hasattr(weight_a, "to_local") else weight_a
            )
            weight_b = (
                weight_b.to_local() if hasattr(weight_b, "to_local") else weight_b
            )
        indices = torch.tensor(
            self.selected_experts, device=weight_a.device, dtype=torch.long
        )
        expanded_a = weight_a.new_zeros(self.num_experts, rank, weight_a.shape[1])
        expanded_b = weight_b.new_zeros(weight_b.shape[0], rank, self.num_experts)
        expanded_a = expanded_a.index_copy(
            0, indices, weight_a.reshape(count, rank, weight_a.shape[1])
        )
        expanded_b = expanded_b.index_copy(
            2, indices, weight_b.reshape(weight_b.shape[0], rank, count)
        )
        return expanded_a.flatten(0, 1), expanded_b.flatten(1, 2)

    @contextmanager
    def _activate_lora(self, active_adapters):
        adapters = [name for name in active_adapters if name in self.lora_A]
        if not adapters:
            yield
            return
        if len(adapters) != 1:
            raise ValueError("MoE-Sieve currently supports one active adapter")
        param = self.get_param()
        proxy = _SelectedFactorsProxy(
            torch.tensor(self.selected_experts, device=param.device, dtype=torch.long),
            *self.get_delta_factors(adapters[0]),
        )
        base_layer = self.get_base_layer()
        nn.utils.parametrize.register_parametrization(
            base_layer, self.parameter_name, proxy
        )
        try:
            with nn.utils.parametrize.cached():
                yield
        finally:
            self._remove_parametrizations()


SelectiveExpertParamWrapper = ParamWrapper


def register_selected_experts(model, config):
    """Install config-local dispatch; replace this boundary with a public API later."""
    if config.moe_sieve_version != 1:
        raise ValueError("Unsupported MoE-Sieve adapter format version")
    if getattr(model, "is_quantized", False) or getattr(
        model, "_moe_experts_quantized", False
    ):
        raise ValueError("MoE-Sieve requires an unquantized base model")
    modules = validate_selection(model, config.moe_sieve_selection)
    by_identity = {id(module): name for name, (module, _) in modules.items()}

    def dispatch(target, adapter_name, config, parameter_name=None, **kwargs):
        base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        name = by_identity.get(id(base))
        if name is None or parameter_name is None:
            raise ValueError("MoE-Sieve dispatch requires a selected expert parameter")
        selected = config.moe_sieve_selection[name]["selected_experts"]
        offset = getattr(base, "local_expert_offset", 0)
        local = [
            i - offset for i in selected if offset <= i < offset + base.num_experts
        ]
        wrapper = SelectiveExpertParamWrapper(
            target,
            adapter_name,
            parameter_name=parameter_name,
            config=config,
            selected_experts=local,
            **kwargs,
        )
        wrapper.global_selected_experts = tuple(selected)
        return wrapper

    config._register_custom_module(
        {type(module): dispatch for module, _ in modules.values()}
    )

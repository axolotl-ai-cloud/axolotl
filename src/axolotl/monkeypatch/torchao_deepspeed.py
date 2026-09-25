"""DeepSpeed preparation for frozen native TorchAO NVFP4 parameters."""

from __future__ import annotations

import types

import torch

from axolotl.monkeypatch.torchao_ddp import prepare_native_nvfp4_components


def _native_deepspeed_names(model):
    return frozenset(getattr(model, "_axolotl_native_nvfp4_deepspeed_names", ()))


def _native_deepspeed_parameters(model, names):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    native = {name: parameters[name] for name in names if name in parameters}
    component_names = frozenset(
        getattr(model, "_axolotl_native_nvfp4_zero3_components", ())
    )
    valid = (
        set(names) == component_names
        and all(parameter.dtype == torch.uint8 for parameter in native.values())
        if component_names
        else all(
            type(parameter).__name__ == "NVFP4Tensor" for parameter in native.values()
        )
    )
    if (
        set(native) != set(names)
        or not valid
        or any(parameter.requires_grad for parameter in native.values())
    ):
        raise ValueError("Native NVFP4 DeepSpeed parameters must remain frozen")
    return native


def _install_native_nvfp4_broadcast_filter(engine_class=None) -> None:
    if engine_class is None:
        from deepspeed.runtime.engine import DeepSpeedEngine

        engine_class = DeepSpeedEngine
    DeepSpeedEngine = engine_class
    from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload

    from axolotl.monkeypatch.torchao_deepspeed_compat import (
        install_native_nvfp4_debug_compat,
        install_native_nvfp4_zero3_dtype_compat,
    )

    install_native_nvfp4_debug_compat(DeepSpeedEngine)
    install_native_nvfp4_zero3_dtype_compat(DeepSpeedZeRoOffload)
    if getattr(DeepSpeedEngine, "_axolotl_native_nvfp4_broadcast_filter", False):
        return
    original_broadcast_model = DeepSpeedEngine._broadcast_model

    def broadcast_model(self):
        names = _native_deepspeed_names(self.module)
        if not names:
            return original_broadcast_model(self)
        _native_deepspeed_parameters(self.module, names)
        sentinel = object()
        previous = self.module.__dict__.get("named_parameters", sentinel)
        original_named_parameters = self.module.named_parameters

        def filtered_named_parameters(*args, **kwargs):
            return (
                (name, parameter)
                for name, parameter in original_named_parameters(*args, **kwargs)
                if name not in names
            )

        self.module.named_parameters = filtered_named_parameters
        try:
            return original_broadcast_model(self)
        finally:
            if previous is sentinel:
                del self.module.named_parameters
            else:
                self.module.named_parameters = previous

    if hasattr(DeepSpeedEngine, "module_state_dict"):
        original_module_state = DeepSpeedEngine.module_state_dict

        def module_state_dict(
            self,
            destination=None,
            prefix="",
            keep_vars=False,
            exclude_frozen_parameters=False,
        ):
            state = original_module_state(
                self,
                destination=destination,
                prefix=prefix,
                keep_vars=keep_vars,
                exclude_frozen_parameters=exclude_frozen_parameters,
            )
            if exclude_frozen_parameters and getattr(
                self.module, "_axolotl_native_nvfp4_deepspeed_names", ()
            ):
                for name, parameter in self.module.named_parameters(
                    remove_duplicate=False
                ):
                    if not parameter.requires_grad:
                        state.pop(prefix + name, None)
            return state

        DeepSpeedEngine.module_state_dict = module_state_dict

    if hasattr(DeepSpeedEngine, "load_module_state_dict"):
        original_load = DeepSpeedEngine.load_module_state_dict

        def load_module_state_dict(
            self,
            checkpoint,
            strict=True,
            custom_load_fn=None,
            fetch_z3_params=False,
            *args,
            **kwargs,
        ):
            native_names = _native_deepspeed_names(self.module)
            if native_names:
                canonical = _native_deepspeed_parameters(self.module, native_names)
                native_ids = {id(parameter) for parameter in canonical.values()}
                native_aliases = {
                    name
                    for name, parameter in self.module.named_parameters(
                        remove_duplicate=False
                    )
                    if id(parameter) in native_ids
                }
                if native_aliases & checkpoint["module"].keys():
                    raise ValueError(
                        "Adapter checkpoint contains frozen native parameters"
                    )
                if fetch_z3_params and not getattr(
                    self.module, "_axolotl_native_nvfp4_zero3_components", ()
                ):
                    raise ValueError(
                        "Native NVFP4 adapter restore does not support ZeRO-3"
                    )
                validate_native_nvfp4_adapter_state(self.module, checkpoint["module"])
                if custom_load_fn is None:
                    custom_load_fn = lambda src, dst: load_native_nvfp4_adapter_state(
                        dst, src
                    )
            return original_load(
                self,
                checkpoint,
                strict,
                custom_load_fn,
                fetch_z3_params,
                *args,
                **kwargs,
            )

        DeepSpeedEngine.load_module_state_dict = load_module_state_dict
    DeepSpeedEngine._broadcast_model = broadcast_model
    DeepSpeedEngine._axolotl_native_nvfp4_broadcast_filter = True


def _zero3_reconstruct_native_weight(module):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    return NVFP4Tensor(
        module._axolotl_nvfp4_qdata,
        module._axolotl_nvfp4_scale_bytes.view(torch.float8_e4m3fn),
        module._axolotl_nvfp4_block_size,
        module._axolotl_nvfp4_orig_dtype,
        None
        if module._axolotl_nvfp4_per_tensor_scale_bytes is None
        else module._axolotl_nvfp4_per_tensor_scale_bytes.view(
            module._axolotl_nvfp4_per_tensor_scale_dtype
        ).reshape(module._axolotl_nvfp4_per_tensor_scale_shape),
        None
        if module._axolotl_nvfp4_act_per_tensor_scale_bytes is None
        else module._axolotl_nvfp4_act_per_tensor_scale_bytes.view(
            module._axolotl_nvfp4_act_per_tensor_scale_dtype
        ).reshape(module._axolotl_nvfp4_act_per_tensor_scale_shape),
        module._axolotl_nvfp4_is_swizzled_scales,
        module._axolotl_nvfp4_use_triton_kernel,
        module._axolotl_nvfp4_act_quant_kwargs,
    )


class _NVFP4Zero3Linear(torch.autograd.Function):
    """Native static or dynamic NVFP4 forward with dequantized-weight input STE."""

    @staticmethod
    def forward(ctx, inputs, bias, module):
        output = torch.nn.functional.linear(
            inputs, _zero3_reconstruct_native_weight(module), bias
        )
        ctx.module = module
        ctx.bias_requires_grad = bias is not None and bias.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight = _zero3_reconstruct_native_weight(ctx.module).dequantize()
        grad_inputs = grad_output.matmul(weight)
        grad_bias = None
        if ctx.bias_requires_grad:
            grad_bias = (
                grad_output
                if grad_output.ndim == 1
                else grad_output.sum(tuple(range(grad_output.ndim - 1)))
            )
        return grad_inputs, grad_bias, None


def _zero3_native_forward(module, inputs):
    return _NVFP4Zero3Linear.apply(inputs, module.bias, module)


def _install_zero3_native_components(module, weight, source=None) -> tuple[str, str]:
    import torch

    if weight.ndim != 2:
        raise ValueError("ZeRO-3 native NVFP4 supports dense 2-D weights only")
    module._parameters.pop("weight")
    module.weight = torch.empty(weight.shape, dtype=weight.orig_dtype, device="meta")
    if source is None:
        qdata = torch.nn.Parameter(weight.qdata.detach(), False)
        scale_bytes = torch.nn.Parameter(weight.scale.detach().view(torch.uint8), False)
        per_tensor_scale_bytes = (
            None
            if weight.per_tensor_scale is None
            else weight.per_tensor_scale.detach().reshape(-1).view(torch.uint8)
        )
        per_tensor_scale_dtype = (
            None if weight.per_tensor_scale is None else weight.per_tensor_scale.dtype
        )
        per_tensor_scale_shape = (
            None
            if weight.per_tensor_scale is None
            else tuple(weight.per_tensor_scale.shape)
        )
        act_per_tensor_scale_bytes = (
            None
            if weight.act_per_tensor_scale is None
            else weight.act_per_tensor_scale.detach().reshape(-1).view(torch.uint8)
        )
        act_per_tensor_scale_dtype = (
            None
            if weight.act_per_tensor_scale is None
            else weight.act_per_tensor_scale.dtype
        )
        act_per_tensor_scale_shape = (
            None
            if weight.act_per_tensor_scale is None
            else tuple(weight.act_per_tensor_scale.shape)
        )
        act_quant_kwargs = weight.act_quant_kwargs
    else:
        qdata = source._axolotl_nvfp4_qdata
        scale_bytes = source._axolotl_nvfp4_scale_bytes
        per_tensor_scale_bytes = source._axolotl_nvfp4_per_tensor_scale_bytes
        per_tensor_scale_dtype = source._axolotl_nvfp4_per_tensor_scale_dtype
        per_tensor_scale_shape = source._axolotl_nvfp4_per_tensor_scale_shape
        act_per_tensor_scale_bytes = source._axolotl_nvfp4_act_per_tensor_scale_bytes
        act_per_tensor_scale_dtype = source._axolotl_nvfp4_act_per_tensor_scale_dtype
        act_per_tensor_scale_shape = source._axolotl_nvfp4_act_per_tensor_scale_shape
        act_quant_kwargs = source._axolotl_nvfp4_act_quant_kwargs
    module.register_parameter("_axolotl_nvfp4_qdata", qdata)
    module.register_parameter("_axolotl_nvfp4_scale_bytes", scale_bytes)
    module.register_buffer(
        "_axolotl_nvfp4_per_tensor_scale_bytes",
        per_tensor_scale_bytes,
        persistent=False,
    )
    module._axolotl_nvfp4_per_tensor_scale_dtype = per_tensor_scale_dtype
    module._axolotl_nvfp4_per_tensor_scale_shape = per_tensor_scale_shape
    module.register_buffer(
        "_axolotl_nvfp4_act_per_tensor_scale_bytes",
        act_per_tensor_scale_bytes,
        persistent=False,
    )
    module._axolotl_nvfp4_act_per_tensor_scale_dtype = act_per_tensor_scale_dtype
    module._axolotl_nvfp4_act_per_tensor_scale_shape = act_per_tensor_scale_shape
    module._axolotl_nvfp4_act_quant_kwargs = act_quant_kwargs
    module._axolotl_nvfp4_block_size = weight.block_size
    module._axolotl_nvfp4_orig_dtype = weight.orig_dtype
    module._axolotl_nvfp4_is_swizzled_scales = weight.is_swizzled_scales
    module._axolotl_nvfp4_use_triton_kernel = weight.use_triton_kernel
    module.forward = types.MethodType(_zero3_native_forward, module)
    return ("_axolotl_nvfp4_qdata", "_axolotl_nvfp4_scale_bytes")


def prepare_native_nvfp4_zero3(model, device) -> bool:
    """Replace dense native weights with raw byte ZeRO-3 components."""
    del device
    weights = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if type(parameter).__name__ == "NVFP4Tensor"
    ]
    if not weights:
        return False
    if any(parameter.requires_grad or parameter.ndim != 2 for _, parameter in weights):
        return False
    modules = dict(model.named_modules(remove_duplicate=False))
    targets = []
    for name, weight in weights:
        module_name, separator, parameter_name = name.rpartition(".")
        if not separator:
            module_name, parameter_name = "", name
        module = modules.get(module_name)
        if parameter_name != "weight" or not isinstance(module, torch.nn.Linear):
            return False
        targets.append((module_name, module, weight))
    names = []
    sources = {}
    installed = {}
    for module_name, module, weight in targets:
        source = sources.get(id(weight))
        component_names = installed.get(id(module))
        if component_names is None:
            component_names = _install_zero3_native_components(module, weight, source)
            installed[id(module)] = component_names
        sources.setdefault(id(weight), module)
        names.extend(
            f"{module_name + '.' if module_name else ''}{component}"
            for component in component_names
        )
    model._axolotl_native_nvfp4_deepspeed_names = frozenset(names)
    model._axolotl_native_nvfp4_zero3_components = frozenset(names)
    _install_native_nvfp4_broadcast_filter()
    return True


def prepare_native_nvfp4_deepspeed(model, device, zero_stage: int) -> bool:
    """Prepare frozen native NVFP4 parameters before a supported DeepSpeed engine."""
    if zero_stage == 3:
        return prepare_native_nvfp4_zero3(model, device)
    if zero_stage not in (1, 2):
        return False
    names = prepare_native_nvfp4_components(model, device)
    if not names:
        return False
    model._axolotl_native_nvfp4_deepspeed_names = frozenset(names)
    _install_native_nvfp4_broadcast_filter()
    return True


def _required_adapter_state(model):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    model_keys = set(model.state_dict())
    required = model_keys - parameters.keys()
    required.update(
        name for name, parameter in parameters.items() if parameter.requires_grad
    )
    return required, model_keys


def validate_native_nvfp4_adapter_state(model, state_dict) -> set[str]:
    """Reject missing trainables before loading an adapter checkpoint."""
    expected, model_keys = _required_adapter_state(model)
    saved = set(state_dict)
    missing = expected - saved
    unexpected = saved - model_keys
    if missing or unexpected:
        raise ValueError(
            "Native NVFP4 DeepSpeed adapter checkpoint keys do not match trainable parameters: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return expected


def load_native_nvfp4_adapter_state(model, state_dict) -> None:
    """Restore trainables and buffers while allowing a separately loaded frozen base."""
    expected = validate_native_nvfp4_adapter_state(model, state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    missing_trainable = set(result.missing_keys) & expected
    if missing_trainable or result.unexpected_keys:
        raise ValueError(
            "Native NVFP4 DeepSpeed adapter checkpoint did not restore strictly"
        )


def native_nvfp4_zero3_peft_state_dict(model, *, collect_on_this_rank=True):
    """Collect the selected PEFT adapter state without gathering frozen base tensors."""
    if not getattr(model, "_axolotl_native_nvfp4_zero3_components", ()):
        return None
    try:
        import deepspeed
        import torch.distributed as dist
        from peft.utils.save_and_load import get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError(
            "Native NVFP4 ZeRO-3 export requires DeepSpeed and PEFT"
        ) from exc

    active = getattr(model, "active_adapter", "default")
    if isinstance(active, (list, tuple)):
        if len(active) != 1:
            raise ValueError(
                "Native NVFP4 ZeRO-3 export supports one active PEFT adapter"
            )
        active = active[0]
    if set(model.peft_config) != {active}:
        raise ValueError(
            "Native NVFP4 ZeRO-3 export supports one configured PEFT adapter"
        )

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    named_buffers = dict(model.named_buffers(remove_duplicate=False))
    component_names = frozenset(
        getattr(model, "_axolotl_native_nvfp4_zero3_components", ())
    )
    local_state = {
        name: value
        for name, value in model.state_dict(keep_vars=True).items()
        if name not in component_names
    }
    selected = get_peft_model_state_dict(
        model, state_dict=local_state, adapter_name=active
    )
    source_by_value = {}
    for name, value in local_state.items():
        source_by_value.setdefault(id(value), []).append(name)
    source_names = []
    for name, value in selected.items():
        aliases = source_by_value.get(id(value))
        if aliases is None:
            raise ValueError(f"Could not map PEFT export key {name!r} to a parameter")
        source_names.extend(aliases)
    if not source_names:
        raise ValueError("Native NVFP4 ZeRO-3 adapter state is empty")

    full_state = {}
    sources_by_parameter = {}
    for name in sorted(set(source_names)):
        parameter = named_parameters.get(name)
        buffer = named_buffers.get(name)
        source = parameter if parameter is not None else buffer
        if source is None:
            raise ValueError(f"PEFT export key {name!r} is not model state")
        sources_by_parameter.setdefault(id(source), (source, []))[1].append(name)
    for source, aliases in sources_by_parameter.values():
        if hasattr(source, "ds_id"):
            with deepspeed.zero.GatheredParameters(source):
                if source.numel() != source.ds_numel:
                    raise ValueError(
                        f"ZeRO-3 did not gather adapter parameter {aliases[0]!r}"
                    )
                if collect_on_this_rank:
                    value = source.detach().cpu().clone()
        elif collect_on_this_rank:
            value = source.detach().cpu().clone()
        if collect_on_this_rank:
            full_state.update(dict.fromkeys(aliases, value))
    if dist.is_initialized():
        dist.barrier()
    error = None
    if collect_on_this_rank:
        try:
            adapter_state = get_peft_model_state_dict(
                model, state_dict=full_state, adapter_name=active
            )
            if adapter_state.keys() != selected.keys():
                raise ValueError(
                    "Native NVFP4 ZeRO-3 adapter export key set changed after gather"
                )
        except Exception as exc:  # pylint: disable=broad-except
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((item for item in errors if item is not None), None)
    if error is not None:
        raise RuntimeError(f"Native NVFP4 ZeRO-3 adapter collection failed: {error}")
    return full_state if collect_on_this_rank else {}

# Copyright 2026 Axolotl AI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Shard bitsandbytes parametrized experts without requantizing their weights."""

import math
import os

import torch
import torch.distributed as dist
import torch.nn.utils.parametrize as P


def expert_storage(module, name):
    parametrizations = getattr(module, "parametrizations", {})
    if name in parametrizations:
        stack = parametrizations[name]
        for entry in stack:
            if type(entry).__name__ == "Bnb4bitParametrization":
                return stack.original, tuple(entry.quant_state.shape), entry
            if type(entry).__name__ == "Bnb8bitParametrization":
                return stack.original, tuple(stack.original.shape), entry
    parameter = getattr(module, name, None)
    return parameter, tuple(parameter.shape) if torch.is_tensor(parameter) else (), None


def scatter_quantized_expert(module, name, e_local, ep_rank_of):
    rank = dist.get_rank()
    original, shape, parametrization = expert_storage(module, name)
    header, error, scales = None, None, None
    if rank == 0 and parametrization is not None:
        try:
            if type(parametrization).__name__ == "Bnb4bitParametrization":
                import bitsandbytes.functional as F

                state = parametrization.quant_state
                local_values = e_local * math.prod(shape[1:])
                if local_values % state.blocksize:
                    raise ValueError(
                        "EP expert slice cuts a bitsandbytes weight quantization block"
                    )
                if original.dtype != torch.uint8:
                    raise ValueError(
                        "Parametrized EP experts require uint8 packed 4-bit storage"
                    )
                metadata = state.as_dict(packed=False)
                metadata.pop("absmax")
                metadata.pop("nested_absmax", None)
                scales = state.absmax
                # A cut through a nested scale block only requires unpacking scales, not weights.
                nested = (
                    state.nested
                    and (local_values // state.blocksize) % state.state2.blocksize == 0
                )
                if state.nested and not nested:
                    scales = (
                        F.dequantize_blockwise(state.absmax, state.state2)
                        + state.offset
                    )
                    metadata = {
                        k: v for k, v in metadata.items() if not k.startswith("nested_")
                    }
                metadata = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in metadata.items()
                }
                header = dict(
                    mode="4bit",
                    shape=shape,
                    metadata=metadata,
                    scale_dtype=str(scales.dtype),
                    nested=nested,
                    device_type=original.device.type,
                )
            else:
                header = dict(
                    mode="8bit", shape=shape, device_type=original.device.type
                )
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
    message = [(header, error)]
    dist.broadcast_object_list(message, src=0)
    header, error = message[0]
    if error:
        raise ValueError(error)
    if header is None:
        return False
    device = original.device
    if device.type == "meta" or device.type != header["device_type"]:
        device = (
            torch.device(
                "cuda", int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            )
            if header["device_type"] == "cuda"
            else torch.device("cpu")
        )
    shape = (e_local, *header["shape"][1:])

    def scatter(component, count, dtype):
        output = torch.empty(count, dtype=dtype, device=device)
        chunks = None
        if rank == 0:
            flat = component.reshape(-1)
            chunks = [
                flat[ep * count : (ep + 1) * count].contiguous().to(device)
                for ep in ep_rank_of
            ]
        dist.scatter(output, scatter_list=chunks, src=0)
        return output

    if header["mode"] == "4bit":
        import bitsandbytes.functional as F
        from bitsandbytes.nn.parametrize import (
            Bnb4bitParametrization,
            _register_parametrization_hooks,
        )

        metadata = dict(header["metadata"], shape=shape)
        values = math.prod(shape)
        packed = scatter(
            original if rank == 0 else None, values // 2, torch.uint8
        ).reshape(-1, 1)
        metadata["absmax"] = scatter(
            scales,
            values // metadata["blocksize"],
            getattr(torch, header["scale_dtype"].removeprefix("torch.")),
        )
        if header["nested"]:
            metadata["nested_absmax"] = scatter(
                parametrization.quant_state.state2.absmax if rank == 0 else None,
                values // metadata["blocksize"] // metadata["nested_blocksize"],
                torch.float32,
            )
        new_parametrization = Bnb4bitParametrization(
            F.QuantState.from_dict(metadata, device=device)
        )
    else:
        from axolotl.monkeypatch.moe_quant import Bnb8bitParametrization

        packed = scatter(
            original if rank == 0 else None, math.prod(shape), torch.int8
        ).reshape(shape)
        stats = scatter(
            parametrization.row_stats if rank == 0 else None,
            math.prod(shape[:-1]),
            torch.float32,
        )
        new_parametrization = Bnb8bitParametrization(stats)
    new_parameter = torch.nn.Parameter(packed, requires_grad=False)
    if parametrization is not None:
        module.parametrizations[name].original = new_parameter
        if header["mode"] == "4bit":
            parametrization.quant_state = new_parametrization.quant_state
        else:
            parametrization.row_stats = new_parametrization.row_stats
    else:
        setattr(module, name, new_parameter)
        P.register_parametrization(module, name, new_parametrization, unsafe=True)
        if header["mode"] == "4bit":
            _register_parametrization_hooks(module, name)
        else:
            from axolotl.monkeypatch.moe_quant import (
                _register_parametrization_cache_hooks,
            )

            if not getattr(module, "_axolotl_8bit_hooks_registered", False):
                _register_parametrization_cache_hooks(module)
                module._axolotl_8bit_hooks_registered = True
    P._cache.pop((id(module), name), None)
    return True

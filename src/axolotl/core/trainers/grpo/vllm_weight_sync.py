"""Push trainer weights to a `vllm serve` server over TRL's VLLMClient."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import wraps

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

WeightMetadata = list[tuple[str, str, list[int]]]


def init_communicator_lazily(vllm_client, device) -> None:
    """Defer the NCCL rendezvous from client construction to the first weight push.

    Opening the communicator while the trainer is still being built makes rank 0
    allocate device state the other DDP ranks don't have.
    """
    update_named_params = vllm_client.update_named_params

    @wraps(update_named_params)
    def _update_named_params(*args, **kwargs):
        if vllm_client.communicator is None:
            vllm_client.init_communicator(device=device)
        return update_named_params(*args, **kwargs)

    vllm_client.update_named_params = _update_named_params


def quantize_fp8(
    weight: torch.Tensor, scale_inv_like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `axolotl.kernels.quantize.dequantize_fp8`: fresh scales in the
    layout of `scale_inv_like`, with ceil-div tail blocks."""
    weight = weight.float()
    if scale_inv_like.numel() == 1:
        scale_inv = weight.abs().amax().clamp(min=1e-12) / FP8_MAX
        quantized = weight / scale_inv
        scale_inv = scale_inv.reshape(scale_inv_like.shape)
    elif scale_inv_like.dim() == 2 and weight.dim() == 2:
        sr, sc = scale_inv_like.shape
        rows, cols = weight.shape
        br, bc = -(-rows // sr), -(-cols // sc)
        padded = torch.nn.functional.pad(weight, (0, sc * bc - cols, 0, sr * br - rows))
        blocks = padded.reshape(sr, br, sc, bc)
        scale_inv = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX
        quantized = (blocks / scale_inv[:, None, :, None]).reshape(padded.shape)
        quantized = quantized[:rows, :cols]
    else:
        raise NotImplementedError(
            f"Unsupported FP8 scale layout {tuple(scale_inv_like.shape)} for weight "
            f"{tuple(weight.shape)}"
        )
    quantized = quantized.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return quantized, scale_inv.to(scale_inv_like.dtype)


@dataclass
class _SyncEntry:
    name: str
    param_name: str
    param: torch.Tensor
    lora: tuple[nn.Module, str] | None = None
    # (vLLM name, current scale) when the LoRA-merged weight is requantized to FP8
    fp8_scale: tuple[str, torch.Tensor] | None = None


def _vllm_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@torch.no_grad()
def _merged_tensors(
    entry: _SyncEntry, layer: nn.Module, adapter: str
) -> list[tuple[str, torch.Tensor]]:
    lora_a = layer.lora_A[adapter].weight
    lora_b = layer.lora_B[adapter].weight
    delta = (lora_b @ lora_a) * layer.scaling[adapter]
    if getattr(layer, "fan_in_fan_out", False):
        delta = delta.T
    if entry.fp8_scale is None:
        return [(entry.name, entry.param.data + delta.to(entry.param.dtype))]

    from axolotl.kernels.quantize import dequantize_fp8

    scale_name, scale_inv = entry.fp8_scale
    merged = dequantize_fp8(entry.param.data, scale_inv, torch.bfloat16)
    weight, new_scale_inv = quantize_fp8(merged + delta.to(torch.bfloat16), scale_inv)
    return [(entry.name, weight), (scale_name, new_scale_inv)]


def peft_weights_for_vllm(
    model: nn.Module, fix_name
) -> tuple[WeightMetadata, Iterator[tuple[str, torch.Tensor]]]:
    """Every base weight of a PEFT model, under vLLM's names, with active LoRA
    deltas folded in out of place so base weights are never modified.

    All tensors are sent, not just the LoRA-touched ones: vLLM's layerwise reload
    materializes any fused layer (qkv_proj, gate_up_proj) that receives only
    some of its tensors from uninitialized memory.

    Returns the `(name, dtype, shape)` metadata `VLLMClient.update_named_params`
    announces up front, and an iterator that materializes one tensor at a time.
    """
    lora_layers: dict[str, tuple[nn.Module, str]] = {}
    for module_name, module in model.base_model.model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "active_adapters"):
            continue
        adapter = module.active_adapters[0]
        if adapter not in module.lora_A:
            continue
        if module.use_dora.get(adapter, False):
            raise NotImplementedError(
                "Out-of-place weight sync doesn't support DoRA; set "
                "`trl.vllm_lora_sync: true` to load the adapter in vLLM instead."
            )
        lora_layers[module_name] = (module, adapter)

    params = dict(model.named_parameters())
    fp8_lora_scales: set[str] = set()
    entries: list[_SyncEntry] = []
    for param_name, param in params.items():
        raw_name = param_name.removeprefix("base_model.model.").replace(
            ".base_layer", ""
        )
        if model.prefix in raw_name or "original_module" in raw_name:
            continue
        if param.__class__.__name__ == "Params4bit":
            raise NotImplementedError(
                "Out-of-place weight sync can't stream bitsandbytes 4-bit weights; "
                "set `trl.vllm_lora_sync: true` for QLoRA."
            )
        entry = _SyncEntry(
            name=fix_name(raw_name, extra_prefixes=["modules_to_save.default."]),
            param_name=param_name,
            param=param,
        )
        if raw_name.endswith(".weight") and raw_name[: -len(".weight")] in lora_layers:
            entry.lora = lora_layers[raw_name[: -len(".weight")]]
            if param.dtype == FP8_DTYPE:
                scale_param = param_name.removesuffix(".weight") + ".weight_scale_inv"
                entry.fp8_scale = (entry.name + "_scale_inv", params[scale_param].data)
                fp8_lora_scales.add(scale_param)
        entries.append(entry)
    # Streamed right after their weight, with the requantized values.
    entries = [e for e in entries if e.param_name not in fp8_lora_scales]

    metadata: WeightMetadata = []
    for entry in entries:
        metadata.append(
            (entry.name, _vllm_dtype(entry.param.dtype), list(entry.param.shape))
        )
        if entry.fp8_scale is not None:
            scale_name, scale_inv = entry.fp8_scale
            metadata.append(
                (scale_name, _vllm_dtype(scale_inv.dtype), list(scale_inv.shape))
            )

    def stream() -> Iterator[tuple[str, torch.Tensor]]:
        for entry in entries:
            if entry.lora is None:
                yield entry.name, entry.param.data
            else:
                yield from _merged_tensors(entry, *entry.lora)

    return metadata, stream()


def load_lora_adapter(vllm_client, adapter_path: str, timeout: float) -> bool:
    """(Re)load a LoRA adapter on the server under the served model's name.

    vLLM resolves a request's `model` against loaded adapters before the base
    model, so every later request for that name, from the trainer or from an
    agent server, generates with the adapter.
    """
    response = vllm_client.session.post(
        f"{vllm_client.base_url}/v1/load_lora_adapter",
        json={
            "lora_name": vllm_client.model,
            "lora_path": adapter_path,
            "load_inplace": True,
        },
        timeout=timeout,
    )
    if response.status_code == 200:
        return True
    LOG.warning(
        "Failed to load LoRA adapter into vLLM: %s %s. The server must run with "
        "`--enable-lora` and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 (`axolotl vllm-serve` "
        "sets both when `trl.vllm_lora_sync: true`).",
        response.status_code,
        response.text,
    )
    return False

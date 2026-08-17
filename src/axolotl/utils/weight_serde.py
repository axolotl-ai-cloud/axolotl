"""Serialize / deserialize tensors for HTTP and IPC weight sync.

NumPy doesn't support bfloat16, so bf16 tensors are cast to fp16 on the wire
and reconstructed at the destination.  All encode/decode helpers live here so
the logic isn't duplicated across trl_vllm.py, vllm_serve_lora.py, and
vllm_worker_ext.py.
"""

import base64

import torch


def encode_for_http(name: str, weight: torch.Tensor) -> dict:
    """Encode a named parameter for JSON transport over HTTP.

    Returns a dict with keys: name, dtype (original), shape, data (base64).
    bf16 tensors are sent as fp16 bytes; the original dtype is preserved in
    the ``dtype`` field so the receiver can cast back.
    """
    w_cpu = weight.contiguous().cpu()
    orig_dtype = str(weight.dtype)
    if w_cpu.dtype == torch.bfloat16:
        w_cpu = w_cpu.half()
    raw = w_cpu.numpy().tobytes()
    return {
        "name": name,
        "dtype": orig_dtype,
        "shape": list(weight.shape),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_from_http(entry: dict) -> tuple[str, torch.Tensor]:
    """Decode an HTTP-encoded weight entry back to a named tensor.

    Infers wire dtype from byte count (bf16 arrives as fp16) and casts to the
    original dtype stored in ``entry["dtype"]``.
    """
    target_dtype = getattr(torch, entry["dtype"].split(".")[-1])
    shape = tuple(entry["shape"])
    raw = base64.b64decode(entry["data"])

    n_elements = 1
    for s in shape:
        n_elements *= s
    wire_bytes_per_elem = len(raw) // max(n_elements, 1)
    if wire_bytes_per_elem == 2:
        wire_dtype = torch.float16
    elif wire_bytes_per_elem == 4:
        wire_dtype = torch.float32
    else:
        wire_dtype = target_dtype

    weight = torch.frombuffer(bytearray(raw), dtype=wire_dtype).reshape(shape)
    if wire_dtype != target_dtype:
        weight = weight.to(target_dtype)
    return entry["name"], weight


def encode_for_ipc(name: str, weight: torch.Tensor) -> dict:
    """Encode a tensor for vLLM's multiproc IPC (raw bytes, no base64).

    Returns a dict with keys: name, data (bytes), dtype (wire), target_dtype
    (original), shape.  bf16 tensors are serialized as fp16.
    """
    w = weight.contiguous()
    target_dtype = str(w.dtype).split(".")[-1]
    if w.dtype == torch.bfloat16:
        w = w.half()
    wire_dtype = str(w.dtype).split(".")[-1]
    return {
        "name": name,
        "data": w.numpy().tobytes(),
        "dtype": wire_dtype,
        "target_dtype": target_dtype,
        "shape": list(weight.shape),
    }


def decode_from_ipc(entry: dict) -> tuple[str, torch.Tensor]:
    """Decode an IPC-encoded weight entry back to a named tensor.

    Handles optional ``target_dtype`` for backward compatibility with older
    serve code that may not include it.
    """
    wire_dtype = getattr(torch, entry["dtype"])
    weight = torch.frombuffer(bytearray(entry["data"]), dtype=wire_dtype).reshape(
        entry["shape"]
    )
    target_dtype = entry.get("target_dtype")
    if target_dtype and target_dtype != entry["dtype"]:
        weight = weight.to(getattr(torch, target_dtype))
    return entry["name"], weight

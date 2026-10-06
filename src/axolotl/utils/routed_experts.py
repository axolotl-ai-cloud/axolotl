"""Wire format for vLLM's per-token MoE expert ids, shared by the vLLM server and trainer."""

import base64
import io

import numpy as np


def encode_routed_experts(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def decode_routed_experts(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)

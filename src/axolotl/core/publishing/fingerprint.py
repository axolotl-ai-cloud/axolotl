"""Identify the merge numerics that a delta chain must hold fixed."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

import axolotl
from axolotl.utils.fp32_precision import configured_cuda_matmul_precision

MERGE_SOURCES = (
    "cli/utils/lora_merge.py",
    "cli/utils/native_nvfp4_merge.py",
    "cli/utils/param_wrapper_merge.py",
    "integrations/kernels/libs/sonicmoe/nvfp4_quant.py",
    "utils/fp32_precision.py",
    "integrations/kernels/libs/sonicmoe/nvfp4_lora.py",
)


@lru_cache(maxsize=1)
def merge_code_hash() -> str:
    root = Path(axolotl.__file__).parent
    digest = hashlib.sha256()
    for name in MERGE_SOURCES:
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def float32_precision() -> dict[str, str]:
    """Read the live TF32 state rather than the config that requested it.

    Read on the publisher thread while training may hold ``ieee_fp32_matmul``, so
    the CUDA matmul precision is the configured one, not that override.
    """
    root = getattr(torch.backends, "fp32_precision", None)
    if root is None:
        return {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        }
    # "none" defers to the parent setting, and torch resolves it in place on first
    # use; unset everywhere means IEEE. The legacy matmul precision is derived
    # from these and raises once backends are set apart, so it is not recorded.
    root = "ieee" if str(root) == "none" else str(root)
    state = {"backends": root}
    for name, precision in (
        ("cuda_matmul", configured_cuda_matmul_precision()),
        ("cudnn", str(getattr(torch.backends.cudnn, "fp32_precision", "none"))),
    ):
        state[name] = root if precision == "none" else precision
    return state


def _torchao_version() -> str | None:
    try:
        import torchao
    except ImportError:
        return None
    return getattr(torchao, "__version__", "unknown")


def merge_fingerprint(
    *,
    export: str,
    base_identity: str,
    merge_path: str | None = None,
    merge_compile: bool | None = None,
    merge_compile_precision: str | None = None,
    scale_mode: str | None = None,
    device: str | torch.device | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe everything that can change merged bytes for the same adapter.

    Call at merge time so the TF32 state is the one the merge actually runs under.
    """
    fingerprint: dict[str, Any] = {
        "export": export,
        "base_identity": base_identity,
        "torch": torch.__version__,
    }
    if export != "lora":
        fingerprint.update(
            merge_path=merge_path,
            merge_compile="eager"
            if not merge_compile
            else f"compiled-{merge_compile_precision or 'strict'}",
            scale_mode=scale_mode,
            device=None if device is None else torch.device(device).type,
            float32_precision=float32_precision(),
            torchao=_torchao_version(),
            merge_code=merge_code_hash(),
        )
    if extra:
        fingerprint["extra"] = extra
    return fingerprint


def fingerprint_digest(fingerprint: dict[str, Any] | None) -> str | None:
    if fingerprint is None:
        return None
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

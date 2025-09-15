"""
Placeholder for PyTorch 2.8+ grouped GEMM MoE path.
Currently probes availability; full integration to be implemented.
"""

from __future__ import annotations


def available() -> bool:
    try:
        import torch  # noqa: F401

        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        return ver >= (2, 8)
    except Exception:
        return False

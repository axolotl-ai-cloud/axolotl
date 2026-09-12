"""Frozen NVFP4 W4A16 dense ``Linear`` via the Marlin grouped kernel (single expert).

Non-expert quantization: swap a frozen ``nn.Linear`` for 4-bit NVFP4 weights that run through the
SAME validated Marlin W4A16 kernel the grouped MoE experts use — 4-bit weight memory + bf16 tensor-
core compute on any sm80+ GPU (no FP4 hardware needed). A dense Linear is just the grouped GEMM with
one expert (E=1): all rows route to expert 0, M padded up to the Marlin block (64).

Weights are quantized once (at swap time, before FSDP wrap) via torchao's two-level NVFP4 quant and
prepped into Marlin layout eagerly; the bf16 weight is then dropped. The module is intentionally NOT
an ``nn.Linear`` subclass so PEFT leaves it frozen (non-experts carry no LoRA).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import load_ext
from .prep import (
    marlin_make_workspace_new,
    marlin_moe_gemm,
    prepare_nvfp4_weight_for_marlin,
)

_BSM = 64  # Marlin MoE block size; M is padded to a multiple of this


class MarlinW4A16Linear(nn.Module):
    """Frozen NVFP4 W4A16 replacement for ``nn.Linear`` (bf16 act/out, 4-bit weight)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        self.out_features, self.in_features = weight.shape
        # The Marlin repack is CUDA-only; non-expert weights may still be on CPU at quant time.
        if weight.device.type != "cuda":
            weight = weight.to("cuda")
        dev = weight.device
        ext = load_ext()

        # Two-level NVFP4 quant (per-tensor fp32 + per-16 e4m3 block scales), one [1,N,K] expert.
        # Per-tensor scale needed for accuracy (rel err ~0.10 two-level vs ~0.25 without):
        # pts = amax / (F4_E2M1_MAX * F8E4M3_MAX) = amax / (6 * 448).
        w_bf16 = weight.to(torch.bfloat16)
        pts = (w_bf16.abs().max() / (6.0 * 448.0)).reshape(1).float()
        nv = NVFP4Tensor.to_nvfp4(
            w_bf16[None], per_tensor_scale=pts, is_swizzled_scales=False
        )
        gscale = nv.per_tensor_scale
        qw, sc, g = prepare_nvfp4_weight_for_marlin(
            nv.qdata,
            nv.scale,
            gscale,
            self.out_features,
            self.in_features,
            torch.bfloat16,
            ext.gptq_marlin_repack,
        )
        # frozen -> buffers (move with .to()/.cuda(); FSDP replicates non-experts)
        self.register_buffer("qweight", qw)  # [1, ...] int32 marlin layout
        self.register_buffer("scales", sc)  # [1, ...] bf16
        self.register_buffer("gscale", g)  # [1] fp32
        self.register_buffer(
            "bias", bias.to(torch.bfloat16) if bias is not None else None
        )
        # Device-specific scratch; persistent=False keeps it out of state_dict/checkpoints.
        self.register_buffer(
            "_workspace", marlin_make_workspace_new(dev, 4), persistent=False
        )
        self._route_cache: dict[tuple[int, torch.device], tuple] = {}

    def _route(self, Mt: int, dev: torch.device):
        key = (Mt, dev)
        cached = self._route_cache.get(key)
        if cached is None:
            si = torch.arange(Mt, dtype=torch.int32, device=dev)
            ei = torch.zeros(
                Mt // _BSM, dtype=torch.int32, device=dev
            )  # all -> expert 0
            ntpp = torch.tensor([Mt], dtype=torch.int32, device=dev)
            tw = torch.ones(Mt, 1, dtype=torch.float32, device=dev)
            cached = (si, ei, ntpp, tw)
            self._route_cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_ext()
        orig = x.shape
        xf = x.reshape(-1, self.in_features).to(torch.bfloat16)
        M = xf.shape[0]
        Mt = (M + _BSM - 1) // _BSM * _BSM
        if Mt != M:
            xf = torch.nn.functional.pad(xf, (0, 0, 0, Mt - M))
        si, ei, ntpp, tw = self._route(Mt, xf.device)
        out = marlin_moe_gemm(
            ext,
            xf,
            self.qweight,
            self.scales,
            self.gscale,
            self._workspace,
            si,
            ei,
            ntpp,
            tw,
            _BSM,
            1,
            False,
            Mt,
            self.out_features,
            self.in_features,
        )
        out = out[:M]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4=W4A16-marlin"

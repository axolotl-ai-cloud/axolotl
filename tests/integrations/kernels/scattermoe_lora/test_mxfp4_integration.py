# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
End-to-end integration test for MXFP4 expert weights through the ScatterMoE
LoRA path.

We build a tiny synthetic DeepSeek-V4-style MoE block (E=8, hidden=512,
intermediate=256, top_k=2), MX-quantize the gate/up and down projection
expert weights via ``torchao.MXTensor.to_mx``, then compare two stacks
forward-only:

  1. **Reference** — pure PyTorch per-expert loop using the bf16 dequant of
     the *same* MX weights. This stands in for "stock HF transformers MoE
     with ``Mxfp4Config`` applied" — both stacks read the same physical MX
     packed/scale buffers, so any divergence comes from the Axolotl
     ScatterMoE plumbing (routing flatten/sort, scatter2scatter, fused
     dequant kernel), not from differing weight quantization.

  2. **Axolotl ScatterMoE** — ``parallel_linear_lora`` driven by an
     ``MXWeights`` container, LoRA disabled (A = B = 0). Tests both
     Strategy A (selective dequant to bf16) and Strategy B (fused MX
     Triton kernel) so the spec'd "stock vs scattermoe" parity check
     covers both code paths.

Comparison tolerance is looser than the unit tests (``atol=rtol=5e-3``)
because the per-expert PyTorch reference accumulates in fp32 while the
Triton path emits bf16 outputs whose final cast rounds.
"""

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (
    selective_mx_weights_fwd,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    parallel_linear_lora,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
    get_active_experts,
    remap_expert_indices,
    selective_expert_weights,
    selective_lora_weights,
)

torchao = pytest.importorskip("torchao")
from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MX kernels"
)


# DeepSeek-V4-style tiny config (small enough for fast unit testing)
E = 8
HIDDEN = 512
INTERMEDIATE = 256
TOP_K = 2
M = 16  # batch * seq


def _build_synthetic_moe():
    """Return (gate_up_mx, down_mx, gate_up_ref, down_ref, router_w)
    matching a DeepSeek-V4 expert block:

      * ``gate_up_proj``: per-expert ``[hidden, 2*intermediate]`` (split into
        gate and up halves after the matmul).
      * ``down_proj``:    per-expert ``[intermediate, hidden]``.

    Storage layout matches axolotl's convention ``[E, N, K]`` where K is the
    contraction axis the kernel will block on. ``gate_up`` has K=hidden,
    N=2*intermediate; ``down`` has K=intermediate, N=hidden.

    bf16 reference tensors are the dequantizations of the *same* MX
    buffers, so the only test source of divergence is the kernel paths.
    """
    torch.manual_seed(42)
    # Scale ~ 1/sqrt(fan_in) so per-layer outputs stay in order-1 range and
    # bf16 final-cast noise is not amplified by the magnitude.
    gup_scale = 1.0 / (HIDDEN**0.5)
    down_scale = 1.0 / (INTERMEDIATE**0.5)
    gate_up = (
        torch.randn(E, 2 * INTERMEDIATE, HIDDEN, device=DEVICE, dtype=DTYPE) * gup_scale
    )
    down = torch.randn(E, HIDDEN, INTERMEDIATE, device=DEVICE, dtype=DTYPE) * down_scale

    gate_up_mx = MXTensor.to_mx(
        gate_up, elem_dtype=torch.float4_e2m1fn_x2, block_size=32
    )
    down_mx = MXTensor.to_mx(down, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
    gate_up_ref = gate_up_mx.dequantize(DTYPE).contiguous()
    down_ref = down_mx.dequantize(DTYPE).contiguous()

    router_w = torch.randn(E, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.1
    return gate_up_mx, down_mx, gate_up_ref, down_ref, router_w


def _reference_moe_forward(x, router_w, gate_up_ref, down_ref):
    """Stand-in for stock HF MoE with Mxfp4Config: per-token routing +
    per-expert matmul on dequantized bf16 weights."""
    # Softmax-topk routing
    router_logits = F.linear(x, router_w)  # [M, E]
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected = torch.topk(routing_weights, TOP_K, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    out = torch.zeros_like(x)
    for e in range(E):
        # Tokens routed to expert e: positions (token_id, k_slot)
        mask = selected == e  # [M, TOP_K]
        if not mask.any():
            continue
        token_ids, slot_ids = mask.nonzero(as_tuple=True)
        x_e = x[token_ids]  # [n_e, HIDDEN]
        gup = x_e @ gate_up_ref[e].t()  # [n_e, 2*INTERMEDIATE]
        gate, up = gup.chunk(2, dim=-1)
        h = F.silu(gate) * up
        y_e = h @ down_ref[e].t()  # [n_e, HIDDEN]
        # Weighted accumulate
        w_e = routing_weights[token_ids, slot_ids].unsqueeze(-1)
        out.index_add_(0, token_ids, w_e * y_e)
    return out


class _MockExperts:
    def __init__(self, gate_up, down):
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.num_experts = E


def _axolotl_moe_forward(x, router_w, gate_up_param, down_param, *, strategy: str):
    """Run the Axolotl ScatterMoE LoRA path with LoRA disabled (A=B=0).

    ``strategy='A'``: ``gate_up_param``/``down_param`` are torchao MXTensors;
    we dequantize the active experts to bf16 and call the bf16 kernel.

    ``strategy='B'``: same MXTensors but routed through the fused MX kernel
    via the ``MXWeights`` container.
    """
    # Routing — same softmax+topk shape as the reference
    router_logits = F.linear(x, router_w)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected = torch.topk(routing_weights, TOP_K, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    sei, ssi, eo = flatten_sort_count(selected, num_experts=E)
    active = get_active_experts(sei, E)
    remapped, compact_offsets = remap_expert_indices(sei, eo, active, E)

    # Build LoRA tensors with A=B=0 so the LoRA term is zero.
    rank = 4
    lora_A = torch.zeros(rank * E, HIDDEN, device=DEVICE, dtype=DTYPE)
    lora_B_gup = torch.zeros(2 * INTERMEDIATE, rank * E, device=DEVICE, dtype=DTYPE)
    lora_B_down = torch.zeros(HIDDEN, rank * E, device=DEVICE, dtype=DTYPE)
    lora_A_inter = torch.zeros(rank * E, INTERMEDIATE, device=DEVICE, dtype=DTYPE)
    A_gup_c, B_gup_c = selective_lora_weights(lora_A, lora_B_gup, active, E)
    A_dn_c, B_dn_c = selective_lora_weights(lora_A_inter, lora_B_down, active, E)

    experts = _MockExperts(gate_up_param, down_param)

    if strategy == "A":
        gate_up_W = (
            selective_expert_weights(experts, "gate_up_proj", active)
            .transpose(2, 1)
            .contiguous()
        )
        down_W = (
            selective_expert_weights(experts, "down_proj", active)
            .transpose(2, 1)
            .contiguous()
        )
        gup_W = gate_up_W
        dwn_W = down_W
    elif strategy == "B":
        gup_W = selective_mx_weights_fwd(gate_up_param, active)
        dwn_W = selective_mx_weights_fwd(down_param, active)
    else:
        raise ValueError(strategy)

    gup = parallel_linear_lora(
        x,
        gup_W,
        TOP_K,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_gup_c,
        lora_B=B_gup_c,
        scaling=0.0,
        grouped_in=False,
        grouped_out=True,
        use_fused_dX=True,
        use_fused_gather=True,
    )
    gate, up = gup.chunk(2, dim=-1)
    h = F.silu(gate) * up
    out = parallel_linear_lora(
        h,
        dwn_W,
        1,
        remapped,
        ssi,
        compact_offsets,
        lora_A=A_dn_c,
        lora_B=B_dn_c,
        scaling=0.0,
        gates=routing_weights,
        grouped_in=True,
        grouped_out=False,
        use_fused_dX=True,
        use_fused_gather=True,
    )
    return out


@pytest.mark.parametrize("strategy", ["A", "B"])
def test_mxfp4_moe_block_matches_pytorch_reference(strategy):
    """The Axolotl ScatterMoE MX path must match the per-expert PyTorch
    reference (operating on the same MX dequantized weights) within
    integration-grade tolerance."""
    gate_up_mx, down_mx, gate_up_ref, down_ref, router_w = _build_synthetic_moe()

    torch.manual_seed(7)
    x = torch.randn(M, HIDDEN, device=DEVICE, dtype=DTYPE)

    ref = _reference_moe_forward(x, router_w, gate_up_ref, down_ref)
    out = _axolotl_moe_forward(x, router_w, gate_up_mx, down_mx, strategy=strategy)

    assert ref.shape == out.shape == (M, HIDDEN)
    assert torch.allclose(ref, out, atol=5e-3, rtol=5e-3), (
        f"Strategy {strategy} MoE block diverges from PyTorch reference: "
        f"max abs={(ref - out).abs().max().item():.4e}, "
        f"max rel={((ref - out).abs() / (ref.abs() + 1e-6)).max().item():.4e}"
    )

"""Fused LoRA MLP for DeepSeek-V4's shared-expert (clamped SwiGLU).

V4's ``DeepseekV4MLP`` is ``down(silu(gate(x).clamp(max=L)) * up(x).clamp(-L, L))`` — the
``swiglu_limit`` clamp is what stops axolotl's stock ``apply_lora_mlp_swiglu`` (no clamp)
from applying. ``LoRA_MLP`` is parameterized by the activation fns, so we reuse all of its
LoRA-matmul fusion and just swap in a clamped SwiGLU activation (forward + backward).

Only the shared expert is a plain SwiGLU MLP; the 256 routed experts go through
``scattermoe_lora`` (use_scattermoe). The MLA attention projections have no fused-LoRA
form and stay on PEFT default.
"""

import functools

import torch
import triton
import triton.language as tl

from axolotl.kernels.lora import LoRA_MLP, _apply_dropout, get_lora_parameters


@triton.jit
def _clamped_swiglu_fwd_kernel(
    gate_ptr, up_ptr, out_ptr, n_elements, LIMIT, block_size: tl.constexpr
):
    off = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = off < n_elements
    gate = tl.load(gate_ptr + off, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + off, mask=mask, other=0).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)  # clamp(max=L)
    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)  # clamp(-L, L)
    f = gate * tl.sigmoid(gate)
    tl.store(out_ptr + off, (f * up).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _clamped_swiglu_bwd_kernel(
    grad_out_ptr, gate_ptr, up_ptr, n_elements, LIMIT, block_size: tl.constexpr
):
    off = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = off < n_elements
    grad_out = tl.load(grad_out_ptr + off, mask=mask, other=0).to(tl.float32)
    g_raw = tl.load(gate_ptr + off, mask=mask, other=0).to(tl.float32)
    u_raw = tl.load(up_ptr + off, mask=mask, other=0).to(tl.float32)
    gate = tl.minimum(g_raw, LIMIT)
    up = tl.minimum(tl.maximum(u_raw, -LIMIT), LIMIT)
    sig = tl.sigmoid(gate)
    silu = sig * gate
    h = silu * up
    # clamp gradients are 0 where the input was clamped
    g_mask = (g_raw < LIMIT).to(tl.float32)
    u_mask = ((u_raw > -LIMIT) & (u_raw < LIMIT)).to(tl.float32)
    grad_up = grad_out * silu * u_mask
    grad_gate = grad_out * up * sig * (1.0 + gate * (1.0 - sig)) * g_mask
    ety = grad_out_ptr.dtype.element_ty
    tl.store(grad_out_ptr + off, h.to(ety), mask=mask)
    tl.store(gate_ptr + off, grad_gate.to(ety), mask=mask)
    tl.store(up_ptr + off, grad_up.to(ety), mask=mask)


def _clamped_swiglu_forward(gate, up, limit):
    n = gate.numel()
    out = torch.empty_like(gate)
    grid = lambda m: (triton.cdiv(n, m["block_size"]),)
    _clamped_swiglu_fwd_kernel[grid](gate, up, out, n, float(limit), block_size=1024)
    return out


def _clamped_swiglu_backward(grad_output, gate, up, limit):
    n = grad_output.numel()
    grid = lambda m: (triton.cdiv(n, m["block_size"]),)
    _clamped_swiglu_bwd_kernel[grid](
        grad_output, gate, up, n, float(limit), block_size=1024
    )
    return grad_output, gate, up


def apply_lora_mlp_clamped_swiglu(self, X, inplace: bool = False):
    """Drop-in ``DeepseekV4MLP.forward`` using fused LoRA + clamped SwiGLU (``self.limit``).

    ``inplace=False``: the shared expert's input aliases the MoE residual/routed-experts
    input, so in-place intermediates would corrupt autograd."""
    gateW, gateb, gateQ, gateA, gateB, gateS, gateLB, gateDrop, gateMag = (
        get_lora_parameters(self.gate_proj)
    )
    upW, upb, upQ, upA, upB, upS, upLB, upDrop, upMag = get_lora_parameters(
        self.up_proj
    )
    downW, downb, downQ, downA, downB, downS, downLB, downDrop, downMag = (
        get_lora_parameters(self.down_proj)
    )
    X_drop = _apply_dropout(gateDrop, X, self.training)
    fwd = functools.partial(_clamped_swiglu_forward, limit=self.limit)
    bwd = functools.partial(_clamped_swiglu_backward, limit=self.limit)
    return LoRA_MLP.apply(
        X,
        X_drop,
        gateW,
        gateb,
        gateQ,
        gateA,
        gateB,
        gateS,
        gateLB,
        gateMag,
        upW,
        upb,
        upQ,
        upA,
        upB,
        upS,
        upLB,
        upMag,
        downW,
        downb,
        downQ,
        downA,
        downB,
        downS,
        downLB,
        downMag,
        fwd,
        bwd,
        inplace,
    )


def patch_dsv4_shared_mlp_lora(model):
    """Swap each LoRA'd shared-expert ``DeepseekV4MLP.forward`` for the fused clamped-SwiGLU
    LoRA kernel. Per-instance (only where gate/up/down carry LoRA), like axolotl's core
    lora_kernels. Returns the count patched."""
    import types

    n = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV4MLP":
            continue
        if all(
            hasattr(getattr(module, p, None), "lora_A")
            for p in ("gate_proj", "up_proj", "down_proj")
        ):
            module.forward = types.MethodType(apply_lora_mlp_clamped_swiglu, module)
            n += 1
    return n

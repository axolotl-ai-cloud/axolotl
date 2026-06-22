"""Chunked-dequant grouped MoE for bnb-4bit experts (portable, any GPU).

scattermoe's fused grouped path reads NVFP4/MXFP4 experts at 4-bit, but bnb-nf4 experts have no
4-bit-read kernel, so the naive path dequantizes ALL E experts to bf16 every forward (~48 GB for
gemma4-26B → ~2.7x the VRAM of the nvfp4 path; at 4k seq ~all experts are active so "selective"
dequant doesn't help). This processes experts in CHUNKS: per chunk, dequant only that chunk's
experts to bf16, run the grouped gate_up (+LoRA) → gated activation → grouped down (+LoRA) for that
chunk's tokens under activation checkpointing (re-dequant in backward), then free. Peak bf16
transient is bounded to ``chunk_size / E`` of the full expert set, bringing bnb-MoE memory in line
with the nvfp4 path. Portable: ``torch._grouped_mm`` + bnb dequant, no custom CUDA.

Only the bnb path routes here (gated on ``module.parametrizations`` in the experts forward); the
NVFP4/MXFP4/bf16 paths are untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .selective_dequant import selective_expert_weights

# Experts dequantized per chunk. 32 is a balanced default: on MoEs with many small experts (e.g.
# gemma4, where the per-step peak is activation-dominated) it recovers most of the throughput at no
# extra memory, while still bounding the bf16 weight transient to 32 experts for large-expert models.
# Tune via cfg.moe_dequant_chunk_size (lower for large experts / tiny GPUs; higher for max speed).
_DEFAULT_CHUNK = 32

# Process-global override (set from cfg.moe_dequant_chunk_size); None = fixed default.
_CHUNK_OVERRIDE: int | None = None
# Whether layer-level gradient checkpointing is active (set from cfg.gradient_checkpointing). When
# True the decoder layer already recomputes the whole MoE forward in backward, so the chunk loop's
# bf16 transient is freed per-iteration WITHOUT a per-chunk checkpoint — adding one would nest a
# redundant 3rd recompute of each chunk (real fwd → layer-GC recompute → per-chunk recompute),
# ~1.5x the expert compute. So per-chunk checkpointing is only used when layer GC is OFF.
_LAYER_GC: bool = False


def set_chunk_size_override(n) -> None:
    global _CHUNK_OVERRIDE
    _CHUNK_OVERRIDE = int(n) if n else None


def set_layer_gc_active(flag) -> None:
    global _LAYER_GC
    _LAYER_GC = bool(flag)


# cfg.moe_bnb_fast: route bnb experts through the 1-launch parallel_linear (scatter2scatter) path.
# Default (None/True). It's faster than the chunked torch._grouped_mm path (~1940 vs 1778 tok/s on
# gemma4-26B, fewer kernel launches) AND same low memory (~21 GiB): the dequant'd bf16 is recomputed
# in backward via a recipe instead of saved, so it isn't pinned per-layer. Set moe_bnb_fast: false to
# force the chunked path, which bounds the per-pass dequant transient to chunk_size experts — useful
# for large-expert MoEs / tiny GPUs where even the single-shot active-expert dequant is too big.
_BNB_FAST: bool = True


def set_bnb_fast(flag) -> None:
    global _BNB_FAST
    # None (config unset) keeps the fast default; only an explicit False forces chunked.
    _BNB_FAST = True if flag is None else bool(flag)


def bnb_fast_enabled() -> bool:
    return _BNB_FAST


def _plan_chunking(num_experts):
    """(chunk_size, use_checkpoint) from the balanced default or the cfg.moe_dequant_chunk_size
    override. Lower it for large-expert MoEs on small GPUs; raise it for max throughput. Per-chunk
    checkpointing is used ONLY when layer GC is off (with layer GC on, the chunk loop already bounds
    the transient and an extra checkpoint would just nest a redundant recompute)."""
    chunk = _CHUNK_OVERRIDE if _CHUNK_OVERRIDE is not None else _DEFAULT_CHUNK
    chunk = max(1, min(num_experts, chunk))
    use_ckpt = (chunk < num_experts) and not _LAYER_GC
    return chunk, use_ckpt


def _gated_act(gate, up, act_type, limit):
    if limit is not None:  # DSV4-style clamped SwiGLU
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    if act_type == "gelu_tanh":  # Gemma4 GeGLU
        return F.gelu(gate, approximate="tanh") * up
    return F.silu(gate) * up


def _chunk_forward(
    x, experts, chunk_idx, coff, gA, gB, gs, dA, dB, ds, act_type, limit
):
    """One expert-chunk end-to-end: dequant chunk → grouped gate_up (+LoRA) → act → grouped down
    (+LoRA). Run under ``checkpoint`` so the chunk's bf16 weights live only during fwd/recompute."""
    # dequant ONLY this chunk's experts (frozen base) to bf16: [c, K, N] after transpose
    gub = selective_expert_weights(experts, "gate_up_proj", chunk_idx).transpose(1, 2)
    dnb = selective_expert_weights(experts, "down_proj", chunk_idx).transpose(1, 2)

    gu = torch._grouped_mm(x, gub, offs=coff)  # [Ntok, 2I]
    if gA is not None:
        xa = torch._grouped_mm(x, gA.transpose(1, 2), offs=coff)  # [Ntok, r]
        gu = gu + gs * torch._grouped_mm(xa, gB.transpose(1, 2), offs=coff)

    gate, up = gu.chunk(2, dim=-1)
    h = _gated_act(gate, up, act_type, limit).contiguous()

    dn = torch._grouped_mm(h, dnb, offs=coff)  # [Ntok, H]
    if dA is not None:
        ha = torch._grouped_mm(h, dA.transpose(1, 2), offs=coff)
        dn = dn + ds * torch._grouped_mm(ha, dB.transpose(1, 2), offs=coff)
    return dn


def chunked_bnb_moe(
    hidden,
    idx,
    wts,
    experts,
    gup_lora,
    down_lora,
    num_experts,
    act_type="silu",
    limit=None,
):
    """Grouped MoE over bnb-4bit experts via chunked dequant. ``hidden`` [N,H]; ``idx``/``wts``
    [N,topk]; ``experts`` is the bnb-quantized experts module; ``*_lora`` are scattermoe-layout
    (A[r*E,K], B[out,r*E], scaling) or None. Returns [N,H], differentiable to hidden + LoRA A/B."""
    from .grouped_train import _lora_stack

    N, H = hidden.shape
    dev = hidden.device
    chunk_size, use_ckpt = _plan_chunking(num_experts)

    Agu = Bgu = sgu = Adn = Bdn = sdn = None
    if gup_lora is not None and down_lora is not None:
        two_i = gup_lora[1].shape[0]  # gate_up out = 2I (B is [2I, r*E])
        inter = down_lora[0].shape[1]  # down in = I (A is [r*E, I])
        Agu, Bgu, sgu = _lora_stack(gup_lora, num_experts, H, two_i)  # [E,r,H],[E,2I,r]
        Adn, Bdn, sdn = _lora_stack(down_lora, num_experts, inter, H)  # [E,r,I],[E,H,r]

    # Sort tokens by expert and group (contiguous per expert), like chunked_dequant_grouped_base.
    flat_exp = idx.reshape(-1)
    order = flat_exp.argsort()
    rep = torch.arange(N, device=dev).repeat_interleave(idx.size(1))[order]
    wflat = wts.reshape(-1)[order]
    counts = torch.bincount(flat_exp, minlength=num_experts)
    tok_off = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).tolist()
    flat = hidden[rep]  # [Ntok, H], grouped

    out = hidden.new_zeros(N, H)
    pieces = []
    for c0 in range(0, num_experts, chunk_size):
        c1 = min(c0 + chunk_size, num_experts)
        t0, t1 = int(tok_off[c0]), int(tok_off[c1])
        if t1 == t0:  # no tokens routed to this expert chunk
            continue
        chunk_idx = torch.arange(c0, c1, device=dev)
        coff = counts[c0:c1].cumsum(0).to(torch.int32)
        gA = Agu[c0:c1] if Agu is not None else None
        gB = Bgu[c0:c1] if Bgu is not None else None
        dA = Adn[c0:c1] if Adn is not None else None
        dB = Bdn[c0:c1] if Bdn is not None else None
        args = (
            flat[t0:t1],
            experts,
            chunk_idx,
            coff,
            gA,
            gB,
            sgu,
            dA,
            dB,
            sdn,
            act_type,
            limit,
        )
        # Per-chunk checkpoint only when layer GC is off; under layer GC the loop already bounds the
        # transient and an extra checkpoint nests a redundant recompute (see set_layer_gc_active).
        o = (
            checkpoint(_chunk_forward, *args, use_reentrant=False)
            if use_ckpt
            else _chunk_forward(*args)
        )
        pieces.append(o * wflat[t0:t1, None].to(o.dtype))
    if not pieces:  # empty batch / no routed tokens
        return out
    return out.index_add(0, rep, torch.cat(pieces, 0))

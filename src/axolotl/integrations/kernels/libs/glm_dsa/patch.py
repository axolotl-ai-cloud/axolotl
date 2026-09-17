"""Patch GlmMoeDsaAttention.forward to use the absorbed-MLA + dispatched sparse-gather kernels.

Replaces the model's dense-mask eager/sdpa attention with: build q_abs / k_shared from the MLA
projections (compressed latent + interleaved-RoPE rope part), call ``mla_attn`` (dense flash below
the auto-calibrated crossover, sparse gather above), then ``project_value`` + ``o_proj``. The DSA
top-k selection is taken from the model's own indexer (``"full"`` layers) or the previous full
layer's indices (``"shared"`` layers), unchanged. The kv_b_proj weight is read through
``effective_weight`` so the absorption is correct (and differentiable) under both full-parameter
and LoRA training.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .attention_mla_absorb import absorb_query, project_value, split_kv_b
from .context_parallel import all_gather_seq
from .dispatch import mla_attn
from .indexer import indexer_topk


@torch.no_grad()
def fused_indexer_topk(
    indexer,
    hidden_states,
    q_resid,
    position_embeddings,
    attention_mask,
    position_ids,
    group=None,
    seq_q=None,
    seq_k=None,
    q_offset=0,
):
    """Replicates GlmMoeDsaIndexer.forward with the fused scorer (collapses the [B,S,32,T] reduction
    to [B,S,T], the long-context memory win). Forward-only. Under context parallel (``group`` set),
    all-gathers the indexer's keys so local queries score against the GLOBAL keys, and uses the
    global ``position_ids`` for the causal mask. Returns top-k of GLOBAL key positions."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf_glm_dsa

    # transformers <= 5.12 applies non-interleaved rope in the stock indexer; the upstream commit
    # that switched it to interleaved also removed apply_rotary_pos_emb from the module, so the
    # symbol's presence selects the formulation matching the installed indexer.
    rope_fn = getattr(hf_glm_dsa, "apply_rotary_pos_emb", None)
    if rope_fn is None:
        rope_fn = hf_glm_dsa.apply_rotary_pos_emb_interleave

    B, S, _ = hidden_states.shape
    q = indexer.wq_b(q_resid).view(B, S, indexer.n_heads, indexer.head_dim)
    q_rot, q_pass = torch.split(
        q,
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)  # [B,S,1,D]
    k_rot, k_pass = torch.split(
        k,
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    cos, sin = position_embeddings
    q_rot, k_rot = rope_fn(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)  # [B,S_local,32,128]
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B,S_local,128]
    # fp32 before scaling, as in the stock indexer: bf16 rounding here perturbs near-tied
    # scores enough to flip top-k selections
    weights = indexer.weights_proj(
        hidden_states.to(indexer.weights_proj.weight.dtype)
    ).float() * (indexer.n_heads**-0.5)  # [B,S_local,32]
    cp = group is not None and dist.is_initialized() and dist.get_world_size(group) > 1
    if cp:
        k = all_gather_seq(
            k, group
        )  # [B,S_global,128]; indexer is no_grad so no bwd comm needed
        attention_mask = None  # use global position_ids for causality instead
    return indexer_topk(
        q,
        k,
        weights,
        indexer.softmax_scale,
        indexer.index_topk,
        attention_mask,
        position_ids,
        seq_q=seq_q,
        seq_k=seq_k,
        q_offset=q_offset,
    )


def _full(t):
    """All-gather a FSDP2 DTensor param to a plain tensor (sharded weights can't mix with plain in
    the absorption arithmetic). No-op for plain tensors."""
    return t.full_tensor() if type(t).__name__ == "DTensor" else t


def _seq_idx_from_position_ids(position_ids):
    """Per-token document id [B,S] under sample packing, or ``None`` if not packed.

    Multipack resets ``position_ids`` to 0 at each packed document's start, so a non-increasing
    step marks a boundary; ``cumsum(position_ids == 0)`` numbers the documents. Returns ``None``
    when ``position_ids`` is absent or strictly increasing (a single unpacked sequence) so the
    fast sparse path is preserved for ordinary long-context training."""
    if position_ids is None:
        return None
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if (
        position_ids.shape[1] < 2
        or not (position_ids[:, 1:] <= position_ids[:, :-1]).any()
    ):
        return None
    return (position_ids == 0).to(torch.int32).cumsum(-1)


def effective_weight(linear) -> torch.Tensor:
    """The weight the projection actually applies, differentiable wrt trainable params:
    full-parameter -> ``linear.weight``; PEFT-LoRA -> ``base.weight + scaling·(Bᵀ? )`` delta so the
    absorption flows gradients to the adapter. Falls back to ``.weight``."""
    if hasattr(linear, "base_layer"):  # PEFT LoRA layer
        w = _full(linear.base_layer.weight)
        active = getattr(linear, "active_adapters", None) or list(
            getattr(linear, "lora_A", {}).keys()
        )
        for name in active:
            A = _full(linear.lora_A[name].weight)  # [r, in]
            Bm = _full(linear.lora_B[name].weight)  # [out, r]
            w = w + linear.scaling[name] * (Bm @ A)
        return w
    return _full(linear.weight)


def _glm_dsa_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    prev_topk_indices=None,
    **kwargs,
):
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        apply_rotary_pos_emb_interleave,
    )

    group = getattr(self, "_cp_group", None)
    cp = group is not None and dist.is_initialized() and dist.get_world_size(group) > 1
    rank = dist.get_rank(group) if cp else 0

    B, S = hidden_states.shape[:-1]  # S = local query count under CP
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(B, S, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(
        q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed = self.kv_a_proj_with_mqa(hidden_states)
    latent, k_rot = torch.split(
        compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    c_kv = self.kv_a_layernorm(latent)  # [B, S, kv_lora]
    k_rot = k_rot.view(B, 1, S, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(
        q_rot, k_rot, cos, sin
    )  # [B,H,S,64],[B,1,S,64]

    # Per-document ids under sample packing (None for ordinary sequences). seq_q is the local
    # queries' doc ids; seq_k is the keys' doc ids (aligned with k_shared).
    # Under CP the doc ids MUST be derived globally: cumsum(position_ids == 0) on a local chunk
    # restarts at 0 on every rank, so a document crossing a CP boundary would get different (and
    # colliding) ids per rank, masking valid remote keys or letting attention cross documents. Gather
    # the global position_ids, number the documents ONCE, keep the full ids for the keys and take this
    # rank's slice for the queries — so a boundary-spanning doc shares one id across ranks. (Also
    # handles a chunk that is entirely mid-document, where the local cumsum would see no reset at all.)
    if cp and position_ids is not None:
        global_position_ids = all_gather_seq(position_ids.unsqueeze(-1), group).squeeze(
            -1
        )
        seq_k = _seq_idx_from_position_ids(global_position_ids)
        seq_q = None if seq_k is None else seq_k[:, rank * S : (rank + 1) * S]
    else:
        seq_q = _seq_idx_from_position_ids(position_ids)
        seq_k = seq_q

    # DSA top-k: this layer's own indexer (full) or the previous full layer's selection (shared).
    # Under CP the indexer gathers global keys; topk indices reference GLOBAL key positions.
    packed = seq_q is not None
    if self.indexer is not None:
        idx_mask = (
            attention_mask[:, 0, :, :]
            if (attention_mask is not None and not cp and not packed)
            else None
        )
        if cp or packed or getattr(self, "_use_fused_indexer", False):
            topk_indices = fused_indexer_topk(
                self.indexer,
                hidden_states,
                q_resid,
                position_embeddings,
                idx_mask,
                position_ids,
                group=group if cp else None,
                seq_q=seq_q,
                seq_k=seq_k,
                q_offset=(rank * S if cp else 0),
            )
        else:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                idx_mask,
                position_ids,
                past_key_values=past_key_values,
            )
    else:
        if prev_topk_indices is None:
            raise ValueError("shared DSA layer needs prev_topk_indices")
        topk_indices = prev_topk_indices

    w_kb_k, w_kb_v = split_kv_b(effective_weight(self.kv_b_proj), self.num_heads)
    q_abs = absorb_query(q_pass, q_rot, w_kb_k)  # [B,H,S_local,576]
    k_shared = torch.cat([c_kv, k_rot.squeeze(1)], dim=-1)  # [B,S_local,576]
    q_offset = 0
    if cp:
        k_shared = all_gather_seq(k_shared, group)  # [B,S_global,576] (differentiable)
        q_offset = rank * S
    out_latent = mla_attn(
        q_abs,
        k_shared,
        topk_indices,
        self.scaling,
        q_offset=q_offset,
        seq_q=seq_q,
        seq_k=seq_k,
    )
    attn = project_value(out_latent, w_kb_v)  # [B,H,S,v_head]
    attn = attn.transpose(1, 2).reshape(B, S, -1).contiguous()
    return self.o_proj(attn), None, topk_indices


def keep_router_fp32(model) -> int:
    """Force the MoE router (gate weight + e_score_correction_bias) to fp32 STORAGE, guaranteeing the
    router never operates below fp32 regardless of the model's compute dtype or any kernelized path.

    Expert selection is discrete (sigmoid -> group top-k), so router precision matters for
    correctness. GlmMoeDsaTopkRouter.forward already upcasts to fp32 for the logit matmul and the
    model keeps e_score_correction_bias fp32, so on a bf16 checkpoint (whose gate weight is *already*
    bf16-rounded at source) this changes no values — it is a guard that the integration's bf16 cast
    can't downcast the router, and it skips the per-forward upcast. The dominant source of routing
    variance is the bf16 RESIDUAL-STREAM router INPUT (inherent to bf16 training), not the router;
    that is addressed model-wide, not here. Returns the router count."""
    import torch as _t

    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "GlmMoeDsaTopkRouter":
            mod.weight.data = mod.weight.data.to(_t.float32)
            bias = getattr(mod, "e_score_correction_bias", None)
            if bias is not None:
                mod.e_score_correction_bias.data = bias.data.to(_t.float32)
            n += 1
    return n


def patch_glm_moe_dsa_attention(
    model, use_fused_indexer: bool = False, cp_group=None
) -> int:
    """Swap every GlmMoeDsaAttention.forward for the absorbed-kernel version. ``use_fused_indexer``
    replaces the eager indexer scoring with the fused [B,S,T] scorer. ``cp_group`` (a torch.distributed
    process group) enables context parallelism: the attention all-gathers the compressed KV +
    indexer-k so each rank's local queries attend the global keys. Returns the count."""
    import types

    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "GlmMoeDsaAttention":
            mod._use_fused_indexer = use_fused_indexer
            mod._cp_group = cp_group
            mod.forward = types.MethodType(_glm_dsa_attention_forward, mod)
            n += 1
    return n

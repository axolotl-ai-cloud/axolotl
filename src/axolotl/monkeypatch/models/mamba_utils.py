"""Shared utilities for Mamba2 SSM sample-packing and context-parallelism patches.

Used by: nemotron_h, falcon_h1, granite_moe_hybrid
"""

import functools

import torch
import torch.distributed as dist


def get_seq_idx(position_ids: torch.Tensor) -> torch.Tensor:
    """Convert position_ids [B, T] → seq_idx [B, T] int32 for mamba-ssm kernels.

    Example: position_ids [[0,1,2,3,0,1,2]] → seq_idx [[0,0,0,0,1,1,1]]

    Under context parallelism a rank may receive a chunk that begins mid-sample
    (position_ids[0] != 0), so the raw cumsum starts at 0 and subtracting 1
    would yield -1 — an invalid value for the Mamba kernels.  Subtracting the
    first element of the cumsum instead normalises every chunk to start at 0
    while still correctly incrementing at every intra-chunk sample boundary.

    Example (CP rank 1, chunk starts mid-sample):
        position_ids [[3,4,5,0,1,2]] → seq_idx [[0,0,0,1,1,1]]
    """
    cumsum = torch.cumsum((position_ids == 0).int(), dim=-1)
    return (cumsum - cumsum[..., :1]).to(torch.int32)


def is_cp_active() -> bool:
    """Return True if context parallelism (ring attention) is active on this rank.

    Zero-cost when CP is not configured: the import guard ensures we only touch
    the distributed group if ring_flash_attn is installed.
    """
    try:
        from axolotl.monkeypatch.ring_attn import get_ring_attn_group

        group = get_ring_attn_group()
        return group is not None and dist.get_world_size(group) > 1
    except (ImportError, RuntimeError):
        return False


def _get_cp_group_and_rank():
    """Return (process_group, local_rank, world_size) for the CP ring."""
    from axolotl.monkeypatch.ring_attn import get_ring_attn_group

    group = get_ring_attn_group()
    return group, dist.get_rank(group), dist.get_world_size(group)


def ring_shift_ssm_state(
    h_final: torch.Tensor,
) -> torch.Tensor:
    """P2P ring: send h_final to rank+1, receive from rank-1 within CP group.

    Uses synchronous send/recv on the ring attention process group.
    Rank 0 in the CP group receives zeros (no previous chunk).

    Args:
        h_final: Final SSM state from this rank's forward pass.
                 Shape is architecture-dependent, typically [B, H, d, n].

    Returns:
        h_prev: SSM state received from rank-1, same shape/dtype as h_final.
                Zero tensor on the first rank in the CP group.
    """
    group, local_rank, world_size = _get_cp_group_and_rank()
    ranks = dist.get_process_group_ranks(group)

    h_prev = torch.zeros_like(h_final)

    if world_size <= 1:
        return h_prev

    prev_global = ranks[(local_rank - 1) % world_size]
    next_global = ranks[(local_rank + 1) % world_size]

    send_op = dist.P2POp(dist.isend, h_final.contiguous(), next_global, group=group)
    recv_op = dist.P2POp(dist.irecv, h_prev, prev_global, group=group)

    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    # Rank 0 in the ring has no true predecessor — zero out received state
    if local_rank == 0:
        h_prev.zero_()

    return h_prev


def mamba2_cp_correction(
    out: torch.Tensor,
    h_final: torch.Tensor,
    C: torch.Tensor,
    cum_A: torch.Tensor,
    h_prev: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CP correction to SSM output using the received state from rank-1.

    SSM output is linear in the initial hidden state, so the contribution of
    h_prev can be added analytically without a second forward pass.

    For each timestep t in the local chunk:
        propagated_state_t = cumA_t * h_prev          [B, H, d, n]
        Δy_t = sum_over_n( C_t * propagated_state_t ) [B, H, d]

    The corrected final state for this rank is:
        h_final_corrected = h_final + cumA_T * h_prev

    Sample packing correctness (seq_idx):
        When sample packing is active, a CP rank may hold multiple packed
        sequences. Only the first sequence (seq_idx == 0) is a continuation
        of the previous rank's chunk — subsequent sequences are brand-new and
        should receive zero correction from h_prev.

        Passing seq_idx masks delta_y to zero for all tokens where
        seq_idx > 0, preventing h_prev state from leaking into unrelated
        packed sequences.

    Args:
        out:       SSM scan output from this rank, shape [B, T, D] where D = H*d.
        h_final:   Final SSM state from this rank, shape [B, H, d, n].
        C:         Output projection matrices, shape [B, T, n_groups, n].
        cum_A:     Cumulative log-transition factors, shape [B, T, H].
                   These are the log-space cumulative sums of A, so
                   exp(cum_A_t) gives the transition matrix from step 0 to t.
        h_prev:    SSM state received from rank-1 (zeros on rank 0).
                   Shape [B, H, d, n].
        num_heads: Number of SSM heads (H).
        head_dim:  Dimension per head (d).
        seq_idx:   Optional sequence index tensor, shape [B, T] int32.
                   When provided, correction is zeroed for tokens where
                   seq_idx > 0 (i.e. sequences that start fresh on this rank).

    Returns:
        corrected_out:     out + Δy, shape [B, T, D].
        corrected_h_final: h_final + cumA_T * h_prev, shape [B, H, d, n].
    """
    if not h_prev.any():
        return out, h_final

    B, T, _ = out.shape
    n_groups = C.shape[2]
    heads_per_group = num_heads // n_groups

    # cum_A: [B, T, H] → transition factors (exponentiate from log-space)
    decay = torch.exp(cum_A).float()  # [B, T, H]

    # Propagate h_prev through cumulative transitions: [B, T, H, d, n]
    prop_state = decay[:, :, :, None, None] * h_prev[:, None, :, :, :].float()

    # C: [B, T, n_groups, n] → expand to heads: [B, T, H, n]
    C_expanded = C.float().repeat_interleave(heads_per_group, dim=2)  # [B, T, H, n]

    # Δy_t = sum_n(C_t * prop_state_t) → [B, T, H, d]
    delta_y = torch.einsum("bthn,bthdn->bthd", C_expanded, prop_state)

    # Mask out correction for tokens belonging to new sequences on this rank.
    # seq_idx == 0 → continuation of the sequence that crossed the CP boundary
    # seq_idx  > 0 → brand-new packed sequence, h_prev is irrelevant to it
    if seq_idx is not None:
        # mask: [B, T, 1, 1] — broadcast over H and d
        mask = (seq_idx == 0).to(delta_y.dtype).unsqueeze(-1).unsqueeze(-1)
        delta_y = delta_y * mask

    # Reshape to [B, T, D] where D = H * d
    delta_y = delta_y.reshape(B, T, num_heads * head_dim).to(out.dtype)

    corrected_out = out + delta_y

    # Correct final state using last-timestep decay.
    # If the last token is in a new sequence (seq_idx > 0 at T-1), h_prev
    # should not propagate into h_final either.
    if seq_idx is not None and seq_idx[:, -1].any():
        # last token belongs to a new sequence — don't corrupt h_final
        corrected_h_final = h_final
    else:
        decay_final = decay[:, -1, :, None, None]  # [B, H, 1, 1]
        corrected_h_final = h_final + (decay_final * h_prev.float()).to(h_final.dtype)

    return corrected_out, corrected_h_final


def ensure_mamba_kernels_loaded(target_module):
    """Eagerly resolve mamba-ssm and causal-conv1d globals on *target_module*.

    Transformers >= 5.5 lazily loads these inside ``Mixer.__init__`` via
    ``lazy_load_kernel``.  Our monkeypatches run *before* model instantiation,
    so the module globals are still ``None``.  This helper triggers the kernel
    resolution early so the patched ``cuda_kernels_forward`` (and
    ``wrap_mamba_scan_for_cp``) can reference them.
    """
    if getattr(target_module, "mamba_chunk_scan_combined", None) is not None:
        return

    try:
        from transformers.integrations.hub_kernels import lazy_load_kernel
        from transformers.utils.import_utils import resolve_internal_import
    except ImportError:
        return

    causal_conv1d = lazy_load_kernel("causal-conv1d")
    if causal_conv1d is not None:
        target_module.causal_conv1d_update = getattr(
            causal_conv1d, "causal_conv1d_update", None
        )
        target_module.causal_conv1d_fn = getattr(
            causal_conv1d, "causal_conv1d_fn", None
        )

    mamba_ssm = lazy_load_kernel("mamba-ssm")
    if mamba_ssm is not None:
        target_module.selective_state_update = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.selective_state_update.selective_state_update",
        )
        target_module.mamba_chunk_scan_combined = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.ssd_combined.mamba_chunk_scan_combined",
        )
        target_module.mamba_split_conv1d_scan_combined = resolve_internal_import(
            mamba_ssm,
            chained_path="ops.triton.ssd_combined.mamba_split_conv1d_scan_combined",
        )

    target_module.is_fast_path_available = all(
        (
            getattr(target_module, "selective_state_update", None),
            getattr(target_module, "mamba_chunk_scan_combined", None),
            getattr(target_module, "mamba_split_conv1d_scan_combined", None),
            getattr(target_module, "causal_conv1d_fn", None),
            getattr(target_module, "causal_conv1d_update", None),
        )
    )


def wrap_mamba_scan_for_cp(target_module):
    """Wrap ``mamba_chunk_scan_combined`` in *target_module* to apply CP correction.

    After the scan, if CP is active the wrapper:
    1. Sends the final SSM state to the next rank via ``ring_shift_ssm_state``.
    2. Computes cumA from the scan's A / dt / dt_bias / dt_softplus args.
    3. Calls ``mamba2_cp_correction`` to add the contribution of h_prev.

    This is installed per-module so it only affects the architecture whose
    modeling file imports ``mamba_chunk_scan_combined``.

    The approach follows Tri Dao's Mamba-2 systems blog: each GPU computes its
    local output and final states, states are passed via P2P, then outputs are
    corrected — no ring attention needed for SSM layers.
    """
    if getattr(target_module, "_cp_scan_wrapped", False):
        return

    ensure_mamba_kernels_loaded(target_module)

    if getattr(target_module, "mamba_chunk_scan_combined", None) is None:
        return

    original_scan = target_module.mamba_chunk_scan_combined

    @functools.wraps(original_scan)
    def _cp_scan_wrapper(*args, **kwargs):
        cp_active = is_cp_active()

        if cp_active:
            kwargs["return_final_states"] = True

        result = original_scan(*args, **kwargs)

        if not cp_active:
            return result

        scan_output, ssm_state = result
        if ssm_state is None:
            return result

        h_prev = ring_shift_ssm_state(ssm_state)

        # Signature: mamba_chunk_scan_combined(x, dt, A, B, C, ...)
        # Extract from kwargs first, fall back to positional args.
        dt_arg = kwargs.get("dt", args[1] if len(args) > 1 else None)
        A_arg = kwargs.get("A", args[2] if len(args) > 2 else None)
        C_arg = kwargs.get("C", args[4] if len(args) > 4 else None)
        if dt_arg is None or A_arg is None or C_arg is None:
            raise ValueError(
                "wrap_mamba_scan_for_cp requires dt, A, C to be passed "
                f"positionally (got {len(args)} positional args) or as kwargs."
            )
        dt_bias = kwargs.get("dt_bias")
        dt_softplus = kwargs.get("dt_softplus", False)
        seq_idx = kwargs.get("seq_idx")

        if dt_softplus:
            dt_eff = torch.nn.functional.softplus(
                dt_arg + (dt_bias if dt_bias is not None else 0)
            )
        else:
            dt_eff = dt_arg

        dA = A_arg[None, None, :] * dt_eff
        cum_A = torch.cumsum(dA, dim=1)

        x = args[0]
        num_heads = A_arg.shape[0]
        head_dim = x.shape[3] if x.ndim == 4 else x.shape[2] // num_heads
        B_dim, T_dim = x.shape[0], x.shape[1]

        scan_flat = scan_output.view(B_dim, T_dim, -1)
        scan_flat, ssm_state = mamba2_cp_correction(
            scan_flat,
            ssm_state,
            C_arg,
            cum_A,
            h_prev,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_idx=seq_idx,
        )
        scan_output = scan_flat.view(scan_output.shape)

        return scan_output, ssm_state

    target_module.mamba_chunk_scan_combined = _cp_scan_wrapper
    target_module._cp_scan_wrapped = True

"""Shared utilities for Mamba2 SSM sample-packing and context-parallelism patches.

Used by: nemotron_h, falcon_h1, bamba, granite_moe_hybrid, zamba2
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
    except Exception:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CP correction to SSM output using the received state from rank-1.

    SSM output is linear in the initial hidden state, so the contribution of
    h_prev can be added analytically without a second forward pass.

    For each timestep t in the local chunk:
        propagated_state_t = cumA_t * h_prev          [B, H, d, n]
        Δy_t = sum_over_n( C_t * propagated_state_t ) [B, H, d]

    The corrected final state for this rank is:
        h_final_corrected = h_final + cumA_T * h_prev

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

    Returns:
        corrected_out:     out + Δy, shape [B, T, D].
        corrected_h_final: h_final + cumA_T * h_prev, shape [B, H, d, n].
    """
    if h_prev.abs().max() == 0:
        return out, h_final

    B, T, _ = out.shape
    n_groups = C.shape[2]
    heads_per_group = num_heads // n_groups

    # cum_A: [B, T, H] → transition factors (exponentiate from log-space)
    decay = torch.exp(cum_A).float()  # [B, T, H]

    # Propagate h_prev through cumulative transitions: [B, T, H, d, n]
    # decay[:, :, :, None, None] * h_prev[:, None, :, :, :]
    prop_state = decay[:, :, :, None, None] * h_prev[:, None, :, :, :].float()

    # C: [B, T, n_groups, n] → expand to heads: [B, T, H, n]
    C_expanded = C.float().repeat_interleave(heads_per_group, dim=2)  # [B, T, H, n]

    # Δy_t = sum_n(C_t_h_n * prop_state_t_h_d_n) for each d
    # C_expanded: [B, T, H, n] → [B, T, H, 1, n]
    # prop_state: [B, T, H, d, n]
    # contract over n → [B, T, H, d]
    delta_y = torch.einsum("bthn,bthdn->bthd", C_expanded, prop_state)

    # Reshape to [B, T, D] where D = H * d
    delta_y = delta_y.reshape(B, T, num_heads * head_dim).to(out.dtype)

    corrected_out = out + delta_y

    # Correct final state: h_final + cumA_T * h_prev
    # Use the last timestep's cumulative decay
    decay_final = decay[:, -1, :, None, None]  # [B, H, 1, 1]
    corrected_h_final = h_final + (decay_final * h_prev.float()).to(h_final.dtype)

    return corrected_out, corrected_h_final


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

        # args: x(0), dt(1), A(2), B(3), C(4) — positional in all callers
        dt_arg = args[1]
        A_arg = args[2]
        C_arg = args[4]
        dt_bias = kwargs.get("dt_bias")
        dt_softplus = kwargs.get("dt_softplus", False)

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
        )
        scan_output = scan_flat.view(scan_output.shape)

        return scan_output, ssm_state

    target_module.mamba_chunk_scan_combined = _cp_scan_wrapper

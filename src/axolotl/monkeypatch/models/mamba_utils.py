"""Shared utilities for Mamba2 SSM sample-packing and context-parallelism patches.

Used by: nemotron_h, falcon_h1, bamba, granite_moe_hybrid, zamba2
"""

import torch


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
        return group is not None and torch.distributed.get_world_size(group) > 1
    except Exception:
        return False


def ring_shift_ssm_state(
    h_final: torch.Tensor,
) -> torch.Tensor:
    """P2P ring: send h_final [B, H, d, n] to rank+1, receive from rank-1.

    Returns the SSM state from the previous rank (h_prev), or a zero tensor on
    rank 0 (no previous rank in the ring for the first chunk).

    Args:
        h_final: Final SSM state from this rank's forward pass, shape [B, H, d, n].

    Returns:
        h_prev: SSM state received from rank-1, same shape as h_final.

    TODO: Implement async P2P send/recv using torch.distributed.isend/irecv.
          Use the ring_attn process group, not the default group.
          Overlap with the next layer's forward if possible.
    """
    raise NotImplementedError(
        "ring_shift_ssm_state is not yet implemented. "
        "Context parallelism SSM state passing is planned for a future PR."
    )


def mamba2_cp_correction(
    out: torch.Tensor,
    h_final: torch.Tensor,
    C: torch.Tensor,
    cum_A: torch.Tensor,
    h_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CP correction to SSM output using the received state from rank-1.

    Since SSM output is linear in the initial hidden state, the contribution of
    h_prev (state from the end of the previous rank's chunk) can be added to
    the current rank's output analytically, without a second forward pass:

        Δy_t     = C_t · (cumA_t · h_prev)
        Δh_final = cumA_full · h_prev

    where cumA_t is the cumulative state-transition matrix from step 0 to t.

    Args:
        out:     SSM output from this rank, shape [B, T, D].
        h_final: Final SSM state from this rank, shape [B, H, d, n].
        C:       Output projection matrices, shape [B, T, n_groups, state_size].
        cum_A:   Cumulative transition factors, shape [B, T, H].
        h_prev:  SSM state received from rank-1 (zeros on rank 0).

    Returns:
        corrected_out:    out + Δy,    shape [B, T, D].
        corrected_h_final: h_final + Δh, shape [B, H, d, n].

    TODO: Implement once ring_shift_ssm_state is done. The slow-path
          (cuda_kernels_forward else-branch) already returns ssm_state and the
          intermediate C / cumA tensors needed here.
    """
    raise NotImplementedError(
        "mamba2_cp_correction is not yet implemented. "
        "Context parallelism SSM output correction is planned for a future PR."
    )

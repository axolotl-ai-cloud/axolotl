"""Runtime fix for the sonic-moe EP sentinel backward NaN (router-score grad).

In the `ep-support` kernel build, `_down_projection_backward_act` computes the
router-score grad as `ds[s_scatter_idx] = ds_scattered`, where quack's ``gemm_dgated``
only writes the valid rows ``[0, n_valid)`` of ``ds_scattered`` (grouped rows past
``expert_frequency_offset[E]`` sit outside every expert range). Under EP the dropped
sentinel tail is uninitialized memory and its ``s_scatter_idx`` entries are 0, so
garbage scatters into ``ds[0]`` and sentinel lanes keep garbage: NaN gradients from
step 1.

The replacement routes invalid-tail writes to a trash slot and zeroes them, fully
on-device, with the drop layout and GEMM shapes untouched. Remove this module once the
pinned kernel revision ships the fix (the source check below then auto-disables it).
"""

import functools
import inspect
import sys

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# The unguarded scatter this patch replaces. Absent = fixed upstream, nothing to do.
_BUG_SIGNATURE = "ds[s_scatter_idx] = ds_scattered"


def _make_fixed_down_projection_backward_act(bwd_mod):
    def _down_projection_backward_act(
        dout: torch.Tensor,
        h: torch.Tensor,
        w2: torch.Tensor,
        dh: torch.Tensor,
        ds: torch.Tensor,
        b2: torch.Tensor | None,
        db2: torch.Tensor | None,
        a_prime: torch.Tensor,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        activation_type: str,
    ) -> None:
        assert activation_type in (
            "swiglu",
            "geglu",
        ), f"QuACK gemm_gated only supports glu activations, got {activation_type}"

        s = topk_scores[s_scatter_idx]
        _, _, ds_scattered = bwd_mod.gemm_dgated(
            dout,
            w2.permute(2, 0, 1),
            PreAct=h,
            activation=activation_type,
            dx_out=dh,
            postact_out=a_prime,
            colvec_scale=s,
            colvec_reduce=True,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            dynamic_scheduler=False,
        )

        # EP sentinels: grouped rows [n_valid, TK) sit outside every expert range, so
        # the GEMM never writes their ds_scattered entries (uninitialized memory) and
        # their s_scatter_idx entries are zero-init'd, aliasing slot 0. Route those
        # writes to a trash slot (TK) and zero their values so sentinel lanes get
        # exactly zero score-grad and slot 0 is never corrupted. n_valid stays on
        # device (no sync).
        num_lanes = s_scatter_idx.size(0)
        row_is_valid = (
            torch.arange(num_lanes, device=ds.device) < expert_frequency_offset[-1]
        )
        safe_scatter_idx = torch.where(
            row_is_valid, s_scatter_idx.to(torch.long), num_lanes
        )
        safe_ds_scattered = torch.where(
            row_is_valid, ds_scattered, torch.zeros_like(ds_scattered)
        )

        if db2 is None:
            ds_ext = torch.zeros(num_lanes + 1, device=ds.device, dtype=ds.dtype)
            ds_ext[safe_scatter_idx] = safe_ds_scattered.to(ds.dtype)
            ds.copy_(ds_ext[:num_lanes])
        else:
            hidden = w2.size(0)
            num_experts = expert_frequency_offset.size(0) - 1

            old_ds_partial = torch.zeros(
                num_lanes + 1, 1, device=ds_scattered.device, dtype=ds_scattered.dtype
            )
            old_ds_partial[safe_scatter_idx, 0] = safe_ds_scattered
            old_ds_partial = old_ds_partial[:num_lanes]

            block_h = min(bwd_mod.triton.next_power_of_2(hidden), 2048)
            num_h_blocks = bwd_mod.triton.cdiv(hidden, block_h)
            new_ds_partial = torch.zeros(
                num_lanes, num_h_blocks, dtype=torch.float32, device=ds.device
            )

            bwd_mod.db2_and_ds_kernel[(num_experts, num_h_blocks)](
                dout,
                topk_scores,
                new_ds_partial,
                old_ds_partial,
                b2,
                db2,
                x_gather_idx,
                s_scatter_idx,
                expert_frequency_offset,
                hidden,
                num_experts,
                1,
                BLOCK_H=block_h,
                BLOCK_OLD_DS_PARTIAL_N=1,
            )

            if num_h_blocks == 1:
                ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
            else:
                ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))

    return _down_projection_backward_act


@functools.cache
def apply_sonicmoe_ep_backward_patch() -> bool:
    """Rebind ``_down_projection_backward_act`` in the loaded sonic-moe kernel.

    ``_DownProjection.backward`` resolves it as a module global of
    ``functional/__init__``, which is exactly ``moe_general_routing_inputs.__globals__``.
    Returns True when the patch was applied.
    """
    from transformers.integrations.sonicmoe import _load_sonicmoe_kernel

    kernel = _load_sonicmoe_kernel()
    routing_fn = kernel.moe_general_routing_inputs
    functional_ns = getattr(routing_fn, "__globals__", None)
    bwd_mod = sys.modules.get(f"{routing_fn.__module__}.backward")
    if (
        functional_ns is None
        or bwd_mod is None
        or "_down_projection_backward_act" not in functional_ns
    ):
        LOG.warning(
            "sonic-moe EP backward patch: unexpected kernel layout, cannot patch; "
            "verify grad norms stay finite before trusting EP training"
        )
        return False

    if getattr(
        functional_ns["_down_projection_backward_act"], "_axolotl_patched", False
    ):
        return True

    try:
        source = inspect.getsource(bwd_mod)
    except (OSError, TypeError):
        source = ""
    if _BUG_SIGNATURE not in source:
        LOG.info(
            "sonic-moe kernel has no unguarded ds scatter; assuming the EP sentinel "
            "backward fix landed upstream, not patching"
        )
        return False

    fixed = _make_fixed_down_projection_backward_act(bwd_mod)
    fixed._axolotl_patched = True
    # `_DownProjection.backward` resolves the name from functional/__init__'s globals;
    # rebinding there is sufficient (functional.backward's own copy has no callers).
    functional_ns["_down_projection_backward_act"] = fixed
    LOG.info(
        "patched sonic-moe _down_projection_backward_act: EP sentinel router-grad "
        "NaN fix (drop once upstream ships it)"
    )
    return True

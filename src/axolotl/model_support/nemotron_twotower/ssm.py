"""Mamba-2 mixer evaluation seeded from (and returning) a recurrent state.

The two-tower denoiser runs each Mamba-2 layer over one noisy block, starting from
the context tower's conv and SSM state at the end of the clean prefix (reference:
``NemotronHTwoTowerForCausalLM._denoiser_block_mamba``). The native transformers
mixer has no "start from this state" path outside its single-token decode, so this
module re-expresses the mixer with explicit state in and out:

* the depthwise causal conv is plain ``F.conv1d`` over ``[state | block]``: the
  torch fallback of ``causal_conv1d_fn`` silently ignores ``initial_states``, and
  the block is short, so a kernel buys nothing;
* the SSD scan goes through ``transformers``' ``mamba2_chunk_scan`` entry point,
  which resolves to ``mamba_ssm.mamba_chunk_scan_combined`` when that package is
  installed and to transformers' einsum reference otherwise. Both take
  ``initial_states`` and return the final state. The scan runs in fp32, as the
  reference does for its denoiser blocks (a long seeded state overflows bf16).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN


def causal_conv1d_with_state(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv over ``x`` [batch, length, channels] after ``initial_state``.

    ``weight`` is the ``nn.Conv1d`` weight ``[channels, 1, kernel]``;
    ``initial_state`` holds the ``kernel - 1`` raw inputs that precede ``x``
    (``[batch, channels, kernel - 1]``, oldest first; zeros when ``None``, which is
    the ordinary causal zero padding). Returns the activated output in ``x``'s
    dtype and the new state: the last ``kernel - 1`` raw inputs of ``[state | x]``.
    """
    batch, _, channels = x.shape
    width = weight.shape[-1] - 1
    xt = x.transpose(1, 2)
    if initial_state is None:
        initial_state = xt.new_zeros(batch, channels, width)
    elif initial_state.shape != (batch, channels, width):
        raise ValueError(
            f"conv initial_state must be {(batch, channels, width)}, got {tuple(initial_state.shape)}"
        )
    padded = torch.cat([initial_state.to(xt.dtype), xt], dim=-1)
    out = F.conv1d(
        padded.float(),
        weight.float(),
        bias.float() if bias is not None else None,
        groups=channels,
    )
    if activation is not None:
        out = ACT2FN[activation](out)
    new_state = padded[..., padded.shape[-1] - width :]
    return out.to(x.dtype).transpose(1, 2), new_state


def _chunk_scan():
    from transformers.models.nemotron_h import modeling_nemotron_h

    return modeling_nemotron_h.mamba2_chunk_scan


def mamba2_mixer_with_state(
    mixer,
    hidden_states: torch.Tensor,
    *,
    conv_state: torch.Tensor | None = None,
    ssm_state: torch.Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a native ``NemotronHMamba2Mixer`` over ``hidden_states`` from a given state.

    ``hidden_states`` is the post-norm (and, in the denoiser, post-modulation)
    input ``[batch, length, hidden]``. Projections are called as modules so PEFT
    LoRA on ``in_proj`` / ``out_proj`` applies. Returns
    ``(output, conv_state, ssm_state)`` with ``conv_state`` in the conv's raw-input
    layout ``[batch, conv_dim, kernel - 1]`` and ``ssm_state`` fp32
    ``[batch, heads, head_dim, state]``.

    ``dt_limit`` defaults to the checkpoint's ``time_step_limit`` (0, inf); the
    native mixer clamps at ``time_step_min`` instead, which the reference does not.
    """
    batch, length, _ = hidden_states.shape
    d_inner = mixer.intermediate_size
    conv_dim = mixer.conv_dim
    heads = mixer.num_heads
    head_dim = mixer.head_dim
    groups = mixer.n_groups
    d_state = mixer.ssm_state_size

    projected = mixer.in_proj(hidden_states)
    gate, xbc, dt = torch.split(projected, [d_inner, conv_dim, heads], dim=-1)
    xbc, new_conv_state = causal_conv1d_with_state(
        xbc,
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        mixer.activation,
        conv_state,
    )
    x, b, c = torch.split(xbc, [d_inner, groups * d_state, groups * d_state], dim=-1)
    a = -torch.exp(mixer.A_log.float())
    y, new_ssm_state = _chunk_scan()(
        x.reshape(batch, length, heads, head_dim).float(),
        dt.float().contiguous(),
        a,
        b.reshape(batch, length, groups, d_state).float(),
        c.reshape(batch, length, groups, d_state).float(),
        chunk_size=mixer.chunk_size,
        D=mixer.D.float(),
        dt_bias=mixer.dt_bias.float(),
        initial_states=ssm_state.float() if ssm_state is not None else None,
        dt_softplus=True,
        dt_limit=tuple(dt_limit),
        return_final_states=True,
    )
    y = y.reshape(batch, length, d_inner).to(gate.dtype)
    y = mixer.norm(y, gate)
    return mixer.out_proj(y), new_conv_state, new_ssm_state.float()

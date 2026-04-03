"""Sample-packing and context-parallelism patch for Zamba2 (Mamba2/Attention hybrid).

Threads seq_idx (derived from position_ids) into the Mamba2 SSM kernels so
packed-sequence boundaries reset SSM state.  Upstream hard-codes seq_idx=None.
Zamba2 uses "hybrid" layers (shared attention + mamba) and pure mamba layers.
"""

import importlib

import torch
from torch import nn

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,  # noqa: F401
    wrap_mamba_scan_for_cp,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_zamba2_modeling_packing():
    """Patch Zamba2 for sample packing: seq_idx threading into Mamba2 SSM kernels."""
    try:
        mod = importlib.import_module("transformers.models.zamba2.modeling_zamba2")
    except ImportError:
        LOG.warning("zamba2 not found in transformers, skipping packing patches")
        return

    Zamba2MambaMixer = mod.Zamba2MambaMixer
    Zamba2MambaDecoderLayer = mod.Zamba2MambaDecoderLayer

    def patched_cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask=None,
        seq_idx=None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = (
            2 * self.intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
            + self.num_heads
        )

        if cache_params is not None and cache_params.has_previous_state:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [
                d_mlp,
                d_mlp,
                self.intermediate_size,
                self.conv_dim,
                self.num_heads,
            ]
            _, _, gate, hidden_states_B_C, dt = torch.split(
                in_projected_states, split_projection_dim, dim=-1
            )
            hidden_states_B_C = mod.causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [
                    self.intermediate_size,
                    groups_time_state_size,
                    groups_time_state_size,
                ],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())
            A = (
                A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(
                batch_size, self.num_heads, self.head_dim
            )
            hidden_states = mod.selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(
                batch_size, self.num_heads * self.head_dim
            )
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())
            dt_limit_kwargs = (
                {}
                if self.time_step_limit is None
                else {"dt_limit": self.time_step_limit}
            )
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask == 1)
            else:
                input_not_masked = True

            if (
                self.use_mem_eff_path
                and self.training
                and cache_params is None
                and input_not_masked
                and seq_idx is None
            ):
                out, ssm_state = mod.mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )
            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                if cache_params is not None:
                    hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
                    conv_state = nn.functional.pad(
                        hidden_states_B_C_t,
                        (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0),
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_state)

                if mod.causal_conv1d_fn is None or self.activation not in [
                    "silu",
                    "swish",
                ]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[
                            :, :seq_len
                        ]
                    )
                else:
                    hidden_states_B_C = mod.causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                        seq_idx=seq_idx,
                    ).transpose(1, 2)[:, :seq_len]

                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [
                        self.intermediate_size,
                        groups_time_state_size,
                        groups_time_state_size,
                    ],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(
                        dtype
                    )

                C_reshaped = C.view(batch_size, seq_len, self.n_groups, -1)
                scan_output, ssm_state = mod.mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C_reshaped,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=seq_idx,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    Zamba2MambaMixer.cuda_kernels_forward = patched_cuda_kernels_forward

    def patched_mixer_forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        seq_idx=None,
    ):
        if seq_idx is not None and mod.causal_conv1d_fn is None:
            raise RuntimeError(
                "Zamba2 sample packing requires causal_conv1d_fn. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )
        if mod.is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(
                hidden_states, cache_params, attention_mask, seq_idx=seq_idx
            )
        return self.torch_forward(hidden_states, cache_params, attention_mask)

    Zamba2MambaMixer.forward = patched_mixer_forward

    def patched_mamba_decoder_forward(
        self,
        hidden_states,
        original_hidden_states=None,
        layer_idx=None,
        attention_mask=None,
        causal_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        transformer_hidden_states=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = (
            hidden_states + transformer_hidden_states
            if transformer_hidden_states is not None
            else hidden_states
        )
        hidden_states = self.input_layernorm(hidden_states)

        is_decoding = past_key_values is not None and past_key_values.has_previous_state
        seq_idx = (
            get_seq_idx(position_ids)
            if position_ids is not None and not is_decoding
            else None
        )

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
        )

        self_attn_weights = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs

    Zamba2MambaDecoderLayer.forward = patched_mamba_decoder_forward

    wrap_mamba_scan_for_cp(mod)

    LOG.info("Applied Zamba2 sample packing patch (seq_idx threading into Mamba2)")

"""Sample-packing and context-parallelism patch for Falcon-H1 (parallel Mamba2/Attention hybrid).

Threads seq_idx (derived from position_ids) into the Mamba2 SSM kernels so
packed-sequence boundaries reset SSM state.  Upstream hard-codes seq_idx=None,
which leaks hidden state across boundaries.

Unlike Nemotron-H (which selects block_type per layer), Falcon-H1 runs both
Mamba2 and Attention in **parallel** in every FalconH1DecoderLayer, so we
always need seq_idx for the mamba branch.
"""

import importlib

import torch

from axolotl.monkeypatch.models.mamba_utils import (
    ensure_mamba_kernels_loaded,
    get_seq_idx,
    is_cp_active,
    wrap_mamba_scan_for_cp,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_falcon_h1_modeling_packing():
    """Patch Falcon-H1 for sample packing: seq_idx threading into Mamba2 SSM kernels."""
    try:
        mod = importlib.import_module(
            "transformers.models.falcon_h1.modeling_falcon_h1"
        )
    except ImportError:
        LOG.warning("falcon_h1 not found in transformers, skipping packing patches")
        return

    ensure_mamba_kernels_loaded(mod)

    FalconH1Mixer = mod.FalconH1Mixer
    FalconH1DecoderLayer = mod.FalconH1DecoderLayer

    def patched_cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        seq_idx=None,
    ):
        hidden_states = mod.apply_mask_to_padding_states(hidden_states, attention_mask)
        hidden_states = hidden_states * self.ssm_in_multiplier
        projected_states = self.in_proj(hidden_states)
        projected_states = projected_states * self.mup_vector
        d_to_remove = (
            2 * self.intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
            + self.num_heads
        )

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        if use_precomputed_states:
            d_mlp = (projected_states.squeeze(1).shape[-1] - d_to_remove) // 2
            z0, x0, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
                dim=-1,
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
                z=(
                    gate.view(batch_size, self.num_heads, self.head_dim)
                    if not self.mamba_rms_norm
                    else None
                ),
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(
                batch_size, self.num_heads * self.head_dim
            )
            if self.mamba_rms_norm:
                hidden_states = self.norm(hidden_states, gate)
            if d_mlp > 0:
                hidden_states = torch.cat(
                    [torch.nn.functional.silu(z0) * x0, hidden_states], dim=-1
                )
            out = self.out_proj(hidden_states[:, None, ...])
        else:
            A = -torch.exp(self.A_log.float())
            dt_limit_kwargs = (
                {}
                if self.time_step_limit == (0.0, float("inf"))
                else {"dt_limit": self.time_step_limit}
            )

            if self.training and cache_params is None and not is_cp_active():
                out = mod.mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=(self.norm.weight if self.mamba_rms_norm else None),
                    rmsnorm_eps=(
                        self.norm.variance_epsilon if self.mamba_rms_norm else None
                    ),
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )
            else:
                d_mlp = (
                    projected_states.shape[-1]
                    - 2 * self.intermediate_size
                    - 2 * self.n_groups * self.ssm_state_size
                    - self.num_heads
                ) // 2
                if attention_mask is not None:
                    projected_states = projected_states * attention_mask[..., None]
                _, gate, hidden_states_B_C, dt = projected_states.split(
                    [2 * d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                if cache_params is not None:
                    conv_states = torch.nn.functional.pad(
                        hidden_states_B_C.permute(0, 2, 1),
                        (
                            self.conv_kernel_size - hidden_states_B_C.shape[-2],
                            0,
                        ),
                    )
                    cache_params.update_conv_state(
                        self.layer_idx, conv_states, cache_position
                    )

                time_step = torch.nn.functional.softplus(dt + self.dt_bias)

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

                if (
                    attention_mask is not None
                    and attention_mask.shape[1] > 1
                    and attention_mask.shape[0] > 1
                ):
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(
                        dtype
                    )

                C_reshaped = C.view(batch_size, seq_len, self.n_groups, -1)
                with torch.cuda.device(hidden_states.device):
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
                        **dt_limit_kwargs,
                    )

                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                if self.mamba_rms_norm:
                    out = self.norm(scan_output, gate)
                else:
                    out = scan_output * torch.nn.functional.silu(gate)
                out = self.out_proj(out)
        return out

    FalconH1Mixer.cuda_kernels_forward = patched_cuda_kernels_forward

    def patched_mixer_forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        seq_idx=None,
    ):
        if seq_idx is not None and mod.causal_conv1d_fn is None:
            raise RuntimeError(
                "Falcon-H1 sample packing requires causal_conv1d_fn. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )
        if mod.is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(
                hidden_states,
                cache_params,
                cache_position,
                attention_mask,
                seq_idx=seq_idx,
            )
        if seq_idx is not None:
            raise RuntimeError(
                "Falcon-H1 sample packing requires the CUDA fast path. "
                "Ensure model is on CUDA and mamba-ssm/causal-conv1d are installed."
            )
        dtype = hidden_states.dtype
        if (
            attention_mask is not None
            and attention_mask.shape[1] > 1
            and attention_mask.shape[0] > 1
        ):
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return self.torch_forward(
            hidden_states, cache_params, cache_position, attention_mask
        )

    FalconH1Mixer.forward = patched_mixer_forward

    # Falcon-H1 runs mamba + attention in parallel every layer (no block_type).
    # Compute seq_idx from position_ids and pass to the mamba branch.
    def patched_decoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        mamba_attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        is_decoding = past_key_values is not None and past_key_values.has_previous_state
        seq_idx = (
            get_seq_idx(position_ids)
            if position_ids is not None and not is_decoding
            else None
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=mamba_attention_mask,
            seq_idx=seq_idx,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        attention_hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states * self.attention_in_multiplier,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

        hidden_states = mamba_hidden_states + attention_hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    FalconH1DecoderLayer.forward = patched_decoder_forward

    wrap_mamba_scan_for_cp(mod)

    LOG.info("Applied Falcon-H1 sample packing patch (seq_idx threading into Mamba2)")

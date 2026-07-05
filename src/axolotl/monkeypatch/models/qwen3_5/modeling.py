"""Monkeypatch for Qwen3_5 and Qwen3_5Moe models to pass position_ids to linear attention."""

import importlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from axolotl.monkeypatch.lora_kernels import LINEAR_ATTN_IN_PROJS
from axolotl.monkeypatch.models.fla_compiled_loop import (
    call_self_attn_disabled as _call_self_attn_disabled,
    init_fla_compiled_ops as _init_fla_compiled_ops,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from fla.modules.convolution import (
        causal_conv1d as fla_causal_conv1d,  # FLA >= 0.4.1
    )
except ImportError:
    try:
        from fla.modules.conv import causal_conv1d as fla_causal_conv1d  # FLA < 0.4.1
    except ImportError:
        fla_causal_conv1d = None

# True when the FLA opaque custom-op wrappers registered; they keep the decoder loop free of graph breaks (one break anywhere in the loop makes dynamo skip the whole frame).
_FLA_COMPILED_OPS = False


def get_cu_seqlens(position_ids):
    """
    Compute cumulative sequence lengths from position_ids for FLA varlen kernels.

    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.
    https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/modeling_flash_attention_utils.py#L316

    Qwen3.5 uses MRoPE: position_ids arrive as [axes, B, T]. All axes carry the
    same temporal positions, so axis 0 is used to recover the [B, T] layout.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """
    if position_ids.ndim == 3:
        position_ids = position_ids[0]

    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}
    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)
    return torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )


def _inject_fla_kernels(module, *, compile_boundary: bool = False) -> None:
    """Inject FLA kernels into a modeling module, bypassing is_flash_linear_attention_available."""
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        if compile_boundary:
            from axolotl.monkeypatch.models import gated_delta_net_ops

            gated_delta_net_ops.install_rmsnorm_gated_compile_boundary(
                FusedRMSNormGated, logger=LOG
            )

        module.FusedRMSNormGated = FusedRMSNormGated
        module.chunk_gated_delta_rule = chunk_gated_delta_rule
        module.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        module.is_fast_path_available = True
    except ImportError:
        module.chunk_gated_delta_rule = None
        module.fused_recurrent_gated_delta_rule = None
        module.FusedRMSNormGated = None


def _patched_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    """Decoder layer forward that passes position_ids through to linear attention."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    elif self.layer_type == "full_attention":
        if getattr(self, "gradient_checkpointing", False) and self.training:
            # Under gradient checkpointing the fusion hazard (see below) doesn't occur, so keep attention in-graph — any graph break in the loop body makes dynamo skip the whole frame.
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            # Intentional dynamo.disable boundary (non-GC path, incl. model.eval() under compile): on torch 2.11 + flash-attn, Inductor fused the FA2 backward with the gated o_proj dgrad and corrupted packed-sequence gradients; unreproduced on torch 2.9/2.10 + kernels-FA2 (toy and 0.8B real ckpt, boundary removed = noise floor), guarded by test_fa2_compiled_matches_eager_grads.
            hidden_states, _ = _call_self_attn_disabled(
                self.self_attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):  # MoE returns (hidden_states, router_logits)
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def _la_proj_fwd(module, proj_name, x):
    """Fused kernel when patched (skips peft's bf16->fp32->bf16 round-trip), else peft."""
    apply_fn = getattr(module, f"apply_{proj_name}", None)
    if apply_fn is not None:
        return apply_fn(x)
    return getattr(module, proj_name)(x)


def _la_in_proj_fwd(module, x):
    fused = getattr(module, "apply_in_proj_fused", None)
    if fused is not None:
        return fused(x)
    return {name: getattr(module, name)(x) for name in LINEAR_ATTN_IN_PROJS}


def _make_qwen3_5_gated_delta_forward(apply_mask_fn):
    """Factory for patched Qwen3_5/Qwen3_5Moe GatedDeltaNet forward with packing support."""

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = apply_mask_fn(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # Training, no cache: route the FLA kernels through opaque ops that take position_ids and derive cu_seqlens eagerly, so aten.nonzero never enters the traced loop.
        use_compiled_ops = (
            _FLA_COMPILED_OPS and cache_params is None and not use_precomputed_states
        )
        cu_seqlens = None
        pos_for_varlen = None
        if not use_precomputed_states and position_ids is not None:
            if use_compiled_ops:
                pos_for_varlen = position_ids
            else:
                cu_seqlens = get_cu_seqlens(position_ids=position_ids)

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        # All in-projections share hidden_states; fuse into one autograd node.
        # mixed_qkv stays [B, T, D]; only transposed inside paths that require [B, D, T]
        in_proj = _la_in_proj_fwd(self, hidden_states)
        mixed_qkv = in_proj["in_proj_qkv"]  # [B, T, D]

        z = in_proj["in_proj_z"]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = in_proj["in_proj_b"]
        a = in_proj["in_proj_a"]

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv.transpose(1, 2),
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            ).transpose(1, 2)
        else:
            if cache_params is not None:
                mixed_qkv_t = mixed_qkv.transpose(1, 2)
                cache_params.conv_states[self.layer_idx] = F.pad(
                    mixed_qkv_t,
                    (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0),
                )

            if fla_causal_conv1d is not None and pos_for_varlen is not None:
                # Opaque op (same FLA varlen kernels): traceable, unlike the raw entry whose data-dependent .item() graph-breaks the loop.
                mixed_qkv = torch.ops.axolotl_gdn.gdn_conv(
                    mixed_qkv,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                    pos_for_varlen,
                )
            elif fla_causal_conv1d is not None and cu_seqlens is not None:
                # FLA varlen kernel for packed sequences; input must be contiguous [B, T, D]
                mixed_qkv, _ = fla_causal_conv1d(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )
            else:
                if cu_seqlens is not None and fla_causal_conv1d is None:
                    raise RuntimeError(
                        "Packed sequences require fla.modules.convolution.causal_conv1d "
                        "(cu_seqlens support). Install flash-linear-attention or disable packing."
                    )
                mixed_qkv = F.silu(
                    self.conv1d(mixed_qkv.transpose(1, 2))[:, :, :seq_len]
                ).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            if use_compiled_ops:
                # Opaque op mirroring ChunkGatedDeltaRuleFunction: FLA's public entry is @torch.compiler.disable and would graph-break the loop. pos_for_varlen None = dense.
                # Contiguize here: setup_context saves the op's inputs, so split views would re-pay this copy every backward.
                core_attn_out = torch.ops.axolotl_gdn.gdn_chunk(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    g,
                    beta,
                    key.shape[-1] ** -0.5,  # FLA's default scale
                    pos_for_varlen,
                    True,  # qwen3_5 eager casts g to the input dtype
                )[0]
                last_recurrent_state = None
            else:
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g.to(dtype=query.dtype),
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                    # torch_chunk_gated_delta_rule fallback does not accept cu_seqlens
                    **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
                )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g.to(dtype=query.dtype),
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return _la_proj_fwd(self, "out_proj", core_attn_out)

    return patched_forward


def _apply_packing_patches(
    model_type: str,
    cls_prefix: str,
    forward_factory,
    *,
    torch_compile: bool = False,
) -> None:
    global _FLA_COMPILED_OPS
    module_name = f"transformers.models.{model_type}.modeling_{model_type}"

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return

    # Under torch_compile, route FusedRMSNormGated through its custom-op wrapper too: its eager entry is untraceable (un-meta-able as_strided backward) and would graph-break the loop.
    _inject_fla_kernels(module, compile_boundary=torch_compile)
    _FLA_COMPILED_OPS = _init_fla_compiled_ops(torch_compile)
    if torch_compile and not _FLA_COMPILED_OPS:
        from axolotl.monkeypatch.models import gated_delta_net_ops as fla_ops

        # On FA2 the broken-loop compile regime benches slower than plain eager, so this must be loud.
        LOG.warning(
            f"torch_compile is enabled but the FLA custom ops failed to build "
            f"({fla_ops.fla_ops_build_error()}); the {cls_prefix} decoder loop "
            f"will NOT compile and will fall back to the eager kernels. With "
            f"flash_attention_2 this is typically slower than disabling "
            f"torch_compile entirely."
        )
    getattr(module, f"{cls_prefix}DecoderLayer").forward = _patched_decoder_forward
    gated_cls = getattr(module, f"{cls_prefix}GatedDeltaNet")
    gated_cls.forward = forward_factory(module.apply_mask_to_padding_states)

    LOG.info(
        f"Applied {cls_prefix} packing patch "
        f"(fla_causal_conv1d={'available' if fla_causal_conv1d else 'unavailable'}, "
        f"torch_compile={torch_compile}, "
        f"compiled_loop_fla_ops={_FLA_COMPILED_OPS})"
    )


def patch_qwen3_5_modeling_packing(*, torch_compile: bool = False):
    _apply_packing_patches(
        "qwen3_5",
        "Qwen3_5",
        _make_qwen3_5_gated_delta_forward,
        torch_compile=torch_compile,
    )


def patch_qwen3_5_moe_modeling_packing(*, torch_compile: bool = False):
    _apply_packing_patches(
        "qwen3_5_moe",
        "Qwen3_5Moe",
        _make_qwen3_5_gated_delta_forward,
        torch_compile=torch_compile,
    )


def patch_qwen3_5_vlm_flash_attention():
    """
    Patch _is_packed_sequence to handle Qwen3.5's 3-D MRoPE position_ids.

    transformers passes position_ids as [axes, B, T] to decoder layers, but
    _is_packed_sequence only handles 2-D tensors and mis-classifies the 3-D
    shape as a packed-sequence indicator, causing CUDA errors in the varlen path.
    """
    try:
        import transformers.modeling_flash_attention_utils as fa_utils

        _original = fa_utils._is_packed_sequence

        def _patched(position_ids, batch_size):
            if position_ids is not None and position_ids.ndim != 2:
                return False
            return _original(position_ids, batch_size)

        fa_utils._is_packed_sequence = _patched
        LOG.info("Applied Qwen3.5 VLM flash-attention patch (3-D MRoPE position_ids)")
    except Exception as exc:  # pragma: no cover
        LOG.warning(f"Failed to apply Qwen3.5 VLM flash-attention patch: {exc}")

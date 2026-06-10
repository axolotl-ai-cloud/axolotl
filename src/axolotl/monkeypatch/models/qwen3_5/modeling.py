"""Monkeypatch for Qwen3_5 and Qwen3_5Moe models to pass position_ids to linear attention."""

import importlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

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

_FLA_CAUSAL_CONV_COMPILE_BOUNDARY = False


def set_fla_causal_conv_compile_boundary(enabled: bool) -> None:
    global _FLA_CAUSAL_CONV_COMPILE_BOUNDARY
    _FLA_CAUSAL_CONV_COMPILE_BOUNDARY = bool(enabled)


def _call_fla_causal_conv1d(*, x, weight, bias, activation, cu_seqlens):
    return fla_causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )


try:
    import torch._dynamo as _dynamo

    _call_fla_causal_conv1d_disabled = _dynamo.disable(_call_fla_causal_conv1d)
except Exception:  # pragma: no cover
    _call_fla_causal_conv1d_disabled = _call_fla_causal_conv1d


def _call_self_attn(attn_module, **kwargs):
    return attn_module(**kwargs)


try:
    import torch._dynamo as _dynamo

    _call_self_attn_disabled = _dynamo.disable(_call_self_attn)
except Exception:  # pragma: no cover
    _call_self_attn_disabled = _call_self_attn


# torch.compile-friendly wrapper for FLA's FusedRMSNormGated. Its fused gated-norm
# backward (layer_norm_gated_bwd) calls aten.as_strided in a way torch>=2.12 cannot
# meta-trace ("mismatch in length of strides and shape"). Registering it as a custom
# op with a clean fake impl (empty_like — no as_strided) and an OPAQUE recompute
# backward keeps the op inside the Inductor graph (no graph break), so surrounding
# ops still fuse, while the un-meta-able FLA backward runs eager and is never traced.
_FLA_RMSNORM_GATED_OP = None


def _build_fla_rmsnorm_gated_op():
    # Backward is ITS OWN opaque custom op: AOT-autograd traces the backward graph,
    # so the FLA Triton recompute must also be hidden behind an op or it hits
    # FakeTensors ("Cannot access data pointer").
    @torch.library.custom_op("axolotl_fla::rmsnorm_gated_bwd", mutates_args=())
    def _bwd_op(
        grad: torch.Tensor,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        eps: float,
    ) -> list[torch.Tensor]:
        from fla.modules.fused_norm_gate import rms_norm_gated

        with torch.enable_grad():
            xd, gd, wd = (t.detach().requires_grad_(True) for t in (x, g, weight))
            y = rms_norm_gated(xd, gd, wd, None, activation, eps=eps)
            dx, dg, dw = torch.autograd.grad(y, (xd, gd, wd), grad)
        return [dx, dg, dw]

    @_bwd_op.register_fake
    def _(grad, x, g, weight, activation, eps):
        return [torch.empty_like(x), torch.empty_like(g), torch.empty_like(weight)]

    @torch.library.custom_op("axolotl_fla::rmsnorm_gated", mutates_args=())
    def _op(
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        eps: float,
    ) -> torch.Tensor:
        from fla.modules.fused_norm_gate import rms_norm_gated

        return rms_norm_gated(x, g, weight, None, activation, eps=eps).contiguous()

    @_op.register_fake
    def _(x, g, weight, activation, eps):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    def _setup(ctx, inputs, output):
        x, g, weight, activation, eps = inputs
        ctx.save_for_backward(x, g, weight)
        ctx.activation, ctx.eps = activation, eps

    def _bwd(ctx, grad):
        x, g, weight = ctx.saved_tensors
        dx, dg, dw = _bwd_op(grad.contiguous(), x, g, weight, ctx.activation, ctx.eps)
        return dx, dg, dw, None, None

    _op.register_autograd(_bwd, setup_context=_setup)
    return _op


def _fla_rmsnorm_gated_compiled_forward(
    self, x, g, residual=None, prenorm=False, residual_in_fp32=False
):
    # Custom-op path only for the common Qwen3.5 case (no residual/prenorm, no bias);
    # anything else falls back to FLA's eager implementation.
    if residual is None and not prenorm and self.bias is None:
        return _FLA_RMSNORM_GATED_OP(x, g, self.weight, self.activation, self.eps)
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(
        x,
        g,
        self.weight,
        self.bias,
        self.activation,
        residual=residual,
        eps=self.eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )


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

        if compile_boundary and not getattr(
            FusedRMSNormGated, "_axolotl_compile_boundary", False
        ):
            # FLA's fused gated-norm backward (layer_norm_gated_bwd) calls
            # aten.as_strided in a way torch.compile cannot meta-trace on torch
            # >= 2.12 ("mismatch in length of strides and shape"), so dynamo bails
            # on the whole attention frame. Prefer the custom-op wrapper (keeps it
            # in-graph, no break); fall back to an eager boundary if it can't build.
            global _FLA_RMSNORM_GATED_OP
            try:
                if _FLA_RMSNORM_GATED_OP is None:
                    _FLA_RMSNORM_GATED_OP = _build_fla_rmsnorm_gated_op()
                FusedRMSNormGated.forward = _fla_rmsnorm_gated_compiled_forward
                FusedRMSNormGated._axolotl_compile_boundary = True
            except Exception:  # pragma: no cover
                try:
                    import torch._dynamo as _dyn

                    FusedRMSNormGated.forward = _dyn.disable(FusedRMSNormGated.forward)
                    FusedRMSNormGated._axolotl_compile_boundary = True
                except Exception:
                    pass

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
        # Run flash-attention eager under torch.compile. With gradient
        # checkpointing OFF, Inductor otherwise fuses the FA2 backward with the
        # gated-output/o_proj dgrad into a region that yields corrupt (~1e5)
        # input gradients on packed sequences; the gate multiply right after
        # attention is what makes this region fusable on Qwen3.5 (plain-attention
        # models such as Qwen3-VL are unaffected). Breaking the graph around the
        # one attention call avoids the fusion while the rest stays compiled.
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

        cu_seqlens = None
        if not use_precomputed_states and position_ids is not None:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        # mixed_qkv stays [B, T, D]; only transposed inside paths that require [B, D, T]
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, T, D]

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

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

            if fla_causal_conv1d is not None and cu_seqlens is not None:
                # FLA varlen kernel for packed sequences; input must be contiguous [B, T, D]
                conv_fn = (
                    _call_fla_causal_conv1d_disabled
                    if _FLA_CAUSAL_CONV_COMPILE_BOUNDARY
                    else _call_fla_causal_conv1d
                )
                mixed_qkv, _ = conv_fn(
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

        return self.out_proj(core_attn_out)

    return patched_forward


def _apply_packing_patches(
    model_type: str,
    cls_prefix: str,
    forward_factory,
    *,
    fla_causal_conv_compile_boundary: bool = False,
) -> None:
    module_name = f"transformers.models.{model_type}.modeling_{model_type}"

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return

    set_fla_causal_conv_compile_boundary(fla_causal_conv_compile_boundary)
    _inject_fla_kernels(module, compile_boundary=fla_causal_conv_compile_boundary)
    getattr(module, f"{cls_prefix}DecoderLayer").forward = _patched_decoder_forward
    gated_cls = getattr(module, f"{cls_prefix}GatedDeltaNet")
    gated_cls.forward = forward_factory(module.apply_mask_to_padding_states)

    LOG.info(
        f"Applied {cls_prefix} packing patch "
        f"(fla_causal_conv1d={'available' if fla_causal_conv1d else 'unavailable'}, "
        f"compile_boundary={fla_causal_conv_compile_boundary})"
    )


def patch_qwen3_5_modeling_packing(*, fla_causal_conv_compile_boundary: bool = False):
    _apply_packing_patches(
        "qwen3_5",
        "Qwen3_5",
        _make_qwen3_5_gated_delta_forward,
        fla_causal_conv_compile_boundary=fla_causal_conv_compile_boundary,
    )


def patch_qwen3_5_moe_modeling_packing(
    *, fla_causal_conv_compile_boundary: bool = False
):
    _apply_packing_patches(
        "qwen3_5_moe",
        "Qwen3_5Moe",
        _make_qwen3_5_gated_delta_forward,
        fla_causal_conv_compile_boundary=fla_causal_conv_compile_boundary,
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

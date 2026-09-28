"""Monkeypatches for Qwen4-Exp (Qwen3.8-Flash-Next): QSA indexer vectorization and sample packing.

Three independent packing defects are addressed, all gated on `position_ids` being
available (i.e. never active for unpacked runs or cached decoding):

- the gated DeltaNet reads `cu_seq_lens_q` out of kwargs, which nothing populates
  without flash-attention, so its recurrence and depthwise conv run unbroken
  across packed document boundaries;
- the PLE n-gram embedding segments context by matching `config.eos_token_id`
  literally and never looks at positions, so packed boundaries are invisible to it;
- the PLE dilated depthwise conv mixes the previous document's last 9 tokens into
  the next document's first 9.

The QSA indexer rewrite is a pure speedup and applies unconditionally.
"""

import importlib
import math
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.integrations.accelerate import force_accelerate_hooks

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODULE_NAME = "transformers.models.qwen4_exp.modeling_qwen4_exp"

# private kwargs channel: transformers gives decoder layers no positional info
POSITION_IDS_KEY = "axolotl_position_ids"
CU_SEQLENS_KEY = "axolotl_cu_seqlens"

_ORIGINALS: dict[str, object] = {}
_QSA_ORIGINAL_FORWARD = None
_QSA_PATCHED = False


def _import_modeling():
    try:
        return importlib.import_module(MODULE_NAME)
    except ImportError:
        LOG.warning("qwen4_exp not found in transformers, skipping patch")
        return None


def _load_fla():
    """(causal_conv1d, chunk_gated_delta_rule), each None when flash-linear-attention is missing."""
    try:
        from fla.modules.convolution import (
            causal_conv1d,  # FLA >= 0.4.1
        )
    except ImportError:
        try:
            from fla.modules.conv import causal_conv1d  # FLA < 0.4.1
        except ImportError:
            causal_conv1d = None

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError:
        chunk_gated_delta_rule = None

    return causal_conv1d, chunk_gated_delta_rule


def text_position_ids(position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The [B, T] temporal axis, mirroring Qwen4ExpTextModel.forward's MRoPE unpacking."""
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] in (1, 4):
        return position_ids[0]
    return None


def get_cu_seqlens(position_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """Cumulative sequence lengths over the flattened batch, or None when unpacked.

    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}
    position_ids = position_ids.reshape(-1)
    indices_q = (position_ids == position_ids.min()).nonzero().view(-1)
    if indices_q.numel() < 2:
        return None
    return torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )


# --- Sample packing ---------------------------------------------------------


def _make_text_model_forward(original):
    """Publish the temporal position_ids (and their cu_seqlens) into the layer kwargs."""

    @wraps(original)
    def patched_forward(self, *args, **kwargs):
        if kwargs.get("past_key_values") is None:
            position_ids = text_position_ids(kwargs.get("position_ids"))
            if position_ids is not None:
                cu_seqlens = get_cu_seqlens(position_ids)
                if cu_seqlens is not None:
                    kwargs[POSITION_IDS_KEY] = position_ids
                    kwargs[CU_SEQLENS_KEY] = cu_seqlens
        return original(self, *args, **kwargs)

    return patched_forward


def _patched_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    conv_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    ple_input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    """Decoder layer forward routing packing metadata to the PLE and linear attention."""
    position_ids = kwargs.pop(POSITION_IDS_KEY, None)
    cu_seqlens = kwargs.pop(CU_SEQLENS_KEY, None)

    if self.ple is not None:
        hidden_states = hidden_states + self.ple(
            hidden_states,
            ple_input_ids,
            past_key_values,
            conv_mask=conv_mask,
            position_ids=position_ids,
        )

    hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(
        hidden_states
    )
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states,
            cache_params=past_key_values,
            attention_mask=conv_mask,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)

    hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(
        hidden_states
    )
    hidden_states = self.mlp(hidden_states)

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)
    return hidden_states


def _make_gated_delta_forward(module, fla_causal_conv1d, fla_chunk_gated_delta_rule):
    """Factory for the GatedDeltaNet forward that honours packed document boundaries."""

    @force_accelerate_hooks("conv1d")
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = module.apply_mask_to_padding_states(
            hidden_states, attention_mask
        )

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and (
            cache_params.has_previous_state(self.layer_idx, state_idx=0)
        )
        if use_precomputed_states:
            cu_seqlens = None

        if cu_seqlens is not None and (
            fla_causal_conv1d is None or fla_chunk_gated_delta_rule is None
        ):
            # the transformers fallbacks accept cu_seqlens but ignore it, which would
            # silently mix tokens across packed samples
            raise RuntimeError(
                "Packed sequences require flash-linear-attention (cu_seqlens support "
                "in causal_conv1d and chunk_gated_delta_rule). Install "
                "flash-linear-attention or disable sample_packing."
            )

        def flatten(tensor):
            # FLA's varlen kernels treat the whole batch as one sequence
            return tensor.reshape(1, batch_size * seq_len, *tensor.shape[2:])

        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if (
            use_precomputed_states
            and seq_len == 1
            and not cache_params.layers[self.layer_idx].record_past
        ):
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = module.causal_conv1d_update(
                mixed_qkv.transpose(1, 2),
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            ).transpose(1, 2)
        elif cu_seqlens is not None:
            if cache_params is not None:
                cache_params.update_conv_state(
                    mixed_qkv.transpose(1, 2),
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            mixed_qkv, _ = fla_causal_conv1d(
                x=flatten(mixed_qkv),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            mixed_qkv = mixed_qkv.reshape(batch_size, seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(1, 2)
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = module.causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
            mixed_qkv = mixed_qkv.transpose(1, 2)

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

        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if use_precomputed_states
            else None
        )
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = (
                module.torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        elif cu_seqlens is not None:
            core_attn_out, last_recurrent_state = fla_chunk_gated_delta_rule(
                flatten(query),
                flatten(key),
                flatten(value),
                g=flatten(g),
                beta=flatten(beta),
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            core_attn_out, last_recurrent_state = module.torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)

    return patched_forward


def _patched_shift_right_ignore_eos(
    self,
    token_ids: torch.Tensor,
    shift: int,
    position_in_segment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Upstream shift, with packed boundaries folded into the eos-derived segment starts."""
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
        dim=1,
    )
    segment_start = previous_eos + 1
    eos_position_in_segment = positions.unsqueeze(0) - segment_start
    if position_in_segment is not None:
        # a boundary from either source resets the n-gram context
        eos_position_in_segment = torch.minimum(
            eos_position_in_segment, position_in_segment
        )
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (eos_position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))


def _patched_ngram_forward(
    self,
    input_ids: torch.Tensor,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_ids = input_ids.long()
    has_cached_context = (
        past_key_values is not None
        and past_key_values.has_previous_state(self.layer_idx, state_idx=2)
    )
    if has_cached_context:
        previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
    else:
        previous_context = input_ids.new_full(
            (input_ids.shape[0], self.context_len), self.eos_token_id
        )
    if past_key_values is not None:
        input_ids_to_cache = input_ids
        if not has_cached_context and input_ids.shape[1] < self.context_len:
            input_ids_to_cache = torch.nn.functional.pad(
                input_ids_to_cache,
                (self.context_len - input_ids.shape[1], 0),
                value=self.eos_token_id,
            )
        _ = past_key_values.update_conv_state(
            input_ids_to_cache,
            self.layer_idx,
            state_idx=2,
            conv_kernel_size=self.context_len,
        )

    position_in_segment = None
    if position_ids is not None and not has_cached_context:
        # the eos filler standing in for the missing context is its own segment
        position_in_segment = torch.cat(
            [
                position_ids.new_zeros((position_ids.shape[0], self.context_len)),
                position_ids,
            ],
            dim=-1,
        )

    token_history = torch.cat([previous_context, input_ids], dim=-1)
    shifted_tokens = [
        self._shift_right_ignore_eos(token_history, shift, position_in_segment)
        for shift in range(self.ngram_size)
    ]

    blocks = []
    for ngram in range(2, self.ngram_size + 1):
        start_idx = (ngram - 2) * self.heads_per_ngram
        end_idx = start_idx + self.heads_per_ngram
        mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(
                mixed_ids,
                shifted_tokens[position] * self.layer_multipliers[position],
            )
        head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
        head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
        ngram_ids = torch.remainder(
            mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1)
        )
        blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

    ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
    return (
        self.ngram_embedding(ngram_ids.to(self.ngram_embedding.weight.device))
        .to(ngram_ids.device)
        .flatten(-2)
    )


@force_accelerate_hooks("conv1d")
def _patched_short_conv(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    hidden_states = hidden_states.transpose(1, 2)

    if past_key_values is not None:
        hidden_states = past_key_values.update_conv_state(
            hidden_states,
            self.layer_idx,
            state_idx=1,
            conv_kernel_size=self.short_conv_state_len,
        )

    hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
    hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]

    if position_ids is None:
        hidden_states = F.silu(self.conv1d(hidden_states))
        return hidden_states.transpose(1, 2)

    # taps reaching past the document start must be dropped
    weight = self.conv1d.weight
    dilation = self.conv1d.dilation[0]
    kernel_size = weight.shape[-1]
    output = None
    for tap in range(kernel_size):
        offset = (kernel_size - 1 - tap) * dilation
        start = self.short_conv_state_len - offset
        term = hidden_states[..., start : start + seq_len] * weight[:, 0, tap].view(
            1, -1, 1
        )
        if offset > 0:
            term = term * (position_ids >= offset).unsqueeze(1)
        output = term if output is None else output + term

    return F.silu(output).transpose(1, 2)


def _patched_ple_forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    past_key_values=None,
    conv_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    embeddings = self.ple_embedding(input_ids, past_key_values, position_ids)
    key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    value = self.value_proj(embeddings)
    query_normed = self.norm_query(hidden_states).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
        self.hidden_size
    )
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = self.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    if conv_mask is not None:
        modeling = importlib.import_module(MODULE_NAME)
        gated_value = modeling.apply_mask_to_padding_states(gated_value, conv_mask)
        gated_value_normed = modeling.apply_mask_to_padding_states(
            gated_value_normed, conv_mask
        )
    return gated_value + self._short_conv(
        gated_value_normed, past_key_values, position_ids
    )


def patch_qwen4_exp_modeling_packing():
    """Route packed-sequence boundaries into linear attention and the PLE n-gram layer."""
    modeling = _import_modeling()
    if modeling is None or _ORIGINALS:
        return None

    _ORIGINALS.update(
        text_model=modeling.Qwen4ExpTextModel.forward,
        decoder_layer=modeling.Qwen4ExpTextDecoderLayer.forward,
        gated_delta=modeling.Qwen4ExpTextGatedDeltaNet.forward,
        ngram=modeling.Qwen4ExpTextNGramEmbedding.forward,
        shift_right=modeling.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos,
        ple=modeling.Qwen4ExpTextPLELayer.forward,
        short_conv=modeling.Qwen4ExpTextPLELayer._short_conv,
    )

    modeling.Qwen4ExpTextModel.forward = _make_text_model_forward(
        _ORIGINALS["text_model"]
    )
    modeling.Qwen4ExpTextDecoderLayer.forward = _patched_decoder_forward
    fla_causal_conv1d, fla_chunk_gated_delta_rule = _load_fla()
    modeling.Qwen4ExpTextGatedDeltaNet.forward = _make_gated_delta_forward(
        modeling, fla_causal_conv1d, fla_chunk_gated_delta_rule
    )
    modeling.Qwen4ExpTextNGramEmbedding.forward = _patched_ngram_forward
    modeling.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos = (
        _patched_shift_right_ignore_eos
    )
    modeling.Qwen4ExpTextPLELayer.forward = _patched_ple_forward
    modeling.Qwen4ExpTextPLELayer._short_conv = _patched_short_conv

    LOG.info(
        "Applied Qwen4Exp packing patch "
        f"(fla_causal_conv1d={'available' if fla_causal_conv1d else 'unavailable'})"
    )

    def unpatch():
        if not _ORIGINALS:
            return
        modeling.Qwen4ExpTextModel.forward = _ORIGINALS["text_model"]
        modeling.Qwen4ExpTextDecoderLayer.forward = _ORIGINALS["decoder_layer"]
        modeling.Qwen4ExpTextGatedDeltaNet.forward = _ORIGINALS["gated_delta"]
        modeling.Qwen4ExpTextNGramEmbedding.forward = _ORIGINALS["ngram"]
        modeling.Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos = _ORIGINALS[
            "shift_right"
        ]
        modeling.Qwen4ExpTextPLELayer.forward = _ORIGINALS["ple"]
        modeling.Qwen4ExpTextPLELayer._short_conv = _ORIGINALS["short_conv"]
        _ORIGINALS.clear()

    return unpatch


# --- QSA indexer ------------------------------------------------------------


def _qsa_original_forward():
    """The pristine upstream forward, resolved lazily so the fallback works unpatched too."""
    global _QSA_ORIGINAL_FORWARD
    if _QSA_ORIGINAL_FORWARD is None:
        modeling = importlib.import_module(MODULE_NAME)
        _QSA_ORIGINAL_FORWARD = modeling.Qwen4ExpTextQSAIndexer.forward
    return _QSA_ORIGINAL_FORWARD


def _contiguous_run_bounds(visible: torch.Tensor):
    """(starts, counts, ok) for a [batch, query, kv] bool visibility mask."""
    positions = torch.arange(visible.shape[-1], device=visible.device)
    counts = visible.sum(-1)
    starts = torch.argmax(visible.to(torch.uint8), dim=-1)
    expected = (positions >= starts.unsqueeze(-1)) & (
        positions < (starts + counts).unsqueeze(-1)
    )
    return starts, counts, torch.equal(expected, visible)


@torch.no_grad()
def qsa_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    """Vectorized replacement for the upstream per-(batch, query) Python loop.

    The loop is data-dependent only through each query's visible set. Whenever that
    set is a contiguous run [start, start + count) (true for causal, left/right
    padded and block-diagonal packed masks), the blocks a query pools over are
    start + compress_ratio * j, so the pooled keys depend only on
    start % compress_ratio. That leaves at most compress_ratio distinct pooled key
    sets per sequence, each shared by every query with that phase. Masks that are
    not contiguous runs fall back to the upstream loop.
    """
    modeling = importlib.import_module(MODULE_NAME)

    batch_size, seq_length, _ = hidden_states.shape
    head_dim = self.index_head_dim
    ratio = self.compress_ratio
    device = hidden_states.device

    # kv_length <= token_budget: top-k plus tail is the whole visible set, for any mask.
    # With a cache we still have to fall through to update_indexer below.
    if attention_mask.shape[-1] <= self.token_budget and past_key_values is None:
        return attention_mask[:, :1]

    full_cos, full_sin = position_embeddings
    current_cos, current_sin = (
        full_cos[:, -seq_length:, :],
        full_sin[:, -seq_length:, :],
    )

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [self.index_n_heads * head_dim, self.index_kv_heads * head_dim],
        dim=-1,
    )
    hidden_shape = (batch_size, seq_length, -1, head_dim)
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = modeling.apply_rotary_pos_emb(
        q, cos=current_cos, sin=current_sin, unsqueeze_dim=2
    )

    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        if attention_mask.shape[-1] <= self.token_budget:
            return attention_mask[:, :1]

    visible = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )
    visible = visible[:, 0]
    kv_length = visible.shape[-1]

    starts, counts, contiguous = _contiguous_run_bounds(visible)
    if not contiguous:
        return _qsa_original_forward()(
            self, hidden_states, position_embeddings, attention_mask, past_key_values
        )

    num_blocks = counts // ratio
    first_block = starts // ratio
    phase = starts - ratio * first_block

    token_mask = torch.zeros(
        (batch_size, seq_length, kv_length + 1), device=device, dtype=torch.bool
    )
    queries = q.float().reshape(batch_size, seq_length * self.index_n_heads, head_dim)
    offsets = torch.arange(ratio, device=device)

    for current_phase in torch.unique(phase[num_blocks > 0]).tolist():
        num_pooled = (kv_length - current_phase) // ratio
        if num_pooled <= 0:
            continue
        block_starts = current_phase + ratio * torch.arange(num_pooled, device=device)

        key_groups = raw_keys.index_select(
            1, (block_starts.unsqueeze(-1) + offsets).flatten()
        )
        key_groups = key_groups.view(batch_size, num_pooled, ratio, head_dim)
        pooled_keys = key_groups.float().mean(dim=2).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys)
        block_key_states = modeling.apply_rotary_pos_emb(
            pooled_keys.unsqueeze(2),
            cos=full_cos.index_select(1, block_starts),
            sin=full_sin.index_select(1, block_starts),
            unsqueeze_dim=2,
        ).squeeze(2)

        scores = torch.matmul(queries, block_key_states.float().transpose(-1, -2))
        scores = scores.view(batch_size, seq_length, self.index_n_heads, num_pooled)
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(head_dim)

        block_ids = torch.arange(num_pooled, device=device)
        valid = (
            (phase.unsqueeze(-1) == current_phase)
            & (block_ids >= first_block.unsqueeze(-1))
            & (block_ids < (first_block + num_blocks).unsqueeze(-1))
        )
        masked = torch.where(valid, scores, scores.new_full((), float("-inf")))
        top_idx = masked.topk(min(self.block_topk, num_pooled), dim=-1).indices

        tokens = (current_phase + ratio * top_idx).unsqueeze(-1) + offsets
        tokens = torch.where(valid.gather(-1, top_idx).unsqueeze(-1), tokens, kv_length)
        token_mask.scatter_(-1, tokens.flatten(2), True)

    if ratio > 1:
        tail = (starts + ratio * num_blocks).unsqueeze(-1) + torch.arange(
            ratio - 1, device=device
        )
        tail = torch.where(tail < (starts + counts).unsqueeze(-1), tail, kv_length)
        token_mask.scatter_(-1, tail, True)

    selected_token_mask = token_mask[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(
            selected_token_mask, attention_mask.new_zeros(()), min_dtype
        )
    return selected_token_mask


def patch_qwen4_exp_qsa_indexer():
    """Replace the QSA indexer's per-query Python loop with the vectorized forward."""
    global _QSA_PATCHED

    modeling = _import_modeling()
    if modeling is None or _QSA_PATCHED:
        return None

    original = _qsa_original_forward()
    modeling.Qwen4ExpTextQSAIndexer.forward = qsa_indexer_forward
    _QSA_PATCHED = True

    def unpatch():
        global _QSA_PATCHED

        modeling.Qwen4ExpTextQSAIndexer.forward = original
        _QSA_PATCHED = False

    LOG.info("Applied Qwen4Exp QSA indexer vectorization patch")
    return unpatch

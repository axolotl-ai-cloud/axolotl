"""ScatterMoE experts forward for the transformers ExpertsInterface.

PEFT LoRA on ``gate_up_proj`` / ``down_proj`` is fused into the
ScatterMoE Triton call via ``parallel_linear_lora``.
"""

import os

import torch

from .mx_weights import selective_mx_weights_fwd, selective_nvfp4_weights_fwd

# NVFP4 + LoRA fast path: dequantize the frozen FP4 experts to bf16 ONCE and run the plain
# bf16 LoRA grouped GEMM (fwd+bwd), instead of the fused kernel that re-dequantizes FP4
# inside the GEMM K-loop. The fused FP4 dequant starves the tensor cores; routing through
# the bf16 grouped GEMM is far faster (~15x fwd+bwd at the expert layer on B200/B300) at the
# cost of a transient bf16 weight (~2x the packed-FP4 bytes), kept transient via the same
# recompute-in-backward recipe. Set AXOLOTL_NVFP4_LORA_FUSED=1 to fall back to the fused path.
_NVFP4_LORA_DEQUANT_BF16 = os.environ.get("AXOLOTL_NVFP4_LORA_FUSED", "0") != "1"
# Within that bf16 fast path, also run the fused dX backward kernel (scatter2scatter_lora_dX)
# instead of the non-fused base scatter2scatter + Python LoRA dX — the same backward fusion the
# MXFP4 fused path forces internally. Default on; AXOLOTL_NVFP4_LORA_NONFUSED_BWD=1 reverts to
# the non-fused backward (fallback + A/B measurement). dA/dB are unaffected (the fused-gather
# threshold leaves the grouped-Gram path, which is already optimal at high expert counts).
_NVFP4_LORA_FUSED_BWD = os.environ.get("AXOLOTL_NVFP4_LORA_NONFUSED_BWD", "0") != "1"
# fp8-READ variant of the NVFP4+LoRA fast path (workstation Blackwell / sm_120). Materialize the
# frozen NVFP4 weight as fp8 (e4m3) instead of bf16 and read it in the grouped GEMM, upcasting to
# bf16 in-register (bf16 MMA, bf16 activations) — half the weight bytes on the bandwidth-bound
# expert GEMM (~1.3x fwd+bwd on sm_120). Routed through the split/non-fused kernels (the fused
# LoRA kernel serializes the fp8 upcast and loses). Default OFF: the datacenter gain is small and
# the split overhead regresses there, and it adds ~2.5% to the (already accepted) NVFP4 weight
# error. AXOLOTL_NVFP4_LORA_FP8_READ=1 enables (NVFP4 + LoRA only). See selective_dequant_kernel.
_NVFP4_LORA_FP8_READ = os.environ.get("AXOLOTL_NVFP4_LORA_FP8_READ", "0") == "1"
from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora
from .selective_dequant_kernel import (
    dequant_nvfp4_fp8_triton,
    dequant_nvfp4_full_triton,
)
from .selective_dequant import (
    get_active_experts,
    is_mxfp4_param,
    is_nvfp4_param,
    remap_expert_indices,
    selective_lora_weights,
)


def _has_peft_wrapper(module):
    """Check if a module's parameter has been wrapped by PEFT ParamWrapper."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(module, attr, None)
            if isinstance(param, ParamWrapper):
                return True
    except ImportError:
        pass
    return False


def _unwrap_experts_lora(experts):
    """Extract base weights and LoRA params from a PEFT-wrapped Experts module.

    Returns:
        (base_experts, gup_lora, down_lora) where each lora is
        (lora_A, lora_B, scaling) or None.
    """
    try:
        from peft.tuners.param_wrapper import ParamWrapper
    except ImportError:
        return experts, None, None

    if not isinstance(getattr(experts, "gate_up_proj", None), ParamWrapper):
        return experts, None, None

    base_experts = experts
    gup_lora = None
    down_lora = None

    gup_param = experts.gate_up_proj
    if isinstance(gup_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gup_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            gup_lora = (sm_A, sm_B, scaling)

    down_param = experts.down_proj
    if isinstance(down_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(down_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            down_lora = (sm_A, sm_B, scaling)

    return base_experts, gup_lora, down_lora


def _get_base_param(param):
    """Get the base tensor from a PEFT ParamWrapper or regular Parameter."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        while isinstance(param, ParamWrapper):
            param = param.original_parameter
    except ImportError:
        pass
    return param


def _fp8_read_arch_ok(device=None) -> bool:
    """Whether the fp8-read path should activate on this GPU.

    fp8-read wins only in the weight-bandwidth-bound regime: workstation Blackwell (sm_120,
    GDDR7). On datacenter Blackwell (sm_100 / sm_103, HBM) the expert GEMM is compute/MMA-bound,
    fp8 ≈ bf16, and the split structure is slower than the fused kernel -> a regression. So gate
    fp8-read to the validated sm_120; everywhere else falls back to the universal bf16-read path.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(device) == (12, 0)
    except Exception:
        return False


def _parallel_linear_maybe_lora(
    x,
    weight,
    top_k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    lora_tuple,
    grouped_in,
    grouped_out,
    gates=None,
    expert_biases=None,
    use_fused_bwd=False,
):
    """Call parallel_linear or parallel_linear_lora depending on whether LoRA is active.

    ``use_fused_bwd`` selects the fused dX / gather backward kernels (the NVFP4->bf16
    fast path opts in, matching what the fused MXFP4 path forces internally).
    """
    if lora_tuple is not None:
        lora_A, lora_B, scaling = lora_tuple
        return parallel_linear_lora(
            x,
            weight,
            top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A,
            lora_B,
            scaling,
            expert_biases=expert_biases,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
            gates=gates,
            use_fused_dX=use_fused_bwd,
            use_fused_gather=use_fused_bwd,
        )
    return parallel_linear(
        x,
        weight,
        top_k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        expert_biases=expert_biases,
        grouped_in=grouped_in,
        grouped_out=grouped_out,
        gates=gates,
    )


def _prepare_weights_and_lora(
    gu_param,
    dn_param,
    sorted_expert_idxs,
    expert_offsets,
    num_experts,
    gup_lora,
    down_lora,
    dtype,
    module=None,
):
    """Resolve the gate_up / down weights (+ routing + LoRA) for the grouped GEMM.

    For an MXFP4 or NVFP4 base with LoRA on both projections, keep the weights packed
    (4-bit) and route through the fused kernel: select the active experts and remap the
    routing / LoRA to that compact set (dequant happens inside the kernel K-loop; NVFP4
    reuses the MXWeights container with linear E4M3 block scales). Otherwise return bf16
    ``[E, in, out]`` weights (dequantizing FP4 explicitly when LoRA is absent, since the
    fused kernel is LoRA-only and the FP4 tensors have no transpose).

    Returns ``(gate_up_weight, down_weight, sorted_expert_idxs, expert_offsets,
    gup_lora, down_lora, use_fused_bwd)``. ``use_fused_bwd`` is True only for the
    NVFP4->bf16 fast path, where the fused dX / gather backward kernels apply (as the
    fused MXFP4 path forces internally); the bf16 base/no-LoRA paths leave it False.
    """
    is_mx, is_nv = is_mxfp4_param(gu_param), is_nvfp4_param(gu_param)
    use_fused_bwd = False
    if (
        is_nv
        and _NVFP4_LORA_DEQUANT_BF16
        and _NVFP4_LORA_FP8_READ
        and _fp8_read_arch_ok(_get_base_param(gu_param).device)
        and gup_lora is not None
        and down_lora is not None
    ):
        # NVFP4 + LoRA fp8-READ PATH (sm_120 / workstation): dequant the frozen NVFP4 experts to
        # fp8 (e4m3) once; the base scatter2scatter reads fp8 and upcasts to bf16 in-register
        # (half the bf16 weight bytes -> ~1.3x fwd+bwd on the bandwidth-bound expert GEMM). Forced
        # onto the SPLIT / NON-fused path (use_fused_bwd=False): the FUSED LoRA kernel serializes
        # the fp8->bf16 upcast (~0.7x) while the base scatter pipelines it cleanly (~1.7x), so
        # split (base fp8 + bf16 LoRA fwd, base fp8 dX + bf16 LoRA dX) wins. No per-expert scale
        # (it does not improve output quality; b19). `scatter2scatter_lora` routes fp8 W -> split,
        # and `ScatterMoELoRA.forward` skips the host bf16 upcast for fp8. A .recipe rebuilds the
        # fp8 weight in backward (transient, like the bf16 path).
        gate_up_weight = dequant_nvfp4_fp8_triton(gu_param, scale_mode="none")[0].transpose(2, 1)
        down_weight = dequant_nvfp4_fp8_triton(dn_param, scale_mode="none")[0].transpose(2, 1)
        use_fused_bwd = False  # split/non-fused: fp8 wins only off the fused kernel
        if module is not None:
            gate_up_weight.recipe = lambda m=module: dequant_nvfp4_fp8_triton(
                _get_base_param(m.gate_up_proj), scale_mode="none"
            )[0].transpose(2, 1)
            down_weight.recipe = lambda m=module: dequant_nvfp4_fp8_triton(
                _get_base_param(m.down_proj), scale_mode="none"
            )[0].transpose(2, 1)
        else:
            gate_up_weight.recipe = lambda p=gu_param: dequant_nvfp4_fp8_triton(
                p, scale_mode="none"
            )[0].transpose(2, 1)
            down_weight.recipe = lambda p=dn_param: dequant_nvfp4_fp8_triton(
                p, scale_mode="none"
            )[0].transpose(2, 1)
    elif (
        is_nv
        and _NVFP4_LORA_DEQUANT_BF16
        and gup_lora is not None
        and down_lora is not None
    ):
        # NVFP4 + LoRA FAST PATH: dequant->bf16 once, then the plain bf16 LoRA grouped GEMM.
        # Returning bf16 [E, in, out] (not an MXWeights) flips ScatterMoELoRA to is_mx=False,
        # selecting the bf16 fwd/bwd kernels. A .recipe that rebuilds bf16 keeps the (larger)
        # transient weight recomputed in backward rather than pinned across depth. Routing and
        # LoRA tuples are passed through unchanged (no active-expert compaction needed: the
        # bf16 grouped GEMM indexes the full [E,...] set, and training routing is dense).
        gate_up_weight = dequant_nvfp4_full_triton(gu_param, dtype).transpose(2, 1)
        down_weight = dequant_nvfp4_full_triton(dn_param, dtype).transpose(2, 1)
        # is_mx=False here (plain bf16 Tensor), so ScatterMoELoRA won't auto-force the
        # fused backward the way the MXFP4 path does — opt in explicitly so the bf16 dX
        # runs the fused scatter2scatter_lora_dX kernel instead of base + Python LoRA.
        use_fused_bwd = _NVFP4_LORA_FUSED_BWD
        if module is not None:
            gate_up_weight.recipe = (
                lambda m=module, d=dtype: dequant_nvfp4_full_triton(
                    _get_base_param(m.gate_up_proj), d
                ).transpose(2, 1)
            )
            down_weight.recipe = (
                lambda m=module, d=dtype: dequant_nvfp4_full_triton(
                    _get_base_param(m.down_proj), d
                ).transpose(2, 1)
            )
        else:
            gate_up_weight.recipe = (
                lambda p=gu_param, d=dtype: dequant_nvfp4_full_triton(p, d).transpose(2, 1)
            )
            down_weight.recipe = (
                lambda p=dn_param, d=dtype: dequant_nvfp4_full_triton(p, d).transpose(2, 1)
            )
    elif (is_mx or is_nv) and gup_lora is not None and down_lora is not None:
        select = selective_mx_weights_fwd if is_mx else selective_nvfp4_weights_fwd
        active = get_active_experts(sorted_expert_idxs, num_experts)
        sorted_expert_idxs, expert_offsets = remap_expert_indices(
            sorted_expert_idxs, expert_offsets, active, num_experts
        )
        gate_up_weight = select(gu_param, active)
        down_weight = select(dn_param, active)
        # Recompute-in-backward recipe: the selective gather materializes a per-layer
        # copy of the active experts' packed qdata + fp32 block scale (~2 GB/layer at
        # dense routing) which ScatterMoELoRA would otherwise pin on ctx across the
        # fwd→bwd boundary, growing peak ~linearly with depth. It is reconstructible from
        # the frozen, resident param, so hand the Function a lightweight recipe to rebuild
        # it in backward instead. Cheap to recompute, expensive to store; frees the
        # forward-resident copy under both no-checkpointing and non-reentrant GC.
        #
        # FSDP2-safe: re-read the LIVE module param at backward (via `module`) rather than
        # capturing the forward-time object — FSDP reshards the param after forward and
        # re-gathers the full weight for the layer's backward, so a captured reference
        # would be the half-size shard (wrong dX shape). On single-GPU it's the same param.
        if module is not None:
            gate_up_weight.recipe = (
                lambda m=module, a=active, f=select: f(_get_base_param(m.gate_up_proj), a)
            )
            down_weight.recipe = (
                lambda m=module, a=active, f=select: f(_get_base_param(m.down_proj), a)
            )
        else:
            gate_up_weight.recipe = lambda p=gu_param, a=active, f=select: f(p, a)
            down_weight.recipe = lambda p=dn_param, a=active, f=select: f(p, a)
        gup_lora = (
            *selective_lora_weights(gup_lora[0], gup_lora[1], active, num_experts),
            gup_lora[2],
        )
        down_lora = (
            *selective_lora_weights(down_lora[0], down_lora[1], active, num_experts),
            down_lora[2],
        )
    elif is_mx or is_nv:
        # quantized base without LoRA: the fused kernel is LoRA-only and the FP4
        # tensors have no transpose, so dequantize to bf16 here.
        gate_up_weight = gu_param.dequantize(dtype).transpose(2, 1)
        down_weight = dn_param.dequantize(dtype).transpose(2, 1)
    else:
        gate_up_weight = gu_param.transpose(2, 1)  # bf16 [E, out, in] -> [E, in, out]
        down_weight = dn_param.transpose(2, 1)
    return (
        gate_up_weight,
        down_weight,
        sorted_expert_idxs,
        expert_offsets,
        gup_lora,
        down_lora,
        use_fused_bwd,
    )


def scattermoe_supports_layout(self) -> bool:
    """True iff this experts module uses the standard layout scattermoe handles:
    gate_up concatenated as [E, 2I, H], gated SwiGLU, no expert bias. gpt_oss-style
    experts (interleaved gate/up, transposed [E, H, 2I], expert bias) return False."""
    return not (
        getattr(self, "is_transposed", False)
        or not getattr(self, "is_concatenated", True)
        or getattr(self, "has_bias", False)
        or not getattr(self, "has_gate", True)
    )


def _check_supported_layout(self):
    """Reject expert layouts the fixed transpose/chunk below would miscompute."""
    # gpt_oss-style experts (interleaved gate/up, transposed [E, H, 2I], expert
    # bias) would be silently miscomputed by the fixed transpose/chunk below, so
    # reject rather than corrupt training.
    if not scattermoe_supports_layout(self):
        raise NotImplementedError(
            "scattermoe supports only concatenated, non-transposed, gated, biasless "
            "experts (qwen/mixtral/deepseek/glm/...). This model's experts use an "
            "unsupported layout; use use_sonicmoe or a built-in experts_implementation."
        )


def _is_gptoss_layout(self) -> bool:
    """gpt_oss expert layout: transposed [E, H, 2I] weights, interleaved gate/up,
    per-expert bias, clamped sigmoid-GLU. Handled by the Triton path below."""
    return (
        getattr(self, "is_transposed", False)
        and not getattr(self, "is_concatenated", True)
        and getattr(self, "has_bias", False)
        and getattr(self, "has_gate", True)
        and hasattr(self, "gate_up_proj_bias")
        and hasattr(self, "down_proj_bias")
    )


def _gptoss_glu(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """gpt_oss clamped sigmoid-GLU over interleaved gate/up columns (matches
    ``GptOssExperts._apply_gate``)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def _scattermoe_gptoss_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE forward for gpt_oss-style experts (transposed/interleaved/biased).

    gpt_oss stores weights already in ``[E, in, out]`` form, so no transpose; gate/up
    are interleaved (``[..., ::2]`` / ``[..., 1::2]``); per-expert bias is folded into
    the grouped GEMM; the activation is the clamped sigmoid-GLU. LoRA fuses exactly as
    in the standard path (in/out dims match), and the down bias is added per row before
    the routing-weight combine.
    """
    K = top_k_index.shape[1]
    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    gate_up_weight = _get_base_param(
        self.gate_up_proj
    )  # [E, H, 2I], already [E,in,out]
    down_weight = _get_base_param(self.down_proj)  # [E, I, H]
    gate_up_bias = _get_base_param(self.gate_up_proj_bias)  # [E, 2I]
    down_bias = _get_base_param(self.down_proj_bias)  # [E, H]

    gup_lora, down_lora = None, None
    sm_lora = getattr(self, "_scattermoe_lora", None)
    if sm_lora:
        gup_lora = sm_lora.get("gate_up_proj")
        down_lora = sm_lora.get("down_proj")
    elif _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    gate_up = _parallel_linear_maybe_lora(
        hidden_states,
        gate_up_weight,
        K,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gup_lora,
        grouped_in=False,
        grouped_out=True,
        expert_biases=gate_up_bias,
    )
    h = _gptoss_glu(gate_up, getattr(self, "alpha", 1.702), getattr(self, "limit", 7.0))

    output = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        down_lora,
        grouped_in=True,
        grouped_out=False,
        gates=routing_weights,
        expert_biases=down_bias,
    )
    return output


def scattermoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE experts forward with fused-LoRA support."""
    if not scattermoe_supports_layout(self):
        if _is_gptoss_layout(self):
            return _scattermoe_gptoss_forward(
                self, hidden_states, top_k_index, top_k_weights
            )
        _check_supported_layout(self)  # raises for any other unsupported layout

    K = top_k_index.shape[1]

    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    # Extract LoRA params if PEFT is active
    gup_lora, down_lora = None, None
    sm_lora = getattr(self, "_scattermoe_lora", None)
    if sm_lora:
        gup_lora = sm_lora.get("gate_up_proj")
        down_lora = sm_lora.get("down_proj")
    elif _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    (
        gate_up_weight,
        down_weight,
        sorted_expert_idxs,
        expert_offsets,
        gup_lora,
        down_lora,
        use_fused_bwd,
    ) = _prepare_weights_and_lora(
        _get_base_param(self.gate_up_proj),
        _get_base_param(self.down_proj),
        sorted_expert_idxs,
        expert_offsets,
        self.num_experts,
        gup_lora,
        down_lora,
        hidden_states.dtype,
        module=self,
    )

    # Gate-up projection (with optional LoRA)
    gates_h = _parallel_linear_maybe_lora(
        hidden_states,
        gate_up_weight,
        K,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gup_lora,
        grouped_in=False,
        grouped_out=True,
        use_fused_bwd=use_fused_bwd,
    )
    gates, h = gates_h.chunk(2, dim=-1)
    # Clamped SwiGLU when the model defines a swiglu_limit (e.g. DeepSeek-V4: limit=10) —
    # must match the eager experts' `_apply_gate` (gate.clamp(max=L); up.clamp(-L, L)),
    # otherwise outlier activations blow up the residual stream.
    _limit = getattr(self, "limit", None)
    if _limit is not None:
        gates = gates.clamp(max=_limit)
        h = h.clamp(min=-_limit, max=_limit)
    h = self.act_fn(gates) * h

    # Down projection (with optional LoRA + routing weights)
    output = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        down_lora,
        grouped_in=True,
        grouped_out=False,
        gates=routing_weights,
        use_fused_bwd=use_fused_bwd,
    )

    return output


def scattermoe_experts_forward_ep(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE experts forward for the DeepEP local path, skipping EP sentinels.

    After DeepEP dispatch ``top_k_index`` holds local expert ids in ``[0, E_local)``
    for slots this rank owns and ``-1`` for slots routed to remote ranks. Rather than
    map sentinels to expert 0 / weight 0 and run the full grouped GEMM over all
    ``N*K`` rows (compute-and-mask), drop the sentinel rows so only the valid routed
    rows hit the GEMM + per-row LoRA. Output matches the masked path since sentinel
    slots carry weight 0.

    Runs both projections fully grouped (the sentinel-compacted routing breaks the
    ``L_scattered == X.rows * k`` fan-out contract of the scattered path), with the
    weighted token-combine done via ``index_add_``.
    """
    _check_supported_layout(self)

    N = hidden_states.size(0)
    K = top_k_index.shape[1]
    E_local = self.num_experts

    idx_flat = top_k_index.reshape(-1)
    valid = idx_flat >= 0
    e_v = idx_flat[valid].to(torch.long)
    if e_v.numel() == 0:
        return torch.zeros_like(hidden_states)

    tok = torch.arange(N, device=hidden_states.device).repeat_interleave(K)
    tok_v = tok[valid]
    w_v = top_k_weights.reshape(-1)[valid].to(hidden_states.dtype)

    # Sort valid rows by expert so each expert's rows are contiguous (grouped layout).
    se, order = torch.sort(e_v)
    tok_sorted = tok_v[order].to(torch.int32)
    w_sorted = w_v[order]
    expert_offsets = torch.bincount(e_v, minlength=E_local).cumsum(-1)
    M = se.size(0)
    ss = torch.arange(M, device=hidden_states.device, dtype=torch.int32)

    gup_lora, down_lora = None, None
    sm_lora = getattr(self, "_scattermoe_lora", None)
    if sm_lora:
        gup_lora = sm_lora.get("gate_up_proj")
        down_lora = sm_lora.get("down_proj")
    elif _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    gate_up_weight, down_weight, se, expert_offsets, gup_lora, down_lora, use_fused_bwd = (
        _prepare_weights_and_lora(
            _get_base_param(self.gate_up_proj),
            _get_base_param(self.down_proj),
            se,
            expert_offsets,
            E_local,
            gup_lora,
            down_lora,
            hidden_states.dtype,
        )
    )

    # Pre-gather token rows into expert-grouped order; both projections run grouped.
    grouped_x = hidden_states.index_select(0, tok_sorted.to(torch.long))

    gates_h = _parallel_linear_maybe_lora(
        grouped_x,
        gate_up_weight,
        1,
        se,
        ss,
        expert_offsets,
        gup_lora,
        grouped_in=True,
        grouped_out=True,
        use_fused_bwd=use_fused_bwd,
    )
    gates, h = gates_h.chunk(2, dim=-1)
    _limit = getattr(self, "limit", None)
    if _limit is not None:
        gates = gates.clamp(max=_limit)
        h = h.clamp(min=-_limit, max=_limit)
    h = self.act_fn(gates) * h

    down_out = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        se,
        ss,
        expert_offsets,
        down_lora,
        grouped_in=True,
        grouped_out=True,
        use_fused_bwd=use_fused_bwd,
    )
    down_out = down_out * w_sorted.unsqueeze(-1)

    output = hidden_states.new_zeros((N, down_out.size(-1)))
    output.index_add_(0, tok_sorted.to(torch.long), down_out)
    return output


_SCATTERMOE_PATCHED = False


def _ensure_single_lora_ops():
    """Guarantee a single module instance of the autotuned ``kernels.lora_ops``.

    A second instance (same file imported under a different module name) would carry its own
    Triton ``Autotuner`` caches — duplicate autotuning (wasted) and confusing telemetry where
    the same kernel+shape+dtype appears twice with different configs. We canonicalize on the
    axolotl import path and alias any duplicate in ``sys.modules`` to it, so later imports
    resolve to one module. Idempotent; warns only if a real duplicate is found."""
    import sys

    from axolotl.utils.logging import get_logger

    log = get_logger(__name__)
    canon_name = "axolotl.integrations.kernels.libs.scattermoe_lora.kernels.lora_ops"
    canon = sys.modules.get(canon_name)
    if canon is None:
        return
    canon_file = getattr(canon, "__file__", None)
    aliased = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is canon or name == canon_name or "lora_ops" not in name:
            continue
        if getattr(mod, "__file__", None) == canon_file:
            sys.modules[name] = canon
            aliased += 1
    if aliased:
        log.warning(
            "Aliased %d duplicate lora_ops module instance(s) to %s (single autotune cache)",
            aliased, canon_name,
        )


def register_scattermoe_experts():
    """Register ``"scattermoe"`` in the ExpertsInterface and the validator allowlist.

    Idempotent.
    """
    global _SCATTERMOE_PATCHED

    _ensure_single_lora_ops()

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    ALL_EXPERTS_FUNCTIONS.register("scattermoe", scattermoe_experts_forward)

    # transformers' lazy-import can leave TWO `integrations.moe` module objects (two
    # `ExpertsInterface` classes / registries). `@use_experts_implementation` (used by
    # Gemma4 / DeepSeek-V4 experts) binds its registry as a *default argument*, which can
    # be the OTHER object than the one imported above — so a model forward's
    # `get_interface("scattermoe")` would miss our registration. Register into the exact
    # interface that decorator defaults to as well.
    try:
        import inspect

        from transformers.integrations import use_experts_implementation

        canon = inspect.signature(use_experts_implementation).parameters[
            "experts_interface"
        ].default
        if canon is not None and "scattermoe" not in canon:
            canon.register("scattermoe", scattermoe_experts_forward)
    except (ImportError, KeyError, AttributeError):
        pass

    # Route PEFT target_parameters expert LoRA to the fused kernel (bypassing the
    # parametrization merge) for experts-interface MoEs (Gemma4, DiffusionGemma, DeepSeek-V4).
    try:
        from .experts_lora_fastpath import patch_paramwrapper_fastpath

        patch_paramwrapper_fastpath()
    except (ImportError, AttributeError):
        pass

    # Safety net for any path that still merges `base + delta` on a frozen FP4 base
    # (e.g. merge_and_unload): make aten.add dequantize the FP4 tensor.
    try:
        from .torchao_fp4_add import patch_torchao_fp4_add

        patch_torchao_fp4_add()
    except (ImportError, AttributeError):  # torchao optional / API drift
        pass

    # FSDP2 support for NVFP4Tensor (split/view/as_strided + all-gather hooks) so the
    # quantized expert weights can be sharded across GPUs — torchao ships none.
    try:
        from .nvfp4_fsdp import patch_nvfp4_fsdp

        patch_nvfp4_fsdp()
    except (ImportError, AttributeError):
        pass

    if _SCATTERMOE_PATCHED:
        return

    _original_get_correct = PreTrainedModel.get_correct_experts_implementation

    def _patched_get_correct(self_model, requested_experts: str | None) -> str:
        if requested_experts == "scattermoe":
            return "scattermoe"
        return _original_get_correct(self_model, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = _patched_get_correct
    _SCATTERMOE_PATCHED = True

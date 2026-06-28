"""ScatterMoE experts forward for the transformers ExpertsInterface.

PEFT LoRA on ``gate_up_proj`` / ``down_proj`` is fused into the
ScatterMoE Triton call via ``parallel_linear_lora``.
"""

import functools

import torch
import torch.nn.functional as F

from .mx_weights import selective_mx_weights_fwd, selective_nvfp4_weights_fwd
from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora
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
):
    """Call parallel_linear or parallel_linear_lora depending on whether LoRA is active."""
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
    gup_lora, down_lora)``.
    """
    is_mx, is_nv = is_mxfp4_param(gu_param), is_nvfp4_param(gu_param)
    if (is_mx or is_nv) and gup_lora is not None and down_lora is not None:
        select = selective_mx_weights_fwd if is_mx else selective_nvfp4_weights_fwd
        active = get_active_experts(sorted_expert_idxs, num_experts)
        sorted_expert_idxs, expert_offsets = remap_expert_indices(
            sorted_expert_idxs, expert_offsets, active, num_experts
        )
        gate_up_weight = select(gu_param, active)
        down_weight = select(dn_param, active)
        # Recompute-in-backward recipe: the selective gather is a per-layer copy (~2 GB/layer at
        # dense routing) that ScatterMoELoRA would otherwise pin on ctx across all layers (peak
        # grows with depth). It is reconstructible from the frozen resident param, so hand the
        # Function a recipe to rebuild it in backward (cheap to recompute, expensive to store).
        # FSDP2-safe: re-read the LIVE module param at backward (via `module`), not the forward-time
        # object: FSDP reshards after forward and re-gathers for backward, so a captured reference
        # would be the half-size shard (wrong dX shape). On single-GPU it's the same param.
        if module is not None:
            gate_up_weight.recipe = lambda m=module, a=active, f=select: f(
                _get_base_param(m.gate_up_proj), a
            )
            down_weight.recipe = lambda m=module, a=active, f=select: f(
                _get_base_param(m.down_proj), a
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
        # quantized base without LoRA: the fused kernel is LoRA-only and the FP4 tensors have no
        # transpose, so dequantize to bf16 here.
        gate_up_weight = gu_param.dequantize(dtype).transpose(2, 1)
        down_weight = dn_param.dequantize(dtype).transpose(2, 1)
    else:
        gate_up_weight = gu_param.transpose(2, 1)
        down_weight = dn_param.transpose(2, 1)
    return (
        gate_up_weight,
        down_weight,
        sorted_expert_idxs,
        expert_offsets,
        gup_lora,
        down_lora,
    )


def _detect_act_type(module) -> str:
    """Detect gated-activation type from an experts module's act_fn.

    Returns 'gelu_tanh' for Gemma4-style GeGLU (gelu_pytorch_tanh * up),
    'silu' for DSV4-style clamped SwiGLU (silu(clamp(gate)) * clamp(up)).
    Falls back to 'silu' for any unrecognized activation.
    """
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        return "silu"
    fn_name = (
        getattr(act_fn, "__name__", "") or getattr(type(act_fn), "__name__", "") or ""
    )
    if "gelu" in fn_name.lower():
        return "gelu_tanh"
    try:
        if isinstance(act_fn, functools.partial) and act_fn.func is F.gelu:
            return "gelu_tanh"
    except Exception:
        pass
    # last resort: compare act_fn output to gelu_pytorch_tanh numerically
    try:
        x = torch.tensor([0.5], dtype=torch.float32, device="cpu")
        ref = F.gelu(x, approximate="tanh")
        got = act_fn(x.clone())
        if torch.allclose(ref, got, atol=1e-5):
            return "gelu_tanh"
    except Exception:
        pass
    return "silu"


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
    )  # already [E, in, out], no transpose
    down_weight = _get_base_param(self.down_proj)
    gate_up_bias = _get_base_param(self.gate_up_proj_bias)
    down_bias = _get_base_param(self.down_proj_bias)

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


# thin compat wrappers over the centralized runtime module (runtime.py)
from .runtime import RUNTIME  # noqa: E402


def set_fp4_grouped_mode(mode):
    RUNTIME.fp4_grouped_mode = mode


def set_fp4_dx_prefer_fp8(flag):
    RUNTIME.fp4_dx_prefer_fp8 = bool(flag)


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

    gup_lora, down_lora = None, None
    sm_lora = getattr(self, "_scattermoe_lora", None)
    if sm_lora:
        gup_lora = sm_lora.get("gate_up_proj")
        down_lora = sm_lora.get("down_proj")
    elif _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    # Config-gated grouped fp4 path (NVFP4 experts + LoRA + available fp4 backend); faster +
    # lower-memory than the fused-Triton path at scale.
    _fp4_grouped_mode = RUNTIME.fp4_grouped_mode
    if _fp4_grouped_mode is not None and gup_lora is not None and down_lora is not None:
        gu_base = _get_base_param(self.gate_up_proj)
        dn_base = _get_base_param(self.down_proj)
        if is_nvfp4_param(gu_base):
            from .grouped_train import grouped_fp4_available, grouped_fp4_moe_train

            # No LoRA-capable base GEMM here; the legacy fused-MX kernel SIGSEGVs on Blackwell, so
            # hard-error rather than crash.
            if not grouped_fp4_available(_fp4_grouped_mode):
                _cap = (
                    "sm%d%d" % torch.cuda.get_device_capability()
                    if torch.cuda.is_available()
                    else "cpu"
                )
                raise RuntimeError(
                    f"dsv4_fp4_grouped_mode={_fp4_grouped_mode!r} with LoRA selected the grouped "
                    f"NVFP4 MoE path, but no fused grouped backend resolved on this arch ({_cap}): "
                    "marlin, cutlass, and deepgemm are all unavailable. The legacy fused-MX kernel "
                    "(scatter2scatter_lora_mx) is unsafe on this arch (SIGSEGV on Blackwell), so it "
                    "is not used as a fallback. Run on a supported GPU (sm80+/sm90/sm100/sm120) or "
                    "disable dsv4_fp4_grouped_mode."
                )
            # FSDP-safe: backward re-reads the (re-gathered) params via this recipe
            _recipe = lambda: (  # noqa: E731
                _get_base_param(self.gate_up_proj),
                _get_base_param(self.down_proj),
            )
            # persistent per-module cache so the backend requantizes the frozen weight to mxfp4
            # once, surviving FSDP re-gathers (the gathered param is fresh each step).
            cache = self.__dict__.setdefault("_dg_mxfp4_cache", {})
            return grouped_fp4_moe_train(
                hidden_states,
                top_k_index,
                routing_weights,
                gu_base,
                dn_base,
                gup_lora,
                down_lora,
                getattr(self, "limit", None),
                _fp4_grouped_mode,
                act_type=_detect_act_type(self),
                weight_recipe=_recipe,
                mxfp4_cache=cache,
                prefer_fp8_dx=RUNTIME.fp4_dx_prefer_fp8,
            )

    # NVFP4 experts + LoRA without dsv4_fp4_grouped_mode set would fall through to
    # _prepare_weights_and_lora -> selective_nvfp4_weights_fwd -> scatter2scatter_lora_mx, which
    # SIGSEGVs on Blackwell. Turn that silent crash into an actionable error (the grouped path is the
    # supported route for NVFP4+LoRA); non-Blackwell archs keep the legacy path.
    if (
        _fp4_grouped_mode is None
        and gup_lora is not None
        and down_lora is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10
        and is_nvfp4_param(_get_base_param(self.gate_up_proj))
    ):
        raise RuntimeError(
            "NVFP4 experts with LoRA require the grouped fp4 MoE path on Blackwell (sm100/sm120): "
            "the legacy fused-MX kernel (scatter2scatter_lora_mx) SIGSEGVs on this arch. Set "
            "dsv4_fp4_grouped_mode: nvfp4 in your config to enable the supported grouped GEMM path."
        )

    # BnB-4bit experts have no 4-bit-read kernel; the naive path full-dequants all E experts every
    # forward (~2.7x VRAM). Route to the chunked-dequant grouped MoE (bounded transient). Detect
    # WITHOUT touching self.gate_up_proj (would full-dequant).
    _bnb_experts = (
        hasattr(self, "parametrizations")
        and "gate_up_proj" in self.parametrizations
        and "down_proj" in self.parametrizations
    )
    _bnb_fast = False
    if _bnb_experts:
        from .chunked_bnb import bnb_fast_enabled

        _bnb_fast = bnb_fast_enabled()
    if _bnb_experts and not _bnb_fast:
        from .chunked_bnb import chunked_bnb_moe

        return chunked_bnb_moe(
            hidden_states,
            top_k_index,
            routing_weights,
            self,
            gup_lora,
            down_lora,
            self.num_experts,
            act_type=_detect_act_type(self),
            limit=getattr(self, "limit", None),
        )

    if _bnb_experts:
        # bnb_fast: selective dequant of active experts -> 1-launch parallel_linear instead of the
        # chunked torch._grouped_mm storm. Holds bf16 for backward.
        from .selective_dequant import selective_expert_weights

        active = get_active_experts(sorted_expert_idxs, self.num_experts)
        sorted_expert_idxs, expert_offsets = remap_expert_indices(
            sorted_expert_idxs, expert_offsets, active, self.num_experts
        )
        gate_up_weight = selective_expert_weights(
            self, "gate_up_proj", active
        ).transpose(2, 1)
        down_weight = selective_expert_weights(self, "down_proj", active).transpose(
            2, 1
        )
        # Recompute-in-backward recipe: the selective dequant is a per-layer bf16 copy of the active
        # experts that ScatterMoELoRA would otherwise pin across all layers (~40 GB). The frozen
        # 4-bit param is resident, so re-run the dequant in backward via the closure.
        gate_up_weight.recipe = lambda m=self, a=active: selective_expert_weights(
            m, "gate_up_proj", a
        ).transpose(2, 1)
        down_weight.recipe = lambda m=self, a=active: selective_expert_weights(
            m, "down_proj", a
        ).transpose(2, 1)
        if gup_lora is not None and down_lora is not None:
            gup_lora = (
                *selective_lora_weights(
                    gup_lora[0], gup_lora[1], active, self.num_experts
                ),
                gup_lora[2],
            )
            down_lora = (
                *selective_lora_weights(
                    down_lora[0], down_lora[1], active, self.num_experts
                ),
                down_lora[2],
            )
    else:
        (
            gate_up_weight,
            down_weight,
            sorted_expert_idxs,
            expert_offsets,
            gup_lora,
            down_lora,
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
    )
    gates, h = gates_h.chunk(2, dim=-1)
    # Clamped SwiGLU when the model defines a swiglu_limit (e.g. DeepSeek-V4 limit=10) must match the
    # eager experts' `_apply_gate` (gate.clamp(max=L); up.clamp(-L, L)), else outliers blow up.
    _limit = getattr(self, "limit", None)
    if _limit is not None:
        gates = gates.clamp(max=_limit)
        h = h.clamp(min=-_limit, max=_limit)
    h = self.act_fn(gates) * h

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

    gate_up_weight, down_weight, se, expert_offsets, gup_lora, down_lora = (
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
            aliased,
            canon_name,
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

    # transformers' lazy-import can leave TWO `integrations.moe` registries.
    # `@use_experts_implementation` binds its registry as a default argument, which can be the OTHER
    # object than the one imported above, so register into the decorator's default interface too.
    try:
        import inspect

        from transformers.integrations import use_experts_implementation

        canon = (
            inspect.signature(use_experts_implementation)
            .parameters["experts_interface"]
            .default
        )
        if canon is not None and "scattermoe" not in canon:
            canon.register("scattermoe", scattermoe_experts_forward)
    except (ImportError, KeyError, AttributeError):
        pass

    # Route PEFT target_parameters expert LoRA to the fused kernel (bypassing the parametrization
    # merge) for experts-interface MoEs.
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

    # FSDP2 support for NVFP4Tensor (split/view/as_strided + all-gather hooks) so the quantized
    # expert weights can be sharded across GPUs (torchao ships none).
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

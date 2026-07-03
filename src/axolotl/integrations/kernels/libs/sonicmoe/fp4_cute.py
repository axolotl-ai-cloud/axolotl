"""SM100 grouped NVFP4 (W4A4) GEMM host driver, composed from quack.

Instantiates quack's ``GemmGatedSm100`` / ``GemmDefaultSm100`` with
``sf_vec_size=16`` (block-scaled NVFP4 mainloop + gated or default epilogue,
zero quack kernel changes) and drives it with our own compile path:
quack's stock drivers never marry the blockscaled mainloop with the act/gated
epilogue, and its varlen host helpers guard fp4 off. Verified against quack
f4f54db0 (v0.5.3); pin quack, internals are private API.

Conventions:
- All GEMMs are ``C[e] = A[e] @ B[e]^T`` per expert, A/B K-major, fp32 accum.
- Grouped mode is varlen_m: A rows expert-sorted and packed (NO padding),
  ``cu_seqlens (E+1,) int32``; only SFA storage pads (see sf_layout).
- The gated epilogue consumes INTERLEAVED gate/up along N:
  ``postact[:, j] = act(alpha * H[:, 2j], alpha * H[:, 2j+1])``. Concat-layout
  weights must be row-permuted at load (``sf_layout.gate_up_interleave_perm``),
  and the stored preact D comes out interleaved (deinterleave for backward).
- ``alpha`` scales the fp32 accumulator before the activation and before the
  preact store: use it for the activation global scale; fold the per-expert
  weight ``per_tensor_scale`` into SFB (``sf_layout.fold_per_tensor_scale``).

GPU-only (SM100/SM110); quack/cutlass import lazily.
"""

from __future__ import annotations

import functools

import torch

try:
    from .sf_layout import fold_per_tensor_scale, pack_scales_blocked
except ImportError:  # loaded standalone by the pod smoke scripts
    from sf_layout import (  # type: ignore[no-redef]
        fold_per_tensor_scale,
        pack_scales_blocked,
    )

SF_VEC_SIZE = 16


@functools.lru_cache(maxsize=1)
def fp4_cute_available() -> bool:
    """True iff CUDA SM100/SM110 with quack + cutlass DSL importable."""
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability()[0] not in (10, 11):
            return False
        import cutlass  # noqa: F401
        import quack.gemm_act  # noqa: F401

        return True
    except Exception:
        return False


_COMPILE_CACHE: dict = {}


def _compile_kernel(
    n: int,
    k: int,
    num_experts: int,
    gated: bool,
    activation: str,
    has_d: bool,
    tile_mn: tuple,
    cluster_mn: tuple,
    varlen_m: bool,
    add_to_output: bool,
    sfa_sample: torch.Tensor,
    sfb_sample: torch.Tensor,
):
    """Compile once per static shape family; M (or total_m) stays dynamic."""
    from functools import partial

    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32
    from quack.activation import gate_fn_map
    from quack.blockscaled_gemm_utils import _make_compile_tensor_like
    from quack.compile_utils import make_fake_tensor as fake_tensor
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
    from quack.gemm_act import GemmGatedSm100
    from quack.gemm_default_epi import GemmDefaultSm100
    from quack.gemm_tvm_ffi_utils import (
        compile_gemm_kernel,
        make_fake_scheduler_args,
        make_fake_varlen_args,
        make_scheduler_args,
        make_varlen_args,
    )
    from quack.rounding import RoundingMode
    from quack.varlen_utils import VarlenArguments

    fp4 = cutlass.Float4E2M1FN
    bf16 = cutlass.BFloat16
    e4m3 = cutlass.Float8E4M3FN
    device_capacity = get_device_capacity(sfa_sample.device)

    # All-symbolic fakes (mixing concrete dims with sym M crashes the varlen
    # ragged-TMA construction with std::bad_variant_access). The fp4 K dim
    # needs an explicit even-divisibility hint or the packed-nibble shape
    # check ("stride=1 dim must be divisible by 2") rejects the compile.
    m_sym = cute.sym_int()
    n_sym = cute.sym_int(divisibility=8)
    k_sym = cute.sym_int(divisibility=32)
    l_sym = cute.sym_int()
    pa_sym = cute.sym_int(divisibility=8)
    a_shape = (m_sym, k_sym) if varlen_m else (m_sym, k_sym, l_sym)
    d_shape = (m_sym, n_sym) if varlen_m else (m_sym, n_sym, l_sym)
    pa_shape = (m_sym, pa_sym) if varlen_m else (m_sym, pa_sym, l_sym)
    mA = fake_tensor(fp4, a_shape, leading_dim=1, divisibility=32)
    mB = fake_tensor(fp4, (n_sym, k_sym, l_sym), leading_dim=1, divisibility=32)
    mD = fake_tensor(bf16, d_shape, leading_dim=1, divisibility=8) if has_d else None
    mAux = fake_tensor(bf16, pa_shape, leading_dim=1, divisibility=8) if gated else None

    if gated:
        assert not add_to_output, "add_to_output is only wired for the default epilogue"
        gemm_cls = partial(GemmGatedSm100, sf_vec_size=SF_VEC_SIZE)
        compile_epi_args = GemmGatedSm100.EpilogueArguments(
            mAux,
            gate_fn_map[activation],
            alpha=Float32(0.0),
            rounding_mode=RoundingMode.RN,
        )
    else:
        gemm_cls = partial(GemmDefaultSm100, sf_vec_size=SF_VEC_SIZE)
        # add_to_output is a Constexpr: True swaps the D TMA store for a
        # reduce-add, so the kernel accumulates into the caller's out buffer.
        compile_epi_args = GemmDefaultSm100.EpilogueArguments(
            alpha=Float32(0.0),
            add_to_output=add_to_output,
            rounding_mode=RoundingMode.RN,
        )

    compiled = compile_gemm_kernel(
        gemm_cls,
        fp4,
        tuple(tile_mn),
        (cluster_mn[0], cluster_mn[1], 1),
        False,  # pingpong (SM90-only knob)
        True,  # persistent
        False,  # gather_A: unsupported with blockscaled (SF gather missing upstream)
        True,  # is_dynamic_persistent -> use_clc_persistence on SM100
        device_capacity,
        mA,
        mB,
        mD,
        None,
        compile_epi_args,
        make_fake_scheduler_args(False, False, cute.sym_int()),
        make_fake_varlen_args(varlen_m, False, False, None) or VarlenArguments(),
        mSFA=_make_compile_tensor_like(sfa_sample, e4m3, dynamic_layout=True),
        mSFB=_make_compile_tensor_like(sfb_sample, e4m3, dynamic_layout=True),
    )

    max_active = get_max_active_clusters(cluster_mn[0] * cluster_mn[1])

    def run(a, b, d, postact, sfa, sfb, cu_seqlens, alpha):
        if gated:
            epi_args = GemmGatedSm100.EpilogueArguments(
                postact, None, alpha=Float32(alpha), rounding_mode=None
            )
        else:
            # Constexpr fields (add_to_output, rounding_mode) must be None at
            # call time; the namedtuple default False fails the FFI signature.
            epi_args = GemmDefaultSm100.EpilogueArguments(
                alpha=Float32(alpha), add_to_output=None, rounding_mode=None
            )
        scheduler_args = make_scheduler_args(max_active, 8, None)
        varlen_args = make_varlen_args(cu_seqlens, None, None) or VarlenArguments()
        compiled(a, b, d, None, epi_args, scheduler_args, varlen_args, sfa, sfb)

    return run


def _get_run(key, sfa_sample, sfb_sample):
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = _compile_kernel(*key[:-1], sfa_sample, sfb_sample)
    return _COMPILE_CACHE[key]


def _as_fp4x2(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float4_e2m1fn_x2) if t.dtype == torch.uint8 else t


def _check_dims(n: int, k: int, gated: bool, tile_mn: tuple) -> None:
    assert k % 32 == 0, f"K={k} must be divisible by 32 (fp4 16B alignment)"
    assert n % 8 == 0, f"N={n} must be divisible by 8 (bf16 output alignment)"
    assert tile_mn[0] in (128, 256) and tile_mn[1] in (64, 128, 192, 256), (
        f"blockscaled tile must be M in (128,256), N in (64,128,192,256), got {tile_mn}"
    )
    if gated:
        assert n % 2 == 0 and (n // 2) % 8 == 0, f"gated N={n} needs (N/2) % 8 == 0"


class GroupedNvfp4Gemm:
    """Grouped (varlen_m) NVFP4 W4A4 GEMM engine, one weight slice per expert.

    Build once per (N, K, E, gated, activation, tile); ``set_weights`` once per
    weight tensor; ``forward`` per step with dynamic total_m.
    """

    def __init__(
        self,
        n: int,
        k: int,
        num_experts: int,
        *,
        gated: bool = False,
        activation: str = "swiglu",
        store_preact: bool = True,
        tile_mn: tuple = (128, 128),
        cluster_mn: tuple = (1, 1),
    ):
        _check_dims(n, k, gated, tile_mn)
        self.n, self.k, self.num_experts = n, k, num_experts
        self.gated = gated
        self.activation = activation
        self.store_preact = store_preact
        self.tile_mn = tuple(tile_mn)
        self.cluster_mn = tuple(cluster_mn)
        self._b_operand = None
        self._sfb = None
        self._runs: dict = {}

    def set_weights(
        self,
        qdata: torch.Tensor,
        block_scale: torch.Tensor,
        per_tensor_scale: torch.Tensor | None = None,
    ) -> None:
        """qdata ``(E, N, K/2)`` uint8 K-packed; block_scale ``(E, N, K/16)`` e4m3.

        Gated engines expect gate/up rows already INTERLEAVED (apply
        ``gate_up_interleave_perm`` to both tensors for concat-layout weights).
        per_tensor_scale ``(E,)``/``(E,1,1)``/scalar fp32 is folded into SFB.
        """
        e, n, k2 = qdata.shape
        assert (e, n, k2 * 2) == (self.num_experts, self.n, self.k), (
            f"qdata {tuple(qdata.shape)} vs engine (E={self.num_experts}, N={self.n}, K={self.k})"
        )
        assert block_scale.shape == (e, n, self.k // SF_VEC_SIZE)
        if per_tensor_scale is not None:
            block_scale, _ = fold_per_tensor_scale(block_scale, per_tensor_scale)
        # (E, N, K/2) contiguous -> (N, K/2, E) K-major view, as the kernel wants.
        self._b_operand = _as_fp4x2(qdata.contiguous().permute(1, 2, 0))
        self._sfb = pack_scales_blocked(block_scale)

    def forward(
        self,
        a_packed: torch.Tensor,
        sfa_blocked: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        alpha: float = 1.0,
        preact_out: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        add_to_output: bool = False,
    ):
        """a_packed ``(total_m, K/2)`` uint8/fp4x2, expert-sorted, unpadded.
        sfa_blocked from ``sf_layout.build_varlen_sfa``. cu_seqlens ``(E+1,)``.

        ``add_to_output=True`` (non-gated only) accumulates ``alpha * acc``
        into a caller-provided ``out`` instead of overwriting it.

        Returns ``(postact [total_m, N/2] bf16, preact [total_m, N] bf16 | None)``
        for gated engines, else ``out [total_m, N]`` bf16.
        """
        assert self._b_operand is not None, "call set_weights first"
        if add_to_output:
            assert not self.gated and out is not None
        total_m = a_packed.shape[0]
        assert a_packed.shape[1] * 2 == self.k
        device = a_packed.device
        a = _as_fp4x2(a_packed)
        cu = cu_seqlens.to(device=device, dtype=torch.int32)

        d = None
        postact = None
        if self.gated:
            postact = torch.empty(
                total_m, self.n // 2, dtype=torch.bfloat16, device=device
            )
            if self.store_preact:
                d = (
                    preact_out
                    if preact_out is not None
                    else torch.empty(
                        total_m, self.n, dtype=torch.bfloat16, device=device
                    )
                )
        else:
            d = (
                out
                if out is not None
                else torch.empty(total_m, self.n, dtype=torch.bfloat16, device=device)
            )

        run = self._runs.get(add_to_output)
        if run is None:
            key = (
                self.n,
                self.k,
                self.num_experts,
                self.gated,
                self.activation,
                d is not None,
                self.tile_mn,
                self.cluster_mn,
                True,  # varlen_m
                add_to_output,
                device.index,
            )
            run = _get_run(key, self._sfb.new_zeros(1, 1, 1, 512), self._sfb)
            self._runs[add_to_output] = run
        run(a, self._b_operand, d, postact, sfa_blocked, self._sfb, cu, float(alpha))
        return (postact, d) if self.gated else d


def dense_nvfp4_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    gated: bool = False,
    activation: str = "swiglu",
    alpha: float = 1.0,
    store_preact: bool = True,
    tile_mn: tuple = (128, 128),
    cluster_mn: tuple = (1, 1),
):
    """Batched dense (non-varlen) NVFP4 GEMM: the composition smoke path.

    a_q ``(L, M, K/2)`` uint8, a_scale ``(L, M, K/16)`` e4m3; b likewise with N.
    Returns ``(postact (L,M,N/2), preact (L,M,N) | None)`` if gated else
    ``out (L, M, N)``, all bf16.
    """
    num_l, m, k2 = a_q.shape
    k = 2 * k2
    n = b_q.shape[1]
    _check_dims(n, k, gated, tile_mn)
    device = a_q.device

    a = _as_fp4x2(a_q.contiguous().permute(1, 2, 0))  # (M, K/2, L) K-major
    b = _as_fp4x2(b_q.contiguous().permute(1, 2, 0))
    sfa = pack_scales_blocked(a_scale)
    sfb = pack_scales_blocked(b_scale)

    d3 = postact3 = None
    d = postact = None
    if gated:
        postact3 = torch.empty(num_l, m, n // 2, dtype=torch.bfloat16, device=device)
        postact = postact3.permute(1, 2, 0)
        if store_preact:
            d3 = torch.empty(num_l, m, n, dtype=torch.bfloat16, device=device)
            d = d3.permute(1, 2, 0)
    else:
        d3 = torch.empty(num_l, m, n, dtype=torch.bfloat16, device=device)
        d = d3.permute(1, 2, 0)

    key = (
        n,
        k,
        num_l,
        gated,
        activation,
        d is not None,
        tuple(tile_mn),
        tuple(cluster_mn),
        False,  # varlen_m
        False,  # add_to_output
        device.index,
    )
    run = _get_run(key, sfa, sfb)
    run(a, b, d, postact, sfa, sfb, None, float(alpha))
    return (postact3, d3) if gated else d3

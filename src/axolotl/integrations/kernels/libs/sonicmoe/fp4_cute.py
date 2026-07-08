"""SM100 grouped NVFP4 (W4A4) GEMM host driver, composed from quack.

Instantiates quack's ``GemmGatedSm100`` / ``GemmDefaultSm100`` with
``sf_vec_size=16`` (block-scaled NVFP4 mainloop + gated or default epilogue,
zero quack kernel changes) and drives it with our own compile path:
quack's stock drivers never marry the blockscaled mainloop with the act/gated
epilogue, and its varlen host helpers guard fp4 off. Verified against quack
source at f4f54db0; runtime pin quack-kernels==0.5.0, internals are private API.

Conventions:
- All GEMMs are ``C[e] = A[e] @ B[e]^T`` per expert, A/B K-major, fp32 accum.
- Grouped mode is varlen_m: A rows expert-sorted and packed (NO padding),
  ``cu_seqlens (E+1,) int32``; only SFA storage pads (see sf_layout).
- The gated epilogue consumes INTERLEAVED gate/up along N:
  ``postact[:, j] = act(alpha * H[:, 2j], alpha * H[:, 2j+1])``. Concat-layout
  weights must be row-permuted at load (``sf_layout.gate_up_interleave_perm``),
  and the stored preact D comes out interleaved (deinterleave for backward).
- ``alpha`` scales the fp32 accumulator before the activation and before the
  preact store: use it for the activation global scale. Per-expert weight
  ``per_tensor_scale`` is applied EXACTLY via the epilogue colvec variant
  (``forward(colvec=...)``: a per-row fp32 vector multiplied into the fp32
  accumulator before the D store); the lossy SFB fold
  (``sf_layout.fold_per_tensor_scale``) remains available for A/B debugging.

GPU-only (SM100/SM110); quack/cutlass import lazily.
"""

from __future__ import annotations

import functools

import torch

try:
    from .sf_layout import (
        fold_per_tensor_scale,
        gate_up_interleave_perm,
        pack_scales_blocked,
    )
except ImportError:  # loaded standalone (no package parent)
    from sf_layout import (  # type: ignore[no-redef]
        fold_per_tensor_scale,
        gate_up_interleave_perm,
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


@functools.lru_cache(maxsize=1)
def _rowscale_gemm_cls():
    """``GemmDefaultSm100`` with the stock colvec ADD swapped for an exact fp32
    per-row MULTIPLY on the accumulator (quack's ``vec_multiply``), applied
    before the D convert/store and thus also before the ``add_to_output``
    reduce-add."""
    from typing import Optional

    import cutlass.cute as cute
    import quack.utils as utils
    from cutlass import const_expr
    from quack.epi_ops import vec_multiply
    from quack.gemm_default_epi import GemmDefaultSm100

    class GemmDefaultRowScaleSm100(GemmDefaultSm100):
        @cute.jit
        def epi_visit_subtile(
            self,
            params,
            epi_loop_tensors,
            tRS_rD: cute.Tensor,
            tRS_rC: Optional[cute.Tensor] = None,
        ) -> Optional[cute.Tensor]:
            # alpha (and beta/C) exactly as stock GemmDefaultEpiMixin, but the
            # stock colvec ADD is skipped in favor of the multiply below.
            rD = tRS_rD.load()
            if const_expr(hasattr(params, "alpha") and params.alpha is not None):
                rD *= utils.load_scalar_or_pointer(params.alpha)
            if const_expr(tRS_rC is not None):
                if const_expr(not hasattr(params, "beta") or params.beta is None):
                    rD += tRS_rC.load().to(tRS_rD.element_type)  # type: ignore[union-attr]
                else:
                    beta = utils.load_scalar_or_pointer(params.beta)
                    rD += beta * tRS_rC.load().to(tRS_rD.element_type)  # type: ignore[union-attr]
            tRS_rD.store(rD)
            vec_multiply(self, tRS_rD, epi_loop_tensors.get("mColVecBroadcast"), None)
            return None

    return GemmDefaultRowScaleSm100


@functools.lru_cache(maxsize=1)
def _gated_auxadd_gemm_cls():
    """``GemmGatedSm100`` extended with an exact per-row colvec MULTIPLY (as in
    :func:`_rowscale_gemm_cls`) and a preact-space ``TileLoad`` aux ADD, both on
    the fp32 accumulator BEFORE the gated activation and the preact D store:
    ``postact = act(colvec * (alpha * acc) + aux)``. The aux rides the epilogue
    C load pipeline; its fragment is partitioned against the same register
    layout as D, so the add is element-wise in the interleaved preact space."""
    from typing import Callable, NamedTuple, Optional

    import cutlass
    import cutlass.cute as cute
    import quack.utils as utils
    from cutlass import Float32, Int32, const_expr
    from quack.cute_dsl_utils import mlir_namedtuple
    from quack.epi_ops import (
        ColVecLoad,
        RowVecLoad,
        Scalar,
        TileLoad,
        TileStore,
        vec_multiply,
    )
    from quack.gemm_act import GemmGatedSm100, _gated_epi_tile_fn
    from quack.rounding import RoundingMode

    # Functional NamedTuple form: this module uses `from __future__ import
    # annotations`, so a class-body NamedTuple would store STRING annotations
    # that quack's Constexpr converter later fails to eval (get_type_hints
    # resolves them in THIS module's globals, where cute/cutlass are factory
    # locals). The functional form stores real type objects.
    _EpiArgs = NamedTuple(
        "EpilogueArguments",
        [
            ("mAuxOut", cute.Tensor),
            ("act_fn", cutlass.Constexpr[Optional[Callable]]),
            ("alpha", Optional[Float32 | cute.Tensor]),
            ("beta", Optional[Float32 | cute.Tensor]),
            ("mRowVecBroadcast", Optional[cute.Tensor]),
            ("mColVecBroadcast", Optional[cute.Tensor]),
            ("mAuxAdd", Optional[cute.Tensor]),
            ("rounding_mode", cutlass.Constexpr[int]),
            ("sr_seed", Optional[Int32 | cute.Tensor]),
        ],
    )
    _EpiArgs.__new__.__defaults__ = (  # all fields after mAuxOut
        None,
        None,
        None,
        None,
        None,
        None,
        RoundingMode.RN,
        None,
    )
    _EpiArgs = mlir_namedtuple(_EpiArgs)

    class GemmGatedAuxAddSm100(GemmGatedSm100):
        # Stock GemmGatedMixin ops plus the aux TileLoad; __init_subclass__
        # regenerates EpilogueParams from this tuple.
        _epi_ops = (
            Scalar("alpha"),
            Scalar("beta"),
            Scalar("sr_seed", dtype=Int32),
            RowVecLoad("mRowVecBroadcast"),
            ColVecLoad("mColVecBroadcast"),
            TileStore("mAuxOut", epi_tile_fn=_gated_epi_tile_fn),
            TileLoad("mAuxAdd"),
        )

        # NOTE: the aux tensor must be INTERLEAVED in memory. Applying quack's
        # concat_to_interleave view to the TileLoad instead (the trick the
        # mainloop uses for B) device-crashes with an illegal instruction: the
        # epilogue TMA-load path does not survive the hierarchical N mode
        # (probed on B200; mainloop B and the TileStore are fine).
        EpilogueArguments = _EpiArgs

        @cute.jit
        def epi_visit_subtile(
            self,
            params,
            epi_loop_tensors,
            tRS_rD: cute.Tensor,
            tRS_rC: Optional[cute.Tensor] = None,
        ) -> Optional[cute.Tensor]:
            # alpha (and beta/C) exactly as stock GemmDefaultEpiMixin, but the
            # stock colvec ADD is replaced by the exact multiply.
            rD = tRS_rD.load()
            if const_expr(hasattr(params, "alpha") and params.alpha is not None):
                rD *= utils.load_scalar_or_pointer(params.alpha)
            if const_expr(tRS_rC is not None):
                if const_expr(not hasattr(params, "beta") or params.beta is None):
                    rD += tRS_rC.load().to(tRS_rD.element_type)  # type: ignore[union-attr]
                else:
                    beta = utils.load_scalar_or_pointer(params.beta)
                    rD += beta * tRS_rC.load().to(tRS_rD.element_type)  # type: ignore[union-attr]
            tRS_rD.store(rD)
            vec_multiply(self, tRS_rD, epi_loop_tensors.get("mColVecBroadcast"), None)
            tDrAux = epi_loop_tensors.get("mAuxAdd")
            if const_expr(tDrAux is not None):
                rD2 = tRS_rD.load()
                rD2 += tDrAux.load().to(tRS_rD.element_type)  # type: ignore[union-attr]
                tRS_rD.store(rD2)
            # Gated activation, verbatim from GemmGatedMixin's SM100 branch.
            tRS_rAuxOut_layout = cute.recast_layout(2, 1, tRS_rD.layout)
            tRS_rAuxOut = cute.make_rmem_tensor(
                tRS_rAuxOut_layout.shape, self.acc_dtype
            )
            for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]),
                    (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3]),
                )
            return tRS_rAuxOut

    return GemmGatedAuxAddSm100


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
    has_colvec: bool,
    has_aux: bool,
    concat_b: bool,
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

    assert not concat_b or gated, "concat_b (concat-layout weights) is gated-only"
    if gated:
        assert not add_to_output, "add_to_output is only wired for the default epilogue"
        assert not (has_colvec or has_aux) or varlen_m, (
            "gated colvec / aux add are only wired for varlen_m"
        )
        gated_cls = (
            _gated_auxadd_gemm_cls() if (has_colvec or has_aux) else GemmGatedSm100
        )
        gemm_cls = partial(gated_cls, sf_vec_size=SF_VEC_SIZE)
        mColVec = (
            fake_tensor(Float32, (m_sym,), leading_dim=0, divisibility=4)
            if has_colvec
            else None
        )
        # The aux shares mD's fake (same total_m/N and n-major layout): the
        # interleaved preact-space LoRA delta added before the activation.
        mAuxAdd = (
            fake_tensor(bf16, d_shape, leading_dim=1, divisibility=8)
            if has_aux
            else None
        )
        if has_colvec or has_aux:
            compile_epi_args = gated_cls.EpilogueArguments(
                mAux,
                gate_fn_map[activation],
                alpha=Float32(0.0),
                mColVecBroadcast=mColVec,
                mAuxAdd=mAuxAdd,
                rounding_mode=RoundingMode.RN,
            )
        else:
            compile_epi_args = GemmGatedSm100.EpilogueArguments(
                mAux,
                gate_fn_map[activation],
                alpha=Float32(0.0),
                rounding_mode=RoundingMode.RN,
            )
    else:
        assert not has_aux, "aux add is only wired for the gated epilogue"
        assert not has_colvec or varlen_m, "colvec row scale requires varlen_m"
        default_cls = _rowscale_gemm_cls() if has_colvec else GemmDefaultSm100
        gemm_cls = partial(default_cls, sf_vec_size=SF_VEC_SIZE)
        # Passing a fake mColVecBroadcast at compile time is what activates the
        # ColVecLoad op (host-side filtering happens during the compile trace);
        # it shares m_sym with mD, the 1-D (total_m,) varlen colvec convention.
        mColVec = (
            fake_tensor(Float32, (m_sym,), leading_dim=0, divisibility=4)
            if has_colvec
            else None
        )
        # add_to_output is a Constexpr: True swaps the D TMA store for a
        # reduce-add, so the kernel accumulates into the caller's out buffer.
        compile_epi_args = GemmDefaultSm100.EpilogueArguments(
            alpha=Float32(0.0),
            mColVecBroadcast=mColVec,
            add_to_output=add_to_output,
            rounding_mode=RoundingMode.RN,
        )

    # Zero-copy concat weights: the kernel VIEWS B's concat [gate; up] rows as
    # interleaved (hierarchical (2, N/2) layout, quack's own gated-MLP scheme).
    # SFB must then be packed from row-permuted scales (set_weights handles
    # that). The aux TileLoad is NOT listed: it must arrive interleaved in
    # memory (the view trick crashes the epilogue load path, see the subclass).
    concat_layout = ("B",) if concat_b else None

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
        concat_layout=concat_layout,
    )

    max_active = get_max_active_clusters(cluster_mn[0] * cluster_mn[1])

    def run(a, b, d, postact, sfa, sfb, cu_seqlens, alpha, colvec=None, aux=None):
        if gated:
            # Tensor fields compiled non-None must be non-None on every call.
            assert (colvec is not None) == has_colvec
            assert (aux is not None) == has_aux
            if has_colvec or has_aux:
                epi_args = gated_cls.EpilogueArguments(
                    postact,
                    None,
                    alpha=Float32(alpha),
                    mColVecBroadcast=colvec,
                    mAuxAdd=aux,
                    rounding_mode=None,
                )
            else:
                epi_args = GemmGatedSm100.EpilogueArguments(
                    postact, None, alpha=Float32(alpha), rounding_mode=None
                )
        else:
            # Tensor fields compiled non-None must be non-None on every call.
            assert (colvec is not None) == has_colvec
            # Constexpr fields (add_to_output, rounding_mode) must be None at
            # call time; the namedtuple default False fails the FFI signature.
            epi_args = GemmDefaultSm100.EpilogueArguments(
                alpha=Float32(alpha),
                mColVecBroadcast=colvec,
                add_to_output=None,
                rounding_mode=None,
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
        concat_b: bool = False,
        tile_mn: tuple = (128, 128),
        cluster_mn: tuple = (1, 1),
    ):
        """``concat_b=True`` (gated only): ``set_weights`` takes CONCAT
        [gate; up] rows and the kernel views them interleaved zero-copy
        (quack ``concat_layout``); only the small block-scale copy is
        row-permuted for SFB. The preact D still comes out INTERLEAVED."""
        _check_dims(n, k, gated, tile_mn)
        assert not concat_b or gated, "concat_b is gated-only"
        self.n, self.k, self.num_experts = n, k, num_experts
        self.gated = gated
        self.activation = activation
        self.store_preact = store_preact
        self.concat_b = concat_b
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
        ``gate_up_interleave_perm`` to both tensors for concat-layout weights)
        UNLESS built with ``concat_b=True``, in which case both tensors stay in
        CONCAT [gate; up] layout: qdata is consumed zero-copy through the
        kernel's interleaved view, and only the block scales are row-permuted
        here so SFB rows follow the kernel's logical (interleaved) N order.
        per_tensor_scale ``(E,)``/``(E,1,1)``/scalar fp32 is folded into SFB.
        """
        e, n, k2 = qdata.shape
        assert (e, n, k2 * 2) == (self.num_experts, self.n, self.k), (
            f"qdata {tuple(qdata.shape)} vs engine (E={self.num_experts}, N={self.n}, K={self.k})"
        )
        assert block_scale.shape == (e, n, self.k // SF_VEC_SIZE)
        if self.concat_b:
            perm = gate_up_interleave_perm(n, device=block_scale.device)
            block_scale = block_scale.index_select(1, perm)
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
        colvec: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
        preact_out: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        add_to_output: bool = False,
    ):
        """a_packed ``(total_m, K/2)`` uint8/fp4x2, expert-sorted, unpadded.
        sfa_blocked from ``sf_layout.build_varlen_sfa``. cu_seqlens ``(E+1,)``.

        ``colvec`` ``(total_m,)`` fp32 multiplies the fp32 accumulator per row
        in the epilogue: ``d = colvec * (alpha * acc)``, exact (single bf16
        rounding at the D store). On gated engines it applies before the
        activation and the preact store.

        ``aux`` ``(total_m, N)`` bf16 (gated only) is ADDED to the fp32
        accumulator after the colvec multiply, before the activation and the
        preact store: ``preact = colvec * (alpha * acc) + aux`` in the
        INTERLEAVED gate/up space.

        ``add_to_output=True`` (non-gated only) accumulates ``alpha * acc``
        (times ``colvec`` if given) into a caller-provided ``out`` instead of
        overwriting it.

        Returns ``(postact [total_m, N/2] bf16, preact [total_m, N] bf16 | None)``
        for gated engines, else ``out [total_m, N]`` bf16.
        """
        assert self._b_operand is not None, "call set_weights first"
        if add_to_output:
            assert not self.gated and out is not None
        total_m = a_packed.shape[0]
        if colvec is not None:
            assert colvec.dtype == torch.float32 and colvec.shape == (total_m,)
            colvec = colvec.contiguous()
        if aux is not None:
            assert self.gated, "aux add is gated-engine only"
            assert aux.dtype == torch.bfloat16 and aux.shape == (total_m, self.n)
            aux = aux.contiguous()
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

        run_key = (add_to_output, colvec is not None, aux is not None)
        run = self._runs.get(run_key)
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
                colvec is not None,
                aux is not None,
                self.concat_b,
                device.index,
            )
            run = _get_run(key, self._sfb.new_zeros(1, 1, 1, 512), self._sfb)
            self._runs[run_key] = run
        run(
            a,
            self._b_operand,
            d,
            postact,
            sfa_blocked,
            self._sfb,
            cu,
            float(alpha),
            colvec,
            aux,
        )
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
    """Batched dense (non-varlen) NVFP4 GEMM.

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
        False,  # has_colvec
        False,  # has_aux
        False,  # concat_b
        device.index,
    )
    run = _get_run(key, sfa, sfb)
    run(a, b, d, postact, sfa, sfb, None, float(alpha))
    return (postact3, d3) if gated else d3

"""sm120 CUTLASS NVFP4 grouped-GEMM host wrapper.

Contiguous-grouped MoE: tokens sorted by expert + padded per-expert to TILE(128); ONE grouped
GEMM where each m-tile reads its expert's weight via ``m_indices`` (grouped_kernel.py). Base-only
fp4xfp4 (nvfp4) or fp8xfp4 (mxfp). LoRA + swiglu + routing live in the unified dispatcher; this
file is the base GEMM engine. All cutlass/cute imports are lazy (module loads clean off sm120).
"""

from __future__ import annotations

import torch

TILE = 128

# mode -> (a-dtype-name, sf_vec, sf-dtype-name)
_MODES = {
    "nvfp4": ("Float4E2M1FN", 16, "Float8E4M3FN"),
    "fp8": ("Float8E4M3FN", 32, "Float8E8M0FNU"),
}


def _cd(a, b):
    return (a + b - 1) // b


def _helpers():
    """Lazy cute helpers (operand/SF build + inject). Imported only when the sm120 path runs."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils.blackwell_helpers as sm120_utils
    from cutlass.cute.runtime import from_dlpack

    from .grouped_kernel import cvt_sf_MKL_to_M32x4xrm_K4xrk_L  # noqa: F401

    return (
        cutlass,
        cute,
        cutlass_torch,
        from_dlpack,
        sm120_utils,
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
    )


class GroupedFp4Gemm:
    """C[Mt,N] = sorted A[Mt,K] @ B[m_indices[m_tile]]^T. One launch; expert per m-tile.

    Build once per (Mt, N, K, E, mode); inject weights once (set_weights), activation per-forward.
    """

    def __init__(self, Mt, N, K, E, mode="nvfp4"):
        cutlass, cute, cutlass_torch, from_dlpack, sm120_utils, _ = _helpers()
        from cutlass import BFloat16, Float4E2M1FN, Float32

        from .grouped_kernel import Sm120BlockScaledGemmKernel

        a_name, sf_vec, sf_name = _MODES[mode]
        a_dtype = getattr(cutlass, a_name)
        sf_dtype = getattr(cutlass, sf_name)
        self.Mt, self.N, self.K, self.E, self.mode = Mt, N, K, E, mode
        self.sf_vec, self._fp4_a = sf_vec, a_dtype is Float4E2M1FN

        self.a_ct, self.a_st = self._operand(Mt, K, 1, a_dtype)
        self.b_ct, self.b_st = self._operand(N, K, E, Float4E2M1FN)
        self.sfa_ct, self.sfa_st, self.sfa_g, self.sfa_shape = self._sf(
            Mt, K, 1, sf_vec, sf_dtype
        )
        self.sfb_ct, self.sfb_st, self.sfb_g, self.sfb_shape = self._sf(
            N, K, E, sf_vec, sf_dtype
        )
        c_ref = cutlass_torch.matrix(1, Mt, N, False, BFloat16)
        self.c_ct, self.c_st = cutlass_torch.cute_tensor_like(
            c_ref, BFloat16, is_dynamic_layout=True, assumed_align=16
        )
        self.c_ct.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=1
        )
        self.c_ct = cutlass_torch.convert_cute_tensor(
            c_ref.cuda(), self.c_ct, BFloat16, is_dynamic_layout=True
        )
        self.mi = torch.zeros(Mt // TILE, dtype=torch.int32, device="cuda")
        self.mi_ct = from_dlpack(self.mi, assumed_align=4)
        gemm = Sm120BlockScaledGemmKernel(Float32, sf_vec, (128, 128, 128), (128, 128))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        self.stream = cutlass_torch.default_stream()
        self.compiled = cute.compile(
            gemm,
            self.a_ct,
            self.b_ct,
            self.sfa_ct,
            self.sfb_ct,
            self.c_ct,
            self.mi_ct,
            mac,
            self.stream,
        )

    def _operand(self, mn, k, E, op_dtype):
        cutlass, cute, cutlass_torch, _, _, _ = _helpers()
        from cutlass import Float4E2M1FN, Float32

        ref = cutlass_torch.matrix(E, mn, k, False, Float32)
        ct, st = cutlass_torch.cute_tensor_like(
            ref, op_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ct.mark_compact_shape_dynamic(
            mode=1,
            stride_order=(2, 0, 1),
            divisibility=2 if op_dtype is Float4E2M1FN else 1,
        )
        ct = cutlass_torch.convert_cute_tensor(
            ref.cuda(), ct, op_dtype, is_dynamic_layout=True
        )
        return ct, st

    def _sf(self, mn, k, E, sf_vec, sf_dtype):
        cutlass, cute, cutlass_torch, from_dlpack, _, cvt = _helpers()
        sfk = _cd(k, sf_vec)
        mma_shape = (E, _cd(mn, 128), _cd(sfk, 4), 32, 4, 4)
        nat = cutlass_torch.create_and_permute_torch_tensor(
            (E, mn, sfk),
            torch.float64,
            permute_order=(1, 2, 0),
            init_type=cutlass_torch.TensorInitType.SKIP,
        )
        nat.copy_(
            torch.arange(E * mn * sfk, dtype=torch.float64)
            .reshape(E, mn, sfk)
            .permute(1, 2, 0)
        )
        cf = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float64,
            permute_order=(3, 4, 1, 5, 2, 0),
            init_type=cutlass_torch.TensorInitType.SKIP,
        )
        cvt(from_dlpack(nat), from_dlpack(cf))
        gather = cf.reshape(-1).round().long().cuda()
        cf_e = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=(3, 4, 1, 5, 2, 0),
            init_type=cutlass_torch.TensorInitType.SKIP,
        )
        ct, st = cutlass_torch.cute_tensor_like(
            cf_e, sf_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ct = cutlass_torch.convert_cute_tensor(
            cf_e.cuda(), ct, sf_dtype, is_dynamic_layout=True
        )
        return ct, st, gather, tuple(st.shape)

    @staticmethod
    def _inject_operand(st, qdata):
        n = qdata.numel()
        st.view(torch.uint8).permute(2, 0, 1).reshape(-1)[:n].copy_(
            qdata.reshape(-1).view(torch.uint8)
        )

    @staticmethod
    def _inject_sf(st, gather, shape, scale_flat):
        st.view(torch.uint8).copy_(scale_flat.view(torch.uint8)[gather].reshape(shape))

    def set_weights(self, q, s):  # q:[E,N,K/2], s:[E,N,sfk]; one-time per weight gather
        self._inject_operand(self.b_st, q)
        self._inject_sf(self.sfb_st, self.sfb_g, self.sfb_shape, s.reshape(-1))

    def forward(
        self, a_q, a_s, m_indices
    ):  # a_q:[1,Mt,*], a_s:[1,Mt,sfk], m_indices:[Mt/128]
        self._inject_operand(self.a_st, a_q)
        self._inject_sf(self.sfa_st, self.sfa_g, self.sfa_shape, a_s.reshape(-1))
        self.mi.copy_(m_indices)
        self.compiled(
            self.a_ct,
            self.b_ct,
            self.sfa_ct,
            self.sfb_ct,
            self.c_ct,
            self.mi_ct,
            self.stream,
        )
        return self.c_st[:, :, 0]


def quant_act(x, mode):
    """Fast Triton activation quant for the grouped path -> (qdata[Mt,*], scale[Mt,sfk])."""
    if mode == "nvfp4":
        from .quant_nvfp4 import nvfp4_quant

        return nvfp4_quant(x)
    from .quant_mxfp8 import mxfp8_quant

    return mxfp8_quant(x)

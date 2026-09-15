"""SM100 grouped NVFP4 (W4A4) GEMM host driver over quack's public GEMM.

Drives quack's low-level ``quack.gemm.gemm`` (blockscaled NVFP4, varlen_m) with
our own operand prep (``sf_layout`` blocked scales + triton codecs). The GEMM is
non-gated; per-expert ``per_tensor_scale``, the gated activation, and the LoRA
delta are applied AFTER the matmul on the fp32 output, which is exact and matches
the previous fused-epilogue numerics (the matmul is linear in the per-row scale).

Conventions:
- All GEMMs are ``C[e] = A[e] @ B[e]^T`` per expert, A/B K-major, fp32 accum.
- Grouped mode is varlen_m: A rows expert-sorted and packed (NO padding),
  ``cu_seqlens (E+1,) int32``; only SFA storage pads (see sf_layout).
- Gated engines emit an INTERLEAVED preact ``[g0, u0, g1, u1, ...]``
  (``postact[:, j] = act(preact[:, 2j], preact[:, 2j+1])``): ``set_weights`` with
  ``concat_b=True`` row-permutes the concat ``[gate; up]`` weight to that order,
  so the stored preact and the LoRA aux both live in interleaved space (the
  layout ``nvfp4_lora``'s backward expects).
- ``per_tensor_scale`` is applied EXACTLY as a per-row fp32 ``colvec`` multiply on
  the fp32 GEMM output before the single bf16 round; the lossy SFB fold
  (``sf_layout.fold_per_tensor_scale``) stays available for A/B debugging.

Uses quack's LOW-LEVEL ``quack.gemm.gemm`` (not ``gemm_interface.gemm``: the
custom-op/autotune wrapper adds ~4x per-call Python dispatch at these MoE sizes).

GPU-only (SM100/SM110); quack/cutlass import lazily. Runtime pin
quack-kernels 0.6.1 + nvidia-cutlass-dsl 4.6.0.
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
        import quack.gemm  # noqa: F401

        return True
    except Exception:
        return False


def _check_dims(n: int, k: int, gated: bool) -> None:
    assert k % 32 == 0, f"K={k} must be divisible by 32 (fp4 16B alignment)"
    assert n % 8 == 0, f"N={n} must be divisible by 8 (bf16 output alignment)"
    if gated:
        assert n % 2 == 0 and (n // 2) % 8 == 0, f"gated N={n} needs (N/2) % 8 == 0"


def _as_fp4x2(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float4_e2m1fn_x2) if t.dtype == torch.uint8 else t


def _to_sf6(blocked: torch.Tensor) -> torch.Tensor:
    """``(l, r, rk, 512)`` -> ``(l, r, rk, 32, 4, 4)`` (quack's SF atom, free view)."""
    l, r, rk, _ = blocked.shape
    return blocked.view(l, r, rk, 32, 4, 4)


def _blockscaled_tile(total_m: int, n: int) -> tuple[int, int, int, int]:
    """quack ``blockscaled_default_config`` heuristic -> (tile_m, tile_n, cluster_m, cluster_n)."""
    if total_m >= 512 and n >= 256:
        return 256, 256, 2, 1
    if total_m >= 512 and n >= 128:
        return 256, 128, 2, 1
    return 128, 128, 1, 1


def _apply_gated(activation: str, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    if activation == "swiglu":
        return F.silu(gate) * up
    if activation == "geglu":
        return F.gelu(gate, approximate="tanh") * up
    if activation == "reglu":
        return F.relu(gate) * up
    if activation == "glu":
        return torch.sigmoid(gate) * up
    raise NotImplementedError(f"gated activation {activation!r} not supported")


class GroupedNvfp4Gemm:
    """Grouped (varlen_m) NVFP4 W4A4 GEMM engine, one weight slice per expert.

    Build once per (N, K, E, gated, activation); ``set_weights`` once per weight
    tensor; ``forward`` per step with dynamic total_m.
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
        """``concat_b=True`` (gated only): ``set_weights`` takes CONCAT [gate; up]
        rows and row-permutes them to the interleaved order the gated preact uses.
        ``tile_mn``/``cluster_mn`` are accepted for API compatibility but the tile
        is now derived per call from total_m (quack's blockscaled heuristic)."""
        _check_dims(n, k, gated)
        assert not concat_b or gated, "concat_b is gated-only"
        self.n, self.k, self.num_experts = n, k, num_experts
        self.gated = gated
        self.activation = activation
        self.store_preact = store_preact
        self.concat_b = concat_b
        self._b_operand = None  # (E, N, K/2) fp4x2, K-contiguous
        self._sfb = None  # (E, rn, rk, 32, 4, 4) e4m3

    def set_weights(
        self,
        qdata: torch.Tensor,
        block_scale: torch.Tensor,
        per_tensor_scale: torch.Tensor | None = None,
    ) -> None:
        """qdata ``(E, N, K/2)`` uint8 K-packed; block_scale ``(E, N, K/16)`` e4m3.

        Gated engines built with ``concat_b=True`` take concat ``[gate; up]`` rows
        and row-permute BOTH tensors to interleaved order here. per_tensor_scale
        ``(E,)``/``(E,1,1)``/scalar fp32, if given, is folded into SFB (lossy);
        the exact path leaves it out and applies it as a forward colvec.
        """
        e, n, k2 = qdata.shape
        assert (e, n, k2 * 2) == (self.num_experts, self.n, self.k), (
            f"qdata {tuple(qdata.shape)} vs engine (E={self.num_experts}, N={self.n}, K={self.k})"
        )
        assert block_scale.shape == (e, n, self.k // SF_VEC_SIZE)
        if self.gated and self.concat_b:
            perm = gate_up_interleave_perm(n, device=qdata.device)
            qdata = qdata.index_select(1, perm)
            block_scale = block_scale.index_select(1, perm)
        if per_tensor_scale is not None:
            block_scale, _ = fold_per_tensor_scale(block_scale, per_tensor_scale)
        self._b_operand = _as_fp4x2(qdata.contiguous())  # (E, N, K/2) = (l, n, k)
        self._sfb = _to_sf6(pack_scales_blocked(block_scale))

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
        sfa_blocked ``(1, rm, rk, 512)`` from ``sf_layout.build_varlen_sfa``.

        ``colvec`` ``(total_m,)`` fp32 multiplies the fp32 GEMM output per row
        (exact per-expert pts, single bf16 round). ``aux`` ``(total_m, N)`` bf16
        (gated only) is ADDED to the fp32 preact after the colvec, before the
        activation. ``add_to_output=True`` (non-gated) adds the result into a
        caller ``out`` (the LoRA delta) instead of overwriting.

        Returns ``(postact [total_m, N/2], preact [total_m, N] | None)`` bf16 for
        gated engines, else ``out [total_m, N]`` bf16.
        """
        from quack.gemm import gemm as gemm_low

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
        assert a_packed.shape[1] * 2 == self.k
        device = a_packed.device
        qa = _as_fp4x2(a_packed)
        sfa = _to_sf6(sfa_blocked)
        cu = cu_seqlens.to(device=device, dtype=torch.int32)

        acc = torch.empty(total_m, self.n, dtype=torch.float32, device=device)
        tile_m, tile_n, cluster_m, cluster_n = _blockscaled_tile(total_m, self.n)
        gemm_low(
            qa,
            self._b_operand,
            acc,
            None,
            None,
            tile_m,
            tile_n,
            cluster_m,
            cluster_n,
            1,
            persistent=True,
            is_dynamic_persistent=True,
            cu_seqlens_m=cu,
            SFA=sfa,
            SFB=self._sfb,
        )

        pre = acc
        if alpha != 1.0:
            pre = pre * alpha
        if colvec is not None:
            pre = pre * colvec.unsqueeze(1)

        if self.gated:
            if aux is not None:
                pre = pre + aux.float()
            pv = pre.view(total_m, self.n // 2, 2)
            postact = _apply_gated(self.activation, pv[..., 0], pv[..., 1]).to(
                torch.bfloat16
            )
            d = None
            if self.store_preact:
                d = (
                    preact_out
                    if preact_out is not None
                    else torch.empty(
                        total_m, self.n, dtype=torch.bfloat16, device=device
                    )
                )
                d.copy_(pre)
            return postact, d

        if add_to_output:
            # Add in fp32 with a single bf16 round (matches the old fused
            # reduce-add: out holds the bf16 LoRA delta, result is fp32).
            out.copy_((out.float() + pre).to(out.dtype))
            return out
        d = (
            out
            if out is not None
            else torch.empty(total_m, self.n, dtype=torch.bfloat16, device=device)
        )
        d.copy_(pre)
        return d

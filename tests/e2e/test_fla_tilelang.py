"""Tiny optional GPU kernel smoke test; no model loading or distributed launch."""

from importlib.util import find_spec

import pytest
import torch


def test_fla_tilelang_kernels():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if find_spec("tilelang") is None or find_spec("fla") is None:
        pytest.skip("requires FLA and TileLang")
    import tilelang
    from fla.ops.utils.cumsum import chunk_local_cumsum
    from tilelang import language as T

    @tilelang.jit(out_idx=[1])
    def add_one():
        @T.prim_func
        def kernel(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
            with T.Kernel(1, threads=32):
                for i in T.Parallel(64):
                    B[i] = A[i] + 1

        return kernel

    x = torch.arange(64, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(add_one()(x), x + 1)
    values = x.reshape(1, 64, 1)
    torch.testing.assert_close(
        chunk_local_cumsum(values, chunk_size=64), values.cumsum(1)
    )

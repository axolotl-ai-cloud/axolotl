"""Block-fp8 grouped experts whose dims aren't whole 128-blocks (e.g. Gemma-4's
moe_intermediate_size=704): the marlin packs are skipped and DeepGEMM falls back to the masked
Triton kernel. Backend choice only; the GEMMs need CUDA."""

import pytest
import torch

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels

from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    fp8_grouped,
)

E, N_TOK, TOPK, R = 4, 6, 2, 2


@pytest.mark.parametrize(
    "size_n, size_k, ok",
    [
        (256, 256, True),
        (1408, 2816, True),
        (704, 256, False),  # out dim a multiple of 64, not 128
        (256, 704, False),  # reduction a multiple of 16, not 128
        (256, 128, False),  # a single reduction scale group
    ],
)
def test_marlin_fp8_shape_ok(size_n, size_k, ok):
    assert fp8_grouped._marlin_fp8_shape_ok(size_n, size_k) is ok


@pytest.mark.parametrize(
    "hidden, inter, expected",
    [
        (256, 256, "deepgemm_fp8"),
        (256, 704, "blockfp8_triton"),
        (320, 256, "blockfp8_triton"),
    ],
)
def test_deepgemm_needs_128_aligned_reductions(monkeypatch, hidden, inter, expected):
    monkeypatch.setattr(fp8_grouped, "_fp8_train_backend", lambda: "deepgemm_fp8")
    seen = []

    def record(x, base, *args):
        seen.append(base[0])
        return x.new_zeros(x.size(0), hidden)

    monkeypatch.setattr(fp8_grouped._Fp8BlockGroupedExperts, "apply", record)

    def blocks(n):
        return -(-n // 128)

    fp8 = torch.float8_e4m3fn
    gate_up = torch.zeros(E, 2 * inter, hidden, dtype=fp8)
    down = torch.zeros(E, hidden, inter, dtype=fp8)
    gu_scale = torch.ones(E, blocks(2 * inter), blocks(hidden))
    dn_scale = torch.ones(E, blocks(hidden), blocks(inter))
    lora_gu = (torch.zeros(R * E, hidden), torch.zeros(2 * inter, R * E), 1.0)
    lora_dn = (torch.zeros(R * E, inter), torch.zeros(hidden, R * E), 1.0)

    fp8_grouped.grouped_fp8_moe_train(
        torch.randn(N_TOK, hidden, dtype=torch.bfloat16),
        torch.randint(0, E, (N_TOK, TOPK)),
        torch.full((N_TOK, TOPK), 0.5),
        gate_up,
        gu_scale,
        down,
        dn_scale,
        lora_gu,
        lora_dn,
        None,
    )
    assert seen == [expected]

"""Oracle-equivalence tests for the fused ternary Triton kernels.

Every fused op must be bit-identical to its `quant.py` counterpart on CUDA: the fused
and eager paths are chosen per module at runtime, so any drift would change training
numerics depending on where a layer happens to run.
"""

import pytest
import torch

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.kernels import triton as tk

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="triton ternary kernels need CUDA")

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
LAMBDAS = [0.0, 0.37, 1.0]
SHAPES = [(1, 1), (1, 7), (7, 1), (3, 5), (127, 129), (64, 2048), (33, 4097)]


def ref_pack_codes(codes):
    """Independent 4-codes-per-byte packing: lane weights instead of shifts."""
    lanes = codes.reshape(-1).to(torch.int32) + 1
    pad = -lanes.numel() % quant.CODES_PER_BYTE
    if pad:
        lanes = torch.cat([lanes, lanes.new_zeros(pad)])
    weights = torch.tensor([1, 4, 16, 64], device=lanes.device, dtype=torch.int32)
    return (lanes.view(-1, quant.CODES_PER_BYTE) * weights).sum(-1).to(torch.uint8)


def ulp_neighbourhood(value, count=4):
    """`value` plus its `count` nearest fp32 neighbours on either side."""
    values = [value]
    up = down = torch.tensor(value, dtype=torch.float32)
    for _ in range(count):
        up = torch.nextafter(up, torch.tensor(float("inf")))
        down = torch.nextafter(down, torch.tensor(float("-inf")))
        values += [up.item(), down.item()]
    return values


def rand_codes(numel, device="cuda", seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randint(-1, 2, (numel,), generator=generator, device=device)
        .to(torch.int8)
        .reshape(-1)
    )


def test_cpu_tensors_rejected():
    with pytest.raises(RuntimeError, match="CUDA"):
        tk.fake_quant_weight(torch.randn(4, 4), 1.0, scale=torch.tensor(0.5))
    with pytest.raises(RuntimeError, match="CUDA"):
        tk.act_quant(torch.randn(4, 4))
    with pytest.raises(RuntimeError, match="CUDA"):
        tk.pack_codes(torch.zeros(4, dtype=torch.int8))


def test_is_available_matches_device():
    assert tk.is_available() is CUDA


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_fake_quant_matches_oracle(dtype, shape, lambda_):
    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=dtype)
    out = tk.fake_quant_weight(weight, lambda_)
    assert out.dtype == weight.dtype and out.shape == weight.shape
    assert torch.equal(out, quant.fake_quant_weight(weight, lambda_))


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 4096), (2560, 6912)])
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_fake_quant_large_shapes(shape, lambda_):
    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        tk.fake_quant_weight(weight, lambda_),
        quant.fake_quant_weight(weight, lambda_),
    )


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_fake_quant_adversarial_values(dtype, lambda_):
    finfo = torch.finfo(dtype)
    scale = torch.tensor(0.25, device="cuda")
    values = [
        0.0,
        -0.0,
        0.125,  # exact tie -> round-half-even down to 0
        -0.125,
        0.375,  # exact tie -> round-half-even up to 1
        -0.375,
        0.625,  # exact tie -> 2, clipped to 1
        0.1249999,
        1e4,
        -1e4,
        finfo.tiny,
        -finfo.tiny,
        finfo.smallest_normal / 2,  # subnormal
        finfo.max,
        finfo.eps,
    ]
    weight = torch.tensor(values, device="cuda", dtype=dtype)
    assert torch.equal(
        tk.fake_quant_weight(weight, lambda_, scale=scale),
        quant.fake_quant_weight(weight, lambda_, scale=scale),
    )


@requires_cuda
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_fake_quant_tie_neighbourhood(lambda_):
    """Ties and their fp32 neighbours, where the rounding of `w / s` decides the code."""
    scale = torch.tensor(0.030019685626029968, device="cuda")
    values = []
    for multiple in (0.5, 1.5, 2.5):
        values += ulp_neighbourhood(multiple * scale.item())
        values += [-value for value in ulp_neighbourhood(multiple * scale.item())]
    weight = torch.tensor(values, device="cuda", dtype=torch.float32)
    out = tk.fake_quant_weight(weight, lambda_, scale=scale)
    assert torch.equal(out, quant.fake_quant_weight(weight, lambda_, scale=scale))
    codes = quant.ternary_codes(weight, scale)
    assert set(codes.unique().tolist()) == {-1, 0, 1}


@requires_cuda
def test_fake_quant_sub_eps_scale_uses_the_floor():
    """A learnable scale below SCALE_EPS divides by the floor, as the oracle does."""
    weight = torch.full((8,), 3e-6, device="cuda", dtype=torch.float32)
    scale = torch.tensor(1e-9, device="cuda")
    out = tk.fake_quant_weight(weight, 1.0, scale=scale)
    assert torch.equal(out, quant.fake_quant_weight(weight, 1.0, scale=scale))
    assert torch.count_nonzero(out) == 0


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_fake_quant_lambda_one_is_baked_master(dtype):
    """λ == 1 must produce exactly the `codes · s16` values the exporters re-derive."""
    torch.manual_seed(0)
    weight = torch.randn(256, 512, device="cuda", dtype=dtype)
    scale = quant.absmean_scale(weight)
    baked = quant.dequantize_codes(
        quant.ternary_codes(weight, scale), quant.f16_round_scale(scale), dtype
    )
    out = tk.fake_quant_weight(weight, 1.0)
    assert torch.equal(out, baked)
    assert out.unique().numel() <= 3


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", [8, 128, 256])
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_fake_quant_group_scale(dtype, group_size, lambda_):
    torch.manual_seed(0)
    weight = torch.randn(17, 512, device="cuda", dtype=dtype)
    assert torch.equal(
        tk.fake_quant_weight(weight, lambda_, group_size=group_size),
        quant.fake_quant_weight(weight, lambda_, group_size=group_size),
    )


@requires_cuda
def test_fake_quant_group_size_must_divide():
    weight = torch.randn(4, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not divide"):
        tk.fake_quant_weight(weight, 1.0, group_size=32, scale=torch.ones(4, 3).cuda())


@requires_cuda
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_fake_quant_explicit_scale(lambda_):
    """Learnable-scale mode: the caller's scale is used verbatim, not re-derived."""
    torch.manual_seed(0)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    scale = quant.absmean_scale(weight) * 3.7
    out = tk.fake_quant_weight(weight, lambda_, scale=scale)
    assert torch.equal(out, quant.fake_quant_weight(weight, lambda_, scale=scale))
    if lambda_:
        assert not torch.equal(out, quant.fake_quant_weight(weight, lambda_))


@requires_cuda
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_fake_quant_non_contiguous(lambda_):
    torch.manual_seed(0)
    weight = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not weight.is_contiguous()
    assert torch.equal(
        tk.fake_quant_weight(weight, lambda_), quant.fake_quant_weight(weight, lambda_)
    )


@requires_cuda
def test_fake_quant_deterministic():
    torch.manual_seed(0)
    weight = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    first = tk.fake_quant_weight(weight, 0.5)
    assert torch.equal(first, tk.fake_quant_weight(weight, 0.5))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 127), (8, 2048), (3, 9728), (5, 4096), (2, 3, 64)]
)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_act_quant_matches_oracle(dtype, shape, lambda_):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    out = tk.act_quant(x, lambda_)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out, quant.act_quant(x, lambda_))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_act_quant_scale_floor(dtype, lambda_):
    """All-zero and sub-eps rows fall back to the SCALE_EPS floor."""
    rows = [
        [0.0, -0.0, 0.0, 0.0],
        [1e-7, -1e-7, 3e-8, 0.0],
        [1e-4, -2e-4, 5e-5, 1e-5],
    ]
    x = torch.tensor(rows, device="cuda", dtype=dtype)
    assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_act_quant_adversarial_values(dtype, lambda_):
    finfo = torch.finfo(dtype)
    x = torch.tensor(
        [
            [127.0, -127.0, 0.5, -0.5, 63.5, 1.0, -1.0, 0.0],
            [finfo.tiny, -finfo.tiny, 0.0, -0.0, finfo.eps, 1e3, -1e3, 1.0],
        ],
        device="cuda",
        dtype=dtype,
    )
    assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_act_quant_half_amax_ties(dtype, lambda_):
    """`x = amax / 2` lands on the 63.5 tie: the row scale divides an odd 127."""
    amax = 3.8125
    x = torch.tensor(
        [[amax, -amax, amax / 2, -amax / 2, amax / 4, amax / 8, 0.0, amax / 2]],
        device="cuda",
        dtype=dtype,
    )
    assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))


@requires_cuda
@pytest.mark.parametrize("in_features", [4095, 4096, 4097, 16384])
def test_act_quant_multi_tile_rows(in_features):
    """Rows wider than one tile reduce the absmax over several passes."""
    torch.manual_seed(0)
    x = torch.randn(6, in_features, device="cuda", dtype=torch.bfloat16) * 4
    for lambda_ in (0.37, 1.0):
        assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))


@requires_cuda
def test_act_quant_row_scales_are_independent():
    """A huge outlier in one row must not move any other row's codes."""
    torch.manual_seed(0)
    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    x[2, 7] = 5e3
    assert torch.equal(tk.act_quant(x, 1.0), quant.act_quant(x, 1.0))


@requires_cuda
@pytest.mark.parametrize("lambda_", [0.37, 1.0])
def test_act_quant_non_contiguous(lambda_):
    torch.manual_seed(0)
    x = torch.randn(8, 4, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert not x.is_contiguous()
    assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))


@requires_cuda
def test_act_quant_deterministic():
    torch.manual_seed(0)
    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    first = tk.act_quant(x, 0.5)
    assert torch.equal(first, tk.act_quant(x, 0.5))


@requires_cuda
@pytest.mark.parametrize("numel", [1, 2, 3, 4, 5, 127, 4096, 4097])
def test_pack_codes_matches_oracle(numel):
    codes = rand_codes(numel)
    packed = tk.pack_codes(codes)
    assert packed.dtype == torch.uint8
    assert packed.numel() == -(-numel // quant.CODES_PER_BYTE)
    assert torch.equal(packed, quant.pack_codes(codes))
    assert torch.equal(packed, ref_pack_codes(codes))
    assert torch.equal(quant.unpack_codes(packed, (numel,)), codes)


@requires_cuda
def test_pack_codes_shape_independent():
    codes = rand_codes(4096).view(64, 64)
    assert torch.equal(tk.pack_codes(codes), tk.pack_codes(codes.reshape(-1)))


@requires_cuda
def test_pack_codes_non_contiguous():
    codes = rand_codes(4096).view(64, 64).t()
    assert torch.equal(tk.pack_codes(codes), quant.pack_codes(codes))


@requires_cuda
@pytest.mark.parametrize("numel", [1, 3, 4, 130, 8193])
def test_code_stats_matches_oracle(numel):
    prev, now = rand_codes(numel, seed=1), rand_codes(numel, seed=2)
    packed_prev, packed_now = tk.pack_codes(prev), tk.pack_codes(now)
    flips, zeros, total = tk.code_stats(packed_prev, packed_now, numel)

    assert flips.item() == quant.flip_count(packed_prev, packed_now).item()
    assert flips.item() == (prev != now).sum().item()
    assert zeros.item() / numel == pytest.approx(
        quant.zero_fraction(packed_now, numel).item()
    )
    assert zeros.item() == (now == 0).sum().item()
    assert total.item() == numel
    assert flips.dtype == torch.int64


@requires_cuda
def test_code_stats_padding_never_contributes():
    """A tail that is not a multiple of 4 must not be counted as flips or zeros."""
    prev = torch.tensor([1, 0, -1, 1, 0, -1], dtype=torch.int8, device="cuda")
    now = torch.tensor([1, 0, -1, 1, 0, 0], dtype=torch.int8, device="cuda")
    flips, zeros, total = tk.code_stats(tk.pack_codes(prev), tk.pack_codes(now), 6)
    assert (flips.item(), zeros.item(), total.item()) == (1, 3, 6)


@requires_cuda
def test_code_stats_identical_snapshots():
    codes = rand_codes(1000)
    packed = tk.pack_codes(codes)
    flips, zeros, _ = tk.code_stats(packed, packed, 1000)
    assert flips.item() == 0
    assert zeros.item() == (codes == 0).sum().item()


@requires_cuda
def test_code_stats_length_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        tk.code_stats(
            torch.zeros(4, dtype=torch.uint8, device="cuda"),
            torch.zeros(5, dtype=torch.uint8, device="cuda"),
            16,
        )


@requires_cuda
def test_flip_count_matches_oracle():
    prev, now = rand_codes(4096, seed=3), rand_codes(4096, seed=4)
    packed_prev, packed_now = tk.pack_codes(prev), tk.pack_codes(now)
    flips = tk.flip_count(packed_prev, packed_now)
    assert flips.item() == quant.flip_count(packed_prev, packed_now).item()
    assert flips.item() == (prev != now).sum().item()


@requires_cuda
def test_multi_block_reduction():
    """More packed bytes than one program's block: partial sums must aggregate."""
    numel = 4 * 1024 * 9 + 7
    prev, now = rand_codes(numel, seed=5), rand_codes(numel, seed=6)
    flips, zeros, _ = tk.code_stats(tk.pack_codes(prev), tk.pack_codes(now), numel)
    assert flips.item() == (prev != now).sum().item()
    assert zeros.item() == (now == 0).sum().item()


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_model_shapes_match_oracle(dtype, lambda_):
    """Llama-3.2-1B and Qwen3-4B projection shapes, end to end against the oracle."""
    torch.manual_seed(0)
    for out_features, in_features in ((2048, 8192), (2560, 9728)):
        weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
        assert torch.equal(
            tk.fake_quant_weight(weight, lambda_),
            quant.fake_quant_weight(weight, lambda_),
        )
        x = torch.randn(64, in_features, device="cuda", dtype=dtype)
        assert torch.equal(tk.act_quant(x, lambda_), quant.act_quant(x, lambda_))
        codes = quant.ternary_codes(weight, quant.absmean_scale(weight))
        assert torch.equal(tk.pack_codes(codes), quant.pack_codes(codes))

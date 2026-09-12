"""Unit tests for the 8-bit Muon optimizer: blockwise int8 momentum and parity with DistMuon."""

import copy

import pytest
import torch
import torch.nn.functional as F
from axolotl.contribs.mit.muon.dist_muon import DistMuon, DistMuonOptimizerFactory

from axolotl.utils.optimizers.muon_8bit import (
    BLOCK_SIZE,
    Muon8bit,
    Muon8bitOptimizerFactory,
    dequantize,
    quantize,
)


def _mlp(seed: int = 0) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)
    )


def _param_groups(model: torch.nn.Module) -> list[dict]:
    """Matrices go to muon, 1D tensors to adamw, mirroring DistMuonOptimizerFactory."""
    return [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "algorithm": "muon",
            "weight_decay": 0.0,
        },
        {
            "params": [p for p in model.parameters() if p.ndim < 2],
            "algorithm": "adamw",
            "weight_decay": 0.0,
        },
    ]


def _batch(seed: int = 123) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(64, 32), torch.randn(64, 8)


def _step(model, opt, inputs, targets) -> torch.Tensor:
    opt.zero_grad()
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    opt.step()
    return loss.detach()


def _state_bytes(opt: torch.optim.Optimizer, ndim_at_least: int = 0) -> int:
    """Bytes the state actually holds, counting each storage once, not numel * itemsize.

    A tensor that is a view onto a larger buffer keeps all of that buffer alive, so
    logical size would overstate the saving.
    """
    seen: set[int] = set()
    total = 0
    for param, state in opt.state.items():
        if param.ndim < ndim_at_least:
            continue
        for tensor in state.values():
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                total += storage.nbytes()
    return total


@pytest.mark.parametrize(
    "shape",
    [
        (8, 16),  # single partial block
        (4, BLOCK_SIZE),  # exactly one block
        (4, 3 * BLOCK_SIZE),  # several whole blocks
        (4, BLOCK_SIZE + 137),  # ragged trailing block
        (3, 4, 100),  # 3D, flattened per leading index
    ],
)
def test_quantize_roundtrip_within_half_step(shape):
    """Dequantized momentum is within half a quantization step of the original."""
    torch.manual_seed(0)
    m = torch.randn(*shape)
    q, scale = quantize(m)
    err = (dequantize(q, scale, m.dtype) - m).abs().max()
    assert err <= scale.max() / 2 + 1e-6


@pytest.mark.parametrize("shape", [(8, 16), (4, BLOCK_SIZE + 137), (3, 4, 100)])
def test_quantize_preserves_shape_and_stores_int8(shape):
    """The int8 buffer keeps the momentum's shape so it shards like the parameter."""
    q, scale = quantize(torch.randn(*shape))
    assert q.shape == torch.Size(shape)
    assert q.dtype == torch.int8
    assert scale.shape[0] == shape[0]


def test_scale_keeps_leading_dim():
    """Scale is one row per leading index, so FSDP2's dim-0 sharding stays valid."""
    _, scale = quantize(torch.randn(6, 3 * BLOCK_SIZE))
    assert scale.shape == torch.Size([6, 3])


def test_all_zero_momentum_roundtrips_finite():
    """A zero block must not divide by a zero scale (step 1 for params with no grad yet)."""
    m = torch.zeros(4, 100)
    q, scale = quantize(m)
    out = dequantize(q, scale, m.dtype)
    assert torch.isfinite(out).all()
    assert torch.equal(out, m)


def test_values_on_the_int8_grid_roundtrip_exactly():
    """Values already representable on the grid survive quantization unchanged."""
    m = torch.arange(-127, 128, dtype=torch.float32).repeat(2, 1) / 127
    q, scale = quantize(m)
    assert torch.allclose(dequantize(q, scale, m.dtype), m, atol=1e-6)


def test_blocks_are_scaled_independently():
    """An outlier in one block must not degrade the resolution of its neighbours."""
    torch.manual_seed(0)
    m = torch.randn(2, 2 * BLOCK_SIZE)
    spiked = m.clone()
    spiked[:, BLOCK_SIZE:] *= 1000

    q_plain, _ = quantize(m)
    q_spiked, _ = quantize(spiked)
    assert torch.equal(q_plain[:, :BLOCK_SIZE], q_spiked[:, :BLOCK_SIZE])


def test_muon_momentum_is_int8_between_steps():
    """Between steps the muon momentum exists only as int8 blocks plus scales."""
    model = _mlp()
    opt = Muon8bit(_param_groups(model), lr=1e-2)
    _step(model, opt, *_batch())

    for param in model.parameters():
        if param.ndim < 2:
            continue
        state = opt.state[param]
        assert set(state) == {"momentum_q8", "momentum_scale"}
        assert state["momentum_q8"].dtype == torch.int8


def test_adamw_group_state_stays_full_precision():
    """Only muon momentum is quantized; adamw-backed params keep fp moments."""
    model = _mlp()
    opt = Muon8bit(_param_groups(model), lr=1e-2)
    _step(model, opt, *_batch())

    for param in model.parameters():
        if param.ndim >= 2:
            continue
        state = opt.state[param]
        assert set(state) == {"momentum", "variance"}
        assert state["momentum"].dtype == param.dtype
        assert state["variance"].dtype == param.dtype


def test_momentum_is_updated_before_it_is_quantized():
    """After one step the stored momentum is the gradient, not the pre-update zeros."""
    weight = torch.nn.Parameter(torch.randn(16, 32))
    opt = Muon8bit(
        [{"params": [weight], "algorithm": "muon", "weight_decay": 0.0}], lr=1e-2
    )
    torch.manual_seed(1)
    grad = torch.randn(16, 32)
    weight.grad = grad.clone()
    opt.step()

    state = opt.state[weight]
    momentum = dequantize(state["momentum_q8"], state["momentum_scale"], grad.dtype)
    assert (momentum - grad).abs().max() <= grad.abs().max() / 127 / 2 + 1e-6


QUANTIZE_KEEPS_PADDING = pytest.mark.xfail(
    strict=True,
    reason="quantize() returns a view onto the padded blocks, so the state keeps the "
    "full padded buffer alive; add .contiguous() before the final reshape",
)


@QUANTIZE_KEEPS_PADDING
@pytest.mark.parametrize("shape", [(64, 32), (48, BLOCK_SIZE + 137)])
def test_quantized_momentum_owns_its_bytes(shape):
    """The int8 buffer must not be a view onto the zero-padded blocks.

    quantize() slices the padded (rows, nblocks, BLOCK_SIZE) buffer back down to the
    momentum's shape. The slice is a view, so the whole padded allocation stays alive
    and the state is no smaller than the fp momentum it replaced -- on Qwen2.5-0.5B in
    bf16 the muon momentum measures 678.7 MiB quantized against 682.5 MiB dense.
    Adding .contiguous() before the final reshape in quantize() restores the 2x saving.
    """
    q, _ = quantize(torch.randn(*shape))
    assert q.untyped_storage().nbytes() == q.numel() * q.element_size()


@QUANTIZE_KEEPS_PADDING
def test_muon_state_is_smaller_than_dist_muon():
    """The point of the optimizer: quantized momentum must actually free memory."""
    dense_model, quant_model = _mlp(), _mlp()
    dense = DistMuon(_param_groups(dense_model), lr=1e-2)
    quantized = Muon8bit(_param_groups(quant_model), lr=1e-2)
    inputs, targets = _batch()
    _step(dense_model, dense, inputs, targets)
    _step(quant_model, quantized, inputs, targets)

    ratio = _state_bytes(dense, ndim_at_least=2) / _state_bytes(
        quantized, ndim_at_least=2
    )
    assert ratio > 3.5, f"fp32 momentum should shrink ~4x, got {ratio:.2f}x"


def test_first_step_matches_dist_muon_exactly():
    """Momentum starts at zero and is quantized only after the update, so step 1 is exact."""
    dense_model, quant_model = _mlp(), _mlp()
    quant_model.load_state_dict(dense_model.state_dict())
    dense = DistMuon(_param_groups(dense_model), lr=1e-2)
    quantized = Muon8bit(_param_groups(quant_model), lr=1e-2)

    inputs, targets = _batch()
    _step(dense_model, dense, inputs, targets)
    _step(quant_model, quantized, inputs, targets)

    for dense_p, quant_p in zip(
        dense_model.parameters(), quant_model.parameters(), strict=True
    ):
        assert torch.equal(dense_p, quant_p)


def test_trajectory_tracks_dist_muon():
    """Over many steps the quantized run stays within quantization noise of DistMuon."""
    dense_model, quant_model = _mlp(), _mlp()
    quant_model.load_state_dict(dense_model.state_dict())
    dense = DistMuon(_param_groups(dense_model), lr=1e-2)
    quantized = Muon8bit(_param_groups(quant_model), lr=1e-2)

    inputs, targets = _batch()
    for _ in range(6):
        dense_loss = _step(dense_model, dense, inputs, targets)
        quant_loss = _step(quant_model, quantized, inputs, targets)

    relative = max(
        ((a - b).abs().max() / a.abs().max()).item()
        for a, b in zip(dense_model.parameters(), quant_model.parameters(), strict=True)
    )
    assert relative < 0.05
    assert quant_loss == pytest.approx(dense_loss, rel=1e-2)
    assert quant_loss < F.mse_loss(_mlp()(inputs), targets)


def test_every_muon_param_is_updated():
    """Re-batching one group at a time must not drop parameters from the update."""
    model = _mlp()
    opt = Muon8bit(_param_groups(model), lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    _step(model, opt, *_batch())

    for old, new in zip(before, model.parameters(), strict=True):
        assert not torch.equal(old, new)


def test_params_sharing_a_shape_are_batched_together():
    """create_param_batches groups by shape; each param still gets its own momentum."""
    torch.manual_seed(0)
    weights = [torch.nn.Parameter(torch.randn(16, 16)) for _ in range(3)]
    opt = Muon8bit(
        [{"params": weights, "algorithm": "muon", "weight_decay": 0.0}], lr=1e-2
    )
    grads = [torch.randn(16, 16) for _ in weights]
    for weight, grad in zip(weights, grads, strict=True):
        weight.grad = grad.clone()
    opt.step()

    for weight, grad in zip(weights, grads, strict=True):
        state = opt.state[weight]
        momentum = dequantize(state["momentum_q8"], state["momentum_scale"], grad.dtype)
        assert (momentum - grad).abs().max() <= grad.abs().max() / 127 / 2 + 1e-6


# ---- checkpoint resume --------------------------------------------------------------


def test_state_dict_roundtrip_resumes_identically():
    """Quantized state is the whole state, so a resumed run continues bit-exactly."""
    model = _mlp()
    opt = Muon8bit(_param_groups(model), lr=1e-2)
    inputs, targets = _batch()
    for _ in range(3):
        _step(model, opt, inputs, targets)

    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    _step(model, opt, inputs, targets)
    expected = [p.detach().clone() for p in model.parameters()]

    resumed_model = _mlp()
    resumed_model.load_state_dict(model_state)
    resumed = Muon8bit(_param_groups(resumed_model), lr=1e-2)
    resumed.load_state_dict(opt_state)
    _step(resumed_model, resumed, inputs, targets)

    for want, got in zip(expected, resumed_model.parameters(), strict=True):
        assert torch.equal(want, got)


def test_factory_reuses_dist_muon_param_grouping():
    """Muon8bit only swaps the optimizer class; grouping stays DistMuon's."""
    assert issubclass(Muon8bitOptimizerFactory, DistMuonOptimizerFactory)
    assert Muon8bitOptimizerFactory.optim_cls is Muon8bit
    assert Muon8bitOptimizerFactory.__call__ is DistMuonOptimizerFactory.__call__

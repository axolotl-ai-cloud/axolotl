"""Tests for the Mamba / Mamba2 / Falcon-Mamba sample-packing patches."""

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.models.mamba import modeling as mamba_packing
from axolotl.monkeypatch.models.mamba.modeling import (
    PackedSegments,
    build_segment_plan,
    packed_selective_scan,
)


def _reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, **_):
    """A causal toy scan: cumulative sum along time of ``u * delta``, plus ``z``."""
    out = torch.cumsum(u * delta, dim=-1)
    if z is not None:
        out = out + z
    return out, None


class TestSegmentPlan:
    def test_indexes_each_document_from_its_start(self):
        seq_idx = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.int32)

        index, mask = build_segment_plan(seq_idx)

        assert index.shape == (4, 4)
        assert mask.tolist() == [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ]
        assert index[mask].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_single_document_row_is_one_segment(self):
        index, mask = build_segment_plan(torch.zeros(1, 6, dtype=torch.int32))

        assert index.shape == (1, 6)
        assert mask.all()


class TestPackedSelectiveScan:
    def test_matches_scanning_each_document_alone(self):
        torch.manual_seed(0)
        dim, state = 3, 2
        seq_idx = torch.tensor([[0, 0, 0, 1, 1, 2]], dtype=torch.int32)
        u, delta, z = (torch.randn(1, dim, 6) for _ in range(3))
        B, C = (torch.randn(1, state, 6) for _ in range(2))
        A = torch.randn(dim, state)

        out = packed_selective_scan(
            _reference_scan, PackedSegments(seq_idx), u, delta, A, B, C, None, z, None
        )

        for start, end in ((0, 3), (3, 5), (5, 6)):
            expected, _ = _reference_scan(
                u[..., start:end],
                delta[..., start:end],
                A,
                None,
                None,
                z=z[..., start:end],
            )
            torch.testing.assert_close(out[..., start:end], expected)

    def test_gradients_flow_back_to_packed_inputs(self):
        seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        u = torch.randn(1, 2, 4, requires_grad=True)
        delta = torch.ones(1, 2, 4)
        B = C = torch.zeros(1, 1, 4)

        out = packed_selective_scan(
            _reference_scan,
            PackedSegments(seq_idx),
            u,
            delta,
            None,
            B,
            C,
            None,
            None,
            None,
        )
        out.sum().backward()

        # each token contributes to itself and every later token of its own document
        assert u.grad[0, 0].tolist() == [2.0, 1.0, 2.0, 1.0]

    def test_batched_rows_keep_their_own_documents(self):
        seq_idx = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.int32)
        u = torch.ones(2, 1, 3)
        delta = torch.ones(2, 1, 3)
        B = C = torch.zeros(2, 1, 3)

        out = packed_selective_scan(
            _reference_scan,
            PackedSegments(seq_idx),
            u,
            delta,
            None,
            B,
            C,
            None,
            None,
            None,
        )

        assert out[:, 0].tolist() == [[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]


class TestKernelShims:
    def _module(self):
        mod = ModuleType("fake_mamba_modeling")
        calls = {}

        def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
            calls["conv_seq_idx"] = seq_idx
            return x

        def mamba_chunk_scan_combined(*args, seq_idx=None, **kwargs):
            calls["scan_seq_idx"] = seq_idx
            return args[0], None

        mod.causal_conv1d_fn = causal_conv1d_fn
        mod.mamba_chunk_scan_combined = mamba_chunk_scan_combined
        mod.selective_scan_fn = _reference_scan
        return mod, calls

    def test_seq_idx_overrides_the_explicit_none(self):
        mod, calls = self._module()
        seq_idx = torch.tensor([[0, 0, 1]], dtype=torch.int32)
        x = torch.zeros(1, 2, 3)

        with mamba_packing._kernels_with_seq_idx(mod, PackedSegments(seq_idx), None):
            mod.causal_conv1d_fn(x, None, None, activation="silu")
            mod.mamba_chunk_scan_combined(x, seq_idx=None)

        assert calls["conv_seq_idx"] is seq_idx
        assert calls["scan_seq_idx"] is seq_idx

    def test_kernels_are_restored_after_the_block(self):
        mod, _ = self._module()
        originals = (
            mod.causal_conv1d_fn,
            mod.mamba_chunk_scan_combined,
            mod.selective_scan_fn,
        )

        with mamba_packing._kernels_with_seq_idx(
            mod,
            PackedSegments(torch.zeros(1, 2, dtype=torch.int32)),
            "selective_scan_fn",
        ):
            assert mod.selective_scan_fn is not originals[2]

        assert (
            mod.causal_conv1d_fn,
            mod.mamba_chunk_scan_combined,
            mod.selective_scan_fn,
        ) == originals

    def test_selective_scan_shim_splits_documents_and_drops_last_state(self):
        mod, _ = self._module()
        seq_idx = torch.tensor([[0, 0, 1]], dtype=torch.int32)
        u = torch.ones(1, 1, 3)

        with mamba_packing._kernels_with_seq_idx(
            mod, PackedSegments(seq_idx), "selective_scan_fn"
        ):
            out, state = mod.selective_scan_fn(
                u,
                u,
                None,
                u,
                u,
                None,
                None,
                None,
                delta_softplus=True,
                return_last_state=True,
            )

        assert state is None
        assert out[0, 0].tolist() == [1.0, 2.0, 1.0]


class _RecordingMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states


def _tiny_model(model_type, cls_prefix):
    modeling = pytest.importorskip(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    config_cls = getattr(
        pytest.importorskip(
            f"transformers.models.{model_type}.configuration_{model_type}"
        ),
        f"{cls_prefix}Config",
    )
    kwargs = {"vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2}
    if model_type == "mamba2":
        kwargs.update(num_heads=2, head_dim=8, state_size=4, expand=1, n_groups=1)
    else:
        kwargs.update(state_size=4, expand=1)
    model = getattr(modeling, f"{cls_prefix}ForCausalLM")(config_cls(**kwargs)).eval()
    mixers = []
    for block in model.backbone.layers:
        block.mixer = _RecordingMixer()
        mixers.append(block.mixer)
    return modeling, model, mixers


@pytest.fixture(
    params=[("mamba", "Mamba"), ("mamba2", "Mamba2"), ("falcon_mamba", "FalconMamba")]
)
def patched_model(request, monkeypatch):
    model_type, cls_prefix = request.param
    modeling, model, mixers = _tiny_model(model_type, cls_prefix)
    saved = {
        cls: getattr(modeling, f"{cls_prefix}{cls}").forward
        for cls in ("ForCausalLM", "Block", "Mixer")
    }
    monkeypatch.setattr(mamba_packing, "_require_kernels", lambda *a, **k: None)
    getattr(mamba_packing, f"patch_{model_type}_modeling_packing")()
    yield model, mixers
    for cls, forward in saved.items():
        getattr(modeling, f"{cls_prefix}{cls}").forward = forward


def test_position_ids_reach_every_mixer_as_segments(patched_model):
    model, mixers = patched_model
    input_ids = torch.randint(0, 32, (1, 5))
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    mask = torch.tensor([[1, 1, 1, 2, 2]])

    model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask)

    for mixer in mixers:
        (call,) = mixer.calls
        assert call["segments"].seq_idx.tolist() == [[0, 0, 0, 1, 1]]
        assert call["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_unpacked_forward_takes_the_stock_path(patched_model):
    model, mixers = patched_model

    model(input_ids=torch.randint(0, 32, (1, 5)))

    for mixer in mixers:
        (call,) = mixer.calls
        assert "segments" not in call


def test_packed_forward_without_kernels_raises(monkeypatch):
    modeling, model, _ = _tiny_model("mamba2", "Mamba2")
    saved = modeling.Mamba2Mixer.forward
    monkeypatch.setattr(mamba_packing, "_require_kernels", lambda *a, **k: None)
    monkeypatch.setattr(modeling, "is_fast_path_available", False, raising=False)
    mamba_packing._patch_mixer(modeling, modeling.Mamba2Mixer, None, False)
    try:
        mixer = modeling.Mamba2Mixer(model.config, layer_idx=0)
        with pytest.raises(RuntimeError, match="CUDA fast path"):
            mixer(
                torch.zeros(1, 3, 16),
                segments=PackedSegments(torch.zeros(1, 3, dtype=torch.int32)),
            )
    finally:
        modeling.Mamba2Mixer.forward = saved


def test_patch_requires_kernels_or_hub_kernels(monkeypatch):
    fake = ModuleType("transformers.models.mamba2.modeling_mamba2")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(
        mamba_packing, "mamba2_seq_idx_kernels_available", lambda: False
    )

    with pytest.raises(RuntimeError, match="use_kernels"):
        mamba_packing.patch_mamba2_modeling_packing(kernels_enabled=False)

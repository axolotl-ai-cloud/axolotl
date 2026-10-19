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
    packed_causal_conv,
    packed_selective_scan,
)
from axolotl.monkeypatch.models.mamba_utils import (
    kernel_accepts,
    require_seq_idx_kernels,
)


def _reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, **_):
    """A causal toy scan: cumulative sum along time of ``u * delta``, plus ``z``."""
    out = torch.cumsum(u * delta, dim=-1)
    if z is not None:
        out = out + z
    return out, None


def _covered(plan):
    return sorted(position for index, mask in plan for position in index[mask].tolist())


class TestSegmentPlan:
    def test_covers_every_token_exactly_once(self):
        seq_idx = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.int32)

        plan = build_segment_plan(seq_idx)

        assert _covered(plan) == list(range(10))
        for index, mask in plan:
            assert index.shape == mask.shape

    def test_single_document_row_is_one_segment(self):
        (plan,) = build_segment_plan(torch.zeros(1, 6, dtype=torch.int32))
        index, mask = plan

        assert index.shape == (1, 6)
        assert mask.all()

    def test_padding_is_bounded_by_the_pad_factor(self):
        # one 64-token doc and 32 two-token docs: padding all to 64 would be 16x
        seq_idx = torch.tensor(
            [[0] * 64 + [1 + i // 2 for i in range(64)]], dtype=torch.int32
        )

        plan = build_segment_plan(seq_idx, pad_factor=2.0)

        assert _covered(plan) == list(range(128))
        for index, mask in plan:
            assert index.numel() <= 2.0 * mask.sum().item()

    def test_pad_factor_infinity_makes_one_group(self):
        seq_idx = torch.tensor([[0, 0, 0, 1, 2, 2]], dtype=torch.int32)

        plan = build_segment_plan(seq_idx, pad_factor=float("inf"))

        assert len(plan) == 1
        assert plan[0][0].shape == (3, 3)


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

    def test_many_length_groups_assemble_one_output(self):
        # one 16-token document and eight single-token ones split into two groups
        seq_idx = torch.tensor([[0] * 16 + list(range(1, 9))], dtype=torch.int32)
        u = torch.ones(1, 1, 24)
        delta = torch.ones(1, 1, 24)
        B = C = torch.zeros(1, 1, 24)
        segments = PackedSegments(seq_idx)
        assert len(segments.plan) == 2

        out = packed_selective_scan(
            _reference_scan, segments, u, delta, None, B, C, None, None, None
        )

        assert out[0, 0].tolist() == [float(i) for i in range(1, 17)] + [1.0] * 8


class TestPackedCausalConv:
    def test_matches_convolving_each_document_alone(self):
        modeling = pytest.importorskip("transformers.models.mamba.modeling_mamba")
        torch.manual_seed(0)
        channels, kernel = 3, 4
        seq_idx = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2]], dtype=torch.int32)
        x = torch.randn(1, channels, 9)
        weight = torch.randn(channels, kernel)
        bias = torch.randn(channels)

        out = packed_causal_conv(
            modeling.causal_conv1d_fn,
            PackedSegments(seq_idx),
            x,
            weight,
            bias,
            activation="silu",
        )

        for start, end in ((0, 3), (3, 5), (5, 9)):
            expected = modeling.causal_conv1d_fn(
                x[..., start:end], weight, bias, activation="silu"
            )
            torch.testing.assert_close(out[..., start:end], expected)


class TestKernelIntrospection:
    """The pinned transformers wraps every kernel and filters kwargs to the live one."""

    def _wrapped(self, implementation):
        from transformers.integrations.hub_kernels import (
            use_kernel_func_from_hub_with_fallback,
        )

        fake = ModuleType("fake_kernels_pkg")
        fake.causal_conv1d_fn = implementation
        sys.modules[fake.__name__] = fake
        try:

            @use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", fake.__name__)
            def torch_fallback(
                hidden_states, weight, bias=None, activation=None, **kwargs
            ):
                return hidden_states

            return torch_fallback
        finally:
            del sys.modules[fake.__name__]

    def test_fallback_without_seq_idx_is_reported(self):
        fn = self._wrapped(None)

        assert kernel_accepts(fn, "seq_idx") is False

    def test_kernel_with_seq_idx_is_reported(self):
        def kernel(x, weight, bias=None, seq_idx=None, activation=None):
            return x

        assert kernel_accepts(self._wrapped(kernel), "seq_idx") is True

    def test_plain_function_is_unknown(self):
        assert kernel_accepts(lambda x: x, "seq_idx") is None

    def _kernelized(self, forward):
        """After kernelize the wrapper's forward is the hub kernel itself."""
        func = self._wrapped(None)
        func.forward = forward
        return func

    def test_hub_kernel_signature_is_read_after_kernelize(self):
        def hub_kernel(hidden_states, weight, bias=None, seq_idx=None):
            return hidden_states

        def hub_kernel_without(hidden_states, weight, bias=None):
            return hidden_states

        def opaque(hidden_states, **kwargs):
            return hidden_states

        def pops_from_kwargs(hidden_states, **kwargs):
            # the kernels-community layers take **kwargs and pop seq_idx inside
            _ = kwargs.pop("seq_idx", None)
            return hidden_states

        assert kernel_accepts(self._kernelized(hub_kernel), "seq_idx") is True
        assert kernel_accepts(self._kernelized(hub_kernel_without), "seq_idx") is False
        assert kernel_accepts(self._kernelized(opaque), "seq_idx") is None
        assert kernel_accepts(self._kernelized(pops_from_kwargs), "seq_idx") is True

    def test_require_raises_on_fallback_unless_hub_kernels(self):
        mod = ModuleType("fake_modeling")
        mod.causal_conv1d_fn = self._wrapped(None)

        with pytest.raises(RuntimeError, match="use_kernels"):
            require_seq_idx_kernels(mod, ("causal_conv1d_fn",), "mamba", False)
        require_seq_idx_kernels(mod, ("causal_conv1d_fn",), "mamba", True)

    def test_pinned_transformers_fallbacks_are_detected(self):
        modeling = pytest.importorskip("transformers.models.mamba2.modeling_mamba2")

        if kernel_accepts(modeling.causal_conv1d_fn, "seq_idx") is False:
            with pytest.raises(RuntimeError, match="torch fallback"):
                mamba_packing.patch_mamba2_modeling_packing(kernels_enabled=False)


class TestUnfusedPackedScan:
    FAMILY = {
        "fused": "mamba_inner_fn",
        "scan": "mamba_selective_scan",
        "conv": "causal_conv1d_fn",
    }

    @staticmethod
    def _module(conv):
        mod = ModuleType("fake_mamba_modeling")
        mod.mamba_inner_fn = lambda *a, **k: "fused"
        mod.mamba_selective_scan = _reference_scan
        mod.causal_conv1d_fn = conv
        return mod

    def test_fused_kernel_is_disabled_and_scan_split(self):
        conv_batches = []

        def conv(x, weight, bias=None, activation=None, **_):
            conv_batches.append(x.shape[0])
            return x

        mod = self._module(conv)
        originals = (mod.mamba_inner_fn, mod.mamba_selective_scan, mod.causal_conv1d_fn)
        seq_idx = torch.tensor([[0, 0, 1]], dtype=torch.int32)
        u = torch.ones(1, 1, 3)

        with mamba_packing._unfused_packed_scan(
            mod, PackedSegments(seq_idx), self.FAMILY
        ):
            assert mod.mamba_inner_fn(u) is None
            # the torch fallback conv has no seq_idx, so it goes per document too
            assert mod.causal_conv1d_fn(u, None, seq_idx=seq_idx).shape == u.shape
            assert conv_batches == [2]
            out = mod.mamba_selective_scan(
                u,
                u,
                None,
                u,
                u,
                None,
                None,
                None,
                delta_softplus=True,
                return_last_state=False,
            )

        assert out[0, 0].tolist() == [1.0, 2.0, 1.0]
        assert (
            mod.mamba_inner_fn,
            mod.mamba_selective_scan,
            mod.causal_conv1d_fn,
        ) == originals

    def test_conv_kernel_taking_seq_idx_is_left_alone(self):
        def kernel(hidden_states, weight, bias=None, seq_idx=None):
            return hidden_states

        mod = self._module(TestKernelIntrospection()._wrapped(kernel))
        conv = mod.causal_conv1d_fn
        segments = PackedSegments(torch.tensor([[0, 1]], dtype=torch.int32))

        with mamba_packing._unfused_packed_scan(mod, segments, self.FAMILY):
            assert mod.causal_conv1d_fn is conv


class _RecordingMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states


FAMILIES = [("mamba", "Mamba"), ("mamba2", "Mamba2"), ("falcon_mamba", "FalconMamba")]


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
    return modeling, model


@pytest.fixture(params=FAMILIES, ids=[family[0] for family in FAMILIES])
def patched_model(request, monkeypatch):
    model_type, cls_prefix = request.param
    modeling, model = _tiny_model(model_type, cls_prefix)
    mixers = []
    for block in model.backbone.layers:
        block.mixer = _RecordingMixer()
        mixers.append(block.mixer)
    saved = {
        cls: getattr(modeling, f"{cls_prefix}{cls}").forward
        for cls in ("ForCausalLM", "Block", "Mixer")
    }
    monkeypatch.setattr(mamba_packing, "require_seq_idx_kernels", lambda *a, **k: None)
    getattr(mamba_packing, f"patch_{model_type}_modeling_packing")()
    yield model, mixers
    for cls, forward in saved.items():
        getattr(modeling, f"{cls_prefix}{cls}").forward = forward


def test_position_ids_reach_every_mixer_as_segments(patched_model):
    model, mixers = patched_model
    input_ids = torch.randint(0, 32, (1, 6))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 2, 2, 0]])

    model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask)

    for mixer in mixers:
        (call,) = mixer.calls
        assert call["segments"].seq_idx.tolist() == [[0, 0, 0, 1, 1, 2]]
        # the multipack segment ids are reduced to a padding mask
        assert call["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0]]


def test_unpacked_forward_takes_the_stock_path(patched_model):
    model, mixers = patched_model

    model(input_ids=torch.randint(0, 32, (1, 5)))

    for mixer in mixers:
        (call,) = mixer.calls
        assert "segments" not in call


@pytest.fixture(params=FAMILIES, ids=[family[0] for family in FAMILIES])
def patched_mixer(request, monkeypatch):
    """A real mixer whose module-level kernels are recorders, with the CUDA check bypassed."""
    model_type, cls_prefix = request.param
    modeling, model = _tiny_model(model_type, cls_prefix)
    family = mamba_packing._FAMILIES[model_type]
    seen = {}

    def conv(hidden_states, weight, bias=None, activation=None, seq_idx=None, **_):
        seen["conv_seq_idx"] = seq_idx
        seen["conv_batches"] = seen.get("conv_batches", []) + [hidden_states.shape[0]]
        return hidden_states

    monkeypatch.setattr(modeling, "causal_conv1d_fn", conv)
    if model_type == "mamba2":

        def fused(*args, seq_idx=None, **kwargs):
            seen["fused_seq_idx"] = seq_idx
            return None

        def chunk_scan(hidden_states, *args, seq_idx=None, **kwargs):
            seen["scan_seq_idx"] = seq_idx
            return hidden_states

        monkeypatch.setattr(modeling, "mamba2_split_conv1d_scan_combined", fused)
        monkeypatch.setattr(modeling, "mamba2_chunk_scan", chunk_scan)
    else:

        def fused(*args, **kwargs):
            seen["fused_called"] = True
            return None

        def scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, **kwargs):
            seen["scan_batches"] = seen.get("scan_batches", []) + [u.shape[0]]
            return _reference_scan(u, delta, A, B, C, D, z, delta_bias)[0]

        monkeypatch.setattr(modeling, family["fused"], fused)
        monkeypatch.setattr(modeling, family["scan"], scan)

    saved = modeling.__dict__[f"{cls_prefix}Mixer"].forward
    monkeypatch.setattr(mamba_packing, "_assert_packed_ready", lambda *a, **k: None)
    mamba_packing._patch_mixer(
        modeling, getattr(modeling, f"{cls_prefix}Mixer"), family, model_type
    )
    mixer = model.backbone.layers[0].mixer.train()
    yield model_type, mixer, seen
    getattr(modeling, f"{cls_prefix}Mixer").forward = saved


def test_mixer_threads_seq_idx_into_its_kernels(patched_mixer):
    model_type, mixer, seen = patched_mixer
    seq_idx = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
    hidden = torch.randn(1, 5, 16)

    out = mixer(hidden, segments=PackedSegments(seq_idx))

    assert out.shape == hidden.shape
    if model_type == "mamba2":
        assert seen["conv_seq_idx"] is seq_idx
        assert seen["fused_seq_idx"] is seq_idx
        assert seen["scan_seq_idx"] is seq_idx
    else:
        # the fused row kernel is bypassed; the fallback conv and the scan both see
        # the two documents as a batch
        assert "fused_called" not in seen
        assert seen["conv_batches"] == [2]
        assert seen["scan_batches"] == [2]


@pytest.mark.parametrize("model_type,cls_prefix", FAMILIES[::2])
def test_mamba1_packing_needs_no_cuda_or_kernels(model_type, cls_prefix):
    modeling, model = _tiny_model(model_type, cls_prefix)

    mamba_packing._assert_packed_ready(
        model.backbone.layers[0].mixer,
        modeling,
        mamba_packing._FAMILIES[model_type],
        model_type,
    )


def test_packed_forward_off_cuda_raises(monkeypatch):
    modeling, model = _tiny_model("mamba2", "Mamba2")
    saved = modeling.Mamba2Mixer.forward
    mamba_packing._patch_mixer(
        modeling, modeling.Mamba2Mixer, mamba_packing._FAMILIES["mamba2"], "mamba2"
    )
    try:
        with pytest.raises(RuntimeError, match="CUDA kernels"):
            model.backbone.layers[0].mixer(
                torch.zeros(1, 3, 16),
                segments=PackedSegments(torch.zeros(1, 3, dtype=torch.int32)),
            )
    finally:
        modeling.Mamba2Mixer.forward = saved

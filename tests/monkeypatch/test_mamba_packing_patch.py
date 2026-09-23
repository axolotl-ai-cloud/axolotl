"""Tests for the Mamba2 sample-packing patch."""

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.models.mamba import modeling as mamba_packing
from axolotl.monkeypatch.models.mamba_utils import (
    kernel_accepts,
    require_seq_idx_kernels,
)


class TestKernelIntrospection:
    """``kernel_accepts`` must see through the transformers hub-kernel wrapper."""

    @staticmethod
    def _wrapped(implementation):
        hub = pytest.importorskip("transformers.integrations.hub_kernels")
        fake = ModuleType("fake_causal_conv1d")
        if implementation is not None:
            fake.causal_conv1d_fn = implementation
        sys.modules[fake.__name__] = fake
        try:

            @hub.use_kernel_func_from_hub_with_fallback(
                "causal_conv1d_fn", fake.__name__
            )
            def causal_conv1d_fn(hidden_states, weight, bias=None, **kwargs):
                return hidden_states

        finally:
            sys.modules.pop(fake.__name__, None)
        return causal_conv1d_fn

    def test_fallback_without_seq_idx_is_reported(self):
        assert kernel_accepts(self._wrapped(None), "seq_idx") is False

    def test_kernel_with_seq_idx_is_reported(self):
        def kernel(hidden_states, weight, bias=None, seq_idx=None):
            return hidden_states

        assert kernel_accepts(self._wrapped(kernel), "seq_idx") is True

    def test_plain_function_is_unknown(self):
        assert kernel_accepts(lambda x: x, "seq_idx") is None

    @staticmethod
    def _kernelized(forward):
        """After kernelize the wrapper's forward is the hub kernel itself."""
        func = TestKernelIntrospection._wrapped(None)
        func.forward = forward
        return func

    def test_hub_kernel_signature_is_read_after_kernelize(self):
        def hub_kernel(hidden_states, weight, bias=None, seq_idx=None):
            return hidden_states

        def hub_kernel_without(hidden_states, weight, bias=None):
            return hidden_states

        def opaque(hidden_states, **kwargs):
            return hidden_states

        assert kernel_accepts(self._kernelized(hub_kernel), "seq_idx") is True
        assert kernel_accepts(self._kernelized(hub_kernel_without), "seq_idx") is False
        assert kernel_accepts(self._kernelized(opaque), "seq_idx") is None

    def test_require_raises_on_fallback_unless_hub_kernels(self):
        mod = ModuleType("fake_modeling")
        mod.causal_conv1d_fn = self._wrapped(None)

        with pytest.raises(RuntimeError, match="use_kernels"):
            require_seq_idx_kernels(mod, ("causal_conv1d_fn",), "mamba2", False)
        require_seq_idx_kernels(mod, ("causal_conv1d_fn",), "mamba2", True)

    def test_pinned_transformers_fallbacks_are_detected(self):
        modeling = pytest.importorskip("transformers.models.mamba2.modeling_mamba2")

        if kernel_accepts(modeling.causal_conv1d_fn, "seq_idx") is False:
            with pytest.raises(RuntimeError, match="torch fallback"):
                mamba_packing.patch_mamba2_modeling_packing(kernels_enabled=False)


class _RecordingMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states


def _tiny_model():
    modeling = pytest.importorskip("transformers.models.mamba2.modeling_mamba2")
    config_cls = pytest.importorskip(
        "transformers.models.mamba2.configuration_mamba2"
    ).Mamba2Config
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=8,
        state_size=4,
        expand=1,
        n_groups=1,
    )
    return modeling, modeling.Mamba2ForCausalLM(config).eval()


@pytest.fixture
def patched_model(monkeypatch):
    modeling, model = _tiny_model()
    mixers = []
    for block in model.backbone.layers:
        block.mixer = _RecordingMixer()
        mixers.append(block.mixer)
    saved = {
        cls: getattr(modeling, f"Mamba2{cls}").forward
        for cls in ("ForCausalLM", "Block", "Mixer")
    }
    monkeypatch.setattr(mamba_packing, "require_seq_idx_kernels", lambda *a, **k: None)
    mamba_packing.patch_mamba2_modeling_packing()
    yield model, mixers
    for cls, forward in saved.items():
        getattr(modeling, f"Mamba2{cls}").forward = forward


def test_position_ids_reach_every_mixer_as_seq_idx(patched_model):
    model, mixers = patched_model
    input_ids = torch.randint(0, 32, (1, 6))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 2, 2, 0]])

    model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask)

    for mixer in mixers:
        (call,) = mixer.calls
        assert call["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2]]
        assert call["seq_idx"].dtype == torch.int32
        # the multipack segment ids are reduced to a padding mask
        assert call["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0]]


def test_unpacked_forward_takes_the_stock_path(patched_model):
    model, mixers = patched_model

    model(input_ids=torch.randint(0, 32, (1, 5)))

    for mixer in mixers:
        (call,) = mixer.calls
        assert "seq_idx" not in call


@pytest.fixture
def patched_mixer(monkeypatch):
    """A real mixer whose module-level kernels are recorders, with the CUDA check bypassed."""
    modeling, model = _tiny_model()
    seen = {}

    def conv(hidden_states, weight, bias=None, activation=None, seq_idx=None, **_):
        seen["conv_seq_idx"] = seq_idx
        return hidden_states

    def fused(*args, seq_idx=None, **kwargs):
        seen["fused_seq_idx"] = seq_idx
        return None

    def chunk_scan(hidden_states, *args, seq_idx=None, **kwargs):
        seen["scan_seq_idx"] = seq_idx
        return hidden_states

    monkeypatch.setattr(modeling, "causal_conv1d_fn", conv)
    monkeypatch.setattr(modeling, "mamba2_split_conv1d_scan_combined", fused)
    monkeypatch.setattr(modeling, "mamba2_chunk_scan", chunk_scan)

    saved = modeling.Mamba2Mixer.forward
    monkeypatch.setattr(mamba_packing, "_assert_packed_ready", lambda *a, **k: None)
    mamba_packing._patch_mixer(modeling, modeling.Mamba2Mixer)
    mixer = model.backbone.layers[0].mixer.train()
    yield mixer, seen
    modeling.Mamba2Mixer.forward = saved


def test_mixer_threads_seq_idx_into_its_kernels(patched_mixer):
    mixer, seen = patched_mixer
    seq_idx = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
    hidden = torch.randn(1, 5, 16)

    out = mixer(hidden, seq_idx=seq_idx)

    assert out.shape == hidden.shape
    assert seen["conv_seq_idx"] is seq_idx
    assert seen["fused_seq_idx"] is seq_idx
    assert seen["scan_seq_idx"] is seq_idx


def test_packed_forward_off_cuda_raises():
    modeling, model = _tiny_model()
    saved = modeling.Mamba2Mixer.forward
    mamba_packing._patch_mixer(modeling, modeling.Mamba2Mixer)
    try:
        with pytest.raises(RuntimeError, match="CUDA kernels"):
            model.backbone.layers[0].mixer(
                torch.zeros(1, 3, 16), seq_idx=torch.zeros(1, 3, dtype=torch.int32)
            )
    finally:
        modeling.Mamba2Mixer.forward = saved

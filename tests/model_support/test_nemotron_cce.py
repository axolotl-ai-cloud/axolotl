"""CPU-only tests for the opt-in native Nemotron CCE bridge."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from axolotl.model_support.nemotron_diffusion import _before_model_build
from axolotl.model_support.nemotron_diffusion.cut_cross_entropy import (
    apply_nemotron_cce_patch,
    apply_pending_nemotron_cce_patch,
    get_cce_head,
    get_cce_options,
    patch_nemotron,
)


class Options:
    train_only = False
    c_grad_chunk_size = 0

    def to_kwargs(self):
        return {"impl": "dense", "reduction": "mean"}


class ChunkOptions(Options):
    c_grad_chunk_size = 128


class DefaultChunkKwargOptions(Options):
    def to_kwargs(self):
        return {"impl": "dense", "reduction": "mean", "c_grad_chunk_size": 0}


def install_cce(monkeypatch):
    module = types.ModuleType("cut_cross_entropy")

    def linear_cross_entropy(
        hidden, weight, targets, bias=None, reduction="mean", shift=0, **kwargs
    ):
        del kwargs
        assert reduction == "none" and shift == 0
        logits = functional.linear(hidden, weight, bias)
        return functional.cross_entropy(
            logits.flatten(0, -2), targets.flatten(), reduction="none"
        ).view_as(targets)

    module.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", module)


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_head = torch.nn.Linear(3, 5, bias=False)
        self.embed_tokens = torch.nn.Embedding(5, 3)
        self.embed_tokens.weight.requires_grad_(False)
        self.normal_calls = 0

    def get_output_embeddings(self):
        return self.diffusion_head

    def forward(self, hidden, output_last_hidden_states_only=False, **kwargs):
        del kwargs
        if output_last_hidden_states_only:
            return SimpleNamespace(last_hidden_state=hidden)
        self.normal_calls += 1
        return "normal"


@pytest.fixture(autouse=True)
def isolated_native_class(monkeypatch):
    monkeypatch.setattr(TinyNative, "forward", TinyNative.forward)
    monkeypatch.setattr(TinyNative, "_axolotl_cce_options", None, raising=False)
    monkeypatch.setattr(TinyNative, "_axolotl_cce_patched", False, raising=False)


def test_hook_is_lazy_when_cce_is_off(monkeypatch):
    package = types.ModuleType("cut_cross_entropy")
    transformers = types.ModuleType("cut_cross_entropy.transformers")
    patch = types.ModuleType("cut_cross_entropy.transformers.patch")
    patch.PATCH_FNS = {}
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", package)
    monkeypatch.setitem(sys.modules, "cut_cross_entropy.transformers", transformers)
    monkeypatch.setitem(sys.modules, "cut_cross_entropy.transformers.patch", patch)
    _before_model_build(SimpleNamespace(cfg=SimpleNamespace(cut_cross_entropy=False)))
    assert patch.PATCH_FNS == {}
    _before_model_build(SimpleNamespace(cfg=SimpleNamespace(cut_cross_entropy=True)))
    assert patch.PATCH_FNS["nemotron_labs_diffusion"] == (
        "axolotl.model_support.nemotron_diffusion.cut_cross_entropy",
        "patch_nemotron",
    )


def test_explicit_cce_matches_dense_and_normal_forward_is_unchanged(monkeypatch):
    install_cce(monkeypatch)
    apply_nemotron_cce_patch(TinyNative, DefaultChunkKwargOptions())
    model = TinyNative()
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    assert model(hidden) == "normal"
    assert model.normal_calls == 1
    assert model(hidden, cce_return_hidden_states=True).last_hidden_state is hidden
    loss = model(hidden, cce_targets=targets).loss
    expected = functional.cross_entropy(
        functional.linear(hidden, model.diffusion_head.weight).flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).view_as(targets)
    torch.testing.assert_close(loss, expected)
    loss.sum().backward()
    assert hidden.grad is not None and model.diffusion_head.weight.grad is not None


def test_default_chunk_size_is_not_forwarded_to_older_cce(monkeypatch):
    calls = []
    module = types.ModuleType("cut_cross_entropy")

    def linear_cross_entropy(
        hidden, weight, targets, bias=None, reduction="mean", shift=0, impl=None
    ):
        del impl
        calls.append((reduction, shift))
        logits = functional.linear(hidden, weight, bias)
        return functional.cross_entropy(
            logits.flatten(0, -2), targets.flatten(), reduction="none"
        ).view_as(targets)

    module.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", module)
    apply_nemotron_cce_patch(TinyNative, DefaultChunkKwargOptions())
    model = TinyNative()
    assert model(
        torch.randn(1, 2, 3), cce_targets=torch.tensor([[0, 1]])
    ).loss.shape == (1, 2)
    assert calls == [("none", 0)]


def test_chunked_cce_requires_a_kernel_that_accepts_chunk_size(monkeypatch):
    module = types.ModuleType("cut_cross_entropy")

    def linear_cross_entropy(hidden, weight, targets, bias=None):
        del hidden, weight, targets, bias

    module.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", module)
    apply_nemotron_cce_patch(TinyNative, ChunkOptions())
    with pytest.raises(ImportError, match="c_grad_chunk_size"):
        TinyNative()(torch.randn(1, 2, 3), cce_targets=torch.tensor([[0, 1]]))


def test_chunked_cce_is_forwarded_when_the_kernel_supports_it(monkeypatch):
    received = []
    module = types.ModuleType("cut_cross_entropy")

    def linear_cross_entropy(
        hidden, weight, targets, bias=None, c_grad_chunk_size=None, **kwargs
    ):
        del hidden, weight, bias, kwargs
        received.append(c_grad_chunk_size)
        return torch.zeros_like(targets, dtype=torch.float)

    module.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", module)
    apply_nemotron_cce_patch(TinyNative, ChunkOptions())
    TinyNative()(torch.randn(1, 2, 3), cce_targets=torch.tensor([[0, 1]]))
    assert received == [128]


def test_pending_options_are_bound_once_without_global_leak():
    class First(TinyNative):
        pass

    class Second(TinyNative):
        pass

    First._axolotl_cce_patched = False
    First._axolotl_cce_options = None
    Second._axolotl_cce_patched = False
    Second._axolotl_cce_options = None
    patch_nemotron("nemotron_labs_diffusion", Options())
    apply_pending_nemotron_cce_patch(First)
    apply_pending_nemotron_cce_patch(Second)
    assert get_cce_options(First()) is not None
    with pytest.raises(ValueError, match="not enabled"):
        get_cce_options(Second())


def test_peft_or_ddp_wrapper_is_unwrapped():
    apply_nemotron_cce_patch(TinyNative, Options())
    model = TinyNative()
    wrapper = SimpleNamespace(module=model)
    assert get_cce_options(wrapper) is get_cce_options(model)
    assert get_cce_head(wrapper) is model.diffusion_head
    model.embed_tokens.weight.requires_grad_(True)
    assert get_cce_head(model) is model.diffusion_head


def test_head_lora_or_labels_are_rejected(monkeypatch):
    install_cce(monkeypatch)
    apply_nemotron_cce_patch(TinyNative, Options())
    model = TinyNative()
    with pytest.raises(ValueError, match="does not accept labels"):
        model(
            torch.randn(1, 2, 3),
            cce_targets=torch.ones(1, 2, dtype=torch.long),
            labels=torch.ones(1, 2, dtype=torch.long),
        )

    class LoraHead(torch.nn.Linear):
        lora_A = object()

    model.diffusion_head = LoraHead(3, 5, bias=False)
    with pytest.raises(ValueError, match="unwrapped nn.Linear"):
        get_cce_head(model)


def test_native_nemotron_cce_loss_and_gradients_match_dense(monkeypatch):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from tests.native_source_fixtures import native_source_fixture_path
    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        use_cache=False,
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
    )
    config._attn_implementation = "eager"
    model_class = resolve_nemotron_model_class(source)
    model = model_class(config).eval()
    ids = torch.tensor([[7, 100, 9, 100]])
    targets = torch.tensor([[-100, 8, -100, 10]])
    dense = functional.cross_entropy(
        model(input_ids=ids, use_cache=False).logits.flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).view_as(targets)
    dense.sum().backward()
    expected_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    install_cce(monkeypatch)
    apply_nemotron_cce_patch(model_class, Options())
    outputs = model(input_ids=ids, use_cache=False, cce_targets=targets)
    assert outputs.logits is None
    torch.testing.assert_close(outputs.loss, dense)
    outputs.loss.sum().backward()
    for name, parameter in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(parameter.grad, expected_grads[name])

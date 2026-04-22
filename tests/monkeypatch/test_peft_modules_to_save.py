"""Tests for axolotl.monkeypatch.peft_modules_to_save."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("peft")


@pytest.fixture(scope="module")
def patched():
    from axolotl.monkeypatch.peft_modules_to_save import (
        patch_peft_modules_to_save_kwargs,
    )

    patch_peft_modules_to_save_kwargs()
    yield


@pytest.fixture
def wrap_module(patched):
    """Wrap ``module`` like PEFT wraps a ``modules_to_save`` entry."""
    from peft.utils.other import ModulesToSaveWrapper

    def _wrap(module: nn.Module) -> ModulesToSaveWrapper:
        w = ModulesToSaveWrapper(module, "default")
        w.set_adapter("default")
        return w

    return _wrap


def test_kwargs_only_forward(wrap_module):
    """Gemma 4 vision_tower shape: self.vision_tower(pixel_values=...)."""

    class KwargsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, *, pixel_values):
            return self.lin(pixel_values)

    w = wrap_module(KwargsOnly())
    out = w(pixel_values=torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_positional_only_forward(wrap_module):
    """embed_tokens shape: self.embed_tokens(input_ids)."""

    class PosOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 8)

        def forward(self, x):
            return self.emb(x)

    w = wrap_module(PosOnly())
    out = w(torch.tensor([1, 2, 3]))
    assert out.shape == (3, 8)


def test_mixed_args_kwargs_forward(wrap_module):
    """Module taking both positional and keyword args."""

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x, scale=1.0):
            return self.lin(x) * scale

    w = wrap_module(Mixed())
    out = w(torch.randn(2, 8), scale=2.0)
    assert out.shape == (2, 8)


def test_kwargs_only_disabled_adapter(wrap_module):
    """Passthrough branch (enable_adapters(False)) must also accept kwargs-only."""

    class KwargsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, *, pixel_values):
            return self.lin(pixel_values)

    w = wrap_module(KwargsOnly())
    w.enable_adapters(False)

    out = w(pixel_values=torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_trainable_tokens_wrapper_not_broken(patched):
    """Base-class forward replacement must not break TrainableTokensWrapper
    (positional embed_tokens path must still work)."""
    try:
        from peft.utils.other import TrainableTokensWrapper
    except ImportError:
        pytest.skip("TrainableTokensWrapper not available in installed PEFT")

    try:
        w = TrainableTokensWrapper(
            nn.Embedding(100, 8), "default", token_indices=[0, 1, 2]
        )
    except TypeError as exc:
        pytest.skip(f"TrainableTokensWrapper init signature changed: {exc}")
    w.set_adapter("default")

    out = w(torch.tensor([0, 1, 2, 5]))
    assert out.shape == (4, 8)


def test_mixed_batch_kwargs_only_raises(wrap_module):
    """Document current behavior: adapter_names without positional input is
    unsupported (mixed-batch path is deliberately not kwargs-patched)."""

    class KwargsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, *, pixel_values):
            return self.lin(pixel_values)

    w = wrap_module(KwargsOnly())
    with pytest.raises(TypeError):
        w(pixel_values=torch.randn(2, 8), adapter_names=["default", "default"])


def test_patch_is_idempotent():
    """Applying the patch twice is a no-op."""
    from peft.utils.other import AuxiliaryTrainingWrapper

    from axolotl.monkeypatch.peft_modules_to_save import (
        patch_peft_modules_to_save_kwargs,
    )

    patch_peft_modules_to_save_kwargs()
    first = AuxiliaryTrainingWrapper.forward
    patch_peft_modules_to_save_kwargs()
    second = AuxiliaryTrainingWrapper.forward
    assert first is second, "patch must be idempotent"

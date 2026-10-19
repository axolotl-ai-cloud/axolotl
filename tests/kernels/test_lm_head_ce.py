"""Head-aware fused-CE dispatcher routing (CPU; kernel math is covered per-dtype
in the e2e tests). Verifies which kernel a given lm_head resolves to without
running the tiled CE itself."""

import pytest
import torch.nn as nn

from axolotl.integrations.nvfp4.kernels.lm_head_ce import (
    _is_plain_frozen_linear,
    patch_lm_head_cross_entropy,
)


@pytest.fixture(autouse=True)
def _restore_vocab_block():
    """The bf16 kernel keeps _VOCAB_BLOCK as a module global; snapshot and
    restore it so vocab-block tests don't leak state across tests."""
    import axolotl.integrations.nvfp4.kernels.bf16_fused_ce as bf16_ce

    saved = bf16_ce._VOCAB_BLOCK
    yield
    bf16_ce._VOCAB_BLOCK = saved


class _FakeCausal(nn.Module):
    def __init__(self, head):
        super().__init__()
        self._head = head

    def get_output_embeddings(self):
        return self._head


def _frozen_linear():
    head = nn.Linear(8, 16, bias=False).bfloat16()
    head.weight.requires_grad_(False)
    return head


def test_off_does_not_patch():
    assert patch_lm_head_cross_entropy(_FakeCausal(_frozen_linear()), "off") is None
    assert patch_lm_head_cross_entropy(_FakeCausal(_frozen_linear()), None) is None


def test_is_plain_frozen_linear():
    assert _is_plain_frozen_linear(_frozen_linear()) is True
    assert _is_plain_frozen_linear(nn.Linear(8, 16, bias=False)) is False  # trainable
    biased = nn.Linear(8, 16, bias=True)
    biased.weight.requires_grad_(False)
    assert _is_plain_frozen_linear(biased) is False


def test_auto_routes_plain_frozen_head_to_bf16():
    assert patch_lm_head_cross_entropy(_FakeCausal(_frozen_linear()), "auto") == "bf16"


def test_auto_skips_trainable_head():
    head = nn.Linear(8, 16, bias=False)  # requires_grad=True
    assert patch_lm_head_cross_entropy(_FakeCausal(head), "auto") is None


def test_explicit_fp4_falls_back_on_non_fp4_head():
    # forcing fp4 on a plain bf16 head finds no NVFP4 store -> materialized path
    assert patch_lm_head_cross_entropy(_FakeCausal(_frozen_linear()), "fp4") is None


def test_no_output_embeddings_is_safe():
    class _NoHead(nn.Module):
        def get_output_embeddings(self):
            raise NotImplementedError

    assert patch_lm_head_cross_entropy(_NoHead(), "auto") is None


def test_vocab_block_threads_into_bf16_kernel():
    import axolotl.integrations.nvfp4.kernels.bf16_fused_ce as bf16_ce

    patch_lm_head_cross_entropy(_FakeCausal(_frozen_linear()), "bf16", vocab_block=2048)
    assert bf16_ce._VOCAB_BLOCK == 2048


def test_env_var_wins_over_vocab_block(monkeypatch):
    import axolotl.integrations.nvfp4.kernels.bf16_fused_ce as bf16_ce

    monkeypatch.setenv("AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK", "1024")
    bf16_ce._set_vocab_block(8192)
    assert bf16_ce._VOCAB_BLOCK == 1024

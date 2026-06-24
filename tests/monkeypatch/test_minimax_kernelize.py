"""Tests for the MiniMax M2 ``kernelize()`` / ``use_kernels`` crash fix.

See :mod:`axolotl.monkeypatch.minimax_kernelize` for the rationale.
"""

import pytest

pytest.importorskip(
    "transformers.models.minimax_m2",
    reason="minimax_kernelize patch only matters when MiniMax M2 is available",
)


@pytest.fixture(autouse=True)
def _clean_patch_state():
    """Keep each test isolated from the global attention ``__init__`` patch."""
    from axolotl.monkeypatch.minimax_kernelize import unpatch_minimax_kernelize

    unpatch_minimax_kernelize()
    yield
    unpatch_minimax_kernelize()


def _tiny_m2_model():
    from transformers import AutoModelForCausalLM
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

    cfg = MiniMaxM2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
    )
    return AutoModelForCausalLM.from_config(cfg)


def test_kernelize_crashes_without_patch():
    model = _tiny_m2_model()
    model.train()
    with pytest.raises(ValueError, match=r"not a .torch\.nn\.Module"):
        model.kernelize()


def test_patch_lets_kernelize_succeed():
    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    assert patch_minimax_kernelize() is True
    model = _tiny_m2_model()
    model.train()
    model.kernelize()


def test_patch_is_idempotent():
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

    from axolotl.monkeypatch.minimax_kernelize import patch_minimax_kernelize

    patch_minimax_kernelize()
    patched_init = MiniMaxM2Attention.__init__
    patch_minimax_kernelize()
    assert MiniMaxM2Attention.__init__ is patched_init

"""Tests for the Gemma 4 hybrid-attention mask fix.

These tests pin the single critical behavior: after installing the patch,
``modeling_gemma4.create_causal_mask`` passes an SDPA-overridden config to
the underlying mask builder regardless of what the caller's config says.
This is what keeps full-attention (head_dim=512) global layers from
crashing at long sequence lengths — they need a 4D SDPA-format mask, not
the 2D FA2 mask that would be built from the model-level config.

The tests use a mocked ``create_causal_mask`` so they don't have to load
a real 26B Gemma 4 model or even have access to its weights. What matters
for the bug fix is which config is handed to the mask factory, not the
factory's actual output.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "transformers.models.gemma4",
    reason="gemma4_hybrid_mask patch only matters when Gemma 4 is available",
)


@pytest.fixture
def restore_gemma4_module():
    """Snapshot ``modeling_gemma4.create_causal_mask`` and restore after
    each test so patch state doesn't leak across the suite."""
    from transformers.models.gemma4 import modeling_gemma4

    saved = modeling_gemma4.create_causal_mask
    yield modeling_gemma4
    modeling_gemma4.create_causal_mask = saved
    # Reset the module-level flag so the next test can re-install cleanly.
    from axolotl.monkeypatch import gemma4_hybrid_mask

    gemma4_hybrid_mask._PATCH_APPLIED = False


def test_patch_replaces_create_causal_mask(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    original = modeling_gemma4.create_causal_mask
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_gemma4.create_causal_mask is not original
    assert modeling_gemma4.create_causal_mask._axolotl_original is original, (
        "patched wrapper must expose the original reference for teardown"
    )


def test_patch_is_idempotent(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    wrapper_first = modeling_gemma4.create_causal_mask

    # Second call must not re-wrap the already-wrapped function (which
    # would leak the original reference through a chain of wrappers).
    patch_gemma4_hybrid_mask()
    wrapper_second = modeling_gemma4.create_causal_mask

    assert wrapper_first is wrapper_second


def test_patched_mask_forces_sdpa_config(restore_gemma4_module):
    """Core invariant: when the patched wrapper is called with a config
    that says ``flash_attention_2``, the underlying mask factory receives
    a shallow-copied config whose ``_attn_implementation`` is ``"sdpa"``.

    Without this, the full-attention global layers get a 2D FA2 mask and
    crash at long seq lens with the [B, H, S, S] / [B, S] expand error.
    """
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Swap in a mock BEFORE installing the patch so the wrapper captures
    # it as the "original". The mock records every call so we can inspect
    # what config got passed through.
    mock_factory = MagicMock(name="create_causal_mask", return_value="mask_4d")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    # Caller-supplied config says FA2 (that's the model-level setting).
    caller_config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        head_dim=512,
        some_other_attr="preserved",
    )
    result = modeling_gemma4.create_causal_mask(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    # Wrapper returned whatever the mock returned — no transformation of
    # the result itself.
    assert result == "mask_4d"

    # The mock was called exactly once with a config whose
    # ``_attn_implementation`` is sdpa, NOT the caller's fa2.
    assert mock_factory.call_count == 1
    (passed_config, *_), passed_kwargs = mock_factory.call_args
    assert passed_config._attn_implementation == "sdpa"

    # The wrapper must NOT mutate the caller's config in place — other
    # mask builders (e.g. create_sliding_window_causal_mask) read from
    # the same config and must still see fa2.
    assert caller_config._attn_implementation == "flash_attention_2"

    # Other attributes on the config must be preserved so the underlying
    # factory has everything it needs (head_dim, rope_theta, vocab_size, ...).
    assert passed_config.head_dim == 512
    assert passed_config.some_other_attr == "preserved"


def test_patched_wrapper_passes_through_all_kwargs(restore_gemma4_module):
    """The wrapper must forward positional + keyword args to the original
    unchanged, so transformers' own call-site in Gemma4TextModel.forward
    keeps working across minor transformers-version signature drift."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    mock_factory = MagicMock(return_value="mask")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    modeling_gemma4.create_causal_mask(
        caller_config,
        "positional_arg",
        inputs_embeds="embeds",
        attention_mask="mask_2d",
        past_key_values="cache",
        position_ids="positions",
        or_mask_function="or_fn",
    )

    args, kwargs = mock_factory.call_args
    # First positional (after config override) is preserved.
    assert args[1] == "positional_arg"
    # All kwargs are forwarded untouched.
    assert kwargs["inputs_embeds"] == "embeds"
    assert kwargs["attention_mask"] == "mask_2d"
    assert kwargs["past_key_values"] == "cache"
    assert kwargs["position_ids"] == "positions"
    assert kwargs["or_mask_function"] == "or_fn"


def test_unpatch_restores_original(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    sentinel = MagicMock(name="original")
    modeling_gemma4.create_causal_mask = sentinel
    patch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is not sentinel

    unpatch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is sentinel


def test_unpatch_is_safe_without_prior_patch(restore_gemma4_module):
    from axolotl.monkeypatch.gemma4_hybrid_mask import unpatch_gemma4_hybrid_mask

    # Should be a no-op, no exception.
    unpatch_gemma4_hybrid_mask()


def test_sliding_window_mask_builder_is_not_patched(restore_gemma4_module):
    """Only ``create_causal_mask`` is overridden — the sliding-window
    factory must remain bound to its original to preserve FA2 masks for
    the sliding-attention layers. If we accidentally patch both, the
    sliding layers get SDPA format and lose the FA2 speedup."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_sliding_window_causal_mask"):
        pytest.skip("transformers version has no create_sliding_window_causal_mask")

    sliding_before = modeling_gemma4.create_sliding_window_causal_mask
    patch_gemma4_hybrid_mask()
    sliding_after = modeling_gemma4.create_sliding_window_causal_mask
    assert sliding_after is sliding_before


# ---------------------------------------------------------------------------
# Integration tests with a tiny randomly-initialized Gemma4TextModel.
#
# These do NOT load real 26B weights. They build a ~350k-param Gemma 4 text
# model with 2 layers (one sliding, one full_attention), apply the hybrid
# attention path end-to-end, and run a forward pass with a padded
# attention_mask at a long-ish seq len. The invariant we're pinning is that
# the full_attention layer does not crash with the
#   "Target sizes: [B, H, S, S]. Tensor sizes: [B, S]"
# error — the exact failure that blew up the Gemma 4 MoE 26B pilot at ~7k
# tokens in the FSDP2 training run.
# ---------------------------------------------------------------------------


def _build_tiny_gemma4_text_model():
    """Return a tiny randomly-initialized Gemma4TextModel with mixed layers."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    # Caller-supplied attn impl simulates the pilot config (fa2 at model
    # level). The hybrid patch is what makes this survive long context.
    cfg._attn_implementation = "sdpa"  # start safe; the test toggles fa2 later
    torch.manual_seed(42)
    model = Gemma4TextModel(cfg).eval()
    return model, cfg


def _apply_hybrid_attn_inline(model, cfg):
    """Replicate what ``patch_manager._apply_gemma_hybrid_attention`` does
    to a model, without needing a full PatchManager / pydantic cfg."""
    import copy

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    for layer_idx, layer in enumerate(model.layers):
        if cfg.layer_types[layer_idx] != "sliding_attention":
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                sdpa_cfg = copy.copy(attn.config)
                sdpa_cfg._attn_implementation = "sdpa"
                attn.config = sdpa_cfg
    patch_gemma4_hybrid_mask()


def test_tiny_gemma4_long_context_forward_does_not_crash(restore_gemma4_module):
    """End-to-end invariant: with the hybrid attn patch applied, a tiny
    Gemma4TextModel runs a forward at long context (1024 tokens) with
    real padding in the attention mask, producing the expected output
    shape. This exercises the actual code path that crashed the pilot
    without needing a real 26B checkpoint or CUDA."""
    import torch

    model, cfg = _build_tiny_gemma4_text_model()
    _apply_hybrid_attn_inline(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    # Pad positions in the second row. Without padding, SDPA falls back to
    # ``is_causal=True`` with ``mask=None`` — we need a materialized 4D
    # mask to exercise the actual bug site.
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


def test_patched_create_causal_mask_returns_4d_for_real_config(
    restore_gemma4_module,
):
    """Hit the REAL ``create_causal_mask`` (not a mock) via the wrapper
    and verify the returned mask is a 4D tensor — which is the shape the
    SDPA-patched global layers need. Without the patch and with a
    caller-supplied FA2 config this would return a 2D mask and the layer
    would crash at long context."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    modeling_gemma4 = restore_gemma4_module

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    # Simulate the pilot: caller says flash_attention_2, but global layers
    # were switched to SDPA per-layer. Without the patch, create_causal_mask
    # would return an FA2 2D mask here and the SDPA layer would crash.
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 1024
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0  # force the 4D materialized path
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    mask = modeling_gemma4.create_causal_mask(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dim() == 4, (
        f"expected a 4D SDPA-format mask, got {mask.dim()}D "
        f"shape={tuple(mask.shape)}. The full_attention global layers need "
        "this shape or they crash at long context."
    )
    assert mask.shape[0] == B
    assert mask.shape[-1] == S
    assert mask.shape[-2] == S

    # Caller's config must be untouched — other code paths still read it.
    assert cfg._attn_implementation == "flash_attention_2"

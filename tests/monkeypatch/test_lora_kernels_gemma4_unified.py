"""Tests for ``patch_self_attn_lora`` on the gemma4_unified attention class.

Unlike standard gemma4 (which always skips the LoRA source rewrite), the unified
branch is *conditional* on ``fused_attn_kernel``: with it set the QKV/O kernels
ride the fused forward; without it the rewrite is skipped with a warning. Either
way the source rewrite (``_original_forward``) is never installed."""

import logging

import pytest

pytest.importorskip("triton", reason="importing lora_kernels pulls in triton")
gemma4_unified_modeling = pytest.importorskip(
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
    reason="unified lora-kernel branch only matters when gemma4_unified is available",
)


def _cfg(fused_attn_kernel):
    from axolotl.utils.dict import DictDefault

    return DictDefault({"fused_attn_kernel": fused_attn_kernel, "lora_dropout": 0.0})


@pytest.fixture
def restore_unified_attn():
    """Ensure ``_original_forward`` doesn't leak in/out of these tests."""
    cls = gemma4_unified_modeling.Gemma4UnifiedTextAttention
    had = hasattr(cls, "_original_forward")
    saved = getattr(cls, "_original_forward", None)
    if had:
        del cls._original_forward
    yield cls
    if had:
        cls._original_forward = saved
    elif hasattr(cls, "_original_forward"):
        del cls._original_forward


def _run_patch(monkeypatch, caplog, cls, cfg, level):
    from axolotl.monkeypatch import lora_kernels

    monkeypatch.setattr(lora_kernels, "get_attention_cls_from_config", lambda _cfg: cls)
    logger = logging.getLogger("axolotl.monkeypatch.lora_kernels")
    logger.addHandler(caplog.handler)
    previous_level = logger.level
    logger.setLevel(level)
    try:
        lora_kernels.patch_self_attn_lora(cfg)
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


class TestUnifiedLoraKernelSkip:
    def test_skips_with_warning_when_no_fused_attn_kernel(
        self, restore_unified_attn, monkeypatch, caplog
    ):
        cls = restore_unified_attn
        _run_patch(monkeypatch, caplog, cls, _cfg(False), logging.WARNING)

        assert "fused_attn_kernel" in caplog.text
        assert "skipping" in caplog.text.lower()
        assert not hasattr(cls, "_original_forward"), (
            "unified attention must not get the LoRA source rewrite"
        )

    def test_skips_quietly_with_fused_attn_kernel(
        self, restore_unified_attn, monkeypatch, caplog
    ):
        cls = restore_unified_attn
        _run_patch(monkeypatch, caplog, cls, _cfg(True), logging.INFO)

        assert "fused attention path" in caplog.text
        assert not hasattr(cls, "_original_forward")

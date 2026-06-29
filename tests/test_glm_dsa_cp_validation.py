"""Validation tests for the GLM DSA kernels' context-parallel exemptions.

``context_parallel_size > 1`` normally requires flash attention + ``ring_flash_attn``. The GLM DSA
kernels own context-parallel attention (compressed-KV all-gather + per-rank ``q_offset``), so when
``use_glm_dsa_kernels`` is set the validator skips that requirement and leaves ``ring_attn_func`` None
(the SP context manager still shards the sequence by chunking). These are config-validation checks only
-- no GPU / dist / model needed. ``ring_flash_attn`` is intentionally NOT mocked here: the DSA path must
validate without it installed.
"""

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def _cfg(**extra):
    return DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        learning_rate=1e-3,
        datasets=[{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        sequence_len=2048,
        **extra,
    )


def test_config_validation_recovers_inherited_defaults():
    from pydantic import BaseModel

    from axolotl.utils.config import _model_with_inherited_default_fallback

    class _Base(BaseModel):
        base_model: str
        strict: bool = False

    class _Merged(_Base):
        strict: bool

    cfg = _model_with_inherited_default_fallback(
        _Merged, {"base_model": "HuggingFaceTB/SmolLM2-135M"}
    )

    assert cfg.strict is False


def test_config_validation_recovers_known_none_defaults():
    from pydantic import BaseModel

    from axolotl.utils.config import _model_with_inherited_default_fallback

    class _Merged(BaseModel):
        base_model: str
        xformers_attention: bool | None

    cfg = _model_with_inherited_default_fallback(
        _Merged, {"base_model": "HuggingFaceTB/SmolLM2-135M"}
    )

    assert cfg.xformers_attention is None


def test_config_validation_retry_accepts_field_names():
    from pydantic import BaseModel, ConfigDict, Field

    from axolotl.utils.config import _model_with_inherited_default_fallback

    class _Merged(BaseModel):
        model_config = ConfigDict(validate_by_name=False, validate_by_alias=True)

        base_model: str
        xformers_attention: bool | None = Field(alias="xformersAttention")

    cfg = _model_with_inherited_default_fallback(
        _Merged, {"base_model": "HuggingFaceTB/SmolLM2-135M"}
    )

    assert cfg.xformers_attention is None


class TestGlmDsaContextParallelValidation:
    """The use_glm_dsa_kernels exemptions in check_context_parallel_size / validate_ring_attn_func."""

    def test_dsa_cp_skips_flash_and_ring_requirement(self, monkeypatch):
        """use_glm_dsa_kernels + context_parallel_size>1 validates WITHOUT flash attention and WITHOUT
        ring_flash_attn installed -- the DSA kernels own CP attention."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        cfg = _cfg(
            plugins=["axolotl.integrations.kernels.KernelsPlugin"],
            use_glm_dsa_kernels=True,
            context_parallel_size=2,
        )
        prepare_plugins(cfg)
        out = validate_config(cfg)  # must not raise (no flash, no ring_flash_attn)
        # ring attention is NOT substituted: ring_attn_func stays None so the SP context manager only
        # shards the sequence (register_ring_attn skips the ring_flash_attn import when it is None).
        assert out.ring_attn_func is None

    def test_cp_without_dsa_still_requires_flash(self, monkeypatch):
        """The exemption is scoped to use_glm_dsa_kernels -- plain CP still demands flash attention."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        cfg = _cfg(context_parallel_size=2)  # no DSA kernels, no flash_attention
        with pytest.raises(Exception, match="(?i)flash attention"):
            validate_config(cfg)

    def test_dsa_without_cp_leaves_ring_attn_func_none(self, monkeypatch):
        """use_glm_dsa_kernels with context_parallel_size 1 is a no-op for the CP validators."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        cfg = _cfg(
            plugins=["axolotl.integrations.kernels.KernelsPlugin"],
            use_glm_dsa_kernels=True,
        )
        prepare_plugins(cfg)
        out = validate_config(cfg)
        assert out.ring_attn_func is None

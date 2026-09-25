"""GLM owns its attention kernel while sharing Ringmaster CP configuration guards."""

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
        assert out.ring_attn_func is None

    def test_cp_without_dsa_uses_builtin_plugin(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        cfg = _cfg(context_parallel_size=2)
        out = validate_config(cfg)
        assert out.context_parallel.size == 2
        assert not out.plugins

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


@pytest.mark.parametrize("glm_dsa", [False, True])
@pytest.mark.parametrize("packing", ["sample_packing", "batch_flattening"])
def test_cp_rejects_packed_inputs_for_all_attention_owners(
    monkeypatch, glm_dsa, packing
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = _cfg(
        plugins=["axolotl.integrations.kernels.KernelsPlugin"],
        use_glm_dsa_kernels=glm_dsa,
        context_parallel_size=2,
        **{packing: True},
    )
    prepare_plugins(cfg)
    with pytest.raises(ValueError, match="sample_packing / batch_flattening"):
        validate_config(cfg)


def test_dsa_cp_uses_builtin_sharding_plugin(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = _cfg(use_glm_dsa_kernels=True, context_parallel_size=2)
    out = validate_config(cfg)
    assert out.context_parallel.size == 2
    assert not out.plugins

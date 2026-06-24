"""No-GPU regression tests for the expert_backend intent surface (I1/I2) and the DSV4
attention-LoRA guard (I7).

These need no CUDA and no triton: they only construct pydantic config models. The isolation
tests exercise KernelsArgs directly; the merged tests replicate axolotl's plugin-merge
(``class AxolotlInputConfig(AxolotlInputConfigBase, KernelsArgs)``, see
axolotl.integrations.config.merge_input_args) so the consumer validators run against the full
config where ``experts_implementation`` / ``lora_mlp_kernel`` are real fields. The merged path
is the one that was silently broken: pydantic runs same-mode ``before`` validators in reverse
definition order, so before the fix the consumers ran before expert_backend was canonicalized.
"""

import logging

import pydantic
import pytest

from axolotl.integrations.kernels.args import KernelsArgs


def _merged_cls():
    """Replicate merge_input_args() for the kernels plugin (AxolotlInputConfig + KernelsArgs)."""
    from axolotl.utils.schemas.config import AxolotlInputConfig as Base

    class _Merged(Base, KernelsArgs):
        pass

    return _Merged


def _minimal(**overrides):
    cfg = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "use_kernels": True,
    }
    cfg.update(overrides)
    return cfg


# --- I1/I2: expert_backend must canonicalize BEFORE the downstream consumers ----------------
def test_expert_backend_scattermoe_triggers_dsv4_translation_isolation():
    # The exact combo that regressed: pre-fix, disable_mlp_kernel ran before expert_backend was
    # canonicalized, so use_scattermoe was still unset and dsv4_shared_mlp_lora_kernel stayed None.
    a = KernelsArgs.model_validate(
        {
            "use_kernels": True,
            "expert_backend": "scattermoe",
            "use_dsv4_kernels": True,
            "lora_mlp_kernel": True,
        }
    )
    assert a.use_scattermoe is True
    assert a.dsv4_shared_mlp_lora_kernel is True


def test_expert_backend_matches_legacy_flags_isolation():
    intent = KernelsArgs.model_validate(
        {
            "use_kernels": True,
            "expert_backend": "scattermoe",
            "use_dsv4_kernels": True,
            "lora_mlp_kernel": True,
        }
    )
    legacy = KernelsArgs.model_validate(
        {
            "use_kernels": True,
            "use_scattermoe": True,
            "use_dsv4_kernels": True,
            "lora_mlp_kernel": True,
        }
    )
    assert (intent.use_scattermoe, intent.dsv4_shared_mlp_lora_kernel) == (
        legacy.use_scattermoe,
        legacy.dsv4_shared_mlp_lora_kernel,
    )


@pytest.mark.parametrize("backend", ["scattermoe", "sonicmoe", "eager"])
def test_merged_expert_backend_matches_legacy(backend):
    # The full I3-complete check: on the MERGED config, experts_implementation and lora_mlp_kernel
    # are real fields. The intent surface must produce a byte-identical canonical end state to the
    # equivalent legacy flag for every consumer-derived field.
    Merged = _merged_cls()
    legacy_flag = {
        "scattermoe": {"use_scattermoe": True},
        "sonicmoe": {"use_sonicmoe": True},
        "eager": {},
    }[backend]

    def snap(m):
        return (
            m.use_scattermoe,
            m.use_sonicmoe,
            getattr(m, "experts_implementation", None),
            m.dsv4_shared_mlp_lora_kernel,
            getattr(m, "lora_mlp_kernel", None),
        )

    intent = Merged.model_validate(
        _minimal(expert_backend=backend, use_dsv4_kernels=True, lora_mlp_kernel=True)
    )
    legacy = Merged.model_validate(
        _minimal(**legacy_flag, use_dsv4_kernels=True, lora_mlp_kernel=True)
    )
    assert snap(intent) == snap(legacy)


def test_merged_scattermoe_intent_sets_experts_implementation():
    # Guards the specific I3 symptom: pre-fix this left experts_implementation='eager'.
    Merged = _merged_cls()
    m = Merged.model_validate(_minimal(expert_backend="scattermoe"))
    assert m.use_scattermoe is True
    assert m.experts_implementation == "scattermoe"


# --- I7: attention/module-level LoRA on a DSV4 fused-kernel run is unsupported ---------------
def test_dsv4_attention_lora_rejected_isolation():
    with pytest.raises(pydantic.ValidationError, match="lora_target_modules"):
        KernelsArgs.model_validate(
            {
                "use_kernels": True,
                "use_dsv4_kernels": True,
                "lora_target_modules": ["q_proj"],
            }
        )


def test_dsv4_attention_lora_allowed_with_exclude_isolation(caplog):
    with caplog.at_level(logging.WARNING):
        a = KernelsArgs.model_validate(
            {
                "use_kernels": True,
                "use_dsv4_kernels": True,
                "lora_target_modules": ["q_proj"],
                "lora_exclude_modules": [".*indexer.*"],
            }
        )
    assert a.use_dsv4_kernels is True
    assert any("indexer scorer projections" in r.message for r in caplog.records)


def test_dsv4_lora_target_linear_rejected_isolation():
    # B1: lora_target_linear: true expands (find_all_linear_names) to attention q/k/v/o AFTER
    # validation, so the guard must reject it too, not only lora_target_modules.
    with pytest.raises(pydantic.ValidationError, match="lora_target_linear"):
        KernelsArgs.model_validate(
            {
                "use_kernels": True,
                "use_dsv4_kernels": True,
                "lora_target_linear": True,
            }
        )


def test_dsv4_lora_target_linear_allowed_with_exclude_isolation(caplog):
    with caplog.at_level(logging.WARNING):
        a = KernelsArgs.model_validate(
            {
                "use_kernels": True,
                "use_dsv4_kernels": True,
                "lora_target_linear": True,
                "lora_exclude_modules": [".*indexer.*"],
            }
        )
    assert a.use_dsv4_kernels is True
    assert any("indexer scorer projections" in r.message for r in caplog.records)


def test_dsv4_experts_only_allowed_isolation():
    a = KernelsArgs.model_validate(
        {
            "use_kernels": True,
            "use_dsv4_kernels": True,
            "lora_target_parameters": ["mlp.experts.gate_up_proj"],
        }
    )
    assert a.use_dsv4_kernels is True


def test_attention_lora_without_dsv4_allowed_isolation():
    # The guard must not fire when use_dsv4_kernels is off (normal attention LoRA is fine).
    a = KernelsArgs.model_validate(
        {"use_kernels": True, "lora_target_modules": ["q_proj"]}
    )
    assert a.use_dsv4_kernels is None


def test_merged_dsv4_attention_lora_rejected():
    Merged = _merged_cls()
    with pytest.raises(pydantic.ValidationError, match="lora_target_modules"):
        Merged.model_validate(
            _minimal(use_dsv4_kernels=True, lora_target_modules=["q_proj"])
        )


def test_merged_dsv4_attention_lora_allowed_with_exclude():
    Merged = _merged_cls()
    m = Merged.model_validate(
        _minimal(
            use_dsv4_kernels=True,
            lora_target_modules=["q_proj"],
            lora_exclude_modules=[".*indexer.*"],
        )
    )
    assert m.use_dsv4_kernels is True


def test_merged_dsv4_lora_target_linear_rejected():
    Merged = _merged_cls()
    with pytest.raises(pydantic.ValidationError, match="lora_target_linear"):
        Merged.model_validate(_minimal(use_dsv4_kernels=True, lora_target_linear=True))

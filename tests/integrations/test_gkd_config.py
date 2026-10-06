"""Config validation tests for the GKD plugin args."""

import pydantic
import pytest

from axolotl.integrations.gkd.args import GKDArgs


def _merged_cls():
    """Replicate merge_input_args(): AxolotlInputConfig + GKDArgs."""
    from axolotl.utils.schemas.config import AxolotlInputConfig as Base

    class _Merged(Base, GKDArgs):
        pass

    return _Merged


def _minimal(**overrides):
    cfg = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "gkd_trainer": True,
        "gkd_teacher": "HuggingFaceTB/SmolLM2-1.7B",
    }
    cfg.update(overrides)
    return cfg


def test_defaults_match_reverse_kl_fully_on_policy():
    args = GKDArgs()
    assert args.gkd_lmbda == 1.0
    assert args.gkd_beta == 1.0


def test_teacher_required_when_trainer_enabled():
    with pytest.raises(pydantic.ValidationError, match="gkd_teacher is required"):
        GKDArgs(gkd_trainer=True)


def test_no_validation_when_trainer_disabled():
    GKDArgs(gkd_trainer=False)  # gkd_teacher unset is fine


def test_merged_config_rejects_sample_packing():
    with pytest.raises(
        pydantic.ValidationError, match="incompatible with sample_packing"
    ):
        _merged_cls().model_validate(_minimal(sample_packing=True))


def test_merged_config_accepts_unpacked():
    cfg = _merged_cls().model_validate(_minimal(sample_packing=False))
    assert cfg.gkd_trainer is True
    assert cfg.gkd_teacher == "HuggingFaceTB/SmolLM2-1.7B"

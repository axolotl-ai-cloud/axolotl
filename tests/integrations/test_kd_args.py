"""
Tests for KD plugin config validation.
"""

import pytest
from pydantic import ValidationError

from axolotl.integrations.kd.args import KDArgs
from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss


class KDArgsWithDatasets(KDArgs):
    """KDArgs as it behaves once merged into the full input config."""

    datasets: list = []


def test_kd_disabled_skips_validation():
    args = KDArgs(kd_alpha=5.0)
    assert args.kd_alpha == 5.0


def test_alphas_default_to_a_usable_loss():
    args = KDArgs(kd_trainer=True)

    assert args.kd_alpha == 1.0
    assert args.kd_ce_alpha == 0.0
    assert args.kd_temperature == 1.0
    assert args.kd_normalize_topk is True
    assert args.kd_compiled_kernel is True
    assert args.kd_prepared_targets_alignment == "current"

    # the defaults must be directly usable by the loss, which pre-fix raised
    # TypeError on None weights
    LigerFusedLinearKLTopKLogprobLoss(
        args.kd_ce_alpha,
        args.kd_alpha,
        args.kd_temperature,
        args.kd_beta or 0.0,
    )


def test_kd_trainer_is_boolean():
    assert KDArgs(kd_trainer=1).kd_trainer is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kd_alpha": 1.5}, "kd_alpha"),
        ({"kd_ce_alpha": -0.1}, "kd_ce_alpha"),
        ({"kd_alpha": 0.0, "kd_ce_alpha": 0.0}, "cannot both be 0.0"),
        ({"kd_temperature": 0.0}, "kd_temperature must be positive"),
        ({"kd_beta": 2.0}, "kd_beta"),
        ({"kd_online_topk": 8}, "kd_online_server_base_url"),
        (
            {"kd_online_server_base_url": "http://teacher:8000"},
            "kd_online_topk must be a positive integer",
        ),
        (
            {"kd_online_server_base_url": "http://teacher:8000", "kd_online_topk": 0},
            "kd_online_topk must be a positive integer",
        ),
        ({"kd_temperature_min": 0.5}, "requires kd_online_server_base_url"),
        (
            {
                "kd_online_server_base_url": "http://teacher:8000",
                "kd_online_topk": 8,
                "kd_temperature": 1.0,
                "kd_temperature_min": 2.0,
            },
            "kd_temperature_min",
        ),
        ({"kd_prepared_targets_alignment": "ancient"}, "kd_prepared_targets_alignment"),
        (
            {
                "kd_online_server_base_url": "http://teacher:8000",
                "kd_online_topk": 8,
                "kd_prepared_targets_alignment": "legacy",
            },
            "online teacher produces its targets fresh",
        ),
    ],
)
def test_invalid_kd_configs_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        KDArgs(kd_trainer=True, **kwargs)


def test_online_teacher_conflicts_with_offline_logprob_dataset():
    with pytest.raises(ValidationError, match="offline KD dataset type"):
        KDArgsWithDatasets(
            kd_trainer=True,
            kd_online_server_base_url="http://teacher:8000",
            kd_online_topk=8,
            datasets=[{"path": "foo", "type": "axolotl.integrations.kd.chat_template"}],
        )


def test_online_teacher_allows_plain_datasets():
    args = KDArgsWithDatasets(
        kd_trainer=True,
        kd_online_server_base_url="http://teacher:8000",
        kd_online_topk=8,
        datasets=[{"path": "foo", "type": "chat_template"}],
    )
    assert args.kd_online_topk == 8

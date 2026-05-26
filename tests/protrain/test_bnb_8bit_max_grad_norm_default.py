"""Auto-fill `max_grad_norm: 1.0` for bnb 8-bit optimizers under ProTrain.

The bnb 8-bit moment quantization is sensitive to unbounded gradient norms.
Without `max_grad_norm` set, Mode C 9B full-FT has been observed to NaN-collapse
on NVLink hardware. The args.py validator auto-fills a conservative 1.0 default
unless the user explicitly sets the value.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.args import ProTrainArgs

_PROTRAIN_PLUGIN = "axolotl.integrations.protrain.ProTrainPlugin"


def _base_cfg(optimizer: str, **overrides):
    cfg = {
        "protrain_auto_memory": True,
        "plugins": [_PROTRAIN_PLUGIN],
        "optimizer": optimizer,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize(
    "optimizer", ["adamw_bnb_8bit", "paged_adamw_8bit", "adamw_8bit"]
)
def test_bnb_8bit_autodefaults_max_grad_norm_when_unset(optimizer):
    """No user-set max_grad_norm → validator fills 1.0."""
    data = _base_cfg(optimizer)
    ProTrainArgs._reject_unsupported_optimizer(data)
    assert data.get("max_grad_norm") == 1.0, (
        f"expected max_grad_norm autofilled to 1.0 for {optimizer}, "
        f"got {data.get('max_grad_norm')}"
    )


@pytest.mark.parametrize("explicit_value", [0.5, 1.0, 5.0, 0.0, float("inf")])
def test_bnb_8bit_respects_explicit_max_grad_norm(explicit_value):
    """User-set max_grad_norm is honored (not overwritten)."""
    data = _base_cfg("adamw_bnb_8bit", max_grad_norm=explicit_value)
    ProTrainArgs._reject_unsupported_optimizer(data)
    assert data["max_grad_norm"] == explicit_value, (
        f"user-set max_grad_norm {explicit_value} was overwritten to "
        f"{data['max_grad_norm']}"
    )


def test_bnb_8bit_treats_explicit_none_as_unset():
    """User explicitly setting None is treated as unset → autofill fires."""
    data = _base_cfg("adamw_bnb_8bit", max_grad_norm=None)
    ProTrainArgs._reject_unsupported_optimizer(data)
    # None means "no clipping" in HF Trainer, which is the unsafe path for
    # bnb 8-bit. We treat None identically to absent and autofill 1.0.
    assert data["max_grad_norm"] == 1.0


def test_non_bnb_optimizer_does_not_autofill():
    """adamw_torch / adamw_torch_fused don't need the autofill — moments are
    fp32 and not sensitive to overflow in the same way.
    """
    for optimizer in ["adamw_torch", "adamw_torch_fused", "adamw_apex_fused"]:
        data = _base_cfg(optimizer)
        ProTrainArgs._reject_unsupported_optimizer(data)
        assert "max_grad_norm" not in data, (
            f"max_grad_norm should NOT be autofilled for {optimizer}, "
            f"but found {data.get('max_grad_norm')}"
        )


def test_no_protrain_plugin_does_not_autofill():
    """When ProTrain plugin not registered, the validator is a no-op."""
    data = {
        "protrain_auto_memory": False,
        "plugins": [],
        "optimizer": "adamw_bnb_8bit",
    }
    ProTrainArgs._reject_unsupported_optimizer(data)
    assert "max_grad_norm" not in data

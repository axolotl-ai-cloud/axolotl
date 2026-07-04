"""Schema-validation tests for the top-level ``fsdp_expert_cpu_offload`` knob."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from axolotl.utils.schemas.config import AxolotlInputConfig

_BASE = dict(
    base_model="x",
    datasets=[{"path": "p", "type": "alpaca"}],
    micro_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
)


def _cfg(**kw):
    return AxolotlInputConfig(**{**_BASE, **kw})


def test_valid_fsdp2_no_global_offload():
    cfg = _cfg(
        fsdp_expert_cpu_offload=True,
        fsdp_version=2,
        fsdp_config={"version": 2},
    )
    assert cfg.fsdp_expert_cpu_offload is True


def test_requires_fsdp_config():
    with pytest.raises(ValidationError, match="requires FSDP to be enabled"):
        _cfg(fsdp_expert_cpu_offload=True, fsdp_version=2)


def test_requires_fsdp2():
    with pytest.raises(ValidationError, match="requires fsdp_version: 2"):
        _cfg(
            fsdp_expert_cpu_offload=True,
            fsdp_version=1,
            fsdp_config={"version": 1},
        )


def test_mutually_exclusive_with_global_offload():
    with pytest.raises(ValidationError, match="redundant with"):
        _cfg(
            fsdp_expert_cpu_offload=True,
            fsdp_version=2,
            fsdp_config={"version": 2, "offload_params": True},
        )


def test_unset_and_false_are_permitted():
    assert _cfg().fsdp_expert_cpu_offload is None
    assert _cfg(fsdp_expert_cpu_offload=False).fsdp_expert_cpu_offload is False

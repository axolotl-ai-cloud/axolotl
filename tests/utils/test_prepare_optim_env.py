"""Tests for `prepare_optim_env` distributed timeout wiring."""

import os

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import prepare_optim_env


@pytest.fixture(autouse=True)
def _stub_p2p(monkeypatch):
    # avoid touching CUDA in the unit test
    monkeypatch.setattr(
        "axolotl.utils.trainer.check_cuda_p2p_ib_support", lambda: True
    )


def test_ddp_timeout_sets_nccl_timeout_env(monkeypatch):
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    prepare_optim_env(DictDefault({"ddp_timeout": 7200}))
    assert os.environ["AXOLOTL_NCCL_TIMEOUT"] == "7200"


def test_ddp_timeout_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("AXOLOTL_NCCL_TIMEOUT", "999")
    prepare_optim_env(DictDefault({"ddp_timeout": 7200}))
    assert os.environ["AXOLOTL_NCCL_TIMEOUT"] == "999"


def test_no_ddp_timeout_leaves_env_unset(monkeypatch):
    monkeypatch.delenv("AXOLOTL_NCCL_TIMEOUT", raising=False)
    prepare_optim_env(DictDefault({}))
    assert "AXOLOTL_NCCL_TIMEOUT" not in os.environ

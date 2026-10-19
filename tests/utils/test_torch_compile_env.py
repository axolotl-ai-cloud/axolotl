"""Tests for the torch.compile environment setup."""

import os

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import setup_torch_compile_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION",
    ):
        # setenv first: delenv on an unset var records nothing, so writes by the code under test would leak
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_regional_compilation_defaults_off():
    setup_torch_compile_env(DictDefault(torch_compile=True))

    assert os.environ["ACCELERATE_DYNAMO_BACKEND"] == "INDUCTOR"
    assert "ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION" not in os.environ


def test_regional_compilation_can_be_enabled():
    setup_torch_compile_env(
        DictDefault(torch_compile=True, torch_compile_regional=True)
    )

    assert os.environ["ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION"] == "1"


def test_no_env_without_torch_compile():
    setup_torch_compile_env(DictDefault(torch_compile=False))

    assert "ACCELERATE_DYNAMO_BACKEND" not in os.environ
    assert "ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION" not in os.environ

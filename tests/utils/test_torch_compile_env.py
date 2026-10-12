"""Tests for the torch.compile environment setup."""

import os

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import setup_torch_compile_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ACCELERATE_DYNAMO_BACKEND", raising=False)
    monkeypatch.delenv("ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION", raising=False)


def test_regional_compilation_defaults_on():
    setup_torch_compile_env(DictDefault(torch_compile=True))

    assert os.environ["ACCELERATE_DYNAMO_BACKEND"] == "INDUCTOR"
    assert os.environ["ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION"] == "1"


def test_regional_compilation_can_be_disabled():
    setup_torch_compile_env(
        DictDefault(torch_compile=True, torch_compile_regional=False)
    )

    assert os.environ["ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION"] == "0"


def test_no_env_without_torch_compile():
    setup_torch_compile_env(DictDefault(torch_compile=False))

    assert "ACCELERATE_DYNAMO_BACKEND" not in os.environ
    assert "ACCELERATE_DYNAMO_USE_REGIONAL_COMPILATION" not in os.environ

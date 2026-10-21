"""Tests for is_deepspeed_zero3_config."""

import json

from axolotl.utils.schemas.utils import is_deepspeed_zero3_config


def test_empty_and_none_are_not_zero3():
    assert is_deepspeed_zero3_config(None) is False
    assert is_deepspeed_zero3_config("") is False
    assert is_deepspeed_zero3_config({}) is False


def test_dict_stage_is_compared_as_string():
    assert is_deepspeed_zero3_config({"zero_optimization": {"stage": 3}}) is True
    assert is_deepspeed_zero3_config({"zero_optimization": {"stage": "3"}}) is True
    assert is_deepspeed_zero3_config({"zero_optimization": {"stage": 2}}) is False
    assert is_deepspeed_zero3_config({"zero_optimization": None}) is False


def test_missing_path_falls_back_to_its_name():
    assert is_deepspeed_zero3_config("/nope/zero3_custom.json") is True
    assert is_deepspeed_zero3_config("/nope/zero2.json") is False


def test_existing_file_is_read(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    assert is_deepspeed_zero3_config(str(path)) is True
    path.write_text(json.dumps({"zero_optimization": {"stage": 1}}))
    assert is_deepspeed_zero3_config(str(path)) is False

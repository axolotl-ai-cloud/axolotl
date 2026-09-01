"""Tests for forwarding FP8 settings through ``PatchManager``."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault


def test_fp8_patch_forwards_recipe_and_all_gather(monkeypatch):
    """Pass configured FP8 values to the Trainer patch."""
    calls = {}

    trainer_patch = ModuleType("axolotl.monkeypatch.trainer_accelerator_args")
    trainer_patch.patch_create_accelerate_code_for_fp8 = (  # type: ignore[attr-defined]
        lambda **kwargs: calls.update(kwargs)
    )
    moe_filter = ModuleType("axolotl.monkeypatch.accelerate.float8_moe_filter")
    moe_filter.patch_fp8_exclude_moe_router = (  # type: ignore[attr-defined]
        lambda: calls.update(filter_called=True)
    )
    monkeypatch.setitem(sys.modules, trainer_patch.__name__, trainer_patch)
    monkeypatch.setitem(sys.modules, moe_filter.__name__, moe_filter)

    cfg = DictDefault(
        {
            "fp8": True,
            "fp8_config": {"recipe": "rowwise"},
            "fp8_enable_fsdp_float8_all_gather": False,
        }
    )
    PatchManager(cfg, MagicMock())._apply_fp8_patches()

    assert calls == {
        "enable_fsdp_float8_all_gather": False,
        "fp8_recipe": "rowwise",
        "filter_called": True,
    }

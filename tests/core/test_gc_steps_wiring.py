"""get_callbacks must register GCCallback whenever gc_collect_steps (or legacy gc_steps) is set."""

from unittest.mock import MagicMock, patch

from axolotl.core.builders import HFCausalTrainerBuilder
from axolotl.utils.callbacks import GCCallback
from axolotl.utils.dict import DictDefault


def _get_callbacks(cfg_dict):
    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.cfg = DictDefault(cfg_dict)
    builder.model = MagicMock()
    with (
        patch("axolotl.core.builders.base.PluginManager") as pm,
        patch("axolotl.core.builders.base.TelemetryManager") as tm,
    ):
        pm.get_instance.return_value.add_callbacks_pre_trainer.return_value = []
        tm.get_instance.return_value.enabled = False
        return builder.get_callbacks()


def _gc(callbacks):
    return next((c for c in callbacks if isinstance(c, GCCallback)), None)


def test_gc_collect_steps_alone_registers_callback():
    """gc_collect_steps set without the deprecated gc_steps must still register GCCallback."""
    gc = _gc(_get_callbacks({"gc_collect_steps": 5}))
    assert gc is not None
    assert gc.gc_collect_steps == 5


def test_legacy_gc_steps_still_registers_callback():
    gc = _gc(_get_callbacks({"gc_steps": 7}))
    assert gc is not None
    assert gc.gc_collect_steps == 7


def test_no_gc_config_registers_nothing():
    assert _gc(_get_callbacks({})) is None

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Pending post-trainer callbacks are mutated before the builder appends them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.integrations.base import BasePlugin, PluginManager


def test_builder_forwards_pending_list_not_trainer_handler(monkeypatch):
    captured = {}

    class FakePM:
        def mutate_callbacks_post_trainer(self, cfg, trainer, callbacks=None):
            captured["callbacks"] = callbacks
            captured["trainer"] = trainer
            return ["mutated"]

    class StubBuilder(TrainerBuilderBase):
        def build(self, total_num_steps):
            raise NotImplementedError

    monkeypatch.setattr(PluginManager, "get_instance", classmethod(lambda cls: FakePM()))
    builder = StubBuilder.__new__(StubBuilder)
    builder.cfg = SimpleNamespace(plugins=["x"])
    trainer = MagicMock()
    trainer.callback_handler.callbacks = ["already-on-trainer"]

    assert builder.mutate_callbacks_post_trainer(trainer, ["pending"]) == ["mutated"]
    assert captured["callbacks"] == ["pending"]
    assert captured["trainer"] is trainer


def test_plugin_manager_applies_non_none_mutations():
    class Keep(BasePlugin):
        def mutate_callbacks_post_trainer(self, cfg, trainer, callbacks):
            return None

    class Swap(BasePlugin):
        def mutate_callbacks_post_trainer(self, cfg, trainer, callbacks):
            return [c for c in callbacks if c != "drop"] + ["added"]

    fake = SimpleNamespace(plugins={"keep": Keep(), "swap": Swap()})
    assert PluginManager.mutate_callbacks_post_trainer(
        fake, object(), object(), ["keep", "drop"]
    ) == ["keep", "added"]

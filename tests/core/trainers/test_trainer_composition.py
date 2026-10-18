"""Composing trainer classes with core and plugin mixins at build time."""

import pytest
from transformers import Trainer

from axolotl.core.builders import HFCausalTrainerBuilder
from axolotl.core.trainers import composition
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.composition import (
    compose_trainer_cls,
    register_trainer_mixin,
    trainer_mixins_for,
)
from axolotl.integrations.base import BasePlugin, PluginManager
from axolotl.utils.dict import DictDefault


class Base:
    def describe(self):
        return ["base"]


class Outer:
    def describe(self):
        return ["outer", *super().describe()]


class Inner:
    def describe(self):
        return ["inner", *super().describe()]


class RecordingMixin(Trainer):
    """Module-level so plugins can name it by dotted path."""

    def _save_checkpoint(self, *args, **kwargs):
        self.saved_through_mixin = True
        return super()._save_checkpoint(*args, **kwargs)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(composition, "_CORE_MIXINS", [])
    monkeypatch.setattr(composition, "_COMPOSED", {})


def test_mixins_wrap_the_base_in_the_given_order():
    composed = compose_trainer_cls(Base, [Outer, Inner])

    assert issubclass(composed, Base)
    assert composed.__mro__[:4] == (composed, Outer, Inner, Base)
    assert composed().describe() == ["outer", "inner", "base"]
    assert (composed.__name__, composed.__module__) == (Base.__name__, Base.__module__)


def test_no_mixins_returns_the_base_and_composition_is_cached():
    assert compose_trainer_cls(Base, []) is Base
    first = compose_trainer_cls(Base, [Outer])
    assert compose_trainer_cls(Base, [Outer]) is first
    assert compose_trainer_cls(Base, [Inner]) is not first


def test_duplicates_and_mixins_already_in_the_base_are_skipped():
    already = compose_trainer_cls(Base, [Inner])
    composed = compose_trainer_cls(already, [Outer, Inner, Outer])

    assert composed.__mro__[1:3] == (Outer, already)
    assert compose_trainer_cls(already, [Inner]) is already


def test_conflicting_order_names_the_mixins():
    class Child(Outer):
        pass

    with pytest.raises(TypeError, match="Cannot compose Base with trainer mixins"):
        compose_trainer_cls(Base, [Outer, Child])


class _Plugin(BasePlugin):
    def __init__(self, mixins):
        super().__init__()
        self.mixins = mixins

    def get_trainer_mixins(self, cfg):
        return self.mixins


def test_core_mixins_come_first_then_plugins_in_order():
    register_trainer_mixin(Outer, when=lambda cfg: cfg.enable_outer)
    register_trainer_mixin(Inner)
    manager = PluginManager.get_instance()
    manager.plugins["first"] = _Plugin([f"{__name__}.RecordingMixin"])
    manager.plugins["second"] = _Plugin([Base])

    assert trainer_mixins_for(DictDefault(enable_outer=True)) == [
        Outer,
        Inner,
        RecordingMixin,
        Base,
    ]
    assert trainer_mixins_for(DictDefault(enable_outer=False)) == [
        Inner,
        RecordingMixin,
        Base,
    ]


def test_registering_twice_keeps_one_entry():
    register_trainer_mixin(Outer)
    register_trainer_mixin(Outer)
    assert trainer_mixins_for(DictDefault()) == [Outer]


def test_plugin_mixin_must_be_a_class():
    PluginManager.get_instance().plugins["bad"] = _Plugin([lambda: None])
    with pytest.raises(TypeError, match="is not a class"):
        trainer_mixins_for(DictDefault())


def test_builder_composes_plugin_mixins_into_the_trainer(sft_cfg, model, tokenizer):
    PluginManager.get_instance().plugins["recording"] = _Plugin([RecordingMixin])

    trainer = HFCausalTrainerBuilder(sft_cfg, model, tokenizer).build(100)

    assert isinstance(trainer, AxolotlTrainer)
    assert isinstance(trainer, RecordingMixin)
    assert type(trainer).__mro__[1] is RecordingMixin
    assert type(trainer).__mro__[2] is AxolotlTrainer


def test_builder_without_mixins_keeps_the_selected_class(sft_cfg, model, tokenizer):
    trainer = HFCausalTrainerBuilder(sft_cfg, model, tokenizer).build(100)
    assert type(trainer) is AxolotlTrainer

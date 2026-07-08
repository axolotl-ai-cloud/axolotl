"""Regression tests for the PluginManager.create_optimizer hook wiring.

`OptimizerMixin.create_optimizer` must consult `PluginManager.create_optimizer`
before falling back to the built-in optimizer construction, mirroring how
`SchedulerMixin.create_scheduler` consults `PluginManager.create_lr_scheduler`.
"""

from torch import nn
from torch.optim import SGD, Optimizer
from transformers.trainer import Trainer

from axolotl.core.trainers.mixins.optimizer import OptimizerMixin
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.integrations.base import BasePlugin, PluginManager


class _OptimizerTrainer(OptimizerMixin, Trainer):
    """Minimal Trainer subclass exercising only the optimizer mixin."""


def _build_trainer(tmp_path):
    trainer = _OptimizerTrainer.__new__(_OptimizerTrainer)
    trainer.args = AxolotlTrainingArguments(output_dir=str(tmp_path))
    trainer.model = nn.Linear(4, 4)
    trainer.optimizer = None
    trainer.optimizer_cls_and_kwargs = None
    trainer.lr_scheduler = None
    return trainer


def _register_plugin(plugin):
    PluginManager.get_instance().plugins["dummy_optimizer_plugin"] = plugin


def test_no_plugin_builds_default_optimizer(tmp_path):
    trainer = _build_trainer(tmp_path)

    optimizer = trainer.create_optimizer()

    assert isinstance(optimizer, Optimizer)
    assert optimizer is trainer.optimizer
    assert optimizer.__class__.__name__ == "AdamW"


def test_plugin_returning_none_falls_back_to_default(tmp_path):
    trainer = _build_trainer(tmp_path)

    class _NoopPlugin(BasePlugin):
        def create_optimizer(self, cfg, trainer):
            return None

    _register_plugin(_NoopPlugin())

    optimizer = trainer.create_optimizer()

    assert optimizer.__class__.__name__ == "AdamW"


def test_plugin_optimizer_instance_is_used(tmp_path):
    trainer = _build_trainer(tmp_path)
    plugin_optimizer = SGD(trainer.model.parameters(), lr=0.01)

    class _InstancePlugin(BasePlugin):
        def create_optimizer(self, cfg, trainer):
            return plugin_optimizer

    _register_plugin(_InstancePlugin())

    optimizer = trainer.create_optimizer()

    assert optimizer is plugin_optimizer
    assert trainer.optimizer is plugin_optimizer


def test_plugin_optimizer_factory_is_used(tmp_path):
    trainer = _build_trainer(tmp_path)
    built = {}

    class _Factory:
        def __call__(self, opt_model, training_args, **kwargs):
            optimizer = SGD(opt_model.parameters(), lr=0.02)
            built["optimizer"] = optimizer
            built["opt_model"] = opt_model
            built["training_args"] = training_args
            return optimizer

    class _FactoryPlugin(BasePlugin):
        def create_optimizer(self, cfg, trainer):
            return _Factory()

    _register_plugin(_FactoryPlugin())

    optimizer = trainer.create_optimizer()

    assert optimizer is built["optimizer"]
    assert trainer.optimizer is optimizer
    assert built["opt_model"] is trainer.model
    assert built["training_args"] is trainer.args

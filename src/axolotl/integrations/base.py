# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Base class for all plugins.

A plugin is a reusable, modular, and self-contained piece of code that extends the functionality of Axolotl.
Plugins can be used to integrate third-party models, modify the training process, or add new features.

To create a new plugin, you need to inherit from the BasePlugin class and implement the required methods.
"""

from __future__ import annotations

import collections
import importlib
from typing import TYPE_CHECKING, Callable, OrderedDict, Union

from peft import PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, Trainer

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__, use_environ=True)

if TYPE_CHECKING:
    from axolotl.common.datasets import TrainDatasetMeta


class BasePlugin:
    """Base class for all plugins. Defines the interface for plugin methods.

    A plugin is a reusable, modular, and self-contained piece of code that extends
    the functionality of Axolotl. Plugins can be used to integrate third-party models,
    modify the training process, or add new features.

    To create a new plugin, you need to inherit from the BasePlugin class and
    implement the required methods.

    Note:
        Plugin methods include:
        - register(cfg): Registers the plugin with the given configuration.
        - load_datasets(cfg): Loads and preprocesses the dataset for training.
        - pre_model_load(cfg): Performs actions before the model is loaded.
        - post_model_build(cfg, model): Performs actions after the model is loaded, but
            before LoRA adapters are applied.
        - pre_lora_load(cfg, model): Performs actions before LoRA weights are loaded.
        - post_lora_load(cfg, model): Performs actions after LoRA weights are loaded.
        - post_model_load(cfg, model): Performs actions after the model is loaded,
            inclusive of any adapters.
        - post_trainer_create(cfg, trainer): Performs actions after the trainer is
            created.
        - create_optimizer(cfg, trainer): Creates and returns an optimizer for training.
        - create_lr_scheduler(cfg, trainer, optimizer, num_training_steps): Creates and
            returns a learning rate scheduler.
        - add_callbacks_pre_trainer(cfg, model): Adds callbacks to the trainer before
            training.
        - add_callbacks_post_trainer(cfg, trainer): Adds callbacks to the trainer after
            training.
    """

    def __init__(self):
        """Initializes the BasePlugin."""

    def register(self, cfg: DictDefault):  # pylint: disable=unused-argument
        """Registers the plugin with the given configuration.

        Args:
            cfg: The configuration for the plugin.
        """

    def get_input_args(self) -> str | None:
        """Returns a pydantic model for the plugin's input arguments."""

    def load_datasets(
        self, cfg: DictDefault, preprocess: bool = False
    ) -> Union["TrainDatasetMeta", None]:
        """Loads and preprocesses the dataset for training.

        Args:
            cfg: The configuration for the plugin.
            preprocess: Whether this is the preprocess step of the datasets.

        Returns:
            dataset_meta: The metadata for the training dataset.
        """

    def pre_model_load(self, cfg: DictDefault):  # pylint: disable=unused-argument
        """Performs actions before the model is loaded.

        Args:
            cfg: The configuration for the plugin.
        """

    # pylint: disable=unused-argument
    def post_model_build(self, cfg: DictDefault, model: PreTrainedModel):
        """Performs actions after the model is built/loaded, but before any adapters are applied.

        Args:
            cfg: The configuration for the plugin.
        """

    # pylint: disable=unused-argument
    def pre_lora_load(self, cfg: DictDefault, model: PreTrainedModel):
        """Performs actions before LoRA weights are loaded.

        Args:
            cfg: The configuration for the plugin.
            model: The loaded model.
        """

    # pylint: disable=unused-argument
    def post_lora_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Performs actions after LoRA weights are loaded.

        Args:
            cfg: The configuration for the plugin.
            model: The loaded model.
        """

    # pylint: disable=unused-argument
    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Performs actions after the model is loaded.

        Args:
            cfg: The configuration for the plugin.
            model: The loaded model.
        """

    # pylint: disable=unused-argument
    def get_trainer_cls(self, cfg: DictDefault) -> Trainer | None:
        """Returns a custom class for the trainer.

        Args:
            cfg: The global axolotl configuration.

        Returns:
            The first non-`None` trainer class returned by a plugin.
        """

    # pylint: disable=unused-argument
    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        """Performs actions after the trainer is created.

        Args:
            cfg: The configuration for the plugin.
            trainer: The trainer object for training.
        """

    # pylint: disable=unused-argument
    def create_optimizer(self, cfg: DictDefault, trainer: Trainer) -> Optimizer | None:
        """Creates and returns an optimizer for training.

        Args:
            cfg: The configuration for the plugin.
            trainer: The trainer object for training.

        Returns:
            The created optimizer.
        """

    # pylint: disable=unused-argument
    def create_lr_scheduler(
        self,
        cfg: DictDefault,
        trainer: Trainer,
        optimizer: Optimizer,
        num_training_steps: int,
    ) -> LRScheduler | None:
        """Creates and returns a learning rate scheduler.

        Args:
            cfg: The configuration for the plugin.
            trainer: The trainer object for training.
            optimizer: The optimizer for training.
            num_training_steps: Total number of training steps

        Returns:
            The created learning rate scheduler.
        """

    # pylint: disable=unused-argument
    def add_callbacks_pre_trainer(
        self, cfg: DictDefault, model: PreTrainedModel
    ) -> list[Callable]:
        """Set up callbacks before creating the trainer.

        Args:
            cfg: The configuration for the plugin.
            model: The loaded model.

        Returns:
            A list of callback functions to be added to the `TrainingArgs`.
        """
        return []

    # pylint: disable=unused-argument
    def add_callbacks_post_trainer(
        self, cfg: DictDefault, trainer: Trainer
    ) -> list[Callable]:
        """Adds callbacks to the trainer after creating the trainer. This is useful for
        callbacks that require access to the model or trainer.

        Args:
            cfg: The configuration for the plugin.
            trainer: The trainer object for training.

        Returns:
            A list of callback functions to be added
        """
        return []

    # pylint: disable=unused-argument
    def post_train(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Performs actions after training is complete.

        Args:
            cfg: The axolotl configuration.
            model: The loaded model.
        """

    def post_train_unload(self, cfg: DictDefault):  # pylint: disable=unused-argument
        """Performs actions after training is complete and the model is unloaded.

        Args:
            cfg: The configuration for the plugin.
        """


def load_plugin(plugin_name: str) -> BasePlugin:
    """Loads a plugin based on the given plugin name.

    The plugin name should be in the format "module_name.class_name". This function
    splits the plugin name into module and class, imports the module, retrieves the
    class from the module, and creates an instance of the class.

    Args:
        plugin_name: The name of the plugin to be loaded. The name should be in the
            format "module_name.class_name".

    Returns:
        An instance of the loaded plugin.

    Raises:
        ImportError: If the plugin module cannot be imported.
    """
    # split the plugin name into module and class
    module_name, class_name = plugin_name.rsplit(".", 1)

    # import the module
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as orig_exc:
        try:
            if not module_name.startswith("axolotl.integrations."):
                module = importlib.import_module("axolotl.integrations." + module_name)
            else:
                raise orig_exc
        except ModuleNotFoundError as exc:
            raise orig_exc from exc

    # instantiate the class
    plugin_class = getattr(module, class_name)
    # create an instance of the class
    plugin = plugin_class()

    return plugin


class PluginManager:
    """The `PluginManager` class is responsible for loading and managing plugins. It
    should be a singleton so it can be accessed from anywhere in the codebase.

    Attributes:
        plugins: A list of loaded plugins.

    Note:
        Key methods include:
        - get_instance(): Static method to get the singleton instance of `PluginManager`.
        - register(plugin_name: str): Registers a new plugin by its name.
        - pre_model_load(cfg): Calls the pre_model_load method of all registered plugins.
    """

    plugins: OrderedDict[str, BasePlugin] = collections.OrderedDict()

    _instance: PluginManager | None = None
    _cfg: DictDefault | None = None

    def __new__(cls):
        """Creates a new instance of PluginManager if it doesn't exist yet."""
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance.plugins: OrderedDict[str, BasePlugin] = (
                collections.OrderedDict()
            )
        return cls._instance

    @staticmethod
    def get_instance() -> "PluginManager":
        """Returns the singleton instance of PluginManager. If the instance doesn't
        exist, it creates a new one.
        """
        if PluginManager._instance is None:
            PluginManager()
        return PluginManager._instance  # type: ignore

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg

    def register(self, plugin_name: str):
        """Registers a new plugin by its name.

        Args:
            plugin_name: The name of the plugin to be registered.

        Raises:
            ImportError: If the plugin module cannot be imported.
        """
        try:
            LOG.info(f"Attempting to load plugin: {plugin_name}")
            plugin = load_plugin(plugin_name)
            self.plugins[plugin_name] = plugin
            LOG.info(f"Plugin loaded successfully: {plugin_name}")
        except ImportError:
            LOG.error(f"Failed to load plugin: {plugin_name}")

    def get_input_args(self) -> list[str]:
        """Returns a list of Pydantic classes for all registered plugins' input arguments.'

        Returns:
            A list of Pydantic classes for all registered plugins' input arguments.'
        """
        input_args = []
        for plugin in self.plugins.values():
            input_args_from_plugin = plugin.get_input_args()
            if input_args_from_plugin is not None:
                input_args.append(input_args_from_plugin)
        return input_args

    def load_datasets(
        self, cfg: DictDefault, preprocess: bool = False
    ) -> Union["TrainDatasetMeta", None]:
        """Calls the load_datasets method of each registered plugin.

        Args:
            cfg: The configuration for the plugins.
            preprocess: Whether this is preprocess step of the datasets.

        Returns:
            The dataset metadata loaded from all registered plugins.
        """
        return_ds_meta = None
        for plugin in self.plugins.values():
            dataset_meta = plugin.load_datasets(cfg, preprocess)
            if dataset_meta is not None:
                if return_ds_meta is None:
                    return_ds_meta = dataset_meta
                else:
                    raise RuntimeError("Multiple plugins loaded datasets")
        return return_ds_meta

    def pre_model_load(self, cfg: DictDefault):
        """Calls the pre_model_load method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
        """
        for plugin in self.plugins.values():
            plugin.pre_model_load(cfg)

    def post_model_build(self, cfg: DictDefault, model: PreTrainedModel):
        """Calls the `post_model_build` method of all registered plugins after the
        model has been built / loaded, but before any adapters have been applied.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.
        """
        for plugin in self.plugins.values():
            plugin.post_model_build(cfg, model)

    def pre_lora_load(self, cfg: DictDefault, model: PreTrainedModel):
        """Calls the `pre_lora_load` method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.
        """
        for plugin in self.plugins.values():
            plugin.pre_lora_load(cfg, model)

    def post_lora_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Calls the `post_lora_load` method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.
        """
        for plugin in self.plugins.values():
            plugin.post_lora_load(cfg, model)

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Calls the `post_model_load` method of all registered plugins after the model
        has been loaded inclusive of any adapters.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.
        """
        for plugin in self.plugins.values():
            plugin.post_model_load(cfg, model)

    def get_trainer_cls(self, cfg: DictDefault) -> Trainer | None:
        """Calls the `get_trainer_cls` method of all registered plugins and returns the
        first non-`None` trainer class.

        Args:
            cfg: The configuration for the plugins.

        Returns:
            The first non-`None` trainer class returned by a plugin.
        """
        for plugin in self.plugins.values():
            trainer_cls = plugin.get_trainer_cls(cfg)
            if trainer_cls is not None:
                return trainer_cls
        return None

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        """Calls the `post_trainer_create` method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            trainer: The trainer object for training.
        """
        for plugin in self.plugins.values():
            plugin.post_trainer_create(cfg, trainer)

    def create_optimizer(self, trainer: Trainer) -> Optimizer | None:
        """Calls the `create_optimizer` method of all registered plugins and returns
        the first non-`None` optimizer.

        Args:
            trainer: The trainer object for training.

        Returns:
            The created optimizer, or `None` if none was found.
        """
        for plugin in self.plugins.values():
            optimizer = plugin.create_optimizer(self.cfg, trainer)
            if optimizer is not None:
                return optimizer
        return None

    def create_lr_scheduler(
        self, trainer: Trainer, optimizer: Optimizer, num_training_steps: int
    ) -> LRScheduler | None:
        """Calls the `create_lr_scheduler` method of all registered plugins and returns
        the first non-`None` scheduler.

        Args:
            trainer: The trainer object for training.
            optimizer: The optimizer for training.

        Returns:
            The created learning rate scheduler, or `None` if not found.
        """
        for plugin in self.plugins.values():
            scheduler: LRScheduler | None = plugin.create_lr_scheduler(
                self.cfg,
                trainer=trainer,
                optimizer=optimizer,
                num_training_steps=num_training_steps,
            )
            if scheduler is not None:
                return scheduler
        return None

    def add_callbacks_pre_trainer(
        self, cfg: DictDefault, model: PreTrainedModel
    ) -> list[Callable]:
        """Calls the add_callbacks_pre_trainer method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.

        Returns:
            A list of callback functions to be added to the `TrainingArgs`.
        """
        callbacks = []
        for plugin in self.plugins.values():
            plugin_callbacks = plugin.add_callbacks_pre_trainer(cfg, model)
            if plugin_callbacks:  # if the plugin returned a list of callbacks
                callbacks.extend(plugin_callbacks)
        return callbacks

    def add_callbacks_post_trainer(
        self, cfg: DictDefault, trainer: Trainer
    ) -> list[Callable]:
        """Calls the `add_callbacks_post_trainer` method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            trainer: The trainer object for training.

        Returns:
            A list of callback functions to be added to the `TrainingArgs`.
        """
        callbacks = []
        for plugin in self.plugins.values():
            plugin_callbacks = plugin.add_callbacks_post_trainer(cfg, trainer)
            if plugin_callbacks:
                callbacks.extend(plugin_callbacks)
        return callbacks

    def post_train(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Calls the post_train method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
            model: The loaded model.
        """
        for plugin in self.plugins.values():
            plugin.post_train(cfg, model)

    def post_train_unload(self, cfg: DictDefault):
        """Calls the post_train_unload method of all registered plugins.

        Args:
            cfg: The configuration for the plugins.
        """
        for plugin in self.plugins.values():
            plugin.post_train_unload(cfg)


class BaseOptimizerFactory:
    """Base class for factories to create custom optimizers"""

    def __call__(
        self, opt_model, training_args, **optimizer_kwargs
    ) -> Optimizer | None:
        pass

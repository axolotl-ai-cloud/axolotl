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

"""
Base class for all plugins.

A plugin is a reusable, modular, and self-contained piece of code that extends the functionality of Axolotl.
Plugins can be used to integrate third-party models, modify the training process, or add new features.

To create a new plugin, you need to inherit from the BasePlugin class and implement the required methods.
"""
import importlib
import logging
from typing import List


class BasePlugin:
    """
    Base class for all plugins. Defines the interface for plugin methods.

    Attributes:
    None

    Methods:
    register(cfg): Registers the plugin with the given configuration.
    pre_model_load(cfg): Performs actions before the model is loaded.
    post_model_load(cfg, model): Performs actions after the model is loaded.
    pre_lora_load(cfg, model): Performs actions before LoRA weights are loaded.
    post_lora_load(cfg, model): Performs actions after LoRA weights are loaded.
    create_optimizer(cfg, trainer): Creates and returns an optimizer for training.
    create_lr_scheduler(cfg, trainer, optimizer): Creates and returns a learning rate scheduler.
    add_callbacks_pre_trainer(cfg, model): Adds callbacks to the trainer before training.
    add_callbacks_post_trainer(cfg, trainer): Adds callbacks to the trainer after training.
    """

    def __init__(self):
        """
        Initializes the BasePlugin.
        """

    def register(self, cfg):
        """
        Registers the plugin with the given configuration.

        Parameters:
        cfg (dict): The configuration for the plugin.

        Returns:
        None
        """

    def get_input_args(self):
        """
        Returns a pydantic model for the plugin's input arguments.
        """

    def pre_model_load(self, cfg):
        """
        Performs actions before the model is loaded.

        Parameters:
        cfg (dict): The configuration for the plugin.

        Returns:
        None
        """

    def post_model_load(self, cfg, model):
        """
        Performs actions after the model is loaded.

        Parameters:
        cfg (dict): The configuration for the plugin.
        model (object): The loaded model.

        Returns:
        None
        """

    def pre_lora_load(self, cfg, model):
        """
        Performs actions before LoRA weights are loaded.

        Parameters:
        cfg (dict): The configuration for the plugin.
        model (object): The loaded model.

        Returns:
        None
        """

    def post_lora_load(self, cfg, model):
        """
        Performs actions after LoRA weights are loaded.

        Parameters:
        cfg (dict): The configuration for the plugin.
        model (object): The loaded model.

        Returns:
        None
        """

    def create_optimizer(self, cfg, trainer):
        """
        Creates and returns an optimizer for training.

        Parameters:
        cfg (dict): The configuration for the plugin.
        trainer (object): The trainer object for training.

        Returns:
        object: The created optimizer.
        """

    def create_lr_scheduler(self, cfg, trainer, optimizer):
        """
        Creates and returns a learning rate scheduler.

        Parameters:
        cfg (dict): The configuration for the plugin.
        trainer (object): The trainer object for training.
        optimizer (object): The optimizer for training.

        Returns:
        object: The created learning rate scheduler.
        """

    def add_callbacks_pre_trainer(self, cfg, model):
        """
        Adds callbacks to the trainer before training.

        Parameters:
        cfg (dict): The configuration for the plugin.
        model (object): The loaded model.

        Returns:
        List[callable]: A list of callback functions to be added to the TrainingArgs
        """

    def add_callbacks_post_trainer(self, cfg, trainer):
        """
        Adds callbacks to the trainer after training.

        Parameters:
        cfg (dict): The configuration for the plugin.
        trainer (object): The trainer object for training.

        Returns:
        List[callable]: A list of callback functions to be added to the TrainingArgs
        """


def load_plugin(plugin_name: str) -> BasePlugin:
    """
    Loads a plugin based on the given plugin name.

    The plugin name should be in the format "module_name.class_name".
    This function splits the plugin name into module and class, imports the module,
    retrieves the class from the module, and creates an instance of the class.

    Parameters:
    plugin_name (str): The name of the plugin to be loaded. The name should be in the format "module_name.class_name".

    Returns:
    BasePlugin: An instance of the loaded plugin.

    Raises:
    ImportError: If the plugin module cannot be imported.
    """
    # split the plugin name into module and class
    module_name, class_name = plugin_name.rsplit(".", 1)

    # import the module
    module = importlib.import_module(module_name)
    # instantiate the class
    plugin_class = getattr(module, class_name)
    # create an instance of the class
    plugin = plugin_class()

    return plugin


class PluginManager:
    """
    The PluginManager class is responsible for loading and managing plugins.
    It should be a singleton so it can be accessed from anywhere in the codebase.

    Attributes:
    plugins (List[BasePlugin]): A list of loaded plugins.

    Methods:
    get_instance(): Static method to get the singleton instance of PluginManager.
    register(plugin_name: str): Registers a new plugin by its name.
    pre_model_load(cfg): Calls the pre_model_load method of all registered plugins.
    """

    plugins: List[BasePlugin] = []

    _instance = None

    def __new__(cls):
        """
        Creates a new instance of PluginManager if it doesn't exist yet.
        """
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance.plugins: List[BasePlugin] = []
        return cls._instance

    @staticmethod
    def get_instance() -> "PluginManager":
        """
        Returns the singleton instance of PluginManager.
        If the instance doesn't exist, it creates a new one.
        """
        if PluginManager._instance is None:
            PluginManager()
        return PluginManager._instance  # type: ignore

    def register(self, plugin_name: str):
        """
        Registers a new plugin by its name.

        Parameters:
        plugin_name (str): The name of the plugin to be registered.

        Returns:
        None

        Raises:
        ImportError: If the plugin module cannot be imported.
        """
        try:
            plugin = load_plugin(plugin_name)
            self.plugins.append(plugin)
        except ImportError:
            logging.error(f"Failed to load plugin: {plugin_name}")

    def get_input_args(self):
        """
        Returns a list of Pydantic classes for all registered plugins' input arguments.'

        Returns:
        list[str]: A list of Pydantic classes for all registered plugins' input arguments.'
        """
        input_args = []
        for plugin in self.plugins:
            input_args_from_plugin = plugin.get_input_args()
            if input_args_from_plugin is not None:
                input_args.append(input_args_from_plugin)
        return input_args

    def pre_model_load(self, cfg):
        """
        Calls the pre_model_load method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.

        Returns:
        None
        """
        for plugin in self.plugins:
            plugin.pre_model_load(cfg)

    def post_model_load(self, cfg, model):
        """
        Calls the post_model_load method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.
        model (object): The loaded model.

        Returns:
        None
        """
        for plugin in self.plugins:
            plugin.post_model_load(cfg, model)

    def pre_lora_load(self, cfg, model):
        """
        Calls the pre_lora_load method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.
        model (object): The loaded model.

        Returns:
        None
        """
        for plugin in self.plugins:
            plugin.pre_lora_load(cfg, model)

    def post_lora_load(self, cfg, model):
        """
        Calls the post_lora_load method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.
        model (object): The loaded model.

        Returns:
        None
        """
        for plugin in self.plugins:
            plugin.post_lora_load(cfg, model)

    def create_optimizer(self, cfg, trainer):
        """
        Calls the create_optimizer method of all registered plugins and returns the first non-None optimizer.

        Parameters:
        cfg (dict): The configuration for the plugins.
        trainer (object): The trainer object for training.

        Returns:
        object: The created optimizer, or None if none was found.
        """
        for plugin in self.plugins:
            optimizer = plugin.create_optimizer(cfg, trainer)
            if optimizer is not None:
                return optimizer
        return None

    def create_lr_scheduler(self, cfg, trainer, optimizer):
        """
        Calls the create_lr_scheduler method of all registered plugins and returns the first non-None scheduler.

        Parameters:
        cfg (dict): The configuration for the plugins.
        trainer (object): The trainer object for training.
        optimizer (object): The optimizer for training.

        Returns:
        object: The created learning rate scheduler, or None if none was found.
        """
        for plugin in self.plugins:
            scheduler = plugin.create_lr_scheduler(cfg, trainer, optimizer)
            if scheduler is not None:
                return scheduler
        return None

    def add_callbacks_pre_trainer(self, cfg, model):
        """
        Calls the add_callbacks_pre_trainer method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.
        model (object): The loaded model.

        Returns:
        List[callable]: A list of callback functions to be added to the TrainingArgs.
        """
        callbacks = []
        for plugin in self.plugins:
            callbacks.extend(plugin.add_callbacks_pre_trainer(cfg, model))
        return callbacks

    def add_callbacks_post_trainer(self, cfg, trainer):
        """
        Calls the add_callbacks_post_trainer method of all registered plugins.

        Parameters:
        cfg (dict): The configuration for the plugins.
        trainer (object): The trainer object for training.

        Returns:
        List[callable]: A list of callback functions to be added to the TrainingArgs.
        """
        callbacks = []
        for plugin in self.plugins:
            callbacks.extend(plugin.add_callbacks_post_trainer(cfg, trainer))
        return callbacks

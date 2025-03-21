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
Module to handle merging the plugins' input arguments with the base configurations.

This was moved here to prevent circular imports.
"""

from typing import Any, Dict, List

from axolotl.utils.schemas.config import (
    AxolotlConfigWCapabilities as AxolotlConfigWCapabilitiesBase,
)
from axolotl.utils.schemas.config import AxolotlInputConfig as AxolotlInputConfigBase


def merge_input_args():
    """
    Merges input arguments from registered plugins with the base configurations.

    This function retrieves the input arguments from registered plugins using the PluginManager.
    It then dynamically creates new classes, AxolotlConfigWCapabilities and AxolotlInputConfig,
    that inherit from the base configurations and include the input arguments from the plugins.

    Returns:
    tuple: A tuple containing the newly created classes, AxolotlConfigWCapabilities and AxolotlInputConfig.
    """
    from axolotl.integrations.base import PluginManager

    plugin_manager = PluginManager.get_instance()
    input_args: List[str] = plugin_manager.get_input_args()
    plugin_classes = []
    dynamic_input = ""
    for plugin_args in input_args:
        plugin_module, plugin_cls = plugin_args.rsplit(".", 1)
        dynamic_input += f"from {plugin_module} import {plugin_cls}\n"
        plugin_classes.append(plugin_cls)
    if dynamic_input:
        dynamic_input += f"class AxolotlConfigWCapabilities(AxolotlConfigWCapabilitiesBase, {', '.join(plugin_classes)}):\n    pass\n"
        dynamic_input += f"class AxolotlInputConfig(AxolotlInputConfigBase, {', '.join(plugin_classes)}):\n    pass\n"

        namespace: Dict[Any, Any] = {}
        exec(  # pylint: disable=exec-used  # nosec B102
            dynamic_input, globals(), namespace
        )
        AxolotlInputConfig = namespace[  # pylint: disable=invalid-name
            "AxolotlInputConfig"
        ]
        AxolotlConfigWCapabilities = namespace[  # pylint: disable=invalid-name
            "AxolotlConfigWCapabilities"
        ]
        return AxolotlConfigWCapabilities, AxolotlInputConfig
    return AxolotlConfigWCapabilitiesBase, AxolotlInputConfigBase

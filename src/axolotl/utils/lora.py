# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
module to get the state dict of a merged lora model
"""
import torch
from peft.tuners.tuners_utils import onload_layer
from peft.utils import ModulesToSaveWrapper, _get_submodules


def get_lora_merged_state_dict(
    model: torch.nn.Module,
) -> dict:
    r"""
    Create and return a state_dict that has the LoRA deltas
    merged into the base model’s weights, without modifying `model` in place.

    Arguments:
        model (torch.nn.Module): A model that has LoRA/PEFT adapters attached.

    Returns:
        dict: A state_dict of the merged parameters.
    """

    base_model_prefix = "base_model.model."
    state_dict = {}
    key_list = [key for key, _ in model.named_modules() if model.prefix not in key]
    for key in key_list:
        try:
            _, target, _ = _get_submodules(model, key)
        except AttributeError:
            continue
        with onload_layer(target):
            weight_key = key.replace(base_model_prefix, "") + ".weight"
            bias_key = key.replace(base_model_prefix, "") + ".bias"
            if hasattr(target, "base_layer"):
                target.merge(safe_merge=True, adapter_names=None)
                # get the state_dict of target.base_layer
                layer_state_dict = target.base_layer.state_dict()
                state_dict[weight_key] = layer_state_dict["weight"]
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    new_module.merge(safe_merge=True, adapter_names=None)
                layer_state_dict = new_module.state_dict()
                state_dict[weight_key] = layer_state_dict["weight"]
            elif hasattr(target, "weight"):
                if any(
                    skip in key
                    for skip in [
                        ".original_module",
                        ".modules_to_save",
                        ".base_layer",
                    ]
                ):
                    continue
                layer_state_dict = target.state_dict()
                state_dict[weight_key] = layer_state_dict["weight"]
                if hasattr(target, "bias") and "bias" in layer_state_dict.keys():
                    state_dict[bias_key] = layer_state_dict["bias"]
    return state_dict

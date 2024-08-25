# Copyright 2024 Axolotl AI. All rights reserved.
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
Spectrum Plugin to automatically generate unfrozen parameters based on SNR data.
"""

import json
import logging

import requests

from axolotl.integrations.base import BasePlugin

from .args import SpectrumArgs  # pylint: disable=unused-import. # noqa: F401


def _generate_unfrozen_params_yaml(snr_data, top_fraction=0.5):
    unfrozen_parameters = {}
    for layer_name, info in snr_data.items():
        layer_type = info["type"]
        if layer_type not in unfrozen_parameters:
            unfrozen_parameters[layer_type] = []
        unfrozen_parameters[layer_type].append((layer_name, info["snr"]))
    top_layers_by_type = {}
    for layer_type, layers in unfrozen_parameters.items():
        layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
        num_top_layers = int(len(layers) * top_fraction)
        top_layers_by_type[layer_type] = [
            layer[0] for layer in layers_sorted[:num_top_layers]
        ]
    unfrozen_parameters = [
        "^lm_head.weight$",
        "^model.embed_tokens.weight$",
    ]
    for layer_type, layer_names in top_layers_by_type.items():
        for layer_name in layer_names:
            unfrozen_parameters.append(layer_name)
    return unfrozen_parameters


class SpectrumPlugin(BasePlugin):
    """
    Spectrum Plugin to automatically generate unfrozen parameters based on SNR data.
    """

    base_url = "https://raw.githubusercontent.com/cognitivecomputations/spectrum/main/model_snr_results/"
    base_path = "./model_snr_results/"
    snr_file_template = "snr_results_{model_name_slug}.json"

    def get_input_args(self):
        return "axolotl.integrations.spectrum.SpectrumArgs"

    def pre_model_load(self, cfg):
        if cfg.get("spectrum_model_name"):
            model_name = cfg["spectrum_model_name"]
        else:
            model_name = cfg["base_model"]
        top_fraction = cfg.get("spectrum_top_fraction", 50)
        model_slug = model_name.replace("/", "-").replace("_", "-")
        snr_url = self.base_url + self.snr_file_template.format(
            model_name_slug=model_slug
        )
        snr_path = self.base_path + self.snr_file_template.format(
            model_name_slug=model_slug
        )
        # first check if the files exist locally and read the json
        snr_data = None
        try:
            with open(snr_path, "r", encoding="utf-8") as fin:
                snr_data = json.load(fin)
        except FileNotFoundError:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning(f"Failed to read SNR data from {snr_path}: {exc}")

        if not snr_data:
            try:
                snr_data = requests.get(snr_url, timeout=60).json()
            except requests.exceptions.RequestException as exc:
                logging.warning(f"Failed to fetch SNR data from {snr_url}: {exc}")
                return
            # also catch json parsing errors
            except json.JSONDecodeError as exc:
                logging.warning(f"Failed to parse SNR data from {snr_url}: {exc}")
                return

        unfrozen_parameters = _generate_unfrozen_params_yaml(
            snr_data, top_fraction=top_fraction
        )
        cfg["unfrozen_parameters"] = unfrozen_parameters

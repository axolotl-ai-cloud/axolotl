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
Plugin init to add KD support to Axolotl.
"""
from axolotl.integrations.base import BasePlugin

from .args import KDArgs  # pylint: disable=unused-import. # noqa: F401


class KDPlugin(BasePlugin):
    """
    Plugin for KD support in Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.kd.KDArgs"

    def get_trainer_cls(self, cfg):
        if cfg.kd_trainer:
            from .trainer import AxolotlKDTrainer

            return AxolotlKDTrainer
        return None

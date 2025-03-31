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
Module for the Plugin for Cut Cross Entropy integration with Axolotl.

Cut Cross Entropy is an optimized implementation of cross entropy loss
from Apple's ML team.
"""
import importlib
import logging

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.utils import get_pytorch_version
from axolotl.utils.distributed import zero_only

from .args import CutCrossEntropyArgs  # pylint: disable=unused-import. # noqa: F401

LOG = logging.getLogger("axolotl.integrations.cut_cross_entropy")

_CCE_INSTALL_MESSAGE = (
    "Please install cut_cross_entropy with transformers support using "
    '`pip install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git@24fbe4b5dab9a6c250a014573613c1890190536c"`'
)


class CutCrossEntropyPlugin(BasePlugin):
    """
    Plugin for Cut Cross Entropy integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.cut_cross_entropy.CutCrossEntropyArgs"

    def _check_requirements(self):
        """Check if all requirements are met."""
        # Check PyTorch version

        major, minor, _ = get_pytorch_version()
        if (major, minor) < (2, 4):
            raise ImportError(
                "Cut Cross Entropy requires PyTorch >= 2.4.0. "
                f"Current version: {torch.__version__}"
            )

        # Check if cut_cross_entropy is installed
        cce_spec = importlib.util.find_spec("cut_cross_entropy")
        if cce_spec is None:
            raise ImportError(_CCE_INSTALL_MESSAGE)

        cce_spec_transformers = importlib.util.find_spec(
            "cut_cross_entropy.transformers"
        )
        if cce_spec_transformers is None:
            raise ImportError(_CCE_INSTALL_MESSAGE)

    def pre_model_load(self, cfg):
        """Apply cut cross entropy before model loading if enabled."""
        if cfg.cut_cross_entropy:
            self._check_requirements()

            from axolotl.integrations.cut_cross_entropy.monkeypatch.patch import (
                cce_patch,
            )

            with zero_only():
                LOG.info(
                    f"Applying Cut Cross Entropy to model type: {cfg.model_config_type}"
                )

            # The patch checks model_type internally
            cce_patch(cfg.model_config_type)

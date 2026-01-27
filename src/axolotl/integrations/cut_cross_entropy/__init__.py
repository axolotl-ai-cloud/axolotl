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
from functools import partial

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.utils import get_pytorch_version
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

from .args import CutCrossEntropyArgs as CutCrossEntropyArgs

LOG = get_logger(__name__)

_CCE_INSTALL_MESSAGE = (
    "Please install Axolotl's fork of cut_cross_entropy with transformers support using "
    '`pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f4b5712"`'
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
            raise ImportError(
                "Transformers support is not installed. " + _CCE_INSTALL_MESSAGE
            )

        # Check if Axolotl's cce fork is installed
        try:
            from cut_cross_entropy.transformers.patch import AXOLOTL_CCE_FORK

            if not AXOLOTL_CCE_FORK:
                raise ImportError
        except ImportError as e:
            raise ImportError(
                "Axolotl's fork of cut_cross_entropy is not installed. "
                + _CCE_INSTALL_MESSAGE
            ) from e

    def pre_model_load(self, cfg):
        """Apply cut cross entropy before model loading if enabled."""
        if cfg.cut_cross_entropy:
            self._check_requirements()
            self.patch_llama_like(cfg.model_config_type)

            from cut_cross_entropy.transformers.patch import cce_patch

            LOG.info(
                f"Applying Cut Cross Entropy to model type: {cfg.model_config_type}"
            )

            # The patch checks model_type internally

            cce_patch(
                cfg.model_config_type,
                remote_model_id=cfg.base_model if cfg.trust_remote_code else None,
            )

    def patch_llama_like(
        self,
        model_type: str,
    ) -> None:
        """
        Generic patch for model architectures with causal lm similar to llama
        """
        from cut_cross_entropy.transformers.patch import PATCH_FNS

        def patch_generic(
            maybe_model, patch_options, model_type: str, remote_model_id: str | None
        ):
            import cut_cross_entropy.transformers.llama
            from cut_cross_entropy.transformers.llama import cce_forward

            try:
                # Dynamically import the module and CausalLM class
                module_path = f"transformers.models.{model_type}.modeling_{model_type}"
                model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
                module = __import__(
                    module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"]
                )
                model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")

                cut_cross_entropy.transformers.llama._PATCH_OPTS = patch_options

                model_cls.forward = cce_forward

            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    f"Could not import ForCausalLM class for model_type: {model_type}. "
                    f"Error: {str(e)}"
                ) from e

        if model_type not in PATCH_FNS:
            LOG.warning_once(
                "Setting up generic cce patch for model type: %s", model_type
            )
            LOG.warning_once(
                f"Generic Cut Cross Entropy + {model_type} support is experimental and may not work as expected."
            )
            PATCH_FNS[model_type] = partial(patch_generic, model_type=model_type)

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
import inspect
from functools import partial

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.model_support import check_capability, get_model_support
from axolotl.utils import get_pytorch_version
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

from .args import CutCrossEntropyArgs as CutCrossEntropyArgs

LOG = get_logger(__name__)

_CCE_INSTALL_MESSAGE = (
    "Please install Axolotl's fork of cut_cross_entropy with transformers support using "
    '`pip uninstall -y cut-cross-entropy && pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@latest"`'
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
            check_capability(
                get_model_support(cfg.model_config_type),
                "cut_cross_entropy",
                cfg.model_config_type,
                hint="Disable cut_cross_entropy for this model.",
            )
            self._check_requirements()
            self.patch_llama_like(cfg.model_config_type)

            from cut_cross_entropy.transformers.patch import cce_patch

            LOG.info(
                f"Applying Cut Cross Entropy to model type: {cfg.model_config_type}"
            )

            patch_kwargs = {
                "remote_model_id": cfg.base_model if cfg.trust_remote_code else None,
                "accum_c_fp32": bool(cfg.cut_cross_entropy_accum_c_fp32),
            }
            if cfg.cut_cross_entropy_c_grad_chunk_size:
                if "c_grad_chunk_size" not in inspect.signature(cce_patch).parameters:
                    raise ImportError(
                        "The installed cut_cross_entropy does not support "
                        "`cut_cross_entropy_c_grad_chunk_size`. " + _CCE_INSTALL_MESSAGE
                    )
                patch_kwargs["c_grad_chunk_size"] = self._resolve_c_grad_chunk_size(cfg)

            # The patch checks model_type internally
            cce_patch(cfg.model_config_type, **patch_kwargs)

    def _resolve_c_grad_chunk_size(self, cfg) -> int:
        chunk = cfg.cut_cross_entropy_c_grad_chunk_size
        if chunk != "auto":
            return int(chunk)

        try:
            from cut_cross_entropy import recommend_c_grad_chunk_size
        except ImportError as e:
            raise ImportError(
                "The installed cut_cross_entropy cannot recommend a chunk size. "
                + _CCE_INSTALL_MESSAGE
            ) from e
        if not torch.cuda.is_available():
            raise ValueError(
                "`cut_cross_entropy_c_grad_chunk_size: auto` needs a CUDA device to "
                "size the chunk; set an explicit multiple of 128 instead."
            )

        from axolotl.loaders.utils import load_model_config

        model_config = load_model_config(cfg)
        if hasattr(model_config, "get_text_config"):
            model_config = model_config.get_text_config()

        # tokens the loss sees per local microbatch; CP splits the sequence across ranks
        num_tokens = cfg.micro_batch_size * (
            cfg.sequence_len // max(cfg.context_parallel_size or 1, 1)
        )
        chunk = recommend_c_grad_chunk_size(
            num_tokens=num_tokens,
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            device=torch.cuda.current_device(),
        )
        LOG.info(
            "Cut Cross Entropy classifier-gradient chunk size resolved to %d "
            "(0 keeps the full fp32 accumulator)",
            chunk,
        )
        return chunk

    def patch_llama_like(
        self,
        model_type_to_patch: str,
    ) -> None:
        """
        Generic patch for model architectures with causal lm similar to llama
        """
        from cut_cross_entropy.transformers.patch import PATCH_FNS

        def patch_generic(
            maybe_model,
            patch_options,
            remote_model_id: str | None,
            model_type: str,
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

        if model_type_to_patch not in PATCH_FNS:
            LOG.warning_once(
                "Setting up generic cce patch for model type: %s", model_type_to_patch
            )
            LOG.warning_once(
                f"Generic Cut Cross Entropy + {model_type_to_patch} support is experimental and may not work as expected."
            )
            PATCH_FNS[model_type_to_patch] = partial(
                patch_generic, model_type=model_type_to_patch
            )

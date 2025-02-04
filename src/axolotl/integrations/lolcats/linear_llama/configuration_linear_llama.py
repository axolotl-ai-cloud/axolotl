# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Linear LLaMA model configuration"""

from transformers import LlamaConfig


class LinearLlamaConfig(LlamaConfig):
    """
    This is the configuration class to store the configuration of a [`LinearLlamaModel`].
    It is a modified LlamaConfig that includes additional parameters for linear attention.

    Args:
        attention_config (`dict`):
            Dictionary containing the configuration for linear attention mechanism.
            Expected contents:
                `feature_map` (`str`):
                    The type of feature map to use for linear attention.
                `feature_map_kwargs` (`dict`):
                    Additional arguments for the feature map.
                `learned_kernel` (`str`, *optional*):
                    Type of learned kernel to use, if any.
                `learned_kernel_kwargs` (`dict`, *optional*):
                    Additional arguments for the learned kernel.
                `tie_qk_kernels` (`bool`, *optional*, defaults to False):
                    Whether to tie query and key kernels.
                `rotary_config` (`dict`, *optional*):
                    Configuration for rotary embeddings.
                `train_attention` (`bool`, *optional*, defaults to False):
                    Whether to train attention to match softmax attention.
                `remove_base_attn` (`bool`, *optional*, defaults to True):
                    Whether to remove base attention after initialization.
                `mask_value` (`int`, *optional*, defaults to 0):
                    Value to use for masking.
                `eps` (`float`, *optional*, defaults to 1e-12):
                    Epsilon value for numerical stability.
                `fp32_attention` (`bool`, *optional*, defaults to False):
                    Whether to use fp32 precision for attention computation.
                `track_state_grads` (`bool`, *optional*, defaults to False):
                    Whether to track gradients of attention states.

        **kwargs:
            Additional arguments inherited from LlamaConfig.
    """

    model_type = "linear_llama"

    def __init__(self, attention_config: dict, **kwargs):
        super().__init__(**kwargs)

        # Set default attention config if none provided
        self.attention_config = attention_config

    @classmethod
    def from_llama(cls, llama_config: LlamaConfig, attention_config: dict):
        """
        Instantiate a LinearLlamaConfig from a LlamaConfig and additional attention config.

        Args:
            llama_config (:class:`~transformers.LlamaConfig`):
                The LlamaConfig to inherit from.

            attention_config (`dict`):
                Dictionary containing the configuration for linear attention mechanism.
        """

        return cls(attention_config=attention_config, **llama_config.to_dict())

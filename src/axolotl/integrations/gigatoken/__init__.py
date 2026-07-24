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
Plugin for gigatoken integration with Axolotl.

gigatoken is a fast CPU tokenizer. This plugin attaches a gigatoken-accelerated,
HF-compatible encoder to the tokenizer, which the pretraining/completion path
uses to speed up raw-text tokenization. It does not replace the tokenizer, so
chat-template and other prompt strategies are unaffected.
"""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .args import GigatokenArgs as GigatokenArgs

LOG = get_logger(__name__)


class GigatokenPlugin(BasePlugin):
    """Plugin for gigatoken integration with Axolotl."""

    def get_input_args(self):
        return "axolotl.integrations.gigatoken.GigatokenArgs"

    def post_tokenizer_load(self, cfg, tokenizer):
        if not cfg.gigatoken:
            return None

        import gigatoken as gt

        tokenizer._gigatoken_encoder = gt.Tokenizer(tokenizer).as_hf()
        LOG.info("gigatoken encoder attached for raw-text tokenization")
        return None

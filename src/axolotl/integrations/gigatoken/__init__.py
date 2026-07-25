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
from axolotl.utils.tokenization import set_fast_encoder

from .args import GigatokenArgs as GigatokenArgs

LOG = get_logger(__name__)

_PARITY_SAMPLES = (
    "The quick brown fox jumps over the lazy dog.",
    "héllo 世界 🎉 — ünïcode",
    "  leading and trailing whitespace \t\n",
    "1234567890 !@#$%^&*()[]{}",
)


def _check_parity(tokenizer, encoder) -> None:
    """Reject an encoder that disagrees with the tokenizer it claims to replace.

    Divergence here silently corrupts every token the run trains on, and gigatoken
    ignores unsupported arguments rather than raising, so it can't be caught later.
    """
    samples = list(_PARITY_SAMPLES)
    expected = tokenizer(samples, add_special_tokens=True)["input_ids"]
    actual = encoder(samples, add_special_tokens=True)["input_ids"]

    for text, want, got in zip(samples, expected, actual, strict=True):
        if list(want) != list(got):
            raise RuntimeError(
                f"gigatoken tokenized {text!r} as {list(got)}, but the HuggingFace "
                f"tokenizer produced {list(want)}. Set `gigatoken: false` to train "
                "with the HuggingFace tokenizer instead."
            )


class GigatokenPlugin(BasePlugin):
    """Plugin for gigatoken integration with Axolotl."""

    def get_input_args(self):
        return "axolotl.integrations.gigatoken.GigatokenArgs"

    def post_tokenizer_load(self, cfg, tokenizer):
        if not cfg.gigatoken:
            return None

        import gigatoken as gt

        try:
            encoder = gt.Tokenizer(tokenizer).as_hf()
        except Exception as exc:
            raise RuntimeError(
                "gigatoken could not wrap "
                f"{getattr(tokenizer, 'name_or_path', tokenizer)!r}: {exc}. Its byte "
                "remapping does not support every vocabulary; set `gigatoken: false` "
                "to train with the HuggingFace tokenizer instead."
            ) from exc

        _check_parity(tokenizer, encoder)
        set_fast_encoder(tokenizer, encoder)
        LOG.info("gigatoken encoder attached for raw-text tokenization")
        return None

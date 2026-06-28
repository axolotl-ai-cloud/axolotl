"""
Tests for auto-detection of the actual EOT token for Gemma-like models
where tokenizer.eos_token != the chat template turn terminator.

Covers issue #3754: chat_template strategy never trains the Gemma turn
terminator because <eos> (id=1) is used as the EOT token, but Gemma
templates emit <end_of_turn> (id=107) / <turn|> (id=106).
"""
from unittest.mock import MagicMock

import pytest

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy


# Gemma 2 chat template - uses <end_of_turn>, NOT eos_token variable
GEMMA2_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"
)

# A template that DOES use eos_token variable (e.g., ChatML, Llama3 style)
LLAMA3_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
    "{{ message['content'] | trim + eos_token }}"
    "{% endfor %}"
)

_EOT_ID = 107  # <end_of_turn> Gemma token id
_EOS_ID = 1    # <eos> Gemma real eos_token id
_SENTINEL_IDS = [300, 400, 500]


def _make_gemma_mock_tokenizer():
    """Build a minimal mock tokenizer that mimics Gemma 2 token behavior."""
    tok = MagicMock()
    tok.eos_token = "<eos>"
    tok.eos_token_id = _EOS_ID

    def _encode(text, add_special_tokens=False):
        if "SENTINEL_XYZ_DO_NOT_USE" in str(text):
            return list(_SENTINEL_IDS)
        if text == "<end_of_turn>":
            return [_EOT_ID]
        if text == "<eos>":
            return [_EOS_ID]
        return [999]

    def _decode(token_ids, skip_special_tokens=False):
        mapping = {_EOT_ID: "<end_of_turn>", 108: "\n", _EOS_ID: "<eos>"}
        if len(token_ids) == 1 and token_ids[0] in mapping:
            return mapping[token_ids[0]]
        return ""

    def _apply_chat_template(conv, tokenize=True, add_generation_prompt=False, **kwargs):
        # Simulate:
        # <bos>(2) <start_of_turn>(106) user(100) \n(108) x(150) <end_of_turn>(107) \n(108)
        # <start_of_turn>(106) model(200) \n(108) SENTINEL(300,400,500) <end_of_turn>(107) \n(108)
        return [2, 106, 100, 108, 150, _EOT_ID, 108,
                106, 200, 108, 300, 400, 500, _EOT_ID, 108]

    tok.encode = _encode
    tok.decode = _decode
    tok.apply_chat_template = _apply_chat_template
    return tok


def _make_prompter(chat_template: str):
    prompter = MagicMock()
    prompter.chat_template = chat_template
    prompter.roles = {"user": "user", "assistant": "assistant"}
    prompter.chat_template_msg_variables = {"role", "content"}
    return prompter


class TestGemmaEotAutoDetection:
    """Auto-detection of EOT token when eos_token is absent from chat template."""

    def test_eot_not_found_before_fix(self):
        """
        RED: Without the fix, _eot_token_ids is {1} (<eos>), which never
        appears in Gemma renders, so the turn terminator is never trained.
        After the fix, _eot_token_ids should contain 107 (<end_of_turn>).
        This test asserts the FIXED behaviour; failing it means the fix
        is absent.
        """
        tok = _make_gemma_mock_tokenizer()
        prompter = _make_prompter(GEMMA2_TEMPLATE)

        strategy = ChatTemplateStrategy(
            prompter=prompter,
            tokenizer=tok,
            train_on_inputs=False,
            sequence_len=512,
        )

        # After fix: the wrong <eos> id must NOT be cached as an EOT id
        assert _EOS_ID not in strategy._eot_token_ids, (
            "Bug still present: <eos> (id=1) is used as EOT but never appears "
            "in Gemma renders, so the turn terminator is never trained."
        )

    def test_gemma_eot_auto_detected(self):
        """
        GREEN: After the fix, <end_of_turn> (id=107) is auto-detected and
        stored in _eot_token_ids, so find_first_eot_token will label it.
        """
        tok = _make_gemma_mock_tokenizer()
        prompter = _make_prompter(GEMMA2_TEMPLATE)

        strategy = ChatTemplateStrategy(
            prompter=prompter,
            tokenizer=tok,
            train_on_inputs=False,
            sequence_len=512,
        )

        assert _EOT_ID in strategy._eot_token_ids, (
            "<end_of_turn> (id=107) was not auto-detected as an EOT token. "
            "The Gemma turn terminator will not be trained."
        )
        assert "<end_of_turn>" in strategy.eot_tokens

    def test_explicit_eot_tokens_not_overridden(self):
        """
        When the caller explicitly passes eot_tokens, auto-detection must
        NOT override them.
        """
        tok = _make_gemma_mock_tokenizer()
        prompter = _make_prompter(GEMMA2_TEMPLATE)
        # The template has <end_of_turn> so it will pass validation
        prompter.chat_template = GEMMA2_TEMPLATE.replace(
            "<end_of_turn>", "<custom_eot>"
        )
        tok.encode = lambda text, **kw: [555] if text == "<custom_eot>" else [999]

        strategy = ChatTemplateStrategy(
            prompter=prompter,
            tokenizer=tok,
            train_on_inputs=False,
            sequence_len=512,
            eot_tokens=["<custom_eot>"],
        )

        assert strategy.eot_tokens == ["<custom_eot>"]

    def test_normal_template_with_eos_token_var_unchanged(self):
        """
        Templates that reference eos_token (Llama3 / ChatML style) should
        remain unaffected; no auto-detection should fire.
        """
        tok = MagicMock()
        tok.eos_token = "<|eot_id|>"
        tok.eos_token_id = 128009
        tok.encode = MagicMock(return_value=[128009])
        prompter = _make_prompter(LLAMA3_STYLE_TEMPLATE)

        strategy = ChatTemplateStrategy(
            prompter=prompter,
            tokenizer=tok,
            train_on_inputs=False,
            sequence_len=512,
        )

        # <|eot_id|> IS in the template via `eos_token`, so no auto-detection;
        # the original eot_tokens = ['<|eot_id|>'] should be preserved.
        assert strategy.eot_tokens == ["<|eot_id|>"]
        assert 128009 in strategy._eot_token_ids

    def test_find_first_eot_correctly_locates_gemma_terminator(self):
        """
        End-to-end: find_first_eot_token must return the correct index when
        the rendered sequence contains <end_of_turn> (id=107).
        """
        tok = _make_gemma_mock_tokenizer()
        prompter = _make_prompter(GEMMA2_TEMPLATE)

        strategy = ChatTemplateStrategy(
            prompter=prompter,
            tokenizer=tok,
            train_on_inputs=False,
            sequence_len=512,
        )

        # Simulated rendered assistant turn: content at [0..2], EOT at [3]
        fake_ids = [300, 400, 500, _EOT_ID, 108]
        idx = strategy.find_first_eot_token(fake_ids, start_idx=0)

        assert idx == 3, (
            f"Expected EOT at index 3, got {idx}. "
            "<end_of_turn> (id=107) is not being searched for."
        )

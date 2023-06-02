"""Module testing prompters"""

import unittest

from rathe import AlpacaPromptFormatter, InstructPrompt


class AlpacaPrompterTest(unittest.TestCase):
    """
    Test AlpacaPrompter
    """

    def test_prompt_style_w_none(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt("tell me a joke"), special_tokens={}
        ).to_string()
        # just testing that it uses instruct style
        assert "### Instruction:" in res

    def test_prompt_style_w_instruct(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt("tell me a joke about the following", input="alpacas"),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" in res
        assert "alpacas" in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res
        res = formatter.format(
            InstructPrompt("tell me a joke about the following"), special_tokens={}
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" not in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res

    def test_prompt_style_w_chat(self):
        formatter = AlpacaPromptFormatter()
        res = formatter.format(
            InstructPrompt(
                "tell me a joke about the following", input="alpacas"
            ).as_chat(),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "alpacas" in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res
        res = formatter.format(
            InstructPrompt("tell me a joke about the following").as_chat(),
            special_tokens={},
        ).to_string()
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res
